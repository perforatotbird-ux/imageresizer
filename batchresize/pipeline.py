"""Pipelined batch processing: discover -> read -> decode -> resize -> encode -> write.

Every stage runs in its own thread(s) and is connected by bounded queues, so a
slow disk never starves the GPU and the GPU never stalls waiting for one file.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .cpu_backend import CpuEngine, decode_to_rgb
from .geometry import Plan, compute_box, make_plan
from .imageio_meta import (SourceMeta, has_alpha_hint, is_truncated_jpeg,
                           read_source_meta, splice_metadata)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class RunOptions:
    input_dir: Path
    output_dir: Path
    max_w: int
    max_h: int
    mode: str = "contain"
    quality: int = 80
    subsampling: int = 2
    sizing: str = "resolution"
    sizing_value: float = 0.0
    allow_upscale: bool = False
    workers: int = 8
    batch_size: int = 8
    queue_depth: int = 32
    skip_existing: bool = False
    overwrite: bool = True
    limit: Optional[int] = None
    keep_exif: bool = True
    keep_icc: bool = True
    bg_color: tuple[int, int, int] = (255, 255, 255)
    vram_budget_gb: float = 2.0
    filter_name: str = "lanczos3"
    fallback: bool = True
    allow_truncated: bool = False
    optimize_huffman: bool = False
    fsync: bool = False
    verbose: bool = False


@dataclass
class Job:
    src: Path
    out: Path
    data: bytes
    meta: SourceMeta
    plan: Plan
    has_alpha: bool = False


@dataclass
class Result:
    job: Job
    jpeg: bytes
    out_w: int
    out_h: int
    backend: str
    timings: dict = field(default_factory=dict)


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.processed = 0
        self.failed = 0
        self.skipped = 0
        self.fallback = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self.t_read = 0.0
        self.t_decode = 0.0
        self.t_resize = 0.0
        self.t_encode = 0.0
        self.t_write = 0.0
        self.errors: list[tuple[str, str]] = []

    def error(self, path: Path, exc: BaseException) -> None:
        with self.lock:
            self.failed += 1
            if len(self.errors) < 50:
                self.errors.append((str(path), f"{type(exc).__name__}: {exc}"))

    def add_read(self, seconds: float) -> None:
        with self.lock:
            self.t_read += seconds

    def add(self, res: Result, write_s: float) -> None:
        with self.lock:
            self.processed += 1
            self.bytes_in += len(res.job.data)
            self.bytes_out += len(res.jpeg)
            self.t_decode += res.timings.get("decode", 0.0)
            self.t_resize += res.timings.get("resize", 0.0)
            self.t_encode += res.timings.get("encode", 0.0)
            self.t_write += write_s

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "processed": self.processed,
                "failed": self.failed,
                "skipped": self.skipped,
                "fallback": self.fallback,
                "bytes_in": self.bytes_in,
                "bytes_out": self.bytes_out,
                "t_read": self.t_read,
                "t_decode": self.t_decode,
                "t_resize": self.t_resize,
                "t_encode": self.t_encode,
                "t_write": self.t_write,
                "errors": list(self.errors),
            }


def discover(root: Path, exclude: Optional[Path] = None,
             limit: Optional[int] = None) -> list[Path]:
    """Recursive scan, skipping the output tree. Sorted for determinism."""
    root = Path(root)
    exclude = Path(exclude).resolve() if exclude else None
    found: list[Path] = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            sub = Path(entry.path)
                            if exclude is not None and sub.resolve() == exclude:
                                continue
                            stack.append(sub)
                        elif entry.is_file(follow_symlinks=False):
                            if Path(entry.name).suffix.lower() in SUPPORTED_EXTS:
                                found.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue
    found.sort()
    if limit:
        found = found[:limit]
    return found


def build_output_map(files: Iterable[Path], root: Path,
                     out_root: Path) -> dict[Path, Path]:
    """Map sources to targets, disambiguating name clashes (photo.png/photo.jpg)."""
    used: dict[str, Path] = {}
    mapping: dict[Path, Path] = {}
    for src in files:
        rel = src.relative_to(root)
        target = (out_root / rel).with_suffix(".jpg")
        key = str(target).lower()
        if key in used and used[key] != src:
            i = 2
            while True:
                cand = target.with_name(f"{target.stem}__{i}.jpg")
                if str(cand).lower() not in used:
                    target = cand
                    break
                i += 1
            key = str(target).lower()
        used[key] = src
        mapping[src] = target
    return mapping


_MADE_DIRS: set[str] = set()
_MADE_LOCK = threading.Lock()


def atomic_write(path: Path, data: bytes, fsync: bool = False) -> None:
    """Write to a temp file then rename - never leaves a half-written result."""
    parent = str(path.parent)
    if parent not in _MADE_DIRS:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _MADE_LOCK:
            _MADE_DIRS.add(parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


_EMPTY = object()
_EOF = object()


class Pipeline:
    """Threaded stages connected by bounded queues.

    readers  --jobs-->  processor(s)  --results-->  writers
    """

    def __init__(self, opts: RunOptions, engine, backend: str,
                 progress: bool = True, on_progress=None) -> None:
        self.opts = opts
        self.engine = engine
        self.backend = backend
        self.stats = Stats()
        self.stop = threading.Event()
        self.on_progress = on_progress
        self.jobs: queue.Queue = queue.Queue(maxsize=max(1, opts.queue_depth))
        self.results: queue.Queue = queue.Queue(maxsize=max(1, opts.queue_depth))
        self.discovered = 0
        self.t_start = 0.0
        self._progress_enabled = progress
        self._last_progress = 0.0

        if backend == "cpu":
            self._n_readers = max(1, opts.workers)
            self._n_processors = max(1, opts.workers)
        else:
            self._n_readers = max(1, opts.workers)
            self._n_processors = 1
        self._n_writers = max(2, max(1, opts.workers) // 2)

        self._readers: list[threading.Thread] = []
        self._processors: list[threading.Thread] = []
        self._writers: list[threading.Thread] = []
        self._jobs_eof = 0
        self._eof_lock = threading.Lock()
        self._iter_lock = threading.Lock()
        self._deferred: Optional[Job] = None
        self._deferred_lock = threading.Lock()
        self._sentinels_sent = False
        self._out_map: dict[Path, Path] = {}
        self._file_iter = iter(())

    # -- queue helpers -------------------------------------------------------
    def _put_job(self, job: Job) -> bool:
        while True:
            try:
                self.jobs.put(job, timeout=0.1)
                return True
            except queue.Full:
                if self.stop.is_set():
                    return False

    def _take_job(self, block: bool = True):
        with self._deferred_lock:
            if self._deferred is not None:
                job, self._deferred = self._deferred, None
                return job
        if self.stop.is_set():
            return _EOF
        with self._eof_lock:
            producers_done = self._jobs_eof >= self._n_readers
        if producers_done and self.jobs.empty():
            return _EOF
        try:
            return self.jobs.get(timeout=0.1) if block else self.jobs.get_nowait()
        except queue.Empty:
            return _EMPTY

    def _send_writer_sentinels(self) -> None:
        if self._sentinels_sent:
            return
        self._sentinels_sent = True
        for _ in range(self._n_writers):
            self.results.put(None)

    # -- stages --------------------------------------------------------------
    def _reader(self) -> None:
        try:
            while not self.stop.is_set():
                with self._iter_lock:
                    try:
                        src = next(self._file_iter)
                    except StopIteration:
                        src = None
                if src is None:
                    break
                out = self._out_map[src]
                if (self.opts.skip_existing and out.exists()
                        and not self.opts.overwrite):
                    with self.stats.lock:
                        self.stats.skipped += 1
                    continue
                t0 = time.perf_counter()
                try:
                    data = src.read_bytes()
                    self.stats.add_read(time.perf_counter() - t0)
                    suffix = src.suffix.lower()
                    if (suffix in (".jpg", ".jpeg")
                            and not self.opts.allow_truncated):
                        if is_truncated_jpeg(data):
                            raise ValueError("truncated JPEG (missing EOI marker)")
                    meta = read_source_meta(data)
                    box_w, box_h = compute_box(
                        meta.display_width, meta.display_height,
                        self.opts.sizing, self.opts.max_w, self.opts.max_h,
                        self.opts.sizing_value)
                    plan = make_plan(meta.display_width, meta.display_height,
                                     box_w, box_h,
                                     self.opts.mode, self.opts.allow_upscale)
                    job = Job(src, out, data, meta, plan,
                              has_alpha_hint(data, suffix))
                except Exception as exc:
                    self.stats.error(src, exc)
                    continue
                if not self._put_job(job):
                    break
        finally:
            with self._eof_lock:
                self._jobs_eof += 1

    def _cpu_processor(self) -> None:
        while True:
            item = self._take_job(block=True)
            if item is _EOF:
                break
            if item is _EMPTY:
                continue
            self._process_cpu_one(item)

    def _gpu_processor(self) -> None:
        batch_size = max(1, self.opts.batch_size)
        # Decode (RGB) + float work buffer + output: ~8 bytes per source pixel.
        budget_px = max(1, int(self.opts.vram_budget_gb * (2 ** 30) / 8))
        while True:
            item = self._take_job(block=True)
            if item is _EOF:
                break
            if item is _EMPTY:
                continue
            batch = [item]
            pixels = item.meta.display_width * item.meta.display_height
            while len(batch) < batch_size:
                nxt = self._take_job(block=False)
                if nxt is _EOF or nxt is _EMPTY:
                    break
                npix = nxt.meta.display_width * nxt.meta.display_height
                if batch and pixels + npix > budget_px:
                    with self._deferred_lock:
                        self._deferred = nxt
                    break
                batch.append(nxt)
                pixels += npix
            if self.backend == "hybrid":
                for job in batch:
                    self._process_hybrid_one(job)
            else:
                self._process_gpu_batch(batch)

    def _writer(self) -> None:
        while True:
            item = self.results.get()
            if item is None:
                return
            res: Result = item
            t0 = time.perf_counter()
            try:
                atomic_write(res.job.out, res.jpeg, self.opts.fsync)
            except Exception as exc:
                self.stats.error(res.job.src, exc)
                continue
            self.stats.add(res, time.perf_counter() - t0)

    # -- processing ----------------------------------------------------------
    def _finish(self, job: Job, jpeg: bytes, timings: dict,
                backend: str) -> None:
        if self.opts.keep_exif or self.opts.keep_icc:
            try:
                jpeg = splice_metadata(
                    jpeg,
                    job.meta.exif if self.opts.keep_exif else None,
                    job.meta.icc if self.opts.keep_icc else None,
                )
            except Exception:
                pass
        res = Result(job, jpeg, job.plan.out_w, job.plan.out_h,
                     backend, timings)
        self.results.put(res)

    def _process_cpu_one(self, job: Job) -> None:
        engine = self.engine
        try:
            t0 = time.perf_counter()
            img = engine.decode(job.data)
            t1 = time.perf_counter()
            if img.shape[2] == 4 or job.has_alpha:
                img = engine.composite_bg(img, self.opts.bg_color)
            img = engine.apply_orientation(img, job.meta.orientation)
            t2 = time.perf_counter()
            small = engine.resize(img, job.plan)
            t3 = time.perf_counter()
            jpeg = engine.encode(small, self.opts.quality, self.opts.subsampling)
            t4 = time.perf_counter()
            timings = {"decode": t1 - t0, "resize": t3 - t2,
                       "encode": t4 - t3}
            self._finish(job, jpeg, timings, "cpu")
        except Exception as exc:
            self.stats.error(job.src, exc)

    def _process_gpu_batch(self, batch: list[Job]) -> None:
        if not batch:
            return
        if len(batch) == 1:
            self._process_gpu_one(batch[0])
            return
        try:
            self._process_gpu_batch_inner(batch)
        except Exception as exc:
            msg = str(exc).lower()
            if "memory" in msg or "alloc" in msg:
                mid = len(batch) // 2
                self._process_gpu_batch(batch[:mid])
                self._process_gpu_batch(batch[mid:])
            else:
                for job in batch:
                    self._process_gpu_one(job)

    def _process_gpu_batch_inner(self, batch: list[Job]) -> None:
        engine = self.engine
        t0 = time.perf_counter()
        arrs = engine.decode_batch([j.data for j in batch],
                                   [j.has_alpha for j in batch])
        t1 = time.perf_counter()
        per_file = (t1 - t0) / max(1, len(batch))
        for job, arr in zip(batch, arrs):
            if arr is None:
                raise RuntimeError("GPU decode failed")
            try:
                t2 = time.perf_counter()
                small = engine.resize(arr, job.plan)
                t3 = time.perf_counter()
                jpeg = engine.encode(small, self.opts.quality,
                                     self.opts.subsampling)
                t4 = time.perf_counter()
                timings = {"decode": per_file, "resize": t3 - t2,
                           "encode": t4 - t3}
                self._finish(job, jpeg, timings, "gpu")
            except Exception as exc:
                self._gpu_fallback(job, exc)

    def _process_gpu_one(self, job: Job) -> None:
        try:
            self._process_gpu_batch_inner([job])
        except Exception as exc:
            self._gpu_fallback(job, exc)

    def _process_hybrid_one(self, job: Job) -> None:
        """Host decode (Pillow) -> VRAM -> CUDA resize -> nvJPEG encode."""
        engine = self.engine
        try:
            t0 = time.perf_counter()
            rgb, has_alpha = decode_to_rgb(job.data, self.opts.bg_color)
            t1 = time.perf_counter()
            arr = engine.to_device(rgb)
            arr = engine.apply_orientation(arr, job.meta.orientation)
            t2 = time.perf_counter()
            if int(arr.shape[2]) == 4:
                arr = engine.alpha_to_rgb(arr, self.opts.bg_color)
            small = engine.resize(arr, job.plan)
            t3 = time.perf_counter()
            jpeg = engine.encode(small, self.opts.quality,
                                 self.opts.subsampling)
            t4 = time.perf_counter()
            timings = {"decode": t1 - t0, "resize": t3 - t2,
                       "encode": t4 - t3}
            self._finish(job, jpeg, timings, "hybrid")
        except Exception as exc:
            self.stats.error(job.src, exc)

    def _gpu_fallback(self, job: Job, exc: BaseException) -> None:
        if not self.opts.fallback:
            self.stats.error(job.src, exc)
            return
        try:
            t0 = time.perf_counter()
            rgb, has_alpha = decode_to_rgb(job.data, self.opts.bg_color)
            t1 = time.perf_counter()
            if self.backend == "hybrid":
                arr = self.engine.to_device(rgb)
                arr = self.engine.apply_orientation(arr, job.meta.orientation)
            else:
                arr = self.engine.to_device(rgb)
                if job.meta.orientation != 1:
                    arr = self.engine.apply_orientation(
                        arr, job.meta.orientation)
            if int(arr.shape[2]) == 4:
                arr = self.engine.alpha_to_rgb(arr, self.opts.bg_color)
            t2 = time.perf_counter()
            small = self.engine.resize(arr, job.plan)
            t3 = time.perf_counter()
            jpeg = self.engine.encode(small, self.opts.quality,
                                      self.opts.subsampling)
            t4 = time.perf_counter()
            with self.stats.lock:
                self.stats.fallback += 1
            timings = {"decode": t1 - t0, "resize": t3 - t2,
                       "encode": t4 - t3}
            self._finish(job, jpeg, timings, "fallback")
        except Exception as exc2:
            self.stats.error(job.src, exc2)

    # -- orchestration -------------------------------------------------------
    def _progress(self, t_start: float, final: bool = False) -> None:
        if not self._progress_enabled:
            return
        now = time.perf_counter()
        if not final and now - self._last_progress < 0.25:
            return
        self._last_progress = now
        snap = self.stats.snapshot()
        elapsed = max(1e-9, now - t_start)
        rate = snap["processed"] / elapsed
        done = snap["processed"] + snap["failed"] + snap["skipped"]
        total = self.discovered
        if self.on_progress is not None:
            try:
                self.on_progress({
                    "done": done,
                    "total": total,
                    "processed": snap["processed"],
                    "failed": snap["failed"],
                    "skipped": snap["skipped"],
                    "elapsed": elapsed,
                    "rate": rate,
                    "final": final,
                })
            except Exception:
                pass
            return
        line = (f"\r{done}/{total} files | {rate:.1f} file/s | "
                f"{elapsed:.1f}s | threads: "
                f"readers={self._n_readers} "
                f"processors={self._n_processors} "
                f"writers={self._n_writers}")
        import sys
        sys.stdout.write(line + ("\n" if final else ""))
        sys.stdout.flush()

    def _wait(self, threads: list[threading.Thread], t_start: float) -> None:
        workers = self._readers + self._processors
        while any(t.is_alive() for t in workers):
            self._progress(t_start)
            time.sleep(0.1)
        for t in workers:
            t.join()
        self._send_writer_sentinels()
        for t in self._writers:
            t.join()
        self._progress(t_start, final=True)

    def _shutdown(self) -> None:
        self.stop.set()
        for t in self._readers:
            t.join(timeout=10)
        for t in self._processors:
            t.join()
        self._send_writer_sentinels()
        for t in self._writers:
            t.join()

    def run(self) -> dict:
        opts = self.opts
        self.t_start = time.perf_counter()
        files = discover(opts.input_dir, opts.output_dir, opts.limit)
        self.discovered = len(files)
        self._out_map = build_output_map(files, opts.input_dir,
                                         opts.output_dir)
        self._file_iter = iter(files)
        interrupted = False

        if files:
            for i in range(self._n_readers):
                t = threading.Thread(target=self._reader,
                                     name=f"reader-{i}", daemon=True)
                self._readers.append(t)
            target = (self._cpu_processor if self.backend == "cpu"
                      else self._gpu_processor)
            for i in range(self._n_processors):
                t = threading.Thread(target=target,
                                     name=f"processor-{i}", daemon=True)
                self._processors.append(t)
            for i in range(self._n_writers):
                t = threading.Thread(target=self._writer,
                                     name=f"writer-{i}", daemon=True)
                self._writers.append(t)

            threads = self._readers + self._processors + self._writers
            for t in threads:
                t.start()
            try:
                self._wait(threads, self.t_start)
            except KeyboardInterrupt:
                interrupted = True
                self.stop.set()
                self._shutdown()
            finally:
                for t in threads:
                    if t.is_alive():
                        t.join(timeout=5)

        return self._report(interrupted)

    def _report(self, interrupted: bool) -> dict:
        elapsed = max(1e-9, time.perf_counter() - self.t_start)
        snap = self.stats.snapshot()
        processed = snap["processed"]
        per = (lambda v: round(1000.0 * v / processed, 3)) if processed else (lambda v: 0.0)
        return {
            "input": str(self.opts.input_dir),
            "output": str(self.opts.output_dir),
            "backend": self.backend,
            "engine": self.backend,
            "discovered": self.discovered,
            "processed": processed,
            "failed": snap["failed"],
            "skipped": snap["skipped"],
            "fallback_files": snap["fallback"],
            "bytes_in": snap["bytes_in"],
            "bytes_out": snap["bytes_out"],
            "elapsed_sec": round(elapsed, 3),
            "files_per_sec": round(processed / elapsed, 2),
            "mb_per_sec_in": round(snap["bytes_in"] / 2 ** 20 / elapsed, 2),
            "threads": {
                "readers": self._n_readers,
                "processors": self._n_processors,
                "writers": self._n_writers,
                "total": self._n_readers + self._n_processors + self._n_writers,
            },
            "timings_avg_ms": {
                "read": per(snap["t_read"]),
                "decode": per(snap["t_decode"]),
                "resize": per(snap["t_resize"]),
                "encode": per(snap["t_encode"]),
                "write": per(snap["t_write"]),
            },
            "interrupted": interrupted,
            "errors": snap["errors"],
        }
