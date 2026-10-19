"""Pipelined batch processing: discover -> read -> decode -> resize -> encode -> write.

Every stage runs in its own thread(s) and is connected by bounded queues, so a
slow disk never starves the GPU and the GPU never stalls waiting for one file.
"""

from __future__ import annotations

import json
import hashlib
import math
from contextlib import contextmanager
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .cpu_backend import CpuEngine, decode_to_rgb
from .validation import dimensions, positive
from .safety import (atomic_write, check_roots, path_key, protect_target,
                     source_identities, ResumeState)
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
    max_source_bytes: int = 128 * 2**20
    max_queued_bytes: int = 256 * 2**20
    max_working_bytes: int = 1024 * 2**20
    profile_cuda: bool = False
    cancel_file: Optional[Path] = None

    def __post_init__(self):
        self.input_dir = Path(self.input_dir).resolve()
        self.output_dir = Path(self.output_dir).absolute()
        dimensions(self.max_w, self.max_h)
        if self.mode not in ("contain", "cover") or self.sizing not in (
                "resolution", "percent", "megapixels", "longest", "shortest"):
            raise ValueError("unknown sizing/mode")
        if self.sizing != "resolution":
            positive(self.sizing_value, "sizing value")
        positive(self.vram_budget_gb, "VRAM budget")
        for value, name, cap in [(self.workers, "workers", 64),
                (self.batch_size, "batch size", 128), (self.queue_depth, "queue depth", 256)]:
            positive(value, name)
            if int(value) != value or value > cap:
                raise ValueError(f"{name} must be an integer <= {cap}")
        if self.limit is not None and (self.limit < 1 or int(self.limit) != self.limit):
            raise ValueError("limit must be a positive integer")
        if (isinstance(self.quality, bool) or int(self.quality) != self.quality or
                not 1 <= self.quality <= 100 or isinstance(self.subsampling, bool) or
                self.subsampling not in (0, 1, 2)):
            raise ValueError("invalid JPEG quality/subsampling")
        if self.filter_name not in ("box", "bilinear", "lanczos3"):
            raise ValueError("unknown filter")
        if len(self.bg_color) != 3 or any(int(x) != x or not 0 <= x <= 255 for x in self.bg_color):
            raise ValueError("background must contain three bytes")
        for name in ("max_source_bytes", "max_queued_bytes", "max_working_bytes"):
            value = positive(getattr(self, name), name)
            if int(value) != value:
                raise ValueError(f"{name} must be an integer")
        if self.max_source_bytes > self.max_queued_bytes:
            raise ValueError("source byte limit exceeds queued byte budget")


@dataclass
class Job:
    src: Path
    out: Path
    data: bytes
    meta: SourceMeta
    plan: Plan
    has_alpha: bool = False
    digest: str = ""


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
        self.read_count = 0
        self.backend_counts = {}
        self.extra_timings = {}

    def error(self, path: Path, exc: BaseException) -> None:
        with self.lock:
            self.failed += 1
            if len(self.errors) < 50:
                self.errors.append((str(path), f"{type(exc).__name__}: {exc}"))

    def add_read(self, seconds: float) -> None:
        with self.lock:
            self.t_read += seconds
            self.read_count += 1

    def add(self, res: Result, write_s: float) -> None:
        with self.lock:
            self.processed += 1
            self.backend_counts[res.backend] = self.backend_counts.get(res.backend, 0) + 1
            for key, val in res.timings.items():
                if key not in ("decode", "resize", "encode"):
                    self.extra_timings[key] = self.extra_timings.get(key, 0) + val
            self.bytes_in += len(res.job.data)
            self.bytes_out += len(res.jpeg)
            self.t_decode += res.timings.get("decode", 0.0)
            self.t_resize += res.timings.get("resize", 0.0)
            self.t_encode += res.timings.get("encode", 0.0)
            self.t_write += write_s

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "read_count": self.read_count,
                "backend_counts": dict(self.backend_counts),
                "extra_timings": dict(self.extra_timings),
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
             limit: Optional[int] = None, on_error=None) -> list[Path]:
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
                        # Skip Windows junctions/reparse points also on Python 3.10,
                        # which has no Path.is_junction(). Never traverse aliases.
                        if (entry.is_symlink() or getattr(entry.stat(follow_symlinks=False),
                                                         "st_file_attributes", 0) & 0x400):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            sub = Path(entry.path)
                            if exclude is not None and sub.resolve() == exclude:
                                continue
                            stack.append(sub)
                        elif entry.is_file(follow_symlinks=False):
                            if Path(entry.name).suffix.lower() in SUPPORTED_EXTS:
                                found.append(Path(entry.path))
                    except OSError as exc:
                        if on_error is None:
                            raise
                        on_error(Path(entry.path), exc)
        except OSError as exc:
            if on_error is None:
                raise
            on_error(d, exc)
    found.sort()
    if limit is not None:
        positive(limit, "limit")
        found = found[:limit]
    return found


def build_output_map(files: Iterable[Path], root: Path,
                     out_root: Path, previous=None) -> dict[Path, Path]:
    """Map sources to targets, disambiguating name clashes (photo.png/photo.jpg)."""
    used: dict[str, Path] = {}
    mapping: dict[Path, Path] = {}
    files = list(files)
    previous = previous or {}
    for src, target in previous.items():
        key = str(target).casefold()
        if key in used:
            raise ValueError("duplicate resume output mapping")
        used[key] = src
        mapping[src] = target
    for src in files:
        if src in mapping:
            continue
        rel = src.relative_to(root)
        target = (out_root / rel).with_suffix(".jpg")
        key = str(target).casefold()
        if key in used and used[key] != src:
            i = 2
            while True:
                cand = target.with_name(f"{target.stem}__{i}.jpg")
                if str(cand).casefold() not in used:
                    target = cand
                    break
                i += 1
            key = str(target).casefold()
        used[key] = src
        mapping[src] = target
    return mapping


class _ByteBudget:
    def __init__(self, capacity):
        self.capacity = capacity
        self.used = 0
        self.condition = threading.Condition()

    def acquire(self, size, stop):
        if size > self.capacity:
            raise ValueError(f"job needs {size} bytes; budget is {self.capacity}")
        with self.condition:
            while self.used + size > self.capacity:
                if stop.is_set():
                    return False
                self.condition.wait(.05)
            if stop.is_set():
                return False
            self.used += size
            return True

    def release(self, size):
        with self.condition:
            self.used -= size
            self.condition.notify_all()


_EMPTY = object()
_EOF = object()


class Pipeline:
    """Threaded stages connected by bounded queues.

    readers  --jobs-->  processor(s)  --results-->  writers
    """

    def __init__(self, opts: RunOptions, engine, backend: str,
                 progress: bool = True, on_progress=None, files=None) -> None:
        self.opts = opts
        self._fixed_files = list(files) if files is not None else None
        self.engine = engine
        self.backend = backend
        self.stats = Stats()
        self.stop = threading.Event()
        self.fatal = threading.Event()
        self.writers_done = threading.Event()
        self._terminal = set()
        self._pending = set()
        self._state_lock = threading.Lock()
        self._warnings = []
        self._fatal_errors = []
        self._discovery_errors = []
        self._byte_budget = _ByteBudget(opts.max_queued_bytes)
        self._working_budget = _ByteBudget(opts.max_working_bytes)
        self._reserved = {}
        self._sources = set()
        self._identities = set()
        self._resume = None
        self.scan_seconds = 0.0
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
    def _fail(self, path, exc):
        with self._state_lock:
            if path in self._terminal:
                return
            self._terminal.add(path)
        self.stats.error(path, exc)
        self._release_source(path)

    def _release_source(self, path):
        with self._state_lock:
            size = self._reserved.pop(path, 0)
        if size:
            self._byte_budget.release(size)

    def _guarded(self, target):
        try:
            target()
        except BaseException as exc:
            with self._state_lock:
                self._fatal_errors.append(f"{threading.current_thread().name}: {type(exc).__name__}: {exc}")
            self.fatal.set()
            self.stop.set()

    @contextmanager
    def _working(self, job):
        px = job.meta.width * job.meta.height
        # Conservative application-owned CPU arrays. Native codec workspaces
        # remain implementation-dependent, so this is not an OS RSS limit.
        size = px * (64 if job.has_alpha else 10) + job.plan.out_w * job.plan.out_h * 16
        if not self._working_budget.acquire(size, self.stop):
            raise RuntimeError("processing cancelled")
        try:
            yield
        finally:
            self._working_budget.release(size)

    def _put_job(self, job):
        while not self.stop.is_set():
            try:
                self.jobs.put(job, timeout=.05)
                return True
            except queue.Full:
                pass
        return False

    def _put_result(self, result):
        while not self.fatal.is_set():
            try:
                self.results.put(result, timeout=.05)
                return True
            except queue.Full:
                pass
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

    def _send_writer_sentinels(self):
        # Consumers use timeout + this event: no blocking sentinel insertion.
        self.writers_done.set()

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
                t0 = time.perf_counter()
                try:
                    size = src.stat().st_size
                    if size > self.opts.max_source_bytes:
                        raise ValueError("source exceeds compressed byte limit")
                    if not self._byte_budget.acquire(size, self.stop):
                        break
                    with self._state_lock:
                        self._reserved[src] = size
                    with src.open("rb") as f:
                        data = f.read(size + 1)
                    if len(data) != size:
                        raise ValueError("source changed size during read; retry")
                    self.stats.add_read(time.perf_counter() - t0)
                    if data.startswith(b"\xff\xd8") and not self.opts.allow_truncated:
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
                              meta.has_alpha, hashlib.sha256(data).hexdigest())
                    if out.exists() and not self.opts.overwrite:
                        if self.opts.skip_existing and self._resume.valid(job, self.opts.input_dir):
                            with self._state_lock:
                                self._terminal.add(src)
                            with self.stats.lock:
                                self.stats.skipped += 1
                            self._release_source(src)
                            continue
                        raise FileExistsError("existing output is unverified or configuration changed; use --overwrite")
                except Exception as exc:
                    self._fail(src, exc)
                    continue
                if not self._put_job(job):
                    self._release_source(src)
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
            try:
                with self._working(item):
                    self._process_cpu_one(item)
            except Exception as exc:
                self._fail(item.src, exc)

    def _gpu_processor(self) -> None:
        if hasattr(self.engine, "activate"):
            self.engine.activate()
        budget = int(self.opts.vram_budget_gb * 2**30)
        while True:
            item = self._take_job()
            if item is _EOF:
                break
            if item is _EMPTY:
                continue
            batch = [item]
            size = self._gpu_estimate(item)
            if size > budget:
                self._gpu_fallback(item, MemoryError("single image exceeds soft VRAM budget"))
                continue
            while len(batch) < self.opts.batch_size:
                nxt = self._take_job(False)
                if nxt is _EOF or nxt is _EMPTY:
                    break
                cost = self._gpu_estimate(nxt)
                if size + cost > budget:
                    with self._deferred_lock:
                        self._deferred = nxt
                    break
                batch.append(nxt); size += cost
            if self.backend == "hybrid":
                for job in batch:
                    self._process_hybrid_one(job)
            else:
                self._process_gpu_batch(batch)

    @staticmethod
    def _gpu_estimate(job):
        # Sum is conservative for application-owned arrays, not decoder/pool RSS.
        p = job.plan
        return (job.meta.width * job.meta.height * (8 if job.has_alpha else 4) +
                12 * p.src_h * p.out_w + 3 * p.out_w * p.out_h +
                64 * 2**20)  # bounded table cache + codec headroom per item

    def _writer(self):
        while True:
            try:
                res = self.results.get(timeout=.05)
            except queue.Empty:
                if self.writers_done.is_set() or self.fatal.is_set():
                    return
                continue
            t0 = time.perf_counter()
            try:
                if self.fatal.is_set():
                    self._fail(res.job.src, RuntimeError("worker failure before publication"))
                    continue
                protect_target(res.job.out, self.opts.output_dir, self._sources, self._identities)
                atomic_write(res.job.out, res.jpeg, self.opts.fsync, overwrite=self.opts.overwrite)
                with self._state_lock:
                    self._resume.record(res.job, res.jpeg, self.opts.input_dir)
                    self._terminal.add(res.job.src)
                self.stats.add(res, time.perf_counter() - t0)
            except Exception as exc:
                self._fail(res.job.src, exc)
            finally:
                self._release_source(res.job.src)

    # -- processing ----------------------------------------------------------
    def _finish(self, job: Job, jpeg: bytes, timings: dict,
                backend: str) -> None:
        with self._state_lock:
            if job.src in self._terminal or job.src in self._pending:
                return
        t0 = time.perf_counter()
        icc = job.meta.icc
        if icc and job.meta.mode not in ("RGB", "RGBA", "P"):
            from PIL import ImageCms
            icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        if self.opts.keep_exif or self.opts.keep_icc:
            # Fail this file visibly rather than attach stale EXIF/ICC or drop it silently.
            jpeg = splice_metadata(jpeg, job.meta.exif if self.opts.keep_exif else None,
                    icc if self.opts.keep_icc else None,
                    dimensions=(job.plan.out_w, job.plan.out_h))
        timings["metadata"] = time.perf_counter() - t0
        with self._state_lock:
            if job.src in self._terminal or job.src in self._pending:
                return
            self._pending.add(job.src)
        res = Result(job, jpeg, job.plan.out_w, job.plan.out_h, backend, timings)
        if not self._put_result(res):
            self._fail(job.src, RuntimeError("worker failure prevented publication"))

    def _process_cpu_one(self, job: Job, engine=None, backend="cpu") -> None:
        engine = engine or self.engine
        try:
            t0 = time.perf_counter()
            if job.has_alpha or (job.meta.icc and job.meta.mode not in ("RGB", "RGBA", "P")):
                rgb, _ = decode_to_rgb(job.data)
                img = rgb[:, :, [2, 1, 0, 3]] if rgb.shape[2] == 4 else rgb[:, :, ::-1]
                import numpy as np
                img = np.ascontiguousarray(img)
            else:
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
            timings = {"decode": t1 - t0, "normalize": t2 - t1, "resize": t3 - t2,
                       "encode": t4 - t3}
            self._finish(job, jpeg, timings, backend)
        except Exception as exc:
            self._fail(job.src, exc)

    def _process_gpu_batch(self, batch):
        with self._state_lock:
            batch = [j for j in batch if j.src not in self._terminal and j.src not in self._pending]
        if not batch:
            return
        error = None
        try:
            self._process_gpu_batch_inner(batch)
        except Exception as exc:
            # Leave the except suite (and its traceback allocations) before retry.
            error = f"{type(exc).__name__}: {exc}"
        if error is not None:
            if len(batch) > 1:
                mid = len(batch) // 2
                self._process_gpu_batch(batch[:mid])
                self._process_gpu_batch(batch[mid:])
            else:
                self._gpu_fallback(batch[0], RuntimeError(error))

    def _process_gpu_batch_inner(self, batch):
        # Managed colour conversion cannot copy a CMYK/gray profile onto RGB.
        managed = [j for j in batch if j.meta.icc and j.meta.mode not in ("RGB", "RGBA", "P")]
        for job in managed:
            self._process_hybrid_one(job)
        batch = [j for j in batch if j not in managed]
        if not batch:
            return
        t0 = time.perf_counter()
        arrs = self.engine.decode_batch([j.data for j in batch], [j.has_alpha for j in batch])
        self._sync_profile()
        if len(arrs) != len(batch):
            raise RuntimeError("decoder returned wrong batch length")
        decode_time = (time.perf_counter() - t0) / len(batch)
        for job, arr in zip(batch, arrs):
            if arr is None:
                self._gpu_fallback(job, RuntimeError("GPU decode rejected file"))
                continue
            try:
                t1 = time.perf_counter()
                if int(arr.shape[2]) == 4:
                    arr = self.engine.alpha_to_rgb(arr, self.opts.bg_color)
                elif job.has_alpha:
                    raise ValueError("decoder dropped source alpha")
                self._sync_profile()
                t2 = time.perf_counter()
                small = self.engine.resize(arr, job.plan)
                self._sync_profile()
                t3 = time.perf_counter()
                jpeg = self.engine.encode(small, self.opts.quality, self.opts.subsampling)
                self._sync_profile()
                self._finish(job, jpeg, {"decode": decode_time, "normalize": t2-t1,
                    "resize": t3-t2, "encode": time.perf_counter()-t3}, "nvimgcodec-auto")
            except Exception as exc:
                self._gpu_fallback(job, exc)

    def _sync_profile(self):
        if self.opts.profile_cuda and hasattr(self.engine, "synchronize"):
            self.engine.synchronize()

    def _process_gpu_one(self, job):
        self._process_gpu_batch([job])

    def _process_hybrid_one(self, job):
        try:
            with self._working(job):
                self._process_hybrid_inner(job)
        except Exception as exc:
            self._gpu_fallback(job, exc)

    def _process_hybrid_inner(self, job):
        """Host decode (Pillow) -> VRAM -> CUDA resize -> nvJPEG encode."""
        engine = self.engine
        try:
            t0 = time.perf_counter()
            rgb, has_alpha = decode_to_rgb(job.data, self.opts.bg_color)
            t1 = time.perf_counter()
            arr = engine.to_device(rgb)
            arr = engine.apply_orientation(arr, job.meta.orientation)
            self._sync_profile()
            t2 = time.perf_counter()
            if int(arr.shape[2]) == 4:
                arr = engine.alpha_to_rgb(arr, self.opts.bg_color)
            self._sync_profile()
            small = engine.resize(arr, job.plan)
            self._sync_profile()
            t3 = time.perf_counter()
            jpeg = engine.encode(small, self.opts.quality,
                                 self.opts.subsampling)
            self._sync_profile()
            t4 = time.perf_counter()
            timings = {"decode": t1 - t0, "normalize": t2 - t1, "resize": t3 - t2,
                       "encode": t4 - t3}
            self._sync_profile()
            self._finish(job, jpeg, timings, "hybrid")
        except Exception:
            raise

    def _gpu_fallback(self, job, exc):
        if not self.opts.fallback:
            self._fail(job.src, exc)
            return
        with self._state_lock:
            if job.src in self._terminal or job.src in self._pending:
                return
            if len(self._warnings) < 50:
                self._warnings.append(f"{job.src}: GPU rejected file; full CPU fallback: {exc}")
        with self.stats.lock:
            self.stats.fallback += 1
        try:
            with self._working(job):
                self._process_cpu_one(job, CpuEngine(self.opts.filter_name), "cpu-fallback")
        except Exception as fallback_error:
            self._fail(job.src, fallback_error)

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

    def _wait(self, threads, t_start):
        workers = self._readers + self._processors
        failure_deadline = None
        while any(t.is_alive() for t in workers):
            if self.opts.cancel_file is not None and self.opts.cancel_file.exists():
                raise KeyboardInterrupt
            self._progress(t_start)
            if self.fatal.is_set():
                failure_deadline = failure_deadline or time.monotonic() + 5
                if time.monotonic() >= failure_deadline:
                    break
            time.sleep(.02)
        self._send_writer_sentinels()
        self._join_bounded(threads)
        self._progress(t_start, final=True)

    def _join_bounded(self, threads):
        deadline = time.monotonic() + 5
        for t in threads:
            t.join(timeout=max(0, deadline - time.monotonic()))
        alive = [t.name for t in threads if t.is_alive()]
        if alive:
            self.fatal.set(); self.stop.set()
            self._fatal_errors.append("native operation did not stop: " + ", ".join(alive))

    def _shutdown(self):
        self.stop.set()
        self._join_bounded(self._readers + self._processors)
        self._send_writer_sentinels()
        self._join_bounded(self._writers)

    def run(self) -> dict:
        opts = self.opts
        check_roots(opts.input_dir, opts.output_dir.resolve())
        if not opts.input_dir.is_dir():
            raise ValueError("input directory does not exist")
        config = {k: getattr(opts, k) for k in ("max_w", "max_h", "mode", "quality",
            "subsampling", "sizing", "sizing_value", "allow_upscale", "filter_name",
            "bg_color", "keep_exif", "keep_icc", "optimize_huffman")}
        from . import __version__
        from importlib.metadata import version, PackageNotFoundError
        versions = {}
        for package in ("Pillow", "numpy", "opencv-python-headless", "piexif",
                        "cupy-cuda12x", "nvidia-nvimgcodec-cu12"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        config.update(backend=self.backend, resume_contract=2, application=__version__,
                      libraries=versions, input_root=str(opts.input_dir))
        self._resume = ResumeState(opts.output_dir, config)
        try:
            self._resume.acquire()
            return self._run_locked()
        finally:
            # A live native call can still own buffers or publish a result. Keep
            # the lock for manual recovery instead of admitting a second run.
            alive = any(t.is_alive() for t in self._readers + self._processors + self._writers)
            if not alive:
                self._resume.release()
            for path in list(self._reserved):
                self._release_source(path)

    def _run_locked(self):
        opts = self.opts
        self.t_start = time.perf_counter()
        scan_start = time.perf_counter()
        def scan_error(path, exc):
            self._discovery_errors.append((str(path), str(exc)))
        all_files = (discover(opts.input_dir, opts.output_dir, on_error=scan_error)
                     if self._fixed_files is None else list(self._fixed_files))
        if any(not p.is_relative_to(opts.input_dir) for p in all_files):
            raise ValueError("frozen input escapes input directory")
        files = all_files[:opts.limit] if opts.limit is not None else all_files
        self.scan_seconds = time.perf_counter() - scan_start
        self.discovered = len(files)
        self._sources = {path_key(p) for p in all_files}
        self._identities = source_identities(all_files)
        previous = self._resume.previous_mapping(all_files, opts.input_dir)
        self._out_map = build_output_map(files, opts.input_dir, opts.output_dir, previous)
        for target in self._out_map.values():
            protect_target(target, opts.output_dir, self._sources, self._identities)
        self._file_iter = iter(files)
        interrupted = False

        if files:
            for i in range(self._n_readers):
                t = threading.Thread(target=self._guarded, args=(self._reader,),
                                     name=f"reader-{i}", daemon=True)
                self._readers.append(t)
            target = (self._cpu_processor if self.backend == "cpu"
                      else self._gpu_processor)
            for i in range(self._n_processors):
                t = threading.Thread(target=self._guarded, args=(target,),
                                     name=f"processor-{i}", daemon=True)
                self._processors.append(t)
            for i in range(self._n_writers):
                t = threading.Thread(target=self._guarded, args=(self._writer,),
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
                self._send_writer_sentinels()
                self._join_bounded(threads)

        for src in files:
            if src not in self._terminal:
                self._fail(src, RuntimeError("cancelled" if interrupted else "worker did not complete job"))
        if any(t.is_alive() for t in self._readers + self._processors + self._writers):
            self._fatal_errors.append("live native worker: manifest not saved; output lock retained")
            self.fatal.set()
        else:
            try:
                self._resume.save()
            except Exception as exc:
                self._fatal_errors.append(f"resume manifest not saved: {exc}")
                self.fatal.set()
        return self._report(interrupted)

    def _report(self, interrupted: bool) -> dict:
        elapsed = max(1e-9, time.perf_counter() - self.t_start)
        snap = self.stats.snapshot()
        processed = snap["processed"]
        per = (lambda v: round(1000.0 * v / processed, 3)) if processed else (lambda v: 0.0)
        return {
            "complete": not interrupted and not self.fatal.is_set() and not self._discovery_errors
                        and snap["failed"] == 0,
            "fatal_errors": list(self._fatal_errors),
            "discovery_errors": list(self._discovery_errors),
            "warnings": list(self._warnings)[:50],
            "scan_sec": self.scan_seconds,
            "backend_counts": snap["backend_counts"],
            "codec_backend": getattr(self.engine, "backend_policy", "CPU/Pillow/OpenCV"),
            "timing_semantics": ("synchronized service wall time (profiling changes throughput)"
                if self.opts.profile_cuda else "host service envelopes; GPU launches asynchronous, not kernel execution time"),
            "input_validation": "decoder acceptance + JPEG marker check; not an entropy integrity certificate",
            "vram_budget_semantics": "soft application array estimate; excludes opaque codec/driver/pool allocations",
            "extra_timings_avg_ms": {k: per(v) for k, v in snap["extra_timings"].items()},
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
                "read": round(1000 * snap["t_read"] / max(1, snap["read_count"]), 3),
                "decode": per(snap["t_decode"]),
                "resize": per(snap["t_resize"]),
                "encode": per(snap["t_encode"]),
                "write": per(snap["t_write"]),
            },
            "interrupted": interrupted,
            "errors": snap["errors"],
        }
