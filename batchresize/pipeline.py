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
from .presets import SUBSAMPLING_NAMES, auto_subsampling
from .source_quality import (FALLBACK_QUALITY, FALLBACK_SUBSAMPLING,
                              QUALITY_SOURCE, SUBSAMPLING_SOURCE,
                              estimate_source_quality, read_subsampling,
                              source_quantisation_tables)
from .variants import OutputVariant

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _default_working_bytes() -> int:
    """Soft CPU working-set budget: 2 GiB, capped at a quarter of physical RAM."""
    cap = 2048 * 2**20
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            quarter = int(status.ullTotalPhys // 4)
            return max(512 * 2**20, min(cap, quarter))
    except Exception:
        pass
    return cap


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
    max_working_bytes: int = field(default_factory=_default_working_bytes)
    profile_cuda: bool = False
    cancel_file: Optional[Path] = None
    variants: tuple[OutputVariant, ...] = ()
    sharpen: Any = None
    color_policy: str = "preserve_rgb"
    linear_light: bool = False
    strip_personal_data: bool = False
    keep_datetime: bool = False
    crop_anchor: str = "center"
    crop_focus: Optional[Any] = None
    max_file_size: Optional[int | str] = None
    min_quality: int = 30
    source_fallback_quality: int = FALLBACK_QUALITY
    allow_downscale_on_size_limit: bool = False
    fast_jpeg_decode: bool = False
    processors: Optional[int] = None
    readers: Optional[int] = None
    writers: Optional[int] = None

    def __post_init__(self):
        self.variants = tuple(self.variants)
        if len({variant.id for variant in self.variants}) != len(self.variants):
            raise ValueError("duplicate output variant id")
        self.input_dir = Path(self.input_dir).resolve()
        self.output_dir = Path(self.output_dir).absolute()
        dimensions(self.max_w, self.max_h)
        if self.mode not in ("contain", "cover", "pad") or self.sizing not in (
                "resolution", "percent", "megapixels", "longest", "shortest"):
            raise ValueError("unknown sizing/mode")
        from .geometry import ANCHORS, parse_crop_focus
        anchor_key = str(self.crop_anchor).lower().strip()
        if anchor_key not in ANCHORS:
            raise ValueError(f"unknown crop anchor {self.crop_anchor!r}")
        object.__setattr__(self, "crop_anchor", anchor_key)
        if self.crop_focus is not None:
            object.__setattr__(self, "crop_focus", parse_crop_focus(self.crop_focus))
        if self.max_file_size is not None:
            from .size_limit import parse_size_bytes
            # Validate size parseability
            parsed_bytes = parse_size_bytes(self.max_file_size)
            object.__setattr__(self, "max_file_size", parsed_bytes)
        if (isinstance(self.min_quality, bool) or int(self.min_quality) != self.min_quality or
                not 1 <= self.min_quality <= 100):
            raise ValueError("min_quality must be an integer between 1 and 100")
        if (isinstance(self.source_fallback_quality, bool) or
                int(self.source_fallback_quality) != self.source_fallback_quality or
                not 1 <= self.source_fallback_quality <= 100):
            raise ValueError("source_fallback_quality must be an integer between 1 and 100")
        if not isinstance(self.allow_downscale_on_size_limit, bool):
            raise ValueError("allow_downscale_on_size_limit must be a boolean")
        if not isinstance(self.fast_jpeg_decode, bool):
            raise ValueError("fast_jpeg_decode must be a boolean")
        if self.sizing != "resolution":
            positive(self.sizing_value, "sizing value")
        positive(self.vram_budget_gb, "VRAM budget")
        for value, name, cap in [(self.workers, "workers", 64),
                (self.batch_size, "batch size", 128), (self.queue_depth, "queue depth", 256)]:
            positive(value, name)
            if int(value) != value or value > cap:
                raise ValueError(f"{name} must be an integer <= {cap}")
        for val, name, cap in [(self.processors, "processors", 128),
                               (self.readers, "readers", 128),
                               (self.writers, "writers", 128)]:
            if val is not None:
                positive(val, name)
                if int(val) != val or val > cap:
                    raise ValueError(f"{name} must be an integer <= {cap}")
        if self.limit is not None and (self.limit < 1 or int(self.limit) != self.limit):
            raise ValueError("limit must be a positive integer")
        if self.quality == QUALITY_SOURCE:
            if self.subsampling not in (0, 1, 2, SUBSAMPLING_SOURCE):
                raise ValueError("invalid JPEG quality/subsampling")
        else:
            if (isinstance(self.quality, bool) or int(self.quality) != self.quality or
                    not 1 <= self.quality <= 100 or isinstance(self.subsampling, bool) or
                    self.subsampling not in (0, 1, 2)):
                raise ValueError("invalid JPEG quality/subsampling")
            if self.min_quality > self.quality:
                if self.max_file_size is not None:
                    raise ValueError("min_quality cannot be greater than quality")
                object.__setattr__(self, "min_quality", self.quality)
        from .resampling import SUPPORTED_FILTERS
        if self.filter_name not in SUPPORTED_FILTERS:
            raise ValueError("unknown filter")
        from .sharpen import SharpenConfig, parse_sharpen
        if not isinstance(self.sharpen, SharpenConfig):
            object.__setattr__(self, "sharpen", parse_sharpen(self.sharpen))
        from .color_management import canonical_color_policy
        object.__setattr__(self, "color_policy", canonical_color_policy(self.color_policy))
        if not isinstance(self.linear_light, bool):
            raise ValueError("linear_light must be a boolean")
        if not isinstance(self.strip_personal_data, bool):
            raise ValueError("strip_personal_data must be a boolean")
        if not isinstance(self.keep_datetime, bool):
            raise ValueError("keep_datetime must be a boolean")
        if len(self.bg_color) != 3 or any(int(x) != x or not 0 <= x <= 255 for x in self.bg_color):
            raise ValueError("background must contain three bytes")
        for name in ("max_source_bytes", "max_queued_bytes", "max_working_bytes"):
            value = positive(getattr(self, name), name)
            if int(value) != value:
                raise ValueError(f"{name} must be an integer")
        if self.max_source_bytes > self.max_queued_bytes:
            raise ValueError("source byte limit exceeds queued byte budget")


@dataclass(frozen=True)
class OutputTask:
    variant_id: Optional[str]
    out: Path
    plan: Plan
    quality: int
    subsampling: int
    config_digest: str
    #: Quantisation tables of the source file ("same as source" mode).
    qtables: Optional[tuple[list[int], list[int]]] = None


@dataclass
class Job:
    src: Path
    out: Path
    data: bytes
    meta: SourceMeta
    plan: Plan
    has_alpha: bool = False
    digest: str = ""
    outputs: tuple[OutputTask, ...] = ()
    t_queued: float = 0.0


@dataclass
class Result:
    job: Job
    jpeg: bytes
    out: Path
    out_w: int
    out_h: int
    backend: str
    timings: dict = field(default_factory=dict)
    variant_id: Optional[str] = None
    plan: Optional[Plan] = None
    actual_quality: Optional[int] = None
    target_size_achieved: Optional[bool] = None


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
        self.fallback_reasons = {}
        self.backend_bytes = {}
        self.latencies = []
        self.gpu_service_time = 0.0
        self._counted_sources: set[Path] = set()

    def error(self, path: Path, exc: BaseException, count: int = 1) -> None:
        with self.lock:
            self.failed += count
            if len(self.errors) < 50:
                self.errors.append((str(path), f"{type(exc).__name__}: {exc}"))

    def record_fallback(self, reason: str) -> None:
        with self.lock:
            self.fallback += 1
            # Clean and bound fallback reason key
            key = str(reason).strip()
            if len(key) > 120:
                key = key[:117] + "..."
            self.fallback_reasons[key] = self.fallback_reasons.get(key, 0) + 1

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
            if res.job.src not in self._counted_sources:
                self._counted_sources.add(res.job.src)
                self.bytes_in += len(res.job.data)
            self.bytes_out += len(res.jpeg)

            b_stat = self.backend_bytes.setdefault(res.backend, {"bytes_in": 0, "bytes_out": 0})
            b_stat["bytes_out"] += len(res.jpeg)
            if res.job.src not in self._counted_sources:
                b_stat["bytes_in"] += len(res.job.data)

            if getattr(res.job, "t_queued", 0.0) > 0.0:
                latency_ms = (time.perf_counter() - res.job.t_queued) * 1000.0
                if len(self.latencies) < 10000:
                    self.latencies.append(latency_ms)

            self.gpu_service_time += res.timings.get("gpu_service", 0.0)
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
                "fallback_reasons": dict(self.fallback_reasons),
                "backend_bytes": {k: dict(v) for k, v in self.backend_bytes.items()},
                "latencies": list(self.latencies),
                "gpu_service_time": self.gpu_service_time,
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

# Readers/processors must react to stop promptly; writers may legitimately need
# to drain results that were already queued when processing finished.
JOIN_TIMEOUT = 5.0
WRITER_STALL_TIMEOUT = 30.0

# Backend labels recorded per result. A GPU decode/resize/encode route is any
# non-CPU label; 'hybrid' means host decode + GPU resize/encode.
GPU_DECODE_LABELS = frozenset({"gpu", "gpu-engine", "nvimgcodec-auto"})
GPU_COMPUTE_LABELS = GPU_DECODE_LABELS | frozenset({"hybrid"})


def actual_route_flags(backend_counts) -> tuple[bool, bool]:
    """Return ``(used_gpu_decode, used_gpu_compute)`` from recorded labels.

    Keeping this in one place prevents the report's route strings from drifting
    away from the labels written by ``_finish`` (AUD-001).
    """
    labels = set(backend_counts or ())
    gpu_decode = bool(labels & GPU_DECODE_LABELS)
    gpu_compute = gpu_decode or "hybrid" in labels
    return gpu_decode, gpu_compute


class Pipeline:
    """Threaded stages connected by bounded queues.

    readers  --jobs-->  processor(s)  --results-->  writers
    """

    def __init__(self, opts: RunOptions, engine, backend: str,
                 progress: bool = True, on_progress=None, files=None) -> None:
        self.opts = opts
        self._fixed_files = list(files) if files is not None else None
        if engine is None and backend == "cpu":
            from .cpu_backend import CpuEngine
            engine = CpuEngine(opts.filter_name, opts.optimize_huffman)
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
        self._variant_digests: dict[str, str] = {}
        self._variant_maps: dict[str, dict[Path, Path]] = {}
        self._targets_by_source: dict[Path, dict[Optional[str], Path]] = {}
        self._expected: set[tuple[Path, Optional[str]]] = set()
        self._remaining: dict[Path, int] = {}
        self._outputs_total = 0
        self.scan_seconds = 0.0
        self.on_progress = on_progress
        self.jobs: queue.Queue = queue.Queue(maxsize=max(1, opts.queue_depth))
        self.results: queue.Queue = queue.Queue(maxsize=max(1, opts.queue_depth))
        self.discovered = 0
        self._target_size_achieved: Optional[bool] = None
        self.t_start = 0.0
        self._progress_enabled = progress
        self._last_progress = 0.0
        self._writer_stall_timeout = WRITER_STALL_TIMEOUT

        self._gpu_lock = threading.Lock()
        base_workers = max(1, opts.workers)
        self._n_readers = max(1, opts.readers) if opts.readers is not None else base_workers
        if opts.processors is not None:
            self._n_processors = max(1, opts.processors)
        elif backend == "cpu":
            self._n_processors = base_workers
        else:
            self._n_processors = 1
        self._n_writers = max(1, opts.writers) if opts.writers is not None else max(2, base_workers // 2)

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
        self._fail_source(path, exc)

    def _fail_source(self, path, exc):
        with self._state_lock:
            keys = [key for key in self._expected
                    if key[0] == path and key not in self._terminal]
            if not keys:
                key = (path, None)
                if key not in self._terminal:
                    self._terminal.add(key)
                    keys = [key]
            for key in keys:
                self._terminal.add(key)
            self._remaining[path] = 0
        if keys:
            self.stats.error(path, exc, len(keys))
        self._release_source(path)

    def _fail_output(self, job, task, exc):
        key = (job.src, task.variant_id)
        with self._state_lock:
            if key in self._terminal:
                return
            self._terminal.add(key)
        label = str(job.src)
        if task.variant_id is not None:
            label = f"{label} [{task.variant_id}]"
        self.stats.error(label, exc)

    def _mark_skipped(self, job, task):
        key = (job.src, task.variant_id)
        with self._state_lock:
            if key in self._terminal:
                return
            self._terminal.add(key)
        with self.stats.lock:
            self.stats.skipped += 1
        self._complete_output(job.src)

    def _complete_output(self, path):
        with self._state_lock:
            remaining = max(0, self._remaining.get(path, 1) - 1)
            self._remaining[path] = remaining
        if remaining == 0:
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

    def _job_tasks(self, job):
        if job.outputs:
            return job.outputs
        return (OutputTask(None, job.out, job.plan, self.opts.quality,
                           self.opts.subsampling, ""),)

    def _resolve_source_profile(self, data: bytes, quality, subsampling):
        """Per-file quality/subsampling for the "same as source" mode.

        Returns ``(quality, subsampling, qtables)``. The quantisation tables
        of the source are copied verbatim so that only the geometry changes.
        """
        if quality != QUALITY_SOURCE and subsampling != SUBSAMPLING_SOURCE:
            return quality, subsampling, None
        tables = source_quantisation_tables(data)
        if quality == QUALITY_SOURCE:
            estimated = estimate_source_quality(data)
            quality = (estimated if estimated is not None
                       else self.opts.source_fallback_quality)
        if subsampling == SUBSAMPLING_SOURCE:
            detected = read_subsampling(data)
            subsampling = detected if detected is not None else FALLBACK_SUBSAMPLING
        return quality, subsampling, tables

    @contextmanager
    def _working(self, job):
        px = job.meta.width * job.meta.height
        # Conservative application-owned CPU arrays. Native codec workspaces
        # remain implementation-dependent, so this is not an OS RSS limit.
        out_px = sum(task.plan.out_w * task.plan.out_h
                     for task in self._job_tasks(job))
        size = px * (64 if job.has_alpha else 10) + out_px * 16
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
                    digest = hashlib.sha256(data).hexdigest()
                    targets = self._targets_by_source[src]
                    tasks: list[OutputTask] = []
                    if self.opts.variants:
                        for variant in self.opts.variants:
                            if variant.subsampling == "auto":
                                subsampling = auto_subsampling(variant.quality)
                            elif isinstance(variant.subsampling, int) and variant.subsampling in (0, 1, 2):
                                subsampling = variant.subsampling
                            else:
                                subsampling = SUBSAMPLING_NAMES.get(str(variant.subsampling), 2)
                            box_w, box_h = compute_box(
                                meta.display_width, meta.display_height,
                                variant.sizing, variant.max_w, variant.max_h,
                                variant.sizing_value)
                            plan = make_plan(meta.display_width, meta.display_height,
                                             box_w, box_h, variant.mode,
                                             variant.allow_upscale,
                                             anchor=self.opts.crop_anchor,
                                             focus=self.opts.crop_focus)
                            tasks.append(OutputTask(
                                variant.id, targets[variant.id], plan,
                                variant.quality, subsampling,
                                self._variant_digests[variant.id]))
                    else:
                        box_w, box_h = compute_box(
                            meta.display_width, meta.display_height,
                            self.opts.sizing, self.opts.max_w, self.opts.max_h,
                            self.opts.sizing_value)
                        plan = make_plan(meta.display_width, meta.display_height,
                                         box_w, box_h,
                                         self.opts.mode, self.opts.allow_upscale,
                                         anchor=self.opts.crop_anchor,
                                         focus=self.opts.crop_focus)
                        quality, subsampling, qtables = self._resolve_source_profile(
                            data, self.opts.quality, self.opts.subsampling)
                        tasks.append(OutputTask(
                            None, targets[None], plan, quality,
                            subsampling, self._resume.fingerprint, qtables))
                    job = Job(src, tasks[0].out, data, meta, tasks[0].plan,
                              meta.has_alpha, digest, tuple(tasks),
                              t_queued=time.perf_counter())
                    if not self.opts.overwrite:
                        remaining_tasks = []
                        for task in tasks:
                            if not task.out.exists():
                                remaining_tasks.append(task)
                                continue
                            if (self.opts.skip_existing and
                                    self._resume.valid(
                                        job, self.opts.input_dir,
                                        variant_id=task.variant_id,
                                        config_digest=task.config_digest,
                                        out=task.out, plan=task.plan)):
                                self._mark_skipped(job, task)
                                continue
                            self._fail_output(job, task, FileExistsError(
                                "existing output is unverified or configuration "
                                "changed; --skip-existing skips verified results "
                                "only - re-run with --overwrite to rebuild it"))
                            self._complete_output(src)
                        tasks = remaining_tasks
                    if not tasks:
                        self._release_source(src)
                        continue
                    job.outputs = tuple(tasks)
                    job.out = tasks[0].out
                    job.plan = tasks[0].plan
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
                with self._gpu_lock:
                    self._process_gpu_batch(batch)

    @staticmethod
    def _gpu_estimate(job):
        # Sum is conservative for application-owned arrays, not decoder/pool RSS.
        outputs = job.outputs or (OutputTask(None, job.out, job.plan, 0, 0, ""),)
        resize = sum(12 * task.plan.src_h * task.plan.out_w for task in outputs)
        encoded = sum(3 * task.plan.out_w * task.plan.out_h for task in outputs)
        return (job.meta.width * job.meta.height * (8 if job.has_alpha else 4) +
                resize + encoded +
                64 * 2**20)  # bounded table cache + codec headroom per item

    def _task_digest(self, res):
        if res.variant_id is None:
            return self._resume.fingerprint
        return self._variant_digests[res.variant_id]

    def _writer(self):
        while True:
            try:
                res = self.results.get(timeout=.05)
            except queue.Empty:
                if self.writers_done.is_set() or self.fatal.is_set():
                    return
                continue
            t0 = time.perf_counter()
            task = OutputTask(res.variant_id, res.out, res.plan, 0, 0, "")
            try:
                if self.fatal.is_set():
                    self._fail_output(res.job, task, RuntimeError(
                        "worker failure before publication"))
                    continue
                protect_target(res.out, self.opts.output_dir,
                               self._sources, self._identities)
                atomic_write(res.out, res.jpeg, self.opts.fsync,
                             overwrite=self.opts.overwrite)
                with self._state_lock:
                    self._resume.record(
                        res.job, res.jpeg, self.opts.input_dir,
                        variant_id=res.variant_id,
                        config_digest=self._task_digest(res),
                        out=res.out, plan=res.plan,
                        actual_quality=res.actual_quality,
                        target_size_achieved=res.target_size_achieved)
                    self._terminal.add((res.job.src, res.variant_id))
                self.stats.add(res, time.perf_counter() - t0)
            except Exception as exc:
                self._fail_output(res.job, task, exc)
            finally:
                self._complete_output(res.job.src)

    # -- processing ----------------------------------------------------------
    def _finish(self, job: Job, second, third=None, fourth=None,
                fifth="cpu", actual_quality=None, target_size_achieved=None) -> None:
        if isinstance(second, OutputTask):
            task, jpeg, timings, backend = second, third, fourth or {}, fifth
        else:
            # Legacy private signature: _finish(job, jpeg, timings, backend).
            task = OutputTask(None, job.out, job.plan, self.opts.quality,
                              self.opts.subsampling, "")
            jpeg, timings, backend = second, third or {}, fourth or "cpu"
        key = (job.src, task.variant_id)
        with self._state_lock:
            if key in self._terminal or key in self._pending:
                return
        t0 = time.perf_counter()
        from .color_management import SRGB_ICC_BYTES, COLOR_POLICY_SRGB
        output_icc = None
        if self.opts.keep_icc:
            if self.opts.color_policy == COLOR_POLICY_SRGB:
                output_icc = SRGB_ICC_BYTES
            elif job.meta.icc and job.meta.mode not in ("RGB", "RGBA", "P"):
                output_icc = SRGB_ICC_BYTES
            elif job.meta.icc:
                output_icc = job.meta.icc

        if self.opts.keep_exif or self.opts.keep_icc or self.opts.strip_personal_data:
            # Fail this file visibly rather than attach stale EXIF/ICC or drop it silently.
            jpeg = splice_metadata(
                jpeg,
                job.meta.exif if self.opts.keep_exif else None,
                output_icc,
                dimensions=(task.plan.target_w, task.plan.target_h),
                strip_personal=self.opts.strip_personal_data,
                keep_datetime=self.opts.keep_datetime,
            )
        if self.opts.strip_personal_data:
            from .imageio_meta import verify_metadata_sanitized
            verify_metadata_sanitized(jpeg, keep_datetime=self.opts.keep_datetime)
        timings["metadata"] = time.perf_counter() - t0
        with self._state_lock:
            if key in self._terminal or key in self._pending:
                return
            self._pending.add(key)
            if target_size_achieved is not None:
                if self._target_size_achieved is None:
                    self._target_size_achieved = target_size_achieved
                else:
                    self._target_size_achieved = self._target_size_achieved and target_size_achieved
        res = Result(job, jpeg, task.out, task.plan.target_w, task.plan.target_h,
                     backend, timings, task.variant_id, task.plan,
                     actual_quality=actual_quality,
                     target_size_achieved=target_size_achieved)
        if not self._put_result(res):
            self._fail_output(job, task, RuntimeError(
                "worker failure prevented publication"))
            self._complete_output(job.src)

    def _encode_with_size_limit(self, job: Job, task: OutputTask, small, engine):
        actual_quality = task.quality
        target_size_achieved = None
        if self.opts.max_file_size:
            from .size_limit import optimize_jpeg_quality, parse_size_bytes
            max_bytes = parse_size_bytes(self.opts.max_file_size)

            def splice_check(j: bytes) -> bytes:
                output_icc = None
                if self.opts.keep_icc:
                    from .color_management import SRGB_ICC_BYTES, COLOR_POLICY_SRGB
                    if self.opts.color_policy == COLOR_POLICY_SRGB:
                        output_icc = SRGB_ICC_BYTES
                    elif job.meta.icc and job.meta.mode not in ("RGB", "RGBA", "P"):
                        output_icc = SRGB_ICC_BYTES
                    elif job.meta.icc:
                        output_icc = job.meta.icc
                if self.opts.keep_exif or self.opts.keep_icc or self.opts.strip_personal_data:
                    return splice_metadata(
                        j,
                        job.meta.exif if self.opts.keep_exif else None,
                        output_icc,
                        dimensions=(task.plan.target_w, task.plan.target_h),
                        strip_personal=self.opts.strip_personal_data,
                        keep_datetime=self.opts.keep_datetime,
                    )
                return j

            def encode_fn(im, q):
                return engine.encode(im, q, task.subsampling, task.qtables)

            if task.qtables is not None:
                # "Same as source": the quantisation tables fix the
                # compression, a quality search cannot change it.
                jpeg = splice_check(encode_fn(small, task.quality))
                actual_quality = task.quality
                target_size_achieved = len(jpeg) <= max_bytes
            else:
                jpeg, actual_quality, target_size_achieved = optimize_jpeg_quality(
                    encode_fn,
                    small,
                    initial_quality=task.quality,
                    min_quality=self.opts.min_quality,
                    max_bytes=max_bytes,
                    splice_fn=splice_check,
                )

            if not target_size_achieved and self.opts.allow_downscale_on_size_limit:
                import cv2
                is_gpu_arr = hasattr(engine, "cp") and isinstance(small, engine.cp.ndarray)
                host_img = engine.cp.asnumpy(small) if is_gpu_arr else small
                curr_img = host_img
                for scale_factor in (0.85, 0.70, 0.50, 0.35, 0.25):
                    nw = max(8, int(round(small.shape[1] * scale_factor)))
                    nh = max(8, int(round(small.shape[0] * scale_factor)))
                    if nw >= curr_img.shape[1] and nh >= curr_img.shape[0]:
                        continue
                    curr_img = cv2.resize(host_img, (nw, nh), interpolation=cv2.INTER_AREA)
                    dev_img = engine.to_device(curr_img) if is_gpu_arr else curr_img
                    if task.qtables is not None:
                        trial_jpeg = splice_check(encode_fn(dev_img, task.quality))
                        trial_q = task.quality
                        trial_achieved = len(trial_jpeg) <= max_bytes
                    else:
                        trial_jpeg, trial_q, trial_achieved = optimize_jpeg_quality(
                            encode_fn,
                            dev_img,
                            initial_quality=task.quality,
                            min_quality=self.opts.min_quality,
                            max_bytes=max_bytes,
                            splice_fn=splice_check,
                            max_trials=4,
                            max_seconds=1.0,
                        )
                    if trial_achieved:
                        jpeg = trial_jpeg
                        actual_quality = trial_q
                        target_size_achieved = True
                        task = OutputTask(
                            task.variant_id,
                            task.out,
                            Plan(
                                src_w=task.plan.src_w,
                                src_h=task.plan.src_h,
                                out_w=nw,
                                out_h=nh,
                                x0=task.plan.x0,
                                y0=task.plan.y0,
                                cw=task.plan.cw,
                                ch=task.plan.ch,
                                canvas_w=nw if task.plan.is_pad else 0,
                                canvas_h=nh if task.plan.is_pad else 0,
                                pad_left=0,
                                pad_top=0,
                            ),
                            task.quality,
                            task.subsampling,
                            task.config_digest,
                            task.qtables,
                        )
                        break

            if not target_size_achieved and len(self._warnings) < 50:
                self._warnings.append(
                    f"{job.src}: target file size {max_bytes} bytes unreachable at min_quality {self.opts.min_quality} (actual {len(jpeg)} bytes)"
                )
        else:
            jpeg = engine.encode(small, task.quality, task.subsampling, task.qtables)
        return task, jpeg, actual_quality, target_size_achieved

    def _process_cpu_one(self, job: Job, engine=None, backend="cpu") -> None:
        engine = engine or self.engine
        try:
            import numpy as np
            from .color_management import (
                decode_and_manage_color,
                srgb_to_linear,
                linear_to_srgb,
                premultiply_alpha,
                composite_linear_bg,
                COLOR_POLICY_SRGB,
            )
            from .resampling import resample_cpu_float

            t0 = time.perf_counter()
            # An untagged RGB/RGBA image has no profile to convert; the sRGB
            # policy only needs to attach the standard output profile, which
            # _finish does. Decode it with OpenCV instead of the Pillow/CMS path.
            fast_untagged = (
                not job.meta.icc and job.meta.mode in ("RGB", "RGBA")
            )
            fast_srgb = (
                self.opts.color_policy == COLOR_POLICY_SRGB and fast_untagged
            )
            fast_linear = self.opts.linear_light and fast_untagged
            needs_cms = (
                (self.opts.color_policy == COLOR_POLICY_SRGB and not fast_srgb)
                or (self.opts.linear_light and not fast_linear)
                or (job.has_alpha and job.meta.mode != "RGBA")
                or (job.meta.icc and job.meta.mode not in ("RGB", "RGBA", "P"))
                or job.meta.mode not in ("RGB", "RGBA", "L", "CMYK")
            )
            tasks = self._job_tasks(job)
            reduce_scale = 1
            if not needs_cms and getattr(self.opts, "fast_jpeg_decode", False) and job.data.startswith(b"\xFF\xD8"):
                scales = [
                    max(t.plan.target_w / float(job.meta.display_width),
                        t.plan.target_h / float(job.meta.display_height))
                    for t in tasks
                ]
                max_s = max(scales) if scales else 1.0
                if max_s <= 0.125:
                    reduce_scale = 8
                elif max_s <= 0.25:
                    reduce_scale = 4
                elif max_s <= 0.50:
                    reduce_scale = 2

            if needs_cms:
                img_rgb, has_alpha, _ = decode_and_manage_color(
                    job.data, self.opts.color_policy, self.opts.keep_icc
                )
                if img_rgb.shape[2] == 4:
                    img = img_rgb[:, :, [2, 1, 0, 3]]
                else:
                    img = img_rgb[:, :, ::-1]
                img = np.ascontiguousarray(img)
            else:
                if reduce_scale > 1:
                    try:
                        img = engine.decode(job.data, reduce_scale=reduce_scale)
                    except TypeError:
                        reduce_scale = 1
                        img = engine.decode(job.data)
                else:
                    img = engine.decode(job.data)
                has_alpha = False
            t1 = time.perf_counter()

            if not self.opts.linear_light:
                if img.shape[2] == 4 or job.has_alpha or has_alpha:
                    img = engine.composite_bg(img, self.opts.bg_color)
                img = engine.apply_orientation(img, job.meta.orientation)
            else:
                img = engine.apply_orientation(img, job.meta.orientation)

            t2 = time.perf_counter()
            count = max(1, len(tasks))
            for task in tasks:
                try:
                    t3 = time.perf_counter()
                    exec_plan = task.plan
                    if reduce_scale > 1:
                        from .geometry import scale_plan_for_reduced_decode
                        exec_plan = scale_plan_for_reduced_decode(task.plan, reduce_scale)

                    if self.opts.linear_light:
                        if img.shape[2] == 4 or job.has_alpha or has_alpha:
                            bgr = img[:, :, :3]
                            rgb = bgr[:, :, ::-1]
                            rgb_lin = srgb_to_linear(rgb)
                            alpha = img[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
                            rgba_lin = np.concatenate([rgb_lin, alpha], axis=2)
                            rgba_premul = premultiply_alpha(rgba_lin)
                            res_premul = resample_cpu_float(
                                rgba_premul, exec_plan, self.opts.filter_name
                            )
                            comp_lin = composite_linear_bg(
                                res_premul, self.opts.bg_color, is_premultiplied=True
                            )
                            small = linear_to_srgb(comp_lin)[:, :, ::-1]
                        else:
                            # Channel order is irrelevant to the per-channel LUT
                            # and to the resampler, so stay in BGR and skip two
                            # full-frame channel reversals.
                            res_lin = resample_cpu_float(
                                srgb_to_linear(img), exec_plan, self.opts.filter_name
                            )
                            small = linear_to_srgb(res_lin)
                    else:
                        small = engine.resize(img, exec_plan)

                    if self.opts.sharpen.enabled:
                        from .sharpen import apply_sharpen
                        small = apply_sharpen(small, self.opts.sharpen)
                    if task.plan.is_pad:
                        from .geometry import apply_pad
                        small = apply_pad(small, task.plan, self.opts.bg_color)
                    t4 = time.perf_counter()

                    task, jpeg, actual_quality, target_size_achieved = self._encode_with_size_limit(job, task, small, engine)
                    t5 = time.perf_counter()
                    timings = {"decode": (t1 - t0) / count,
                               "normalize": (t2 - t1) / count,
                               "resize": t4 - t3, "encode": t5 - t4}
                    self._finish(job, task, jpeg, timings, backend,
                                 actual_quality=actual_quality,
                                 target_size_achieved=target_size_achieved)
                except Exception as exc:
                    self._fail_output(job, task, exc)
                    self._complete_output(job.src)
        except Exception as exc:
            self._fail(job.src, exc)

    def _process_gpu_batch(self, batch):
        with self._state_lock:
            batch = [
                j for j in batch
                if any((j.src, task.variant_id) not in self._terminal
                       for task in self._job_tasks(j))
            ]
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
        if self.opts.linear_light:
            for job in batch:
                self._gpu_fallback(job, RuntimeError("linear-light resize is not supported on GPU backend"))
            return

        # Managed colour conversion cannot copy a CMYK/gray profile onto RGB.
        from .color_management import COLOR_POLICY_SRGB
        if self.opts.color_policy == COLOR_POLICY_SRGB:
            managed = [j for j in batch if j.meta.icc or j.meta.mode not in ("RGB", "RGBA")]
        else:
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
                tasks = self._job_tasks(job)
                count = max(1, len(tasks))
                for task in tasks:
                    try:
                        small = self.engine.resize(arr, task.plan)
                        if self.opts.sharpen.enabled:
                            from .sharpen import apply_sharpen
                            small = apply_sharpen(small, self.opts.sharpen)
                        if task.plan.is_pad:
                            from .geometry import apply_pad
                            small = apply_pad(small, task.plan, self.opts.bg_color)
                        self._sync_profile()
                        t3 = time.perf_counter()
                        task, jpeg, actual_quality, target_size_achieved = self._encode_with_size_limit(
                            job, task, small, self.engine)
                        self._sync_profile()
                        self._finish(job, task, jpeg, {
                            "decode": decode_time / count,
                            "normalize": (t2 - t1) / count,
                            "resize": t3 - t2,
                            "encode": time.perf_counter() - t3,
                        }, "nvimgcodec-auto", actual_quality=actual_quality, target_size_achieved=target_size_achieved)
                    except Exception as exc:
                        self._gpu_fallback(job, exc)
                        break
            except Exception as exc:
                self._gpu_fallback(job, exc)

    def _sync_profile(self):
        if self.opts.profile_cuda and hasattr(self.engine, "synchronize"):
            self.engine.synchronize()

    def _process_gpu_one(self, job):
        self._process_gpu_batch([job])

    def _process_hybrid_one(self, job):
        if self.opts.linear_light:
            self._gpu_fallback(job, RuntimeError("linear-light resize is not supported on GPU backend"))
            return

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
            with self._gpu_lock:
                arr = engine.to_device(rgb)
                arr = engine.apply_orientation(arr, job.meta.orientation)
                self._sync_profile()
                t2 = time.perf_counter()
                if int(arr.shape[2]) == 4:
                    arr = engine.alpha_to_rgb(arr, self.opts.bg_color)
                self._sync_profile()
                tasks = self._job_tasks(job)
                count = max(1, len(tasks))
                for task in tasks:
                    small = engine.resize(arr, task.plan)
                    if self.opts.sharpen.enabled:
                        from .sharpen import apply_sharpen
                        small = apply_sharpen(small, self.opts.sharpen)
                    if task.plan.is_pad:
                        from .geometry import apply_pad
                        small = apply_pad(small, task.plan, self.opts.bg_color)
                    self._sync_profile()
                    t3 = time.perf_counter()
                    task, jpeg, actual_quality, target_size_achieved = self._encode_with_size_limit(
                        job, task, small, engine)
                    self._sync_profile()
                    t4 = time.perf_counter()
                    timings = {"decode": (t1 - t0) / count,
                               "normalize": (t2 - t1) / count,
                               "resize": t3 - t2, "encode": t4 - t3}
                    self._sync_profile()
                    self._finish(job, task, jpeg, timings, "hybrid",
                                 actual_quality=actual_quality, target_size_achieved=target_size_achieved)
        except Exception:
            raise

    def _gpu_fallback(self, job, exc):
        if not self.opts.fallback:
            self._fail(job.src, exc)
            return
        with self._state_lock:
            unresolved = [
                task for task in self._job_tasks(job)
                if (job.src, task.variant_id) not in self._terminal
                and (job.src, task.variant_id) not in self._pending
            ]
            if not unresolved:
                return
            if len(self._warnings) < 50:
                self._warnings.append(f"{job.src}: GPU rejected file; full CPU fallback: {exc}")
        self.stats.record_fallback(str(exc))
        try:
            with self._working(job):
                self._process_cpu_one(job, CpuEngine(self.opts.filter_name,
                                                     self.opts.optimize_huffman), "cpu-fallback")
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
        total = self._outputs_total or self.discovered
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
                    "sources_total": self.discovered,
                    "outputs_total": self._outputs_total,
                    "final": final,
                })
            except Exception:
                pass
            return
        line = (f"\r{done}/{total} outputs | {rate:.1f} output/s | "
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
        self._join_bounded(self._readers + self._processors)
        self._join_writers(self._writers)
        self._progress(t_start, final=True)

    def _join_bounded(self, threads):
        deadline = time.monotonic() + JOIN_TIMEOUT
        for t in threads:
            t.join(timeout=max(0, deadline - time.monotonic()))
        alive = [t.name for t in threads if t.is_alive()]
        if alive:
            self.fatal.set(); self.stop.set()
            self._fatal_errors.append("native operation did not stop: " + ", ".join(alive))

    def _join_writers(self, writers):
        """Wait for writers to drain instead of treating a full queue as a hang.

        The queue can still hold results when processing ends, so a fixed join
        timeout is not a valid liveness test. A writer is considered stuck only
        when no file completes for ``_writer_stall_timeout`` seconds.
        """
        if not writers:
            return
        last_done = self._completed_count()
        last_change = time.monotonic()
        while any(t.is_alive() for t in writers):
            now = time.monotonic()
            done = self._completed_count()
            if done != last_done:
                last_done = done
                last_change = now
            elif now - last_change >= self._writer_stall_timeout:
                alive = [t.name for t in writers if t.is_alive()]
                self.fatal.set(); self.stop.set()
                self._fatal_errors.append("native operation did not stop: " + ", ".join(alive))
                return
            time.sleep(.02)

    def _completed_count(self):
        snap = self.stats.snapshot()
        return snap["processed"] + snap["failed"] + snap["skipped"]

    def _shutdown(self):
        self.stop.set()
        self._join_bounded(self._readers + self._processors)
        self._send_writer_sentinels()
        self._join_writers(self._writers)

    def run(self) -> dict:
        opts = self.opts
        check_roots(opts.input_dir, opts.output_dir.resolve())
        if not opts.input_dir.is_dir():
            raise ValueError("input directory does not exist")
        config = {k: getattr(opts, k) for k in ("max_w", "max_h", "mode", "quality",
            "subsampling", "sizing", "sizing_value", "allow_upscale", "filter_name",
            "bg_color", "keep_exif", "keep_icc", "optimize_huffman",
            "color_policy", "linear_light", "strip_personal_data", "keep_datetime",
            "crop_anchor", "crop_focus", "max_file_size", "min_quality", "allow_downscale_on_size_limit",
            "fast_jpeg_decode")}
        config["sharpen"] = opts.sharpen.to_dict()
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
                      libraries=versions, input_root=str(opts.input_dir),
                      variants=[variant.to_profile_dict() for variant in opts.variants])
        self._resume = ResumeState(opts.output_dir, config)
        self._variant_digests = {}
        for variant in opts.variants:
            active = dict(config)
            active["active_variant"] = variant.to_profile_dict()
            self._variant_digests[variant.id] = hashlib.sha256(
                json.dumps(active, sort_keys=True).encode()).hexdigest()
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
        self._variant_maps = {}
        self._targets_by_source = {}
        if opts.variants:
            for variant in opts.variants:
                previous = self._resume.previous_mapping(
                    all_files, opts.input_dir, variant.id)
                self._variant_maps[variant.id] = build_output_map(
                    files, opts.input_dir, opts.output_dir / variant.id, previous)
            self._out_map = self._variant_maps[opts.variants[0].id]
            for src in files:
                self._targets_by_source[src] = {
                    variant.id: self._variant_maps[variant.id][src]
                    for variant in opts.variants
                }
        else:
            previous = self._resume.previous_mapping(all_files, opts.input_dir)
            self._out_map = build_output_map(
                files, opts.input_dir, opts.output_dir, previous)
            for src in files:
                self._targets_by_source[src] = {None: self._out_map[src]}
        for mapping in self._targets_by_source.values():
            for target in mapping.values():
                protect_target(target, opts.output_dir,
                               self._sources, self._identities)
        variant_ids = tuple(variant.id for variant in opts.variants) or (None,)
        self._expected = {
            (src, variant_id)
            for src in files for variant_id in variant_ids
        }
        self._remaining = {src: len(variant_ids) for src in files}
        self._outputs_total = len(self._expected)
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
                self._join_bounded(self._readers + self._processors)
                self._join_writers(self._writers)

        for src in files:
            with self._state_lock:
                done = all(key in self._terminal for key in self._expected
                           if key[0] == src)
            if not done:
                self._fail(src, RuntimeError(
                    "cancelled" if interrupted else "worker did not complete job"))
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
        with self._state_lock:
            sources_complete = sum(
                1 for src in {key[0] for key in self._expected}
                if all(key in self._terminal for key in self._expected
                       if key[0] == src)
            )
        processed = snap["processed"]
        per = (lambda v: round(1000.0 * v / processed, 3)) if processed else (lambda v: 0.0)
        from .observability import calculate_latency_stats
        used_gpu_decode, used_gpu_compute = actual_route_flags(snap["backend_counts"])
        return {
            "complete": not interrupted and not self.fatal.is_set() and not self._discovery_errors
                        and snap["failed"] == 0,
            "fatal_errors": list(self._fatal_errors),
            "discovery_errors": list(self._discovery_errors),
            "warnings": list(self._warnings)[:50],
            "scan_sec": self.scan_seconds,
            "backend_counts": snap["backend_counts"],
            "requested_backend": self.backend,
            "actual_routes": {
                "decode": ("nvImageCodec automatic plugin" if used_gpu_decode
                           else "OpenCV/Pillow CPU"),
                "resize": ("CUDA separable kernel" if used_gpu_compute
                           else "Pillow/CPU float"),
                "encode": ("nvJPEG GPU" if used_gpu_compute
                           else "OpenCV CPU JPEG"),
            },
            "codec_backend": getattr(self.engine, "backend_policy", "CPU/Pillow/OpenCV"),
            "codec_backend_attested": False,
            "fallback_reasons": snap.get("fallback_reasons", {}),
            "backend_bytes": snap.get("backend_bytes", {}),
            "host_wall_time_sec": round(elapsed, 3),
            "latency_stats_ms": calculate_latency_stats(snap.get("latencies", [])),
            "gpu_service_time_ms": round(snap.get("gpu_service_time", 0.0), 3) if snap.get("gpu_service_time") else None,
            "fast_jpeg_decode": getattr(self.opts, "fast_jpeg_decode", False),
            "timing_semantics": ("synchronized service wall time (profiling changes throughput)"
                if self.opts.profile_cuda else "host service envelopes; GPU launches asynchronous, not kernel execution time"),
            "input_validation": "decoder acceptance + JPEG marker check; not an entropy integrity certificate",
            "vram_budget_semantics": "soft application array estimate; excludes opaque codec/driver/pool allocations",
            "extra_timings_avg_ms": {k: per(v) for k, v in snap["extra_timings"].items()},
            "input": str(self.opts.input_dir),
            "output": str(self.opts.output_dir),
            "backend": self.backend,
            "engine": self.backend,
            "filter": self.opts.filter_name,
            "sharpen": self.opts.sharpen.to_dict(),
            "color_policy": self.opts.color_policy,
            "linear_light": self.opts.linear_light,
            "strip_personal_data": self.opts.strip_personal_data,
            "keep_datetime": self.opts.keep_datetime,
            "mode": self.opts.mode,
            "crop_anchor": self.opts.crop_anchor,
            "crop_focus": self.opts.crop_focus,
            "max_file_size": self.opts.max_file_size,
            "min_quality": self.opts.min_quality,
            "allow_downscale_on_size_limit": self.opts.allow_downscale_on_size_limit,
            "target_size_achieved": self._target_size_achieved,
            "discovered": self.discovered,
            "sources_total": self.discovered,
            "sources_complete": sources_complete,
            "outputs_total": self._outputs_total,
            "variants": [variant.to_profile_dict() for variant in self.opts.variants],
            "processed": processed,
            "failed": snap["failed"],
            "skipped": snap["skipped"],
            "fallback_files": snap["fallback"],
            "bytes_in": snap["bytes_in"],
            "bytes_out": snap["bytes_out"],
            "elapsed_sec": round(elapsed, 3),
            "files_per_sec": round(processed / elapsed, 2),
            "outputs_per_sec": round(processed / elapsed, 2),
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
