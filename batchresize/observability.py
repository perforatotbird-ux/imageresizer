"""Observability, telemetry, latency tracking and quality metrics for Stage 8."""

from __future__ import annotations

import math
import os
import platform
import sys
from typing import Optional, Sequence
import numpy as np


def compute_image_metrics(img_a: np.ndarray, img_b: np.ndarray) -> dict[str, float]:
    """Compute objective quality metrics (PSNR and MAE) between two images.

    Both images must have the same shape and dtype uint8.
    """
    if img_a.shape != img_b.shape:
        raise ValueError(f"image shapes must match: {img_a.shape} vs {img_b.shape}")

    diff = np.abs(img_a.astype(np.float64) - img_b.astype(np.float64))
    mae = float(np.mean(diff))
    mse = float(np.mean(diff ** 2))

    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = 10.0 * math.log10((255.0 ** 2) / mse)

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "psnr_db": round(psnr, 2) if math.isfinite(psnr) else 999.0,
    }


def calculate_latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Calculate descriptive statistics for latency samples in milliseconds."""
    if not latencies_ms:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    s = sorted(latencies_ms)
    n = len(s)
    mean = sum(s) / n
    median = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
    p95 = s[p95_idx]

    return {
        "count": n,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "p95": round(p95, 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
    }


def get_system_info() -> dict:
    """Collect hardware, OS and runtime environment metadata."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count() or 1,
        "gpu": None,
    }

    # Attempt to detect GPU via CuPy
    try:
        import cupy as cp
        dev = cp.cuda.Device(0)
        free_bytes, total_bytes = dev.mem_info
        props = cp.cuda.runtime.getDeviceProperties(0)
        device_name = props.get("name", "CUDA Device").decode("utf-8", "ignore") if isinstance(props.get("name"), bytes) else str(props.get("name", "CUDA Device"))
        rt_version = cp.cuda.runtime.runtimeGetVersion()
        driver_version = cp.cuda.runtime.driverGetVersion()
        info["gpu"] = {
            "name": device_name,
            "vram_total_mb": round(total_bytes / (1024 ** 2), 1),
            "vram_free_mb": round(free_bytes / (1024 ** 2), 1),
            "cuda_runtime": f"{rt_version // 1000}.{(rt_version % 1000) // 10}",
            "cuda_driver": f"{driver_version // 1000}.{(driver_version % 1000) // 10}",
        }
    except Exception:
        info["gpu"] = None

    return info


class CudaTimer:
    """Accurate GPU execution timer using CUDA events in the specified stream."""

    def __init__(self, cp_module=None, stream=None):
        self._cp = cp_module
        self._stream = stream
        self._start_ev = None
        self._stop_ev = None
        if self._cp is not None and hasattr(self._cp, "cuda"):
            try:
                self._start_ev = self._cp.cuda.Event()
                self._stop_ev = self._cp.cuda.Event()
            except Exception:
                self._start_ev = None
                self._stop_ev = None

    def start(self) -> None:
        if self._start_ev is not None:
            try:
                self._start_ev.record(self._stream)
            except Exception:
                pass

    def stop(self) -> Optional[float]:
        """Stop timing, synchronize stop event, and return elapsed milliseconds."""
        if self._start_ev is not None and self._stop_ev is not None:
            try:
                self._stop_ev.record(self._stream)
                self._stop_ev.synchronize()
                return float(self._cp.cuda.get_elapsed_time(self._start_ev, self._stop_ev))
            except Exception:
                return None
        return None
