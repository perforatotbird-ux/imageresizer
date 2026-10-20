"""Unit tests for Stage 8 observability, latency statistics, and image quality metrics."""

import math
import numpy as np
import pytest

from batchresize.observability import (
    CudaTimer,
    calculate_latency_stats,
    compute_image_metrics,
    get_system_info,
)


def test_compute_image_metrics_identical():
    img1 = np.full((64, 64, 3), 128, dtype=np.uint8)
    img2 = img1.copy()
    m = compute_image_metrics(img1, img2)
    assert m["mae"] == 0.0
    assert m["mse"] == 0.0
    assert m["psnr_db"] == 999.0


def test_compute_image_metrics_diff():
    img1 = np.zeros((10, 10, 3), dtype=np.uint8)
    img2 = np.full((10, 10, 3), 10, dtype=np.uint8)
    m = compute_image_metrics(img1, img2)
    assert m["mae"] == 10.0
    assert m["mse"] == 100.0
    expected_psnr = 10.0 * math.log10((255.0 ** 2) / 100.0)
    assert pytest.approx(m["psnr_db"], 0.01) == expected_psnr


def test_compute_image_metrics_shape_mismatch():
    img1 = np.zeros((10, 10, 3), dtype=np.uint8)
    img2 = np.zeros((20, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shapes must match"):
        compute_image_metrics(img1, img2)


def test_calculate_latency_stats_empty():
    stats = calculate_latency_stats([])
    assert stats["count"] == 0
    assert stats["mean"] == 0.0
    assert stats["median"] == 0.0
    assert stats["p95"] == 0.0


def test_calculate_latency_stats_single():
    stats = calculate_latency_stats([42.5])
    assert stats["count"] == 1
    assert stats["mean"] == 42.5
    assert stats["median"] == 42.5
    assert stats["min"] == 42.5
    assert stats["max"] == 42.5


def test_calculate_latency_stats_multiple():
    samples = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    stats = calculate_latency_stats(samples)
    assert stats["count"] == 10
    assert stats["min"] == 10.0
    assert stats["max"] == 100.0
    assert stats["mean"] == 55.0
    assert stats["median"] == 55.0
    assert stats["p95"] >= 90.0


def test_get_system_info():
    info = get_system_info()
    assert "os" in info
    assert "platform" in info
    assert "python" in info
    assert "cpu_count" in info
    assert info["cpu_count"] >= 1
    assert "gpu" in info  # None or dict with vram and device name


def test_cuda_timer_mock_or_none():
    timer = CudaTimer(None, None)
    timer.start()
    res = timer.stop()
    assert res is None


def test_cuda_timer_mock_stream():
    class MockCuda:
        def Event(self):
            return self

        def record(self, stream):
            pass

        def synchronize(self):
            pass

    class MockModule:
        def __init__(self):
            self.cuda = MockCuda()

        def __getattr__(self, name):
            if name == "cuda":
                return self.cuda
            raise AttributeError(name)

    mod = MockModule()
    mod.cuda.get_elapsed_time = lambda e1, e2: 12.34
    timer = CudaTimer(mod, None)
    timer.start()
    elapsed = timer.stop()
    assert elapsed == 12.34
