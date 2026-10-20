"""Tests for Stage 8: observability, latency tracking, fast reduced JPEG decode and benchmarking."""

import io
import json
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from batchresize import cli
from batchresize.cpu_backend import CpuEngine
from batchresize.geometry import make_plan, scale_plan_for_reduced_decode
from batchresize.observability import compute_image_metrics
from batchresize.pipeline import Pipeline, RunOptions


def _create_jpeg(path: Path, size=(1600, 1200), color=(120, 180, 220)) -> None:
    img = Image.new("RGB", size, color)
    img.save(path, "JPEG", quality=90)


def _create_png(path: Path, size=(64, 48), color=(255, 0, 0)) -> None:
    img = Image.new("RGB", size, color)
    img.save(path, "PNG")


def test_scale_plan_for_reduced_decode():
    plan = make_plan(1600, 1200, 200, 150, mode="contain")
    scaled2 = scale_plan_for_reduced_decode(plan, 2)
    assert scaled2.src_w == 800
    assert scaled2.src_h == 600
    assert scaled2.out_w == 200
    assert scaled2.out_h == 150

    scaled4 = scale_plan_for_reduced_decode(plan, 4)
    assert scaled4.src_w == 400
    assert scaled4.src_h == 300
    assert scaled4.out_w == 200
    assert scaled4.out_h == 150

    scaled8 = scale_plan_for_reduced_decode(plan, 8)
    assert scaled8.src_w == 200
    assert scaled8.src_h == 150
    assert scaled8.out_w == 200
    assert scaled8.out_h == 150

    no_scale = scale_plan_for_reduced_decode(plan, 1)
    assert no_scale == plan


def test_fast_jpeg_decode_end_to_end(tmp_path):
    src_dir = tmp_path / "src"
    out_full = tmp_path / "out_full"
    out_fast = tmp_path / "out_fast"
    src_dir.mkdir()

    # Create a 1600x1200 image
    _create_jpeg(src_dir / "photo.jpg", size=(1600, 1200))

    # Run full decode
    opts_full = RunOptions(
        input_dir=src_dir,
        output_dir=out_full,
        max_w=200,
        max_h=150,
        fast_jpeg_decode=False,
        workers=1,
    )
    rep_full = Pipeline(opts_full, CpuEngine("lanczos3"), "cpu", progress=False).run()
    assert rep_full["processed"] == 1
    assert rep_full["fast_jpeg_decode"] is False

    # Run fast reduced decode
    opts_fast = RunOptions(
        input_dir=src_dir,
        output_dir=out_fast,
        max_w=200,
        max_h=150,
        fast_jpeg_decode=True,
        workers=1,
    )
    rep_fast = Pipeline(opts_fast, CpuEngine("lanczos3"), "cpu", progress=False).run()
    assert rep_fast["processed"] == 1
    assert rep_fast["fast_jpeg_decode"] is True

    # Verify output sizes and visual closeness
    with Image.open(out_full / "photo.jpg") as im_full:
        assert im_full.size == (200, 150)
        arr_full = np.array(im_full)

    with Image.open(out_fast / "photo.jpg") as im_fast:
        assert im_fast.size == (200, 150)
        arr_fast = np.array(im_fast)

    metrics = compute_image_metrics(arr_full, arr_fast)
    assert metrics["psnr_db"] > 30.0  # Visually close despite fast decode


def test_fast_jpeg_decode_safe_for_png_and_cms(tmp_path):
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    src_dir.mkdir()

    _create_png(src_dir / "sample.png")

    opts = RunOptions(
        input_dir=src_dir,
        output_dir=out_dir,
        max_w=32,
        max_h=32,
        fast_jpeg_decode=True,
        color_policy="web_srgb",
        workers=1,
    )
    rep = Pipeline(opts, CpuEngine("lanczos3"), "cpu", progress=False).run()
    assert rep["processed"] == 1
    assert (out_dir / "sample.jpg").exists()


def test_report_observability_fields(tmp_path):
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    _create_jpeg(src_dir / "a.jpg", size=(100, 100))
    _create_jpeg(src_dir / "b.jpg", size=(120, 120))

    opts = RunOptions(
        input_dir=src_dir,
        output_dir=out_dir,
        max_w=50,
        max_h=50,
        workers=2,
    )
    rep = Pipeline(opts, CpuEngine("lanczos3"), "cpu", progress=False).run()

    # Observability assertions
    assert rep["requested_backend"] == "cpu"
    assert "actual_routes" in rep
    assert "decode" in rep["actual_routes"]
    assert "resize" in rep["actual_routes"]
    assert "encode" in rep["actual_routes"]
    assert rep["codec_backend_attested"] is False
    assert "fallback_reasons" in rep
    assert "backend_bytes" in rep
    assert "cpu" in rep["backend_bytes"]
    assert rep["backend_bytes"]["cpu"]["bytes_out"] > 0
    assert "latency_stats_ms" in rep
    assert rep["latency_stats_ms"]["count"] == 2
    assert rep["latency_stats_ms"]["mean"] >= 0.0
    assert "host_wall_time_sec" in rep


def test_gpu_fallback_reasons_in_report(tmp_path):
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    _create_jpeg(src_dir / "a.jpg", size=(64, 48))

    class MockGpuReject:
        def self_test(self):
            return True, "ok"

        def decode_batch(self, *args):
            raise RuntimeError("CUDA device out of memory simulation")

    opts = RunOptions(
        input_dir=src_dir,
        output_dir=out_dir,
        max_w=32,
        max_h=32,
        workers=1,
    )
    pipe = Pipeline(opts, MockGpuReject(), "gpu", progress=False)
    rep = pipe.run()

    assert rep["processed"] == 1
    assert rep["backend_counts"].get("cpu-fallback", 0) == 1
    assert any("out of memory" in k for k in rep["fallback_reasons"])


def test_benchmark_stage8_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "GpuEngine", lambda **kw: (_ for _ in ()).throw(RuntimeError("no GPU")))
    src_dir = tmp_path / "src"
    bench_dir = tmp_path / "bench"
    src_dir.mkdir()
    _create_jpeg(src_dir / "test.jpg", size=(80, 60))

    opts = RunOptions(
        input_dir=src_dir,
        output_dir=bench_dir,
        max_w=40,
        max_h=30,
        workers=1,
    )
    rc = cli._benchmark(opts, repeats=2, quiet=True)
    assert rc == 0

    bench_json_path = bench_dir / "benchmark.json"
    assert bench_json_path.exists()
    data = json.loads(bench_json_path.read_text(encoding="utf-8"))

    assert "cpu" in data
    cpu = data["cpu"]
    assert "cold_run" in cpu
    assert "warm_runs" in cpu
    assert len(cpu["warm_runs"]) == 1
    assert "cold_time_sec" in cpu
    assert "warm_median_time_sec" in cpu
    assert "system_info" in cpu
    assert cpu["system_info"]["cpu_count"] >= 1
    assert [r["discovered"] for r in cpu["runs"]] == [1, 1]  # IR19 backward-compatibility
