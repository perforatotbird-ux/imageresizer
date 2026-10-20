"""End-to-end integration tests for Stage 6: pad mode, crop anchors/focus, file size limits."""

import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from batchresize import cli
from batchresize.cpu_backend import CpuEngine
from batchresize.pipeline import Pipeline, RunOptions


def _create_gradient_img(path: Path, width: int = 100, height: int = 200):
    # Vertical gradient from black at top to white at bottom
    arr = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    arr = np.repeat(arr, width, axis=1)
    rgb = np.stack([arr, arr, arr], axis=2)
    Image.fromarray(rgb).save(path, "PNG")


def _create_noise_img(path: Path, width: int = 400, height: int = 300):
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, "PNG")


class TestPadModePipeline:
    def test_pad_canvas_and_bg_color(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # 200x100 image into 100x100 pad box
        img_path = src_dir / "wide.png"
        Image.new("RGB", (200, 100), color=(0, 255, 0)).save(img_path)

        opts = RunOptions(
            input_dir=src_dir,
            output_dir=out_dir,
            max_w=100,
            max_h=100,
            mode="pad",
            bg_color=(255, 0, 0),  # Red background
            workers=1,
        )
        report = Pipeline(opts, CpuEngine(), "cpu", progress=False).run()
        assert report["processed"] == 1
        assert report["complete"] is True
        assert report["mode"] == "pad"

        out_file = out_dir / "wide.jpg"
        assert out_file.exists()
        with Image.open(out_file) as im:
            assert im.size == (100, 100)
            arr = np.asarray(im)
            # Top-left corner should be red padding (255, 0, 0)
            # Allow minor JPEG compression tolerance
            assert arr[0, 0, 0] > 240
            assert arr[0, 0, 1] < 15
            assert arr[0, 0, 2] < 15

    def test_pad_resume_validation(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        Image.new("RGB", (200, 100), color=(10, 20, 30)).save(src_dir / "img.png")

        opts = RunOptions(
            input_dir=src_dir,
            output_dir=out_dir,
            max_w=100,
            max_h=100,
            mode="pad",
            workers=1,
            skip_existing=True,
            overwrite=False,
        )
        rep1 = Pipeline(opts, CpuEngine(), "cpu", progress=False).run()
        assert rep1["processed"] == 1
        assert rep1["skipped"] == 0

        # Second run should safely verify and skip
        rep2 = Pipeline(opts, CpuEngine(), "cpu", progress=False).run()
        assert rep2["processed"] == 0
        assert rep2["skipped"] == 1


class TestCropAnchorsAndFocusPipeline:
    def test_crop_anchors_top_vs_bottom(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_top = tmp_path / "out_top"
        out_bot = tmp_path / "out_bot"

        img_path = src_dir / "grad.png"
        _create_gradient_img(img_path, width=100, height=200)

        # Cover into 100x100
        opts_top = RunOptions(
            input_dir=src_dir, output_dir=out_top,
            max_w=100, max_h=100, mode="cover", crop_anchor="top", workers=1
        )
        opts_bot = RunOptions(
            input_dir=src_dir, output_dir=out_bot,
            max_w=100, max_h=100, mode="cover", crop_anchor="bottom", workers=1
        )

        Pipeline(opts_top, CpuEngine(), "cpu", progress=False).run()
        Pipeline(opts_bot, CpuEngine(), "cpu", progress=False).run()

        with Image.open(out_top / "grad.jpg") as im_t, Image.open(out_bot / "grad.jpg") as im_b:
            arr_t = np.asarray(im_t)
            arr_b = np.asarray(im_b)
            # Top anchor should have lower average brightness than bottom anchor
            assert arr_t.mean() < arr_b.mean()

    def test_crop_focus_point(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_f1 = tmp_path / "out_f1"
        out_f2 = tmp_path / "out_f2"

        img_path = src_dir / "grad.png"
        _create_gradient_img(img_path, width=100, height=200)

        opts_f1 = RunOptions(
            input_dir=src_dir, output_dir=out_f1,
            max_w=100, max_h=100, mode="cover", crop_focus=(0.5, 0.1), workers=1
        )
        opts_f2 = RunOptions(
            input_dir=src_dir, output_dir=out_f2,
            max_w=100, max_h=100, mode="cover", crop_focus=(0.5, 0.9), workers=1
        )

        Pipeline(opts_f1, CpuEngine(), "cpu", progress=False).run()
        Pipeline(opts_f2, CpuEngine(), "cpu", progress=False).run()

        with Image.open(out_f1 / "grad.jpg") as im1, Image.open(out_f2 / "grad.jpg") as im2:
            arr1 = np.asarray(im1)
            arr2 = np.asarray(im2)
            assert arr1.mean() < arr2.mean()


class TestSizeLimitPipeline:
    def test_target_file_size_achieved(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"

        img_path = src_dir / "photo.png"
        _create_noise_img(img_path, width=400, height=300)

        # Baseline without size limit
        opts_base = RunOptions(
            input_dir=src_dir, output_dir=out_dir / "base",
            max_w=400, max_h=300, quality=90, workers=1
        )
        rep_base = Pipeline(opts_base, CpuEngine(), "cpu", progress=False).run()
        base_size = (out_dir / "base/photo.jpg").stat().st_size

        # Request a limit of half the baseline size
        target_limit = base_size // 2
        opts_limited = RunOptions(
            input_dir=src_dir, output_dir=out_dir / "limited",
            max_w=400, max_h=300, quality=90, min_quality=20,
            max_file_size=target_limit, workers=1
        )
        rep_limited = Pipeline(opts_limited, CpuEngine(), "cpu", progress=False).run()

        assert rep_limited["target_size_achieved"] is True
        out_file = out_dir / "limited/photo.jpg"
        actual_size = out_file.stat().st_size
        assert actual_size <= target_limit

        # Verify state file record
        state_file = out_dir / "limited/.batchresize-state.json"
        state = json.loads(state_file.read_text())
        entry = list(state["entries"].values())[0]
        assert entry["target_size_achieved"] is True
        assert "actual_quality" in entry
        assert entry["actual_quality"] < 90

    def test_unreachable_limit_emits_warning_preserves_dimensions(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"

        img_path = src_dir / "photo.png"
        _create_noise_img(img_path, width=300, height=300)

        opts = RunOptions(
            input_dir=src_dir, output_dir=out_dir,
            max_w=300, max_h=300, quality=80, min_quality=50,
            max_file_size=500,  # Impossible 500 bytes
            allow_downscale_on_size_limit=False,
            workers=1,
        )
        report = Pipeline(opts, CpuEngine(), "cpu", progress=False).run()

        assert report["target_size_achieved"] is False
        assert any("unreachable" in w for w in report["warnings"])
        out_file = out_dir / "photo.jpg"
        assert out_file.exists()
        with Image.open(out_file) as im:
            # Resolution preserved
            assert im.size == (300, 300)

    def test_allow_downscale_on_size_limit(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"

        img_path = src_dir / "photo.png"
        _create_noise_img(img_path, width=300, height=300)

        opts = RunOptions(
            input_dir=src_dir, output_dir=out_dir,
            max_w=300, max_h=300, quality=80, min_quality=40,
            max_file_size=3000,
            allow_downscale_on_size_limit=True,
            workers=1,
        )
        report = Pipeline(opts, CpuEngine(), "cpu", progress=False).run()

        assert report["target_size_achieved"] is True
        out_file = out_dir / "photo.jpg"
        assert out_file.stat().st_size <= 3000
        with Image.open(out_file) as im:
            # Verified downscaled resolution
            assert im.size[0] < 300


class TestCliStage6Integration:
    def test_cli_stage6_options_roundtrip(self, tmp_path, monkeypatch):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"

        Image.new("RGB", (100, 100), color=(128, 128, 128)).save(src_dir / "test.png")

        argv = [
            "batchresize",
            "-i", str(src_dir),
            "-o", str(out_dir),
            "--mode", "pad",
            "--crop-anchor", "top-left",
            "--max-file-size", "50KiB",
            "--min-quality", "35",
            "--allow-downscale-on-size-limit",
            "--backend", "cpu",
            "--quiet",
        ]
        monkeypatch.setattr("sys.argv", argv)
        exit_code = cli.main()
        assert exit_code == 0

        report_file = out_dir / "report.json"
        assert report_file.exists()
        report = json.loads(report_file.read_text())
        assert report["mode"] == "pad"
        assert report["crop_anchor"] == "top-left"
        assert report["max_file_size"] == 51200
        assert report["min_quality"] == 35
        assert report["allow_downscale_on_size_limit"] is True
        assert report["target_size_achieved"] is True
