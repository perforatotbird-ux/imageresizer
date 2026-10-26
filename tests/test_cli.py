import pytest

from pathlib import Path

from batchresize.cli import build_parser, main, parse_color
from batchresize.presets import (auto_subsampling, resolve_quality, resolve_size)


def test_quality_preset_values():
    assert resolve_quality("compact", None) == 65
    assert resolve_quality("balanced", None) == 80
    assert resolve_quality("high", None) == 90
    assert resolve_quality("very_high", None) == 95
    assert resolve_quality(None, 42) == 42


@pytest.mark.parametrize("bad", [0, 101, -5])
def test_custom_quality_out_of_range(bad):
    with pytest.raises(ValueError):
        resolve_quality(None, bad)


def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        resolve_quality("ultra", None)
    with pytest.raises(ValueError):
        resolve_size("huge", None, None)


def test_size_presets():
    assert resolve_size("thumb_256", None, None) == (256, 256)
    assert resolve_size("small_512", None, None) == (512, 512)
    assert resolve_size("web_1024", None, None) == (1024, 1024)
    assert resolve_size("large_1600", None, None) == (1600, 1600)
    assert resolve_size(None, 800, 600) == (800, 600)


def test_half_custom_size_rejected():
    with pytest.raises(ValueError):
        resolve_size(None, 800, None)


def test_auto_subsampling_curve():
    assert auto_subsampling(65) == 2
    assert auto_subsampling(85) == 2
    assert auto_subsampling(90) == 1
    assert auto_subsampling(94) == 1
    assert auto_subsampling(95) == 0
    assert auto_subsampling(100) == 0


def test_colour_parsing():
    assert parse_color("#FFFFFF") == (255, 255, 255)
    assert parse_color("#000") == (0, 0, 0)
    assert parse_color("10,20,30") == (10, 20, 30)
    with pytest.raises(Exception):
        parse_color("nope")


def test_cli_uses_fixed_folders():
    args = build_parser().parse_args([])
    assert Path(args.input).name == "input"
    assert Path(args.output).name == "output"


def test_cli_rejects_same_in_out(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    with pytest.raises(SystemExit):
        main(["-i", str(tmp_path), "-o", str(tmp_path)])


def test_cli_rejects_bad_quality(tmp_path):
    with pytest.raises(SystemExit):
        main(["-i", str(tmp_path), "-o", str(tmp_path / "o"),
              "--jpeg-quality", "0"])


def test_cli_custom_size_needs_width_height(tmp_path):
    with pytest.raises(SystemExit):
        main(["-i", str(tmp_path), "-o", str(tmp_path / "o"), "--size", "custom"])


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.mode == "contain"
    assert args.backend == "auto"
    assert args.subsampling == "auto"
    assert args.workers >= 1
    assert args.batch_size >= 1


def test_cli_working_budget_override():
    from batchresize.cli import _make_options
    args = build_parser().parse_args(["--working-mb", "512"])
    opts = _make_options(args, Path("input"), Path("output"))
    assert opts.max_working_bytes == 512 * 2**20


def test_cli_sharpen_enabled_without_profile():
    from batchresize.cli import _make_options
    args = build_parser().parse_args(["--sharpen", "standard"])
    opts = _make_options(args, Path("input"), Path("output"))
    assert opts.sharpen.enabled is True
    assert opts.sharpen.preset == "standard"


def test_cli_low_jpeg_quality_does_not_crash_validation():
    from batchresize.cli import _make_options
    args = build_parser().parse_args(["--jpeg-quality", "20"])
    opts = _make_options(args, Path("input"), Path("output"))
    assert opts.quality == 20
    assert opts.min_quality <= 20


def test_cli_custom_size_with_sizing_percent():
    from batchresize.cli import _make_options
    args = build_parser().parse_args(["--size", "custom", "--sizing", "percent", "--sizing-value", "50"])
    opts = _make_options(args, Path("input"), Path("output"))
    assert opts.sizing == "percent"
    assert opts.sizing_value == 50.0


def test_cli_concurrency_arguments_explicit():
    from batchresize.cli import _make_options
    args = build_parser().parse_args(["--workers", "8", "--processors", "16", "--readers", "4", "--writers", "2"])
    opts = _make_options(args, Path("input"), Path("output"))
    assert opts.workers == 8
    assert opts.processors == 16
    assert opts.readers == 4
    assert opts.writers == 2


def test_pipeline_thread_allocation():
    from batchresize.cli import _make_options
    from batchresize.pipeline import Pipeline
    args_default = build_parser().parse_args(["--workers", "8"])
    opts_default = _make_options(args_default, Path("input"), Path("output"))

    # CPU default
    p_cpu = Pipeline(opts_default, None, "cpu", progress=False, files=[])
    assert p_cpu._n_readers == 8
    assert p_cpu._n_processors == 8
    assert p_cpu._n_writers == 4

    # GPU default
    p_gpu = Pipeline(opts_default, None, "gpu", progress=False, files=[])
    assert p_gpu._n_readers == 8
    assert p_gpu._n_processors == 1
    assert p_gpu._n_writers == 4

    # Explicit override
    args_custom = build_parser().parse_args(["--workers", "8", "--processors", "12", "--readers", "6", "--writers", "3"])
    opts_custom = _make_options(args_custom, Path("input"), Path("output"))
    p_custom_cpu = Pipeline(opts_custom, None, "cpu", progress=False, files=[])
    assert p_custom_cpu._n_readers == 6
    assert p_custom_cpu._n_processors == 12
    assert p_custom_cpu._n_writers == 3

    p_custom_gpu = Pipeline(opts_custom, None, "gpu", progress=False, files=[])
    assert p_custom_gpu._n_readers == 6
    assert p_custom_gpu._n_processors == 12
    assert p_custom_gpu._n_writers == 3
