"""Regression tests for the "same as source" quality mode."""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from batchresize.cpu_backend import CpuEngine
from batchresize.geometry import make_plan
from batchresize.pipeline import Pipeline, RunOptions
from batchresize.source_quality import (FALLBACK_QUALITY, QUALITY_SOURCE,
                                        SUBSAMPLING_SOURCE, dezigzag,
                                        estimate_source_quality, read_subsampling,
                                        read_quantisation_tables,
                                        source_quantisation_tables)

cv2 = pytest.importorskip("cv2")


def jpeg_bytes(quality=80, subsampling=0, size=(64, 48), seed=1):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    factor = {0: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
              1: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422,
              2: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420}[subsampling]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality,
                                          cv2.IMWRITE_JPEG_SAMPLING_FACTOR, factor])
    assert ok
    return buf.tobytes()


@pytest.mark.parametrize("quality", [100, 95, 90, 85, 80, 70, 60, 50, 30, 10, 1])
def test_quality_estimator_matches_the_encoder(quality):
    assert estimate_source_quality(jpeg_bytes(quality)) == quality


def test_subsampling_is_detected():
    assert read_subsampling(jpeg_bytes(80, 0)) == 0
    assert read_subsampling(jpeg_bytes(80, 1)) == 1
    assert read_subsampling(jpeg_bytes(80, 2)) == 2


def test_tables_are_returned_in_natural_order():
    tables = source_quantisation_tables(jpeg_bytes(80, 0))
    assert tables is not None
    luma, chroma = tables
    assert len(luma) == 64 and len(chroma) == 64
    # natural order: the first entries are the low frequencies
    assert luma[0] <= luma[63]
    assert all(1 <= value <= 255 for value in luma + chroma)


def test_dezigzag_is_inverse_of_the_file_order():
    raw = jpeg_bytes(75, 0)
    file_order = read_quantisation_tables(raw)[0]
    natural = dezigzag(file_order)
    again = [natural[i] for i in
             (0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
              12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
              35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
              58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63)]
    assert again == file_order


def test_non_jpeg_has_no_tables():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buffer, "PNG")
    assert source_quantisation_tables(buffer.getvalue()) is None
    assert estimate_source_quality(buffer.getvalue()) is None


def test_pipeline_keeps_the_quantisation_tables_of_the_source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jpg").write_bytes(jpeg_bytes(70, 0, (200, 150)))
    out = tmp_path / "out"
    opts = RunOptions(src, out, 100, 100, workers=1, quality=QUALITY_SOURCE,
                      subsampling=SUBSAMPLING_SOURCE)
    report = Pipeline(opts, CpuEngine(opts.filter_name), "cpu", False).run()
    assert report["processed"] == 1 and report["failed"] == 0
    source_tables = source_quantisation_tables((src / "a.jpg").read_bytes())
    output_tables = source_quantisation_tables((out / "a.jpg").read_bytes())
    assert output_tables == source_tables
    assert read_subsampling((out / "a.jpg").read_bytes()) == 0


def test_source_mode_keeps_the_chroma_layout_of_the_source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jpg").write_bytes(jpeg_bytes(80, 2, (200, 150)))
    out = tmp_path / "out"
    opts = RunOptions(src, out, 100, 100, workers=1, quality=QUALITY_SOURCE,
                      subsampling=SUBSAMPLING_SOURCE)
    report = Pipeline(opts, CpuEngine(opts.filter_name), "cpu", False).run()
    assert report["processed"] == 1
    assert read_subsampling((out / "a.jpg").read_bytes()) == 2


def test_source_mode_falls_back_for_sources_without_tables(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 90)).save(buffer, "PNG")
    (src / "a.png").write_bytes(buffer.getvalue())
    out = tmp_path / "out"
    opts = RunOptions(src, out, 32, 32, workers=1, quality=QUALITY_SOURCE,
                      subsampling=SUBSAMPLING_SOURCE,
                      source_fallback_quality=55)
    report = Pipeline(opts, CpuEngine(opts.filter_name), "cpu", False).run()
    assert report["processed"] == 1 and report["failed"] == 0
    assert (out / "a.jpg").exists()


def test_cli_accepts_the_source_preset(tmp_path):
    from batchresize.cli import build_parser, _make_options
    src = tmp_path / "src"
    src.mkdir()
    args = build_parser().parse_args([
        "-i", str(src), "-o", str(tmp_path / "out"), "--backend", "cpu",
        "--quality", "source"])
    opts = _make_options(args, src, tmp_path / "out")
    assert opts.quality == QUALITY_SOURCE
    assert opts.subsampling == SUBSAMPLING_SOURCE
    assert opts.source_fallback_quality == FALLBACK_QUALITY
