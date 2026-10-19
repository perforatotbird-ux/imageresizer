"""Pipeline integration tests for Stage 4: filters and output sharpening."""

import json
from pathlib import Path

from PIL import Image

from batchresize.cpu_backend import CpuEngine
from batchresize.pipeline import Pipeline, RunOptions


def sample_png(size=(64, 48), color=(40, 160, 80)):
    import io
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def create_source(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "photo.png").write_bytes(sample_png())
    return root


class TestFiltersAndSharpenPipeline:
    def test_pipeline_with_mitchell_filter(self, tmp_path):
        src = create_source(tmp_path)
        out = tmp_path / "out_mitchell"
        opts = RunOptions(src, out, 32, 32, workers=1, filter_name="mitchell")
        p = Pipeline(opts, CpuEngine("mitchell"), "cpu", progress=False)
        rep = p.run()
        assert rep["complete"]
        assert rep["filter"] == "mitchell"
        dest = out / "photo.jpg"
        assert dest.exists()
        with Image.open(dest) as im:
            assert im.size == (32, 24)

    def test_pipeline_with_area_filter(self, tmp_path):
        src = create_source(tmp_path)
        out = tmp_path / "out_area"
        opts = RunOptions(src, out, 32, 32, workers=1, filter_name="area")
        p = Pipeline(opts, CpuEngine("area"), "cpu", progress=False)
        rep = p.run()
        assert rep["complete"]
        assert rep["filter"] == "area"
        dest = out / "photo.jpg"
        assert dest.exists()

    def test_pipeline_with_nearest_filter(self, tmp_path):
        src = create_source(tmp_path)
        out = tmp_path / "out_nearest"
        opts = RunOptions(src, out, 32, 32, workers=1, filter_name="nearest")
        p = Pipeline(opts, CpuEngine("nearest"), "cpu", progress=False)
        rep = p.run()
        assert rep["complete"]
        assert rep["filter"] == "nearest"
        dest = out / "photo.jpg"
        assert dest.exists()

    def test_pipeline_with_sharpen_standard(self, tmp_path):
        src = create_source(tmp_path)
        out = tmp_path / "out_sharpen"
        opts = RunOptions(src, out, 32, 32, workers=1, sharpen="standard")
        p = Pipeline(opts, CpuEngine("lanczos3"), "cpu", progress=False)
        rep = p.run()
        assert rep["complete"]
        assert rep["sharpen"]["enabled"] is True
        assert rep["sharpen"]["preset"] == "standard"
        dest = out / "photo.jpg"
        assert dest.exists()

    def test_sharpen_change_invalidates_resume(self, tmp_path):
        src = create_source(tmp_path)
        out = tmp_path / "out_resume"
        # Run 1: sharpen off
        opts1 = RunOptions(src, out, 32, 32, workers=1, sharpen="off")
        p1 = Pipeline(opts1, CpuEngine("lanczos3"), "cpu", progress=False)
        rep1 = p1.run()
        assert rep1["processed"] == 1

        # Run 2: same settings with skip_existing -> skipped
        opts2 = RunOptions(src, out, 32, 32, workers=1, sharpen="off", skip_existing=True, overwrite=False)
        p2 = Pipeline(opts2, CpuEngine("lanczos3"), "cpu", progress=False)
        rep2 = p2.run()
        assert rep2["skipped"] == 1

        # Run 3: sharpen standard with skip_existing -> must NOT skip, must reprocess due to fingerprint mismatch!
        opts3 = RunOptions(src, out, 32, 32, workers=1, sharpen="standard", skip_existing=True, overwrite=True)
        p3 = Pipeline(opts3, CpuEngine("lanczos3"), "cpu", progress=False)
        rep3 = p3.run()
        assert rep3["processed"] == 1
        assert rep3["skipped"] == 0
