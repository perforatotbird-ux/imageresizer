"""End-to-end behaviour of multiple output variants per source."""

import json
from pathlib import Path

from PIL import Image

from batchresize.pipeline import Pipeline, RunOptions
from batchresize.variants import OutputVariant


def png(size=(64, 48), color=(30, 180, 90)):
    import io
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def source(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "a.png").write_bytes(png())
    return root


def variants():
    return (OutputVariant(id="thumb", max_w=48, max_h=48, mode="cover",
                          quality=80),
            OutputVariant(id="web", max_w=96, max_h=96, mode="contain",
                          quality=85))


def run(src, out, **kw):
    opts = RunOptions(src, out, 128, 128, workers=1,
                      variants=variants(), **kw)
    p = Pipeline(opts, None, "cpu", progress=False)
    return p.run()


class TestVariants:
    def test_writes_a_subdir_per_variant(self, tmp_path):
        src = source(tmp_path)
        out = tmp_path / "out"
        rep = run(src, out)
        assert rep["complete"]
        assert rep["outputs_total"] == 2
        thumb = out / "thumb" / "a.jpg"
        web = out / "web" / "a.jpg"
        assert thumb.exists() and web.exists()
        with Image.open(thumb) as im:
            assert im.size == (48, 48)
        with Image.open(web) as im:
            assert im.size == (64, 48)

    def test_report_lists_variants(self, tmp_path):
        src = source(tmp_path)
        out = tmp_path / "out"
        rep = run(src, out)
        assert len(rep["variants"]) == 2
        assert {item["id"] for item in rep["variants"]} == {"thumb", "web"}

    def test_resume_skips_verified_variants(self, tmp_path):
        src = source(tmp_path)
        out = tmp_path / "out"
        run(src, out)
        rep = run(src, out, skip_existing=True, overwrite=False)
        assert rep["skipped"] == 2
        assert rep["processed"] == 0

    def test_manifest_has_per_variant_entries(self, tmp_path):
        src = source(tmp_path)
        out = tmp_path / "out"
        run(src, out)
        state = json.loads((out / ".batchresize-state.json").read_text())
        assert state["entries"]["a.png::thumb"]["variant"] == "thumb"
        assert state["entries"]["a.png::web"]["variant"] == "web"
        assert (out / "thumb" / "a.jpg").exists() and (out / "web" / "a.jpg").exists()

    def test_no_single_output_when_variants_present(self, tmp_path):
        src = source(tmp_path)
        out = tmp_path / "out"
        run(src, out)
        assert not (out / "a.jpg").exists()
        assert (out / "thumb" / "a.jpg").exists()
        assert (out / "web" / "a.jpg").exists()

    def test_partial_failure_preserves_successful_variant(self, tmp_path, monkeypatch):
        src = source(tmp_path)
        out = tmp_path / "out"
        from batchresize.cpu_backend import CpuEngine
        orig_resize = CpuEngine.resize

        def mock_resize(self, img, plan):
            if plan.out_w == 48 and plan.out_h == 48:
                raise RuntimeError("thumb variant failed")
            return orig_resize(self, img, plan)

        monkeypatch.setattr(CpuEngine, "resize", mock_resize)
        rep = run(src, out)
        assert not rep["complete"]
        assert rep["processed"] == 1
        assert rep["failed"] == 1
        assert not (out / "thumb" / "a.jpg").exists()
        assert (out / "web" / "a.jpg").exists()

    def test_cancel_leaves_no_corrupt_files(self, tmp_path):
        src = source(tmp_path)
        out = tmp_path / "out"
        cancel_file = tmp_path / "cancel.flag"
        cancel_file.touch()
        rep = run(src, out, cancel_file=cancel_file)
        assert rep["interrupted"] or not rep["complete"]
        assert rep["processed"] == 0
        assert not (out / "thumb" / "a.jpg").exists()
        assert not (out / "web" / "a.jpg").exists()