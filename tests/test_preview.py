"""Preview worker contract: same pipeline, bounded display, protocol survives errors."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import make_rgb, save_jpeg

ROOT = Path(__file__).resolve().parents[1]


def worker_python() -> Path:
    for candidate in (ROOT / ".venv" / "Scripts" / "python.exe",
                      ROOT / ".venv" / "bin" / "python"):
        if candidate.exists():
            return candidate
    pytest.skip("project virtual environment is not available")


class Worker:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "preview-worker"
        self.root.mkdir()
        self.proc = subprocess.Popen(
            [str(worker_python()), "-u", "-m", "batchresize.preview_worker",
             str(self.root)],
            cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)

    def ask(self, payload: dict) -> dict:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, "preview worker closed its protocol stream"
        return json.loads(line)

    def close(self) -> None:
        if self.proc.poll() is None and self.proc.stdin is not None:
            try:
                self.proc.stdin.write('{"cmd":"quit"}\n')
                self.proc.stdin.flush()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)


def params(**overrides) -> dict:
    data = {
        "max_w": 320, "max_h": 320, "mode": "contain", "quality": 82,
        "subsampling": 2, "sizing": "resolution", "sizing_value": 0.0,
        "allow_upscale": False, "keep_exif": True, "keep_icc": True,
        "bg_color": [255, 255, 255], "vram_gb": 2.0, "filter": "lanczos3",
        "fallback": True, "allow_truncated": False,
        "optimize_huffman": False, "backend": "cpu",
    }
    data.update(overrides)
    return data


def request(source: Path, request_id: int = 1, **overrides) -> dict:
    return {"id": request_id, "source": str(source), "view": "fit",
            "box_w": 292, "box_h": 190, "params": params(**overrides)}


@pytest.fixture()
def worker(tmp_path):
    instance = Worker(tmp_path)
    yield instance
    instance.close()


def test_preview_summary_reports_size_and_reduction():
    from batchresize.preview import preview_summary
    short, detail = preview_summary({
        "source_w": 5760, "source_h": 3840, "source_bytes": 1_200_000,
        "result_w": 1024, "result_h": 683, "jpeg_bytes": 95_000,
    })
    assert "5760 × 3840 px" in short
    assert "1024 × 683 px" in short
    assert "Уменьшение" in detail
    assert "по площади" in detail
    assert "файл −" in detail


def test_preview_uses_the_real_pipeline(worker, tmp_path):
    from batchresize.cpu_backend import CpuEngine
    from batchresize.pipeline import Pipeline, RunOptions

    source = tmp_path / "source.jpg"
    save_jpeg(source, make_rgb(900, 600, 4))
    response = worker.ask(request(source))
    assert response["ok"], response
    assert (response["result_w"], response["result_h"]) == (320, 213)

    out = tmp_path / "direct"
    opts = RunOptions(input_dir=source.parent, output_dir=out, max_w=320,
                      max_h=320, quality=82, subsampling=2, workers=1,
                      batch_size=1, queue_depth=2)
    Pipeline(opts, CpuEngine("lanczos3"), "cpu", progress=False,
             files=[source]).run()
    direct = out / "source.jpg"
    digest = hashlib.sha256(direct.read_bytes()).hexdigest()
    assert response["jpeg_sha256"] == digest
    assert Path(response["result"]["path"]).is_file()
    assert Path(response["original"]["path"]).is_file()


def test_preview_honours_exif_orientation(worker, tmp_path):
    source = tmp_path / "rotated.jpg"
    save_jpeg(source, make_rgb(640, 480, 8), orientation=6)
    response = worker.ask(request(source))
    assert response["ok"], response
    assert (response["source_w"], response["source_h"]) == (480, 640)
    assert (response["result_w"], response["result_h"]) == (240, 320)
    assert (response["original"]["w"], response["original"]["h"]) == (142, 190)


def test_preview_zoom_view_returns_scaled_images(worker, tmp_path):
    source = tmp_path / "zoom.jpg"
    save_jpeg(source, make_rgb(900, 600, 11))
    payload = request(source, 1)
    payload.update(view="zoom", zoom=0.5, max_pixels=24_000_000)
    response = worker.ask(payload)
    assert response["ok"], response
    assert response["view"] == "zoom"
    assert (response["original"]["w"], response["original"]["h"]) == (450, 300)
    assert response["result"]["w"] == 160
    assert response["approximate"] is False


def test_preview_zoom_view_caps_large_previews(worker, tmp_path):
    source = tmp_path / "zoom-cap.jpg"
    save_jpeg(source, make_rgb(900, 600, 12))
    payload = request(source, 1)
    payload.update(view="zoom", zoom=4.0, max_pixels=1_000_000)
    response = worker.ask(payload)
    assert response["ok"], response
    assert response["approximate"] is True
    assert response["original"]["w"] * response["original"]["h"] <= 1_000_000


def test_preview_reuses_result_when_only_display_box_changes(worker, tmp_path):
    source = tmp_path / "cache.jpg"
    save_jpeg(source, make_rgb(500, 400, 9))
    first = worker.ask(request(source, 1))
    second_payload = request(source, 2)
    second_payload["box_w"] = 180
    second_payload["box_h"] = 120
    second = worker.ask(second_payload)
    assert first["ok"] and second["ok"], (first, second)
    assert first["reused"] is False and second["reused"] is True
    assert first["jpeg_sha256"] == second["jpeg_sha256"]
    assert (second["result"]["w"], second["result"]["h"]) == (150, 120)


def test_preview_reports_errors_and_keeps_working(worker, tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not a jpeg" * 20)
    failed = worker.ask(request(bad, 1))
    assert failed["ok"] is False
    assert failed["error"]

    good = tmp_path / "good.jpg"
    save_jpeg(good, make_rgb(280, 180, 10))
    ok = worker.ask(request(good, 2))
    assert ok["ok"], ok


def test_preview_reports_requested_gpu_route(worker, tmp_path, gpu_available):
    if not gpu_available:
        pytest.skip("GPU is unavailable")
    source = tmp_path / "gpu.jpg"
    save_jpeg(source, make_rgb(800, 500, 12))
    response = worker.ask(request(source, backend="gpu"))
    assert response["ok"], response
    assert response["requested_backend"] == "gpu"
    assert response["backend"] in ("gpu", "hybrid")
    assert "gpu" in response["route"].lower() or "nvimg" in response["route"].lower()


def test_preview_worker_cleans_its_temp_tree(worker):
    imports = ROOT / "testdata" / "photos" / "Mixed" / "tiny.jpg"
    if not imports.is_file():
        pytest.skip("run tools/make_testset.py first")
    response = worker.ask(request(imports))
    assert response["ok"], response
    root = worker.root
    worker.close()
    assert not root.exists()
