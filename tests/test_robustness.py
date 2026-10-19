import time

import pytest

from batchresize import cli
from batchresize.gpu_backend import GpuUnavailable
from batchresize.pipeline import Pipeline, RunOptions
from conftest import make_rgb, run_cli, save_jpeg


def test_gpu_backend_fails_loudly(tmp_path, monkeypatch):
    src = tmp_path / "photos"
    src.mkdir()
    save_jpeg(src / "a.jpg", make_rgb(200, 150))

    def boom(*a, **kw):
        raise GpuUnavailable("simulated: no CUDA device")

    monkeypatch.setattr(cli, "GpuEngine", boom)
    code = cli.main(["-i", str(src), "-o", str(tmp_path / "out"),
                     "--backend", "gpu", "--size", "thumb_256"])
    assert code == 2


def test_auto_backend_falls_back_to_cpu(tmp_path, monkeypatch, capsys):
    src = tmp_path / "photos"
    src.mkdir()
    save_jpeg(src / "a.jpg", make_rgb(200, 150))

    def boom(*a, **kw):
        raise GpuUnavailable("simulated: no CUDA device")

    monkeypatch.setattr(cli, "GpuEngine", boom)
    code = cli.main(["-i", str(src), "-o", str(tmp_path / "out"),
                     "--backend", "auto", "--size", "thumb_256"])
    assert code == 0
    assert "cpu" in capsys.readouterr().out
    assert (tmp_path / "out" / "a.jpg").exists()


def test_hybrid_backend(tmp_path, gpu_available):
    if not gpu_available:
        pytest.skip("GPU backend unavailable")
    src = tmp_path / "photos"
    src.mkdir()
    save_jpeg(src / "a.jpg", make_rgb(800, 600))
    (src / "b.png").parent.mkdir(exist_ok=True)
    make_rgb(600, 400).save(src / "b.png")
    code, rep = run_cli(src, tmp_path / "out", "--size", "small_512",
                        "--backend", "hybrid")
    assert code == 0
    assert rep["processed"] == 2
    assert rep["backend"] == "hybrid"


def test_keyboard_interrupt_keeps_finished_files(tmp_path, gpu_available):
    """Ctrl+C must leave already-written results on disk and still report."""
    src = tmp_path / "photos"
    src.mkdir()
    for i in range(40):
        save_jpeg(src / f"p{i:02d}.jpg", make_rgb(1200, 900, i))

    opts = RunOptions(input_dir=src, output_dir=tmp_path / "out",
                      max_w=512, max_h=512, workers=4, batch_size=2,
                      overwrite=True)
    if gpu_available:
        from batchresize.gpu_backend import GpuEngine

        engine, kind = GpuEngine(), "gpu"
    else:
        from batchresize.cpu_backend import CpuEngine

        engine, kind = CpuEngine(), "cpu"

    pipe = Pipeline(opts, engine, kind)
    original_wait = Pipeline._wait

    def wait_then_interrupt(self, threads, t_start):
        """Behave like Ctrl+C arriving once a few files are already written."""
        deadline = time.time() + 10
        while self.stats.snapshot()["processed"] < 3 and time.time() < deadline:
            time.sleep(0.01)
        raise KeyboardInterrupt

    Pipeline._wait = wait_then_interrupt
    try:
        rep = pipe.run()
    finally:
        Pipeline._wait = original_wait

    written = list((tmp_path / "out").rglob("*.jpg"))
    assert len(written) >= 3
    assert rep["processed"] == len(written), (rep["processed"], len(written))
    assert not list((tmp_path / "out").rglob("*.tmp"))


def test_very_large_image(tmp_path, gpu_available):
    """~50 MP source must not blow up VRAM/RAM accounting."""
    if not gpu_available:
        pytest.skip("GPU backend unavailable")
    src = tmp_path / "photos"
    src.mkdir()
    save_jpeg(src / "huge.jpg", make_rgb(8000, 6000), quality=88)
    code, rep = run_cli(src, tmp_path / "out", "--size", "web_1024",
                        "--backend", "gpu", "--batch-size", "4")
    assert code == 0
    assert rep["processed"] == 1
    from PIL import Image

    with Image.open(tmp_path / "out" / "huge.jpg") as im:
        assert im.size == (1024, 768)


def test_vram_budget_splits_batches(tmp_path, gpu_available):
    if not gpu_available:
        pytest.skip("GPU backend unavailable")
    src = tmp_path / "photos"
    src.mkdir()
    for i in range(4):
        save_jpeg(src / f"p{i}.jpg", make_rgb(3000, 2000, i))
    code, rep = run_cli(src, tmp_path / "out", "--size", "small_512",
                        "--backend", "gpu", "--batch-size", "8",
                        "--vram-budget", "0.05")
    assert code == 0
    assert rep["processed"] == 4
