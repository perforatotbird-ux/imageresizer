"""Real Tk preview integration. Skips when Tk is unavailable."""

from __future__ import annotations

import time

import pytest

tk = pytest.importorskip("tkinter", reason="Tkinter not installed in this interpreter")
from batchresize.gui import BatchResizeUI


@pytest.fixture()
def app():
    try:
        instance = BatchResizeUI()
        instance.update()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    errors = []
    instance.report_callback_exception = lambda *e: errors.append(e)
    yield instance
    for timer in instance.tk.call("after", "info"):
        instance.after_cancel(timer)
    instance._stop_preview()
    try:
        instance.destroy()
    except tk.TclError:
        pass
    assert not errors, errors


def wait_for_preview(app, prefix="Готово", timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.update()
        if app.preview_status_var.get().startswith(prefix):
            app.update()
            return
        time.sleep(0.02)
    pytest.fail(f"preview did not reach {prefix!r}: {app.preview_status_var.get()}")


def test_preview_panel_shows_real_result(app, tmp_path):
    from PIL import Image

    source = tmp_path / "Фото 01.jpg"
    Image.new("RGB", (640, 480), (30, 90, 180)).save(source, "JPEG", quality=92)
    app.input_var.set(str(tmp_path))
    app.output_var.set(str(tmp_path / "out"))
    app.backend_var.set("cpu")
    app.workers_var.set("1")
    app.update()

    app.preview_source_var.set(str(source))
    wait_for_preview(app)
    assert "640 × 480" in app.preview_info_var.get()
    assert "640 × 480 → 640 × 480" in app.preview_info_var.get()
    assert app.view._preview_images and app.view._preview_data
    assert app.view.preview_canvas.find_all()

    app.width_var.set("320")
    app.height_var.set("240")
    wait_for_preview(app)
    assert "640 × 480 → 320 × 240" in app.preview_info_var.get()


def test_preview_modes_and_view_buttons(app, tmp_path):
    from PIL import Image

    source = tmp_path / "widths.png"
    Image.new("RGB", (800, 400), (90, 30, 30)).save(source)
    app.input_var.set(str(tmp_path))
    app.output_var.set(str(tmp_path / "out"))
    app.backend_var.set("cpu")
    app.preview_source_var.set(str(source))
    wait_for_preview(app)

    app.preview_mode_var.set("До / после")
    wait_for_preview(app)
    texts = [app.view.preview_canvas.itemcget(item, "text")
             for item in app.view.preview_canvas.find_all()
             if app.view.preview_canvas.type(item) == "text"]
    assert "До" in texts and "После" in texts
    app._set_preview_view("actual")
    assert app.preview_view_var.get() == "fit"
    assert app.view.preview_actual_button.instate(["disabled"])

    app.preview_mode_var.set("Результат")
    wait_for_preview(app)
    app._set_preview_view("actual")
    assert app.preview_view_var.get() == "actual"
    assert app.view.preview_actual_button.instate(["!disabled"])


def test_preview_close_stops_worker_and_removes_temp(app, tmp_path):
    from PIL import Image

    source = tmp_path / "close.jpg"
    Image.new("RGB", (320, 200), (10, 120, 30)).save(source, "JPEG")
    app.backend_var.set("cpu")
    app.preview_source_var.set(str(source))
    wait_for_preview(app)
    assert app._preview_client is not None
    temp_root = app._preview_temp.name
    app._stop_preview()
    assert app._preview_client is None
    assert not __import__("pathlib").Path(temp_root).exists()
