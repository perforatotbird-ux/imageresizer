"""Real Tk preview integration. Skips when Tk is unavailable."""

from __future__ import annotations

import time
from pathlib import Path

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
    assert "640 × 480 px" in app.preview_info_var.get()
    assert "→  640 × 480 px" in app.preview_info_var.get()
    assert app.view._preview_images and app.view._preview_data
    assert app.view.preview_canvas.find_all()

    app.width_var.set("320")
    app.height_var.set("240")
    wait_for_preview(app)
    assert "→  320 × 240 px" in app.preview_info_var.get()


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


def wait_for_large_result(app, result_w, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.update()
        window = app._large_preview
        if (window is not None and window._data is not None
                and int(window._data.get("result_w", 0)) == result_w):
            app.update()
            return window
        time.sleep(0.02)
    pytest.fail(f"large preview did not reach result width {result_w}")


def wait_for_large(app, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.update()
        window = app._large_preview
        if window is not None and window.status_var.get().startswith("Готово"):
            app.update()
            return window
        time.sleep(0.02)
    pytest.fail("large preview did not become ready")


def wait_for_source(app, expected, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.update()
        if Path(app.preview_source_var.get()).resolve() == Path(expected).resolve():
            app.update()
            return
        time.sleep(0.02)
    pytest.fail(f"preview source was not auto-selected: {app.preview_source_var.get()}")


def test_input_folder_autoselects_first_image(app, tmp_path):
    from PIL import Image

    folder = tmp_path / "photos"
    folder.mkdir()
    Image.new("RGB", (800, 600), (10, 80, 160)).save(folder / "b.jpg")
    Image.new("RGB", (800, 600), (160, 80, 10)).save(folder / "a.jpg")
    app.output_var.set(str(tmp_path / "out"))
    app.width_var.set("400")
    app.height_var.set("300")
    app.input_var.set(str(folder))
    wait_for_source(app, folder / "a.jpg")
    wait_for_preview(app)
    info = app.preview_info_var.get()
    assert "Уменьшение" in info
    assert "400 × 300" in info

    app._open_large_preview()
    window = wait_for_large(app)
    assert "Уменьшение" in window.info_var.get()
    assert "400 × 300" in window.info_var.get()
    window.close()


def test_large_preview_zooms_both_panes(app, tmp_path):
    from PIL import Image

    source = tmp_path / "large.png"
    Image.new("RGB", (800, 600), (40, 100, 160)).save(source)
    app.input_var.set(str(tmp_path))
    app.output_var.set(str(tmp_path / "out"))
    app.backend_var.set("cpu")
    app.preview_source_var.set(str(source))
    wait_for_preview(app)

    app._open_large_preview()
    window = wait_for_large(app)
    assert set(window._images) == {"original", "result"}
    assert window._data["source_w"] == 800

    window.actual()
    window = wait_for_large(app)
    assert window._zoom == 1.0
    assert window._data["view"] == "viewport"
    assert window._images["original"].height() == 600
    assert window._images["original"].width() <= window._box()[0]

    window.zoom_by(-1)
    window = wait_for_large(app)
    assert window._zoom == 0.75
    assert (window._images["original"].width(),
            window._images["original"].height()) == (600, 450)

    window.fit()
    wait_for_large(app)
    app.width_var.set("400")
    app.height_var.set("300")
    window = wait_for_large_result(app, 400)
    assert (window._images["result"].width(),
            window._images["result"].height()) == (400, 300)

    window.close()
    assert app._large_preview is None


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
