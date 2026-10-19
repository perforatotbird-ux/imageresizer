"""Tk profile integration. Skips when Tk is unavailable."""

from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter", reason="Tkinter not installed in this interpreter")
from batchresize.gui import BatchResizeUI
from batchresize.profiles import builtin_profiles, find_profile


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


def profile(profile_id: str):
    return next(item for item in builtin_profiles() if item.id == profile_id)


def test_apply_builtin_profile_updates_controls(app):
    app._apply_profile(profile("builtin-thumbnails"))
    app.update()
    assert app.sizing_var.get() == "resolution"
    assert (app.width_var.get(), app.height_var.get()) == ("320", "320")
    assert app.mode_var.get() == "cover"
    assert app.jpeg_quality_var.get() == "80"
    assert app.profile_combo.current() >= 0
    assert "применён" in app.profile_status_var.get()


def test_apply_profile_with_off_preset_quality(app):
    app._apply_profile(profile("builtin-screenshots-text"))
    app.update()
    assert app.jpeg_quality_var.get() == "92"
    assert app.subsampling_var.get() == "444"
    assert app.quality_var.get() == "custom"


def test_manual_edit_marks_profile_changed(app):
    app._apply_profile(profile("builtin-thumbnails"))
    app.width_var.set("333")
    app.update()
    assert "изменён" in app.profile_status_var.get()


def test_profile_from_ui_roundtrip(app):
    app._apply_profile(profile("builtin-product-cards"))
    item = app._profile_from_ui("Мой тестовый профиль")
    assert item.name == "Мой тестовый профиль"
    assert item.quality == 85
    assert item.mode == "cover"
    assert (item.width, item.height) == (1000, 1000)
    assert item.id


def test_save_and_reload_user_profile(app, tmp_path):
    app._profiles_path = tmp_path / "profiles.json"
    app._user_profiles = [
        app._profile_from_profile(profile("builtin-web-photo"), "Веб-копия")]
    assert app._save_profile_store()
    app._load_profile_store()
    assert find_profile(app._all_profiles, "Веб-копия") is not None
