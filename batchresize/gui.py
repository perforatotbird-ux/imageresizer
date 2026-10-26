"""Desktop UI for batchresize.

The window itself only needs the standard library plus tkinter, so it can run
on any Python installation. The actual work is delegated to a child process
using the project's virtual environment (``.venv``), which is the interpreter
that has CuPy, nvImageCodec and the rest of the GPU stack installed.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from .presets import ALL_QUALITY_PRESETS, SIZE_PRESETS, auto_subsampling
from .source_quality import QUALITY_SOURCE, SUBSAMPLING_SOURCE
from .preview import PreviewClient
from .profiles import (Profile, ProfileError, build_profile, builtin_profiles,
                       default_profiles_path, export_profile, find_profile,
                       import_into, load_profiles, read_profile_file,
                       save_profiles, unique_id, unique_name)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "input"
DEFAULT_OUTPUT = PROJECT_DIR / "output"

SIZING_MODES = [
    ("resolution", "\u041f\u043e \u0433\u0440\u0430\u043d\u0438\u0446\u0430\u043c \xb7 \u0448\u0438\u0440\u0438\u043d\u0430 \xd7 \u0432\u044b\u0441\u043e\u0442\u0430"),
    ("percent", "\u0412 \u043f\u0440\u043e\u0446\u0435\u043d\u0442\u0430\u0445 \u043e\u0442 \u0438\u0441\u0445\u043e\u0434\u043d\u0438\u043a\u0430"),
    ("megapixels", "\u041f\u043e \u043a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u0443 \u043c\u0435\u0433\u0430\u043f\u0438\u043a\u0441\u0435\u043b\u0435\u0439"),
    ("longest", "\u041f\u043e \u0434\u043b\u0438\u043d\u043d\u043e\u0439 \u0441\u0442\u043e\u0440\u043e\u043d\u0435"),
    ("shortest", "\u041f\u043e \u043a\u043e\u0440\u043e\u0442\u043a\u043e\u0439 \u0441\u0442\u043e\u0440\u043e\u043d\u0435"),
]

QUALITY_CHOICES = [
    ("source", "\u041a\u0430\u043a \u0443 \u0438\u0441\u0445\u043e\u0434\u043d\u0438\u043a\u0430 \xb7 \u0442\u043e\u043b\u044c\u043a\u043e resize"),
    ("maximum", "\u041c\u0430\u043a\u0441\u0438\u043c\u0430\u043b\u044c\u043d\u043e\u0435 \xb7 q=95"),
    ("high", "\u0412\u044b\u0441\u043e\u043a\u043e\u0435 \xb7 q=90"),
    ("balanced", "\u0411\u0430\u043b\u0430\u043d\u0441 \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u0430 \u0438 \u0440\u0430\u0437\u043c\u0435\u0440\u0430 \xb7 q=80"),
    ("small", "\u041a\u043e\u043c\u043f\u0430\u043a\u0442\u043d\u044b\u0439 \u0444\u0430\u0439\u043b \xb7 q=60"),
    ("light", "\u041b\u0451\u0433\u043a\u043e\u0435 \u0441\u0436\u0430\u0442\u0438\u0435 \xb7 q=90"),
    ("medium", "\u0421\u0440\u0435\u0434\u043d\u0435\u0435 \u0441\u0436\u0430\u0442\u0438\u0435 \xb7 q=80"),
    ("strong", "\u0421\u0438\u043b\u044c\u043d\u043e\u0435 \u0441\u0436\u0430\u0442\u0438\u0435 \xb7 q=65"),
    ("extreme", "\u041c\u0430\u043a\u0441\u0438\u043c\u0430\u043b\u044c\u043d\u043e\u0435 \u0441\u0436\u0430\u0442\u0438\u0435 \xb7 q=50"),
]

SUBSAMPLING_CHOICES = [
    ("auto", "\u0410\u0432\u0442\u043e \xb7 \u0437\u0430\u0432\u0438\u0441\u0438\u0442 \u043e\u0442 \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u0430 JPEG"),
    ("444", "4:4:4 \xb7 \u043f\u043e\u043b\u043d\u0430\u044f \u0446\u0432\u0435\u0442\u043e\u0432\u0430\u044f \u0434\u0435\u0442\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f"),
    ("422", "4:2:2 \xb7 \u0431\u0430\u043b\u0430\u043d\u0441 \u0434\u0435\u0442\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u0438 \u0438 \u0440\u0430\u0437\u043c\u0435\u0440\u0430"),
    ("420", "4:2:0 \xb7 \u043c\u0435\u043d\u044c\u0448\u0438\u0439 \u0444\u0430\u0439\u043b"),
]

PROGRESS_RE = re.compile(
    r"(\d+)/(\d+) (?:files|outputs) \| ([\d.]+) (?:file|output)/s \| ([\d.]+)s \| "
    r"threads: readers=(\d+) processors=(\d+) writers=(\d+)"
)


def find_worker_python() -> Path | None:
    """Interpreter that owns the GPU stack (the project venv)."""
    candidates = [
        PROJECT_DIR / ".venv" / "Scripts" / "python.exe",
        PROJECT_DIR / ".venv" / "bin" / "python",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


class BatchResizeUI(tk.Tk):
    def __init__(self) -> None:
        # Must precede Tk creation; no effect on non-Windows platforms.
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except (AttributeError, OSError):
                pass
        super().__init__()
        self.title("imageresizer \u2014 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0439")
        self.geometry("1220x900")
        self.minsize(1060, 740)

        self._events: queue.Queue = queue.Queue(maxsize=1024)
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._start_time = 0.0
        self._stopping = False

        self.worker = find_worker_python() or Path(sys.executable)

        self.input_var = tk.StringVar(value=str(DEFAULT_INPUT))
        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT))
        self.recursive_note = tk.StringVar(value="")

        self.sizing_var = tk.StringVar(value="resolution")
        self.preset_var = tk.StringVar(value="web_1024")
        self.width_var = tk.StringVar(value="1024")
        self.height_var = tk.StringVar(value="1024")
        self.sizing_value_var = tk.StringVar(value="100")
        self.mode_var = tk.StringVar(value="contain")
        self.crop_anchor_var = tk.StringVar(value="center")
        self.sharpen_preset_var = tk.StringVar(value="off")
        self.allow_upscale_var = tk.BooleanVar(value=False)
        self.filter_var = tk.StringVar(value="lanczos3")

        self.quality_var = tk.StringVar(value="balanced")
        self.jpeg_quality_var = tk.StringVar(value="80")
        self.subsampling_var = tk.StringVar(value="auto")
        self.bg_color_var = tk.StringVar(value="#FFFFFF")
        self.max_file_size_var = tk.StringVar(value="")
        self.min_quality_var = tk.StringVar(value="30")
        self.allow_downscale_var = tk.BooleanVar(value=False)
        self.strip_personal_var = tk.BooleanVar(value=False)
        self.web_srgb_var = tk.BooleanVar(value=False)
        self.linear_light_var = tk.BooleanVar(value=False)
        self.write_mode_var = tk.StringVar(value="overwrite")

        cpu_cores = os.cpu_count() or 8
        self.backend_var = tk.StringVar(value="auto")
        self.workers_var = tk.StringVar(value=str(min(8, cpu_cores)))
        self.processors_var = tk.StringVar(value="")
        self.readers_var = tk.StringVar(value=str(min(8, cpu_cores)))
        self.writers_var = tk.StringVar(value="4")
        self.batch_var = tk.StringVar(value="8")
        self.queue_var = tk.StringVar(value="32")
        self.vram_var = tk.StringVar(value="2.0")

        def _sync_workers_to_readers(*_):
            if self.readers_var.get() == "8" or not self.readers_var.get():
                self.readers_var.set(self.workers_var.get())
        self.workers_var.trace_add("write", _sync_workers_to_readers)
        self.limit_var = tk.StringVar(value="")

        self.exif_var = tk.BooleanVar(value=True)
        self.icc_var = tk.BooleanVar(value=True)
        self.skip_existing_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=True)
        self.truncated_var = tk.BooleanVar(value=False)
        self.fallback_var = tk.BooleanVar(value=True)
        self.huffman_var = tk.BooleanVar(value=False)
        self.fsync_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="\u0413\u043e\u0442\u043e\u0432\u043e \u043a \u0440\u0430\u0431\u043e\u0442\u0435")
        self.stats_var = tk.StringVar(value="")
        self.threads_var = tk.StringVar(value="")

        self.profile_var = tk.StringVar(value="")
        self.profile_status_var = tk.StringVar(value="\u041f\u0440\u043e\u0444\u0438\u043b\u044c \u043d\u0435 \u0432\u044b\u0431\u0440\u0430\u043d")
        self._profiles_path = default_profiles_path()
        self._user_profiles: list[Profile] = []
        self._all_profiles: list[Profile] = []
        self._active_profile_id: str | None = None
        self._applying_profile = False
        self._profile_errors: list[str] = []

        self.preview_source_var = tk.StringVar(value="")
        self.preview_mode_var = tk.StringVar(value="\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442")
        self.preview_view_var = tk.StringVar(value="fit")
        self.preview_status_var = tk.StringVar(value="\u0418\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435 \u043d\u0435 \u0432\u044b\u0431\u0440\u0430\u043d\u043e")
        self.preview_info_var = tk.StringVar(value="")
        self.preview_meta_var = tk.StringVar(value="")
        self._preview_client: PreviewClient | None = None
        self._preview_temp = tempfile.TemporaryDirectory(prefix="batchresize-preview-")
        self._preview_request_id = 0
        self._preview_after = None
        self._preview_suspended = False
        self._preview_has_result = False

        self._load_profile_store()
        self._build()
        self._on_quality_change()
        self._on_preset_change()
        self._sync_sizing_mode()
        self._sync_fit_mode()
        self._sync_size_limit_mode()
        self._install_ui_traces()
        self._install_preview_traces()
        self._install_profile_traces()
        for error in self._profile_errors:
            self._log_line(error)
        self.after(100, self._drain_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout --------------------------------------------------------------
    def _build(self) -> None:
        from .ui_view import ModernView
        self.view = ModernView(self, SIZING_MODES, QUALITY_CHOICES,
                               SUBSAMPLING_CHOICES, SIZE_PRESETS)
        self._install_log_context_menu()

    # -- UI helpers ----------------------------------------------------------
    def _browse(self, var: tk.StringVar) -> None:
        initial = var.get() or str(PROJECT_DIR)
        path = filedialog.askdirectory(initialdir=initial)
        if path:
            var.set(path)

    # -- task profiles -------------------------------------------------------
    def _load_profile_store(self) -> None:
        self._user_profiles, self._profile_errors = load_profiles(self._profiles_path)
        combined = builtin_profiles()
        seen = {profile.id for profile in combined}
        for profile in self._user_profiles:
            if profile.id in seen:
                self._profile_errors.append(
                    f"\u041f\u0440\u043e\u0444\u0438\u043b\u044c {profile.id!r} \u0441\u043e\u0432\u043f\u0430\u0434\u0430\u0435\u0442 \u0441\u043e \u0432\u0441\u0442\u0440\u043e\u0435\u043d\u043d\u044b\u043c \u0438 \u043f\u0440\u043e\u043f\u0443\u0449\u0435\u043d.")
                continue
            combined.append(profile)
            seen.add(profile.id)
        self._all_profiles = combined
        if hasattr(self, "profile_combo"):
            self.profile_combo["values"] = self._profile_labels()
            if self._active_profile_id:
                self._set_profile_selection(self._active_profile_id)
            else:
                self.profile_var.set("")

    def _profile_labels(self) -> list[str]:
        labels: list[str] = []
        for profile in self._all_profiles:
            kind = "\u0432\u0441\u0442\u0440\u043e\u0435\u043d\u043d\u044b\u0439" if profile.builtin else "\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c\u0441\u043a\u0438\u0439"
            label = f"{profile.name} \xb7 {kind}"
            if label in labels:
                label = f"{label} [{profile.id}]"
            labels.append(label)
        return labels

    def _set_profile_selection(self, profile_id: str) -> None:
        labels = self._profile_labels()
        for index, profile in enumerate(self._all_profiles):
            if profile.id == profile_id:
                self.profile_combo["values"] = labels
                self.profile_combo.current(index)
                self.profile_var.set(labels[index])
                return
        self.profile_var.set("")

    def _on_profile_selected(self, _event=None) -> None:
        index = self.profile_combo.current()
        if 0 <= index < len(self._all_profiles):
            self._apply_profile(self._all_profiles[index])

    def _apply_profile(self, profile: Profile) -> None:
        self._applying_profile = True
        try:
            if profile.sizing == "resolution":
                index = next((i for i, (key, _text) in enumerate(SIZING_MODES)
                              if key == "resolution"), 0)
                self.sizing_var.set("resolution")
                self.sizing_combo.current(index)
                preset = next((name for name, dims in SIZE_PRESETS.items()
                               if dims == (profile.width, profile.height)), "custom")
                self.preset_var.set(preset)
                self.preset_combo.set(preset)
                self.width_var.set(str(profile.width))
                self.height_var.set(str(profile.height))
            else:
                index = next((i for i, (key, _text) in enumerate(SIZING_MODES)
                              if key == profile.sizing), 0)
                self.sizing_var.set(profile.sizing)
                self.sizing_combo.current(index)
                self.preset_var.set("custom")
                self.preset_combo.set("custom")
                value = profile.sizing_value
                self.sizing_value_var.set(
                    str(int(value)) if float(value).is_integer() else f"{value:g}")
            self.mode_var.set(profile.mode)
            self.allow_upscale_var.set(profile.allow_upscale)
            self.filter_var.set(profile.filter)
            quality = profile.quality
            self.jpeg_quality_var.set(str(quality))
            self.quality_scale.set(quality)
            match = next(((key, text) for key, text in QUALITY_CHOICES
                          if key in ALL_QUALITY_PRESETS
                          and ALL_QUALITY_PRESETS[key] == quality), None)
            if match:
                self.quality_var.set(match[0])
                self.quality_combo.set(match[1])
            else:
                self.quality_var.set("custom")
                self.quality_combo.set("")
            if hasattr(self, '_sync_quality_mode'): self._sync_quality_mode()
            sub_index = next((i for i, (key, _text) in enumerate(SUBSAMPLING_CHOICES)
                              if key == profile.subsampling), 0)
            self.subsampling_var.set(profile.subsampling)
            self.subsampling_combo.current(sub_index)
            self.bg_color_var.set(profile.bg_color)
            self.exif_var.set(profile.keep_exif)
            self.icc_var.set(profile.keep_icc)
            if hasattr(self, "crop_anchor_var") and getattr(profile, "crop_position", None):
                self.crop_anchor_var.set(profile.crop_position)
            if hasattr(self, "sharpen_preset_var") and getattr(profile, "sharpen", None):
                sh = profile.sharpen
                preset = sh.get("preset", "off") if isinstance(sh, dict) else getattr(sh, "preset", "off")
                self.sharpen_preset_var.set(preset if preset in ("weak", "standard") else "off")
            if hasattr(self, "web_srgb_var"):
                self.web_srgb_var.set(getattr(profile, "color_policy", "preserve_rgb") == "web_srgb")
            if hasattr(self, "linear_light_var"):
                self.linear_light_var.set(bool(getattr(profile, "linear_light", False)))
            if hasattr(self, "strip_personal_var"):
                self.strip_personal_var.set(bool(getattr(profile, "strip_personal_data", False)))
        finally:
            self._applying_profile = False
        self._active_profile_id = profile.id
        self._set_profile_selection(profile.id)
        note = f"\u041f\u0440\u043e\u0444\u0438\u043b\u044c \xab{profile.name}\xbb \u043f\u0440\u0438\u043c\u0435\u043d\u0451\u043d"
        if profile.variants:
            note += " \xb7 \u0432\u0430\u0440\u0438\u0430\u043d\u0442\u044b \u043f\u043e\u044f\u0432\u044f\u0442\u0441\u044f \u043f\u043e\u0441\u043b\u0435 \u044d\u0442\u0430\u043f\u0430 3"
        self.profile_status_var.set(note)
        self._schedule_preview()

    def _install_profile_traces(self) -> None:
        for var in (self.width_var, self.height_var, self.sizing_var,
                    self.sizing_value_var, self.mode_var, self.allow_upscale_var,
                    self.filter_var, self.jpeg_quality_var, self.subsampling_var,
                    self.bg_color_var, self.exif_var, self.icc_var):
            var.trace_add("write", self._on_profile_setting_changed)

    def _on_profile_setting_changed(self, *_args) -> None:
        if self._applying_profile or not self._active_profile_id:
            return
        profile = find_profile(self._all_profiles, self._active_profile_id)
        if profile is not None:
            self.profile_status_var.set(f"\u041f\u0440\u043e\u0444\u0438\u043b\u044c \xab{profile.name}\xbb \u0438\u0437\u043c\u0435\u043d\u0451\u043d")

    def _profile_values_from_ui(self) -> dict:
        width = int(self.width_var.get())
        height = int(self.height_var.get())
        value = float(self.sizing_value_var.get())
        quality = int(self.jpeg_quality_var.get())
        sizing = self.sizing_var.get()
        if sizing == "resolution":
            size = {"mode": "resolution", "width": width, "height": height,
                    "value": None}
        else:
            size = {"mode": sizing, "width": None, "height": None, "value": value}
        raw_anchor = self.crop_anchor_var.get().split(" ")[0].strip() if hasattr(self, "crop_anchor_var") else "center"
        raw_sharpen = self.sharpen_preset_var.get().split(" ")[0].strip() if hasattr(self, "sharpen_preset_var") else "off"
        return {
            "size": size,
            "mode": self.mode_var.get(),
            "allow_upscale": self.allow_upscale_var.get(),
            "filter_name": self.filter_var.get(),
            "quality": quality,
            "subsampling": self.subsampling_var.get(),
            "bg_color": self.bg_color_var.get(),
            "keep_exif": self.exif_var.get(),
            "keep_icc": self.icc_var.get(),
            "crop_position": raw_anchor or "center",
            "sharpen": raw_sharpen if raw_sharpen in ("weak", "standard") else "off",
            "color_policy": "web_srgb" if (hasattr(self, "web_srgb_var") and self.web_srgb_var.get()) else "preserve_rgb",
            "linear_light": self.linear_light_var.get() if hasattr(self, "linear_light_var") else False,
            "strip_personal": self.strip_personal_var.get() if hasattr(self, "strip_personal_var") else False,
            "keep_datetime": False,
        }

    def _profile_from_ui(self, name: str, profile_id: str | None = None) -> Profile:
        values = self._profile_values_from_ui()
        return build_profile(
            name, size=values["size"], mode=values["mode"],
            allow_upscale=values["allow_upscale"],
            filter_name=values["filter_name"], quality=values["quality"],
            subsampling=values["subsampling"], bg_color=values["bg_color"],
            keep_exif=values["keep_exif"], keep_icc=values["keep_icc"],
            crop_position=values["crop_position"],
            sharpen=values["sharpen"],
            color_policy=values["color_policy"],
            linear_light=values["linear_light"],
            strip_personal=values["strip_personal"],
            keep_datetime=values["keep_datetime"],
            profile_id=profile_id or unique_id(self._all_profiles, name))

    def _profile_from_profile(self, source: Profile, name: str,
                              profile_id: str | None = None) -> Profile:
        return build_profile(
            name, size=source.size, mode=source.mode,
            allow_upscale=source.allow_upscale, filter_name=source.filter,
            quality=source.quality, subsampling=source.subsampling,
            bg_color=source.bg_color, keep_exif=source.keep_exif,
            keep_icc=source.keep_icc,
            crop_position=getattr(source, "crop_position", "center"),
            sharpen=getattr(source, "sharpen", None),
            color_policy=getattr(source, "color_policy", "preserve_rgb"),
            linear_light=getattr(source, "linear_light", False),
            strip_personal=getattr(source, "strip_personal_data", False),
            keep_datetime=getattr(source, "keep_datetime", False),
            profile_id=profile_id or unique_id(self._all_profiles, name),
            description=source.description, variants=source.variants)

    def _save_profile_store(self) -> bool:
        try:
            save_profiles(self._profiles_path, self._user_profiles)
        except (OSError, ProfileError) as exc:
            messagebox.showerror("\u041f\u0440\u043e\u0444\u0438\u043b\u0438", f"\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u0441\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u043f\u0440\u043e\u0444\u0438\u043b\u0438:\n{exc}")
            return False
        self._load_profile_store()
        return True

    def _replace_user_profile(self, profile: Profile) -> None:
        self._user_profiles = [profile if item.id == profile.id else item
                               for item in self._user_profiles]

    def _save_profile(self) -> None:
        active = (find_profile(self._all_profiles, self._active_profile_id)
                  if self._active_profile_id else None)
        try:
            if active is not None and not active.builtin:
                profile = self._profile_from_ui(active.name, profile_id=active.id)
                self._replace_user_profile(profile)
            else:
                initial = f"{active.name} (\u043c\u043e\u0439)" if active is not None else "\u041c\u043e\u0439 \u043f\u0440\u043e\u0444\u0438\u043b\u044c"
                name = simpledialog.askstring("\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u043f\u0440\u043e\u0444\u0438\u043b\u044c",
                                              "\u041d\u0430\u0437\u0432\u0430\u043d\u0438\u0435 \u043f\u0440\u043e\u0444\u0438\u043b\u044f:", initialvalue=initial,
                                              parent=self)
                if not name or not name.strip():
                    return
                name = name.strip()
                existing = find_profile(self._user_profiles, name)
                if existing is not None:
                    if not messagebox.askyesno(
                            "\u041f\u0440\u043e\u0444\u0438\u043b\u0438", f"\u041f\u0440\u043e\u0444\u0438\u043b\u044c \xab{existing.name}\xbb \u0443\u0436\u0435 \u0435\u0441\u0442\u044c. \u0417\u0430\u043c\u0435\u043d\u0438\u0442\u044c?",
                            parent=self):
                        return
                    profile = self._profile_from_ui(name, profile_id=existing.id)
                    self._replace_user_profile(profile)
                else:
                    profile = self._profile_from_ui(name)
                    self._user_profiles.append(profile)
        except (ValueError, ProfileError) as exc:
            messagebox.showerror("\u041f\u0440\u043e\u0444\u0438\u043b\u0438", str(exc), parent=self)
            return
        if self._save_profile_store():
            self._set_profile_selection(profile.id)
            self._apply_profile(profile)

    def _duplicate_profile(self) -> None:
        active = (find_profile(self._all_profiles, self._active_profile_id)
                  if self._active_profile_id else None)
        if active is None:
            messagebox.showinfo("\u041f\u0440\u043e\u0444\u0438\u043b\u0438", "\u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u0432\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u043f\u0440\u043e\u0444\u0438\u043b\u044c.", parent=self)
            return
        name = simpledialog.askstring("\u0414\u0443\u0431\u043b\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u043f\u0440\u043e\u0444\u0438\u043b\u044c", "\u041d\u0430\u0437\u0432\u0430\u043d\u0438\u0435 \u043a\u043e\u043f\u0438\u0438:",
                                      initialvalue=f"{active.name} (\u043a\u043e\u043f\u0438\u044f)",
                                      parent=self)
        if not name or not name.strip():
            return
        try:
            profile = self._profile_from_profile(active, name.strip())
        except (ValueError, ProfileError) as exc:
            messagebox.showerror("\u041f\u0440\u043e\u0444\u0438\u043b\u0438", str(exc), parent=self)
            return
        self._user_profiles.append(profile)
        if self._save_profile_store():
            self._set_profile_selection(profile.id)
            self._apply_profile(profile)

    def _rename_profile(self) -> None:
        active = (find_profile(self._all_profiles, self._active_profile_id)
                  if self._active_profile_id else None)
        if active is None or active.builtin:
            messagebox.showinfo("\u041f\u0440\u043e\u0444\u0438\u043b\u0438",
                                "\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c\u0441\u043a\u0438\u0439 \u043f\u0440\u043e\u0444\u0438\u043b\u044c \u0434\u043b\u044f \u043f\u0435\u0440\u0435\u0438\u043c\u0435\u043d\u043e\u0432\u0430\u043d\u0438\u044f.",
                                parent=self)
            return
        name = simpledialog.askstring("\u041f\u0435\u0440\u0435\u0438\u043c\u0435\u043d\u043e\u0432\u0430\u0442\u044c \u043f\u0440\u043e\u0444\u0438\u043b\u044c", "\u041d\u043e\u0432\u043e\u0435 \u043d\u0430\u0437\u0432\u0430\u043d\u0438\u0435:",
                                      initialvalue=active.name, parent=self)
        if not name or not name.strip() or name.strip() == active.name:
            return
        try:
            profile = self._profile_from_profile(active, name.strip(),
                                                 profile_id=active.id)
        except (ValueError, ProfileError) as exc:
            messagebox.showerror("\u041f\u0440\u043e\u0444\u0438\u043b\u0438", str(exc), parent=self)
            return
        self._replace_user_profile(profile)
        if self._save_profile_store():
            self._set_profile_selection(profile.id)
            self._apply_profile(profile)

    def _delete_profile(self) -> None:
        active = (find_profile(self._all_profiles, self._active_profile_id)
                  if self._active_profile_id else None)
        if active is None or active.builtin:
            messagebox.showinfo("\u041f\u0440\u043e\u0444\u0438\u043b\u0438",
                                "\u0412\u0441\u0442\u0440\u043e\u0435\u043d\u043d\u044b\u0439 \u043f\u0440\u043e\u0444\u0438\u043b\u044c \u043d\u0435\u043b\u044c\u0437\u044f \u0443\u0434\u0430\u043b\u0438\u0442\u044c. \u0414\u0443\u0431\u043b\u0438\u0440\u0443\u0439\u0442\u0435 \u0435\u0433\u043e.",
                                parent=self)
            return
        if not messagebox.askyesno("\u0423\u0434\u0430\u043b\u0438\u0442\u044c \u043f\u0440\u043e\u0444\u0438\u043b\u044c",
                                   f"\u0423\u0434\u0430\u043b\u0438\u0442\u044c \u043f\u0440\u043e\u0444\u0438\u043b\u044c \xab{active.name}\xbb?",
                                   parent=self):
            return
        self._user_profiles = [item for item in self._user_profiles
                               if item.id != active.id]
        self._active_profile_id = None
        self.profile_status_var.set("\u041f\u0440\u043e\u0444\u0438\u043b\u044c \u043d\u0435 \u0432\u044b\u0431\u0440\u0430\u043d")
        if self._save_profile_store():
            self.profile_var.set("")

    def _import_profiles(self) -> None:
        path = filedialog.askopenfilename(
            title="\u0418\u043c\u043f\u043e\u0440\u0442 \u043f\u0440\u043e\u0444\u0438\u043b\u044f", parent=self,
            filetypes=[("JSON", "*.json"), ("\u0412\u0441\u0435 \u0444\u0430\u0439\u043b\u044b", "*.*")])
        if not path:
            return
        try:
            imported = read_profile_file(Path(path))
        except (OSError, ValueError, ProfileError) as exc:
            messagebox.showerror("\u0418\u043c\u043f\u043e\u0440\u0442 \u043f\u0440\u043e\u0444\u0438\u043b\u044f", str(exc), parent=self)
            return
        before = {profile.id for profile in self._user_profiles}
        existing = [*builtin_profiles(), *self._user_profiles]
        combined = import_into(existing, imported)
        self._user_profiles = [profile for profile in combined
                               if not profile.builtin]
        if not self._save_profile_store():
            return
        added = [profile for profile in self._user_profiles
                 if profile.id not in before]
        if added:
            self._set_profile_selection(added[0].id)
            self._apply_profile(added[0])
        messagebox.showinfo("\u0418\u043c\u043f\u043e\u0440\u0442 \u043f\u0440\u043e\u0444\u0438\u043b\u044f",
                            f"\u0418\u043c\u043f\u043e\u0440\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u043e \u043f\u0440\u043e\u0444\u0438\u043b\u0435\u0439: {len(added)}", parent=self)

    def _export_profile(self) -> None:
        profile = (find_profile(self._all_profiles, self._active_profile_id)
                   if self._active_profile_id else None)
        if profile is None:
            name = simpledialog.askstring("\u042d\u043a\u0441\u043f\u043e\u0440\u0442 \u043f\u0440\u043e\u0444\u0438\u043b\u044f", "\u041d\u0430\u0437\u0432\u0430\u043d\u0438\u0435 \u043f\u0440\u043e\u0444\u0438\u043b\u044f:",
                                          initialvalue="\u041c\u043e\u0439 \u043f\u0440\u043e\u0444\u0438\u043b\u044c", parent=self)
            if not name or not name.strip():
                return
            try:
                profile = self._profile_from_ui(name.strip())
            except (ValueError, ProfileError) as exc:
                messagebox.showerror("\u042d\u043a\u0441\u043f\u043e\u0440\u0442 \u043f\u0440\u043e\u0444\u0438\u043b\u044f", str(exc), parent=self)
                return
        path = filedialog.asksaveasfilename(
            title="\u042d\u043a\u0441\u043f\u043e\u0440\u0442 \u043f\u0440\u043e\u0444\u0438\u043b\u044f", parent=self, defaultextension=".json",
            initialfile=f"{profile.id}.json",
            filetypes=[("JSON", "*.json"), ("\u0412\u0441\u0435 \u0444\u0430\u0439\u043b\u044b", "*.*")])
        if not path:
            return
        try:
            export_profile(Path(path), profile)
        except (OSError, ProfileError) as exc:
            messagebox.showerror("\u042d\u043a\u0441\u043f\u043e\u0440\u0442 \u043f\u0440\u043e\u0444\u0438\u043b\u044f", str(exc), parent=self)
            return
        messagebox.showinfo("\u042d\u043a\u0441\u043f\u043e\u0440\u0442 \u043f\u0440\u043e\u0444\u0438\u043b\u044f",
                            f"\u041f\u0440\u043e\u0444\u0438\u043b\u044c \u0441\u043e\u0445\u0440\u0430\u043d\u0451\u043d:\n{path}", parent=self)

    # -- preview -------------------------------------------------------------
    def _install_preview_traces(self) -> None:
        self.preview_source_var.trace_add("write", self._schedule_preview)
        self.preview_mode_var.trace_add("write", self._on_preview_mode_change)
        self.preview_view_var.trace_add("write", self._schedule_preview)
        for var in (self.width_var, self.height_var, self.sizing_var,
                    self.sizing_value_var, self.mode_var, self.allow_upscale_var,
                    self.filter_var, self.jpeg_quality_var, self.subsampling_var,
                    self.bg_color_var, self.backend_var, self.vram_var,
                    self.exif_var, self.icc_var, self.truncated_var,
                    self.fallback_var, self.huffman_var):
            var.trace_add("write", self._schedule_preview)

    def _choose_preview_file(self) -> None:
        initial = self.preview_source_var.get().strip()
        initial_dir = str(Path(initial).parent) if initial else str(PROJECT_DIR)
        path = filedialog.askopenfilename(
            title="\u0418\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435 \u0434\u043b\u044f \u043f\u0440\u0435\u0432\u044c\u044e",
            initialdir=initial_dir,
            filetypes=[("\u0418\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
                       ("\u0412\u0441\u0435 \u0444\u0430\u0439\u043b\u044b", "*.*")])
        if path:
            self.preview_source_var.set(path)

    def _set_preview_view(self, view: str) -> None:
        if self.preview_mode_var.get() == "\u0414\u043e / \u043f\u043e\u0441\u043b\u0435":
            view = "fit"
        self.preview_view_var.set(view)
        self.view.set_preview_view(view)

    def _on_preview_mode_change(self, *_args) -> None:
        if self.preview_mode_var.get() == "\u0414\u043e / \u043f\u043e\u0441\u043b\u0435":
            self.preview_view_var.set("fit")
            self.view.set_preview_view("fit")
        self._schedule_preview()

    def _on_preview_resize(self, _event=None) -> None:
        if self._preview_has_result and self.preview_source_var.get().strip():
            self._schedule_preview()

    def _schedule_preview(self, *_args) -> None:
        if self._preview_suspended or self._is_running():
            return
        if self._preview_after is not None:
            try:
                self.after_cancel(self._preview_after)
            except tk.TclError:
                pass
            self._preview_after = None
        source = self.preview_source_var.get().strip()
        if not source:
            self._clear_preview()
            return
        if self._preview_has_result and self.preview_status_var.get().startswith("\u0413\u043e\u0442\u043e\u0432\u043e"):
            self.preview_status_var.set("\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u044b \u2014 \u043e\u0431\u043d\u043e\u0432\u043b\u044f\u0435\u043c \u043f\u0440\u0435\u0432\u044c\u044e\u2026")
        self._preview_after = self.after(320, self._request_preview)

    def _preview_box(self) -> tuple[int, int]:
        canvas = self.view.preview_canvas
        width = max(80, canvas.winfo_width())
        height = max(80, canvas.winfo_height())
        if width < 120:
            width, height = 292, 190
        if self.preview_mode_var.get() == "\u0414\u043e / \u043f\u043e\u0441\u043b\u0435":
            width = max(60, (width - 10) // 2)
        return width, height

    def _min_quality(self) -> int:
        """Effective quality floor for the size limit (blank means 30)."""
        raw = self.min_quality_var.get().strip()
        return int(raw) if raw else 30

    def _preview_params(self) -> dict:
        from .cli import parse_color
        from .validation import dimensions, positive
        from .size_limit import parse_size_bytes

        width = int(self.width_var.get())
        height = int(self.height_var.get())
        quality = int(self.jpeg_quality_var.get())
        value = float(self.sizing_value_var.get())
        vram = float(self.vram_var.get())
        dimensions(width, height)
        positive(vram, "VRAM budget")
        sizing = self.sizing_var.get()
        if sizing != "resolution":
            positive(value, "\u0417\u043d\u0430\u0447\u0435\u043d\u0438\u0435 \u043c\u0430\u0441\u0448\u0442\u0430\u0431\u0430")
        if not 1 <= quality <= 100:
            raise ValueError("\u041a\u0430\u0447\u0435\u0441\u0442\u0432\u043e \u0434\u043e\u043b\u0436\u043d\u043e \u0431\u044b\u0442\u044c \u043e\u0442 1 \u0434\u043e 100.")
        subsampling = {"auto": auto_subsampling(quality), "444": 0,
                       "422": 1, "420": 2}[self.subsampling_var.get()]
        source_mode = self.quality_var.get() == QUALITY_SOURCE
        if source_mode:
            # The preview file provides its own quality and chroma layout.
            subsampling = SUBSAMPLING_SOURCE
        max_file_size = self.max_file_size_var.get().strip()
        max_bytes = parse_size_bytes(max_file_size) if max_file_size else None
        min_quality = min(self._min_quality(), quality)
        return {
            "max_w": width,
            "max_h": height,
            "mode": self.mode_var.get(),
            "crop_anchor": self.crop_anchor_var.get().split(" ")[0].strip(),
            "sharpen": self.sharpen_preset_var.get().split(" ")[0].strip(),
            "linear_light": bool(self.linear_light_var.get()),
            "web_srgb": bool(self.web_srgb_var.get()),
            "quality": QUALITY_SOURCE if source_mode else quality,
            "source_fallback_quality": quality,
            "subsampling": subsampling,
            "sizing": sizing,
            "sizing_value": value,
            "allow_upscale": bool(self.allow_upscale_var.get()),
            "keep_exif": bool(self.exif_var.get()),
            "keep_icc": bool(self.icc_var.get()),
            "bg_color": list(parse_color(self.bg_color_var.get())),
            "vram_gb": vram,
            "filter": self.filter_var.get(),
            "fallback": bool(self.fallback_var.get()),
            "allow_truncated": bool(self.truncated_var.get()),
            "optimize_huffman": bool(self.huffman_var.get()),
            "backend": self.backend_var.get(),
            "max_file_size": max_bytes,
            "min_quality": min_quality,
        }

    def _request_preview(self) -> None:
        self._preview_after = None
        if self._preview_suspended or self._is_running():
            return
        source = Path(self.preview_source_var.get().strip()).expanduser()
        if not source.is_file():
            self.preview_status_var.set("\u0424\u0430\u0439\u043b \u043f\u0440\u0435\u0432\u044c\u044e \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d")
            self.view.clear_preview("\u0424\u0430\u0439\u043b \u043f\u0440\u0435\u0432\u044c\u044e \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d")
            return
        try:
            params = self._preview_params()
        except (ValueError, KeyError) as exc:
            self.preview_status_var.set(f"\u041f\u0440\u043e\u0432\u0435\u0440\u044c\u0442\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b: {exc}")
            return
        if self._preview_client is None:
            self._preview_client = PreviewClient(
                self.worker, PROJECT_DIR, Path(self._preview_temp.name), self._events)
        width, height = self._preview_box()
        view = self.preview_view_var.get()
        if self.preview_mode_var.get() == "\u0414\u043e / \u043f\u043e\u0441\u043b\u0435":
            view = "fit"
        payload = {"source": str(source), "view": view,
                   "box_w": width, "box_h": height, "params": params}
        self._preview_request_id = self._preview_client.request(payload)
        self.preview_status_var.set("\u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u043f\u0440\u0435\u0432\u044c\u044e\u2026")

    def _handle_preview(self, payload: dict) -> None:
        kind = payload.get("kind")
        if kind == "log":
            self._log_line(f"preview: {payload.get('text', '')}")
            return
        if kind == "dead":
            self._log_line(f"preview: {payload.get('error', 'worker stopped')}")
            if not self._preview_suspended:
                self.preview_status_var.set(str(payload.get("error", "\u041f\u0440\u0435\u0432\u044c\u044e \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u043e")))
            return
        if kind == "invalidated":
            return
        if payload.get("id") != self._preview_request_id:
            return
        if not payload.get("ok"):
            error = str(payload.get("error", "\u043d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d\u0430\u044f \u043e\u0448\u0438\u0431\u043a\u0430"))
            self.preview_status_var.set(f"\u041e\u0448\u0438\u0431\u043a\u0430 \u043f\u0440\u0435\u0432\u044c\u044e: {error}")
            self._log_line(f"preview: {error}")
            return
        if not self.view.show_preview(payload):
            self.preview_status_var.set("\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u043f\u043e\u043a\u0430\u0437\u0430\u0442\u044c \u043f\u0440\u0435\u0432\u044c\u044e")
            return
        self._preview_has_result = True
        source_w = int(payload.get("source_w", 0))
        source_h = int(payload.get("source_h", 0))
        result_w = int(payload.get("result_w", 0))
        result_h = int(payload.get("result_h", 0))
        kb = int(payload.get("jpeg_bytes", 0)) / 1024
        self.preview_info_var.set(
            f"{source_w} \xd7 {source_h} \u2192 {result_w} \xd7 {result_h} px  \xb7  {kb:.0f} \u041a\u0438\u0411")
        route = str(payload.get("route") or payload.get("backend") or "\u2014")
        requested = str(payload.get("requested_backend") or "\u2014")
        approximate = "  \xb7  \u043f\u0440\u0438\u0431\u043b\u0438\u0436\u0451\u043d\u043d\u043e\u0435" if payload.get("approximate") else ""
        self.preview_meta_var.set(f"\u0414\u0432\u0438\u0436\u043e\u043a: {route}  \xb7  \u0437\u0430\u043f\u0440\u043e\u0448\u0435\u043d: {requested}{approximate}")
        self.preview_status_var.set("\u0413\u043e\u0442\u043e\u0432\u043e" + (" \xb7 \u043f\u0440\u0438\u0431\u043b\u0438\u0436\u0451\u043d\u043d\u043e\u0435 100%" if payload.get("approximate") else ""))

    def _clear_preview(self) -> None:
        self._preview_has_result = False
        self.preview_status_var.set("\u0418\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435 \u043d\u0435 \u0432\u044b\u0431\u0440\u0430\u043d\u043e")
        self.preview_info_var.set("")
        self.preview_meta_var.set("")
        self.view.clear_preview()

    def _pause_preview(self) -> None:
        self._preview_suspended = True
        if self._preview_after is not None:
            try:
                self.after_cancel(self._preview_after)
            except tk.TclError:
                pass
            self._preview_after = None
        if self._preview_client is not None:
            self._preview_client.cancel()
        self.preview_status_var.set("\u041f\u0440\u0435\u0432\u044c\u044e \u043f\u0440\u0438\u043e\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u043e \u0432\u043e \u0432\u0440\u0435\u043c\u044f \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0438")

    def _resume_preview(self) -> None:
        self._preview_suspended = False
        if self.preview_source_var.get().strip():
            self._schedule_preview()

    def _stop_preview(self) -> None:
        if self._preview_after is not None:
            try:
                self.after_cancel(self._preview_after)
            except tk.TclError:
                pass
            self._preview_after = None
        client, self._preview_client = self._preview_client, None
        if client is not None:
            client.close()
        try:
            self._preview_temp.cleanup()
        except (OSError, PermissionError):
            pass

    def _install_ui_traces(self) -> None:
        self.sizing_var.trace_add("write", lambda *_: self._sync_sizing_mode())
        self.mode_var.trace_add("write", lambda *_: self._sync_fit_mode())
        self.max_file_size_var.trace_add("write", lambda *_: self._sync_size_limit_mode())
        self.skip_existing_var.trace_add("write", self._on_legacy_write_var_changed)
        self.overwrite_var.trace_add("write", self._on_legacy_write_var_changed)

    def _on_legacy_write_var_changed(self, *_args) -> None:
        if self.skip_existing_var.get() and not self.overwrite_var.get():
            if self.write_mode_var.get() != "skip":
                self.write_mode_var.set("skip")
        elif self.overwrite_var.get():
            if self.write_mode_var.get() != "overwrite":
                self.write_mode_var.set("overwrite")

    def _sync_write_mode(self) -> None:
        if self.write_mode_var.get() == "overwrite":
            self.overwrite_var.set(True)
            self.skip_existing_var.set(False)
        else:
            self.overwrite_var.set(False)
            self.skip_existing_var.set(True)

    def _sync_sizing_mode(self) -> None:
        mode = self.sizing_var.get()
        is_res = (mode == "resolution")
        res_state = ["!disabled"] if is_res else ["disabled"]
        val_state = ["disabled"] if is_res else ["!disabled"]
        if hasattr(self, "preset_combo"):
            try:
                self.preset_combo.state(res_state)
            except Exception:
                pass
        if hasattr(self, "view"):
            for attr in ("width_entry", "height_entry"):
                w = getattr(self.view, attr, None)
                if w is not None:
                    try:
                        w.state(res_state)
                    except Exception:
                        pass
            w_val = getattr(self.view, "sizing_value_entry", None)
            if w_val is not None:
                try:
                    w_val.state(val_state)
                except Exception:
                    pass
            lbl = getattr(self.view, "sizing_value_label", None)
            if lbl is not None:
                labels_map = {
                    "resolution": "\u0417\u043d\u0430\u0447\u0435\u043d\u0438\u0435 \u043c\u0430\u0441\u0448\u0442\u0430\u0431\u0430 (\u043d\u0435 \u0442\u0440\u0435\u0431\u0443\u0435\u0442\u0441\u044f \u0434\u043b\u044f \u0444\u0438\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u043e\u0433\u043e WxH)",
                    "percent": "\u041c\u0430\u0441\u0448\u0442\u0430\u0431 \u043e\u0442 \u0438\u0441\u0445\u043e\u0434\u043d\u043e\u0433\u043e \u0440\u0430\u0437\u043c\u0435\u0440\u0430, % (\u043d\u0430\u043f\u0440\u0438\u043c\u0435\u0440, 80)",
                    "megapixels": "\u0426\u0435\u043b\u0435\u0432\u043e\u0435 \u0440\u0430\u0437\u0440\u0435\u0448\u0435\u043d\u0438\u0435, \u041c\u041f (\u043d\u0430\u043f\u0440\u0438\u043c\u0435\u0440, 2.5)",
                    "longest": "\u0414\u043b\u0438\u043d\u0430 \u043d\u0430\u0438\u0431\u043e\u043b\u044c\u0448\u0435\u0439 \u0441\u0442\u043e\u0440\u043e\u043d\u044b, px (\u043d\u0430\u043f\u0440\u0438\u043c\u0435\u0440, 1280)",
                    "shortest": "\u0414\u043b\u0438\u043d\u0430 \u043d\u0430\u0438\u043c\u0435\u043d\u044c\u0448\u0435\u0439 \u0441\u0442\u043e\u0440\u043e\u043d\u044b, px (\u043d\u0430\u043f\u0440\u0438\u043c\u0435\u0440, 720)",
                }
                lbl.configure(text=labels_map.get(mode, "\u0417\u043d\u0430\u0447\u0435\u043d\u0438\u0435 \u043c\u0430\u0441\u0448\u0442\u0430\u0431\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f"))
            hint = getattr(self.view, "sizing_mode_hint", None)
            if hint is not None:
                if is_res:
                    hint.configure(
                        text="\u0420\u0435\u0436\u0438\u043c \u0444\u0438\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u043e\u0433\u043e \u0440\u0430\u0437\u0440\u0435\u0448\u0435\u043d\u0438\u044f: \u0430\u043a\u0442\u0438\u0432\u043d\u044b \u043f\u0440\u0435\u0441\u0435\u0442 \u0438 \u0440\u0430\u0437\u043c\u0435\u0440\u044b W\xd7H \u043d\u0438\u0436\u0435.",
                        foreground="#2563EB"
                    )
                else:
                    m_desc = {'percent':'\u043f\u0440\u043e\u0446\u0435\u043d\u0442\u0430\u0445 (%)','megapixels':'\u043c\u0435\u0433\u0430\u043f\u0438\u043a\u0441\u0435\u043b\u044f\u0445 (\u041c\u041f)','longest':'\u0434\u043b\u0438\u043d\u043d\u043e\u0439 \u0441\u0442\u043e\u0440\u043e\u043d\u0435 (px)','shortest':'\u043a\u043e\u0440\u043e\u0442\u043a\u043e\u0439 \u0441\u0442\u043e\u0440\u043e\u043d\u0435 (px)'}.get(mode, mode)
                    hint.configure(
                        text=f"\u0410\u043a\u0442\u0438\u0432\u0435\u043d \u043c\u0430\u0441\u0448\u0442\u0430\u0431 \u0432 {m_desc}. \u041f\u0440\u0435\u0441\u0435\u0442\u044b \u0438 \u0433\u0440\u0430\u043d\u0438\u0446\u044b W\xd7H \u043d\u0438\u0436\u0435 \u043e\u0442\u043a\u043b\u044e\u0447\u0435\u043d\u044b.",
                        foreground="#D97706"
                    )

    def _sync_fit_mode(self) -> None:
        mode = self.mode_var.get()
        is_contain = (mode == "contain")
        if hasattr(self, "crop_anchor_combo"):
            try:
                self.crop_anchor_combo.state(["disabled"] if is_contain else ["readonly", "!disabled"])
            except Exception:
                pass
        if hasattr(self, "view") and hasattr(self.view, "crop_anchor_hint"):
            try:
                if is_contain:
                    self.view.crop_anchor_hint.configure(
                        text="\u0422\u043e\u0447\u043a\u0430 \u043f\u0440\u0438\u0432\u044f\u0437\u043a\u0438 \u043d\u0435 \u0442\u0440\u0435\u0431\u0443\u0435\u0442\u0441\u044f \u043f\u0440\u0438 \u043f\u043e\u043b\u043d\u043e\u043c \u0432\u043f\u0438\u0441\u044b\u0432\u0430\u043d\u0438\u0438 (contain).",
                        foreground="#64748B"
                    )
                else:
                    action = "\u043e\u0431\u0440\u0435\u0437\u043a\u0438 \u0438\u0437\u043b\u0438\u0448\u043a\u043e\u0432" if mode == "cover" else "\u0440\u0430\u0437\u043c\u0435\u0449\u0435\u043d\u0438\u044f \u043d\u0430 \u043a\u0430\u043d\u0432\u0435 \u0441 \u043f\u043e\u043b\u044f\u043c\u0438"
                    self.view.crop_anchor_hint.configure(
                        text=f"\u0422\u043e\u0447\u043a\u0430 \u043f\u0440\u0438\u0432\u044f\u0437\u043a\u0438 \u0434\u043b\u044f {action}.",
                        foreground="#2563EB"
                    )
            except Exception:
                pass

    def _sync_size_limit_mode(self) -> None:
        raw = self.max_file_size_var.get().strip()
        has_limit = bool(raw)
        state = ["!disabled"] if has_limit else ["disabled"]
        if hasattr(self, "view"):
            if hasattr(self.view, "min_quality_entry"):
                try:
                    self.view.min_quality_entry.state(state)
                except Exception:
                    pass
            if hasattr(self.view, "allow_downscale_check"):
                try:
                    self.view.allow_downscale_check.state(state)
                except Exception:
                    pass
            if hasattr(self.view, "min_quality_hint"):
                try:
                    if has_limit:
                        hint = "\u041f\u043e\u0434\u0431\u043e\u0440 \u0440\u0430\u0437\u043c\u0435\u0440\u0430 \u0444\u0430\u0439\u043b\u0430 \u0432\u044b\u043f\u043e\u043b\u043d\u044f\u0435\u0442\u0441\u044f \u043d\u0430 CPU. \u0423\u043a\u0430\u0437\u044b\u0432\u0430\u0439\u0442\u0435 \u0435\u0434\u0438\u043d\u0438\u0446\u044b: \u043d\u0430\u043f\u0440\u0438\u043c\u0435\u0440 800KB \u0438\u043b\u0438 1.5MB."
                        color = "#2563EB"
                        if raw.isdigit():
                            val = int(raw)
                            if val < 1024:
                                hint = f"\u0412\u043d\u0438\u043c\u0430\u043d\u0438\u0435: '{val}' \u043e\u0437\u043d\u0430\u0447\u0430\u0435\u0442 {val} \u0431\u0430\u0439\u0442 (\u043c\u0435\u043d\u0435\u0435 1 \u041a\u0411)! \u0412\u043e\u0437\u043c\u043e\u0436\u043d\u043e, \u0432\u044b \u0438\u043c\u0435\u043b\u0438 \u0432 \u0432\u0438\u0434\u0443 '{val}KB'?"
                                color = "#DC2626"
                        self.view.min_quality_hint.configure(text=hint, foreground=color)
                    else:
                        self.view.min_quality_hint.configure(
                            text="\u041f\u043e\u0440\u043e\u0433 \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u0430 \u043d\u0435 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u0442\u0441\u044f \u0431\u0435\u0437 \u043e\u0433\u0440\u0430\u043d\u0438\u0447\u0435\u043d\u0438\u044f \u0440\u0430\u0437\u043c\u0435\u0440\u0430 \u0444\u0430\u0439\u043b\u0430.",
                            foreground="#64748B"
                        )
                except Exception:
                    pass

    def _on_sizing_change(self, _event=None) -> None:
        idx = self.sizing_combo.current()
        if 0 <= idx < len(SIZING_MODES):
            self.sizing_var.set(SIZING_MODES[idx][0])
        self._sync_sizing_mode()

    def _on_preset_change(self, _event=None) -> None:
        name = self.preset_combo.get()
        if name in SIZE_PRESETS:
            w, h = SIZE_PRESETS[name]
            self.width_var.set(str(w))
            self.height_var.set(str(h))

    def _on_quality_change(self, _event=None) -> None:
        label = self.quality_combo.get()
        for key, text in QUALITY_CHOICES:
            if text == label:
                self.quality_var.set(key)
                if key == globals().get('QUALITY_SOURCE', 'source'):
                    if hasattr(self, '_sync_quality_mode'): self._sync_quality_mode()
                    if hasattr(self, "view") and hasattr(self.view, "quality_metric_label"):
                        self.view.quality_metric_label.configure(text="Auto")
                    return
                value = ALL_QUALITY_PRESETS[key]
                self.jpeg_quality_var.set(str(value))
                self.quality_scale.set(value)
                if hasattr(self, "view") and hasattr(self.view, "quality_metric_label"):
                    self.view.quality_metric_label.configure(text=str(value))
                if hasattr(self, '_sync_quality_mode'): self._sync_quality_mode()
                return

    def _sync_quality_mode(self) -> None:
        """The quality slider is meaningless in the "same as source" mode."""
        source = self.quality_var.get() == QUALITY_SOURCE
        scale = getattr(self, "quality_scale", None)
        if scale is not None:
            try:
                scale.state(["disabled"] if source else ["!disabled"])
            except tk.TclError:
                pass
        if hasattr(self, "view") and hasattr(self.view, "quality_metric_label"):
            self.view.quality_metric_label.configure(text="Auto" if source else self.jpeg_quality_var.get())

    def _on_quality_scale(self, value) -> None:
        val = int(round(float(value)))
        self.jpeg_quality_var.set(str(val))
        if hasattr(self, "quality_combo"):
            match_label = None
            for key, text in QUALITY_CHOICES:
                if key != "source" and key in ALL_QUALITY_PRESETS and ALL_QUALITY_PRESETS[key] == val:
                    match_label = text
                    self.quality_var.set(key)
                    break
            if match_label:
                self.quality_combo.set(match_label)
            else:
                self.quality_var.set("custom")
                self.quality_combo.set("")
        if hasattr(self, "view") and hasattr(self.view, "quality_metric_label"):
            self.view.quality_metric_label.configure(text=str(val))

    def _on_subsampling_change(self, _event=None) -> None:
        idx = self.subsampling_combo.current()
        self.subsampling_var.set(SUBSAMPLING_CHOICES[idx][0])

    def _log_line(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 10000:
            self.log.delete("1.0", "1001.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _copy_log(self) -> None:
        try:
            text = self.log.get("sel.first", "sel.last")
        except tk.TclError:
            text = self.log.get("1.0", "end-1c")
        if text.strip():
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status_var.set("\u0416\u0443\u0440\u043d\u0430\u043b \u0441\u043a\u043e\u043f\u0438\u0440\u043e\u0432\u0430\u043d \u0432 \u0431\u0443\u0444\u0435\u0440 \u043e\u0431\u043c\u0435\u043d\u0430")
        else:
            messagebox.showinfo("\u0416\u0443\u0440\u043d\u0430\u043b \u0441\u043e\u0431\u044b\u0442\u0438\u0439", "\u0416\u0443\u0440\u043d\u0430\u043b \u043f\u0443\u0441\u0442.", parent=self)

    def _save_log(self) -> None:
        text = self.log.get("1.0", "end-1c")
        if not text.strip():
            messagebox.showinfo("\u0421\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0435 \u0436\u0443\u0440\u043d\u0430\u043b\u0430", "\u0416\u0443\u0440\u043d\u0430\u043b \u043f\u0443\u0441\u0442.", parent=self)
            return
        default_name = f"batchresize_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        path = filedialog.asksaveasfilename(
            title="\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u0436\u0443\u0440\u043d\u0430\u043b \u0432 \u0444\u0430\u0439\u043b",
            parent=self,
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("\u0422\u0435\u043a\u0441\u0442\u043e\u0432\u044b\u0435 \u0444\u0430\u0439\u043b\u044b", "*.txt"), ("\u041b\u043e\u0433-\u0444\u0430\u0439\u043b\u044b", "*.log"), ("\u0412\u0441\u0435 \u0444\u0430\u0439\u043b\u044b", "*.*")]
        )
        if not path:
            return
        try:
            Path(path).write_text(text, encoding="utf-8")
            self.status_var.set(f"\u0416\u0443\u0440\u043d\u0430\u043b \u0441\u043e\u0445\u0440\u0430\u043d\u0451\u043d: {Path(path).name}")
            messagebox.showinfo("\u0421\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0435 \u0436\u0443\u0440\u043d\u0430\u043b\u0430", f"\u0416\u0443\u0440\u043d\u0430\u043b \u0443\u0441\u043f\u0435\u0448\u043d\u043e \u0441\u043e\u0445\u0440\u0430\u043d\u0451\u043d:\n{path}", parent=self)
        except OSError as exc:
            messagebox.showerror("\u0421\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0435 \u0436\u0443\u0440\u043d\u0430\u043b\u0430", f"\u041e\u0448\u0438\u0431\u043a\u0430 \u043f\u0440\u0438 \u0441\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u0438 \u0444\u0430\u0439\u043b\u0430:\n{exc}", parent=self)

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.status_var.set("\u0416\u0443\u0440\u043d\u0430\u043b \u043e\u0447\u0438\u0449\u0435\u043d")

    def _install_log_context_menu(self) -> None:
        menu = tk.Menu(self.log, tearoff=0)
        menu.add_command(label="\u041a\u043e\u043f\u0438\u0440\u043e\u0432\u0430\u0442\u044c", command=self._copy_log)
        menu.add_command(label="\u0412\u044b\u0434\u0435\u043b\u0438\u0442\u044c \u0432\u0441\u0451", command=self._select_all_log)
        menu.add_separator()
        menu.add_command(label="\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u0432 \u0444\u0430\u0439\u043b\u2026", command=self._save_log)
        menu.add_command(label="\u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c", command=self._clear_log)

        def show_menu(event):
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        self.log.bind("<Button-3>", show_menu)

    def _select_all_log(self):
        self.log.tag_add("sel", "1.0", "end")
        return "break"

    def _count_files(self):
        src = Path(self.input_var.get())
        dst = Path(self.output_var.get()).resolve()
        if not src.is_dir():
            self.recursive_note.set("\u041f\u0430\u043f\u043a\u0430 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u0430")
            return
        self.recursive_note.set("\u041f\u043e\u0434\u0441\u0447\u0451\u0442 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0439\u2026")
        def count():
            n = 0
            errors = []
            for base, dirs, files in os.walk(src, followlinks=False, onerror=errors.append):
                dirs[:] = [d for d in dirs if (Path(base) / d).resolve() != dst
                    and not (Path(base) / d).is_symlink()
                    and not getattr(Path(base) / d, "is_junction", lambda: False)()]
                n += sum(Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"} for f in files)
            self._events.put(("count", f"\u0418\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0439: {n} \xb7 \u041e\u0448\u0438\u0431\u043e\u043a \u0441\u043a\u0430\u043d\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f: {len(errors)}"))
        threading.Thread(target=count, daemon=True).start()

    def _open_output(self) -> None:
        path = Path(self.output_var.get()).expanduser()
        if not path.is_dir():
            messagebox.showinfo("imageresizer", "\u041f\u0430\u043f\u043a\u0430 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432 \u0435\u0449\u0451 \u043d\u0435 \u0441\u043e\u0437\u0434\u0430\u043d\u0430. \u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0443.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("imageresizer", f"\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u043e\u0442\u043a\u0440\u044b\u0442\u044c \u043f\u0430\u043f\u043a\u0443: {exc}")

    # -- process control -----------------------------------------------------
    def _base_command(self) -> list[str]:
        return [str(self.worker), "-u", "-m", "batchresize"]

    def _doctor(self) -> None:
        if self._is_running():
            return
        self._log_line("-" * 60)
        self._log_line("\u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u044f")
        if not self.view.log_visible:
            self.view.toggle_log()
        self._run_process(self._base_command() + ["--doctor"], "checking")

    def _build_arguments(self) -> list[str] | None:
        src = Path(self.input_var.get())
        dst = Path(self.output_var.get())
        if not src.is_dir():
            messagebox.showerror("batchresize",
                                 f"\u0418\u0441\u0445\u043e\u0434\u043d\u0430\u044f \u043f\u0430\u043f\u043a\u0430 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u0430:\n{src}")
            return None
        if src.resolve() == dst.resolve():
            messagebox.showerror("batchresize",
                                 "\u0418\u0441\u0445\u043e\u0434\u043d\u0430\u044f \u043f\u0430\u043f\u043a\u0430 \u0438 \u043f\u0430\u043f\u043a\u0430 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432 \u0434\u043e\u043b\u0436\u043d\u044b \u043e\u0442\u043b\u0438\u0447\u0430\u0442\u044c\u0441\u044f.")
            return None

        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            quality = int(self.jpeg_quality_var.get())
            value = float(self.sizing_value_var.get())
            workers = int(self.workers_var.get())
            raw_proc = self.processors_var.get().strip() if hasattr(self, "processors_var") else ""
            processors = int(raw_proc) if raw_proc and raw_proc.isdigit() and int(raw_proc) > 0 else None
            raw_readers = self.readers_var.get().strip() if hasattr(self, "readers_var") else ""
            readers = int(raw_readers) if raw_readers and raw_readers.isdigit() and int(raw_readers) > 0 else workers
            raw_writers = self.writers_var.get().strip() if hasattr(self, "writers_var") else ""
            writers = int(raw_writers) if raw_writers and raw_writers.isdigit() and int(raw_writers) > 0 else max(2, workers // 2)
            batch = int(self.batch_var.get())
            depth = int(self.queue_var.get())
            vram = float(self.vram_var.get())
            min_quality = self._min_quality()
        except ValueError:
            messagebox.showerror("batchresize",
                                 "\u0427\u0438\u0441\u043b\u043e\u0432\u044b\u0435 \u043f\u043e\u043b\u044f \u0434\u043e\u043b\u0436\u043d\u044b \u0441\u043e\u0434\u0435\u0440\u0436\u0430\u0442\u044c \u0447\u0438\u0441\u043b\u0430.")
            return None

        try:
            from .validation import dimensions, positive
            from .cli import parse_color
            from .size_limit import parse_size_bytes
            dimensions(width, height)
            positive(vram, "VRAM budget")
            caps = [(workers, "\u041f\u043e\u0442\u043e\u043a\u0438", 128), (batch, "Batch", 128), (depth, "\u041e\u0447\u0435\u0440\u0435\u0434\u044c", 256)]
            if processors is not None:
                caps.append((processors, "\u041f\u043e\u0442\u043e\u043a\u0438 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0438", 128))
            if readers:
                caps.append((readers, "\u041f\u043e\u0442\u043e\u043a\u0438 \u0447\u0442\u0435\u043d\u0438\u044f", 128))
            if writers:
                caps.append((writers, "\u041f\u043e\u0442\u043e\u043a\u0438 \u0437\u0430\u043f\u0438\u0441\u0438", 128))
            for val, name, cap in caps:
                if not 1 <= val <= cap:
                    raise ValueError(f"{name}: \u0434\u043e\u043f\u0443\u0441\u0442\u0438\u043c\u043e \u043e\u0442 1 \u0434\u043e {cap}")
            if self.sizing_var.get() != "resolution":
                positive(value, "\u0417\u043d\u0430\u0447\u0435\u043d\u0438\u0435 \u043c\u0430\u0441\u0448\u0442\u0430\u0431\u0430")
            if not 1 <= quality <= 100:
                raise ValueError("\u041a\u0430\u0447\u0435\u0441\u0442\u0432\u043e \u0434\u043e\u043b\u0436\u043d\u043e \u0431\u044b\u0442\u044c \u043e\u0442 1 \u0434\u043e 100.")
            if not 1 <= min_quality <= 100:
                raise ValueError("\u041c\u0438\u043d\u0438\u043c\u0430\u043b\u044c\u043d\u043e\u0435 \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u043e \u0434\u043e\u043b\u0436\u043d\u043e \u0431\u044b\u0442\u044c \u043e\u0442 1 \u0434\u043e 100.")
            max_file_size = self.max_file_size_var.get().strip()
            if max_file_size:
                if max_file_size.isdigit() and int(max_file_size) < 1024:
                    raise ValueError(f"\u041e\u0433\u0440\u0430\u043d\u0438\u0447\u0435\u043d\u0438\u0435 \u0440\u0430\u0437\u043c\u0435\u0440\u0430 '{max_file_size}' \u043e\u0437\u043d\u0430\u0447\u0430\u0435\u0442 {max_file_size} \u0431\u0430\u0439\u0442 (\u043c\u0435\u043d\u0435\u0435 1 \u041a\u0411). "
                                     f"\u0415\u0441\u043b\u0438 \u0432\u044b \u0438\u043c\u0435\u043b\u0438 \u0432 \u0432\u0438\u0434\u0443 \u043a\u0438\u043b\u043e\u0431\u0430\u0439\u0442\u044b, \u0443\u043a\u0430\u0436\u0438\u0442\u0435 '{max_file_size}KB'.")
                parse_size_bytes(max_file_size)
            parse_color(self.bg_color_var.get())
            if src.resolve().is_relative_to(dst.resolve()):
                raise ValueError("\u041f\u0430\u043f\u043a\u0430 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432 \u043d\u0435 \u043c\u043e\u0436\u0435\u0442 \u0431\u044b\u0442\u044c \u0438\u0441\u0445\u043e\u0434\u043d\u043e\u0439 \u0438\u043b\u0438 \u0435\u0451 \u0440\u043e\u0434\u0438\u0442\u0435\u043b\u044c\u0441\u043a\u043e\u0439 \u043f\u0430\u043f\u043a\u043e\u0439.")
        except ValueError as exc:
            messagebox.showerror("imageresizer", str(exc))
            return None

        # The floor can never exceed the selected quality - otherwise the
        # quality search would have an empty range.
        min_quality = min(min_quality, quality)

        args = self._base_command()
        args += ["-i", str(src), "-o", str(dst)]
        args += ["--size", "custom", "--width", str(width),
                 "--height", str(height)]
        args += ["--sizing", self.sizing_var.get(),
                 "--sizing-value", str(value)]
        mode = self.mode_var.get()
        args += ["--mode", mode]
        if mode in ("cover", "pad"):
            raw_anchor = self.crop_anchor_var.get().split(" ")[0].strip()
            if raw_anchor:
                args += ["--crop-anchor", raw_anchor]
        raw_sharpen = self.sharpen_preset_var.get().split(" ")[0].strip()
        if raw_sharpen in ("weak", "standard"):
            args += ["--sharpen", raw_sharpen]
        if self.allow_downscale_var.get():
            args.append("--allow-downscale-on-size-limit")
        if self.strip_personal_var.get():
            args.append("--strip-personal-data")
        if self.web_srgb_var.get():
            args.append("--web-srgb")
        if self.linear_light_var.get():
            args.append("--linear-light")
        if self.quality_var.get() == QUALITY_SOURCE:
            # The quality comes from every source file; the slider value stays
            # the fallback for sources without JPEG quantisation tables.
            args += ["--quality", QUALITY_SOURCE, "--jpeg-quality", str(quality)]
        else:
            args += ["--jpeg-quality", str(quality)]
        args += ["--subsampling", self.subsampling_var.get()]
        args += ["--filter", self.filter_var.get()]
        args += ["--backend", self.backend_var.get()]
        args += ["--workers", str(workers)]
        if processors is not None:
            args += ["--processors", str(processors)]
        if readers != workers:
            args += ["--readers", str(readers)]
        if writers != max(2, workers // 2):
            args += ["--writers", str(writers)]
        args += ["--batch-size", str(batch)]
        args += ["--queue-depth", str(depth)]
        args += ["--vram-budget", str(vram)]
        args += ["--bg-color", self.bg_color_var.get()]

        if self.allow_upscale_var.get():
            args.append("--allow-upscale")
        if not self.exif_var.get():
            args.append("--no-exif")
        if not self.icc_var.get():
            args.append("--no-icc")
        if self.skip_existing_var.get():
            args.append("--skip-existing")
        if not self.overwrite_var.get():
            args.append("--skip-existing")
        if self.truncated_var.get():
            args.append("--allow-truncated")
        if not self.fallback_var.get():
            args.append("--no-fallback")
        if self.huffman_var.get():
            args.append("--optimize-huffman")
        if self.fsync_var.get():
            args.append("--fsync")

        limit = self.limit_var.get().strip()
        if limit:
            if not limit.isdigit() or int(limit) < 1:
                messagebox.showerror("batchresize",
                                     "\u041b\u0438\u043c\u0438\u0442 \u0444\u0430\u0439\u043b\u043e\u0432 \u0434\u043e\u043b\u0436\u0435\u043d \u0431\u044b\u0442\u044c \u043f\u043e\u043b\u043e\u0436\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u043c \u0446\u0435\u043b\u044b\u043c \u0447\u0438\u0441\u043b\u043e\u043c.")
                return None
            args += ["--limit", limit]

        if max_file_size:
            args += ["--max-file-size", max_file_size,
                     "--min-quality", str(min_quality)]
        elif min_quality < 30:
            args += ["--min-quality", str(min_quality)]

        if self._active_profile_id:
            from .profiles import default_profiles_path, find_profile, load_profiles, builtin_profiles
            user_profs, _ = load_profiles(default_profiles_path())
            all_profs = builtin_profiles() + user_profs
            prof = find_profile(all_profs, self._active_profile_id)
            if prof:
                if prof.variants:
                    for v in prof.variants:
                        ov = OutputVariant.from_profile_dict(v)
                        args += ["--variant", ov.to_spec()]
                if prof.crop_position and prof.crop_position != "center":
                    args += ["--crop-anchor", prof.crop_position]
                if prof.color_policy and prof.color_policy != "preserve_rgb":
                    args += ["--color-policy", prof.color_policy]
                if prof.linear_light:
                    args.append("--linear-light")
                if prof.strip_personal_data:
                    args.append("--strip-personal-data")
                if prof.keep_datetime:
                    args.append("--keep-datetime")

        return args

    def _start(self) -> None:
        if self._is_running():
            messagebox.showinfo("batchresize", "\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u0443\u0436\u0435 \u0432\u044b\u043f\u043e\u043b\u043d\u044f\u0435\u0442\u0441\u044f.")
            return
        args = self._build_arguments()
        if args is None:
            return

        self._log_line("=" * 62)
        self._log_line(" ".join(args[3:]))
        self.progress.configure(value=0, maximum=100)
        self.view.metrics.set("0 \u0444\u0430\u0439\u043b\u043e\u0432  \xb7  \u041f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u043a\u0430\u2026")
        self.status_var.set("\u041f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u043a\u0430 \u043a \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0435\u2026")
        self.stats_var.set("")
        self.threads_var.set("")
        self._run_process(args, "running")

    def _run_process(self, args: list[str], mode: str) -> None:
        if not Path(args[0]).exists():
            messagebox.showerror(
                "batchresize",
                f"\u0418\u043d\u0442\u0435\u0440\u043f\u0440\u0435\u0442\u0430\u0442\u043e\u0440 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d:\n{args[0]}\n\n"
                "\u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 install.bat.")
            return
        self._pause_preview()
        self._start_time = time.perf_counter()
        self._stopping = False
        try:
            self._cancel_dir = tempfile.TemporaryDirectory(prefix="batchresize-cancel-")
            self._cancel_file = Path(self._cancel_dir.name) / "cancel"
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                       BATCHRESIZE_CANCEL_FILE=str(self._cancel_file))
            self._proc = subprocess.Popen(
                args, cwd=str(PROJECT_DIR), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            if hasattr(self, "_cancel_dir"):
                self._cancel_dir.cleanup()
            messagebox.showerror("batchresize", f"\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c \u0440\u0430\u0431\u043e\u0447\u0438\u0439 \u043f\u0440\u043e\u0446\u0435\u0441\u0441: {exc}")
            return

        self._mode = mode
        self.view.set_running(True)
        self.view.status_label.configure(foreground="#2361D1")
        if mode == "checking":
            self.view.metrics.set("\u0414\u0438\u0430\u0433\u043d\u043e\u0441\u0442\u0438\u043a\u0430 \u0434\u043e\u0441\u0442\u0443\u043f\u043d\u043e\u0433\u043e \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u044f\u2026")
        self.status_var.set("\u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u044f\u2026" if mode == "checking"
                            else "\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430\u2026")
        self._reader = threading.Thread(target=self._pump_output,
                                        args=(self._proc,), daemon=True)
        self._reader.start()

    def _pump_output(self, proc: subprocess.Popen) -> None:
        buffer = b""
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buffer += chunk
            if len(buffer) > 65536 and b"\n" not in buffer and b"\r" not in buffer:
                self._events.put(("line", buffer[:65536].decode("utf-8", "replace")))
                buffer = buffer[65536:]
            while True:
                cut = -1
                for sep in (b"\r\n", b"\r", b"\n"):
                    idx = buffer.find(sep)
                    if idx != -1 and (cut == -1 or idx < cut):
                        cut = idx
                if cut == -1:
                    break
                line = buffer[:cut]
                buffer = buffer[cut + 1:]
                if line:
                    text = line.decode("utf-8", "replace").strip()
                    if text:
                        self._events.put(("line", text))
        if buffer:
            text = buffer.decode("utf-8", "replace").strip()
            if text:
                self._events.put(("line", text))
        code = proc.wait()
        proc.stdout.close()
        self._events.put(("exit", code))

    def _is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _cancel(self) -> None:
        if not self._is_running():
            return
        self._stopping = True
        self.status_var.set("\u0417\u0430\u0432\u0435\u0440\u0448\u0430\u0435\u043c \u0442\u0435\u043a\u0443\u0449\u0438\u0435 \u043e\u043f\u0435\u0440\u0430\u0446\u0438\u0438\u2026")
        self.view.status_label.configure(foreground="#865310")
        assert self._proc is not None
        self._cancel_file.touch()
        self._log_line("\u0417\u0430\u043f\u0440\u043e\u0448\u0435\u043d\u0430 \u043e\u0442\u043c\u0435\u043d\u0430. \u0421\u043e\u0445\u0440\u0430\u043d\u044f\u044e\u0442\u0441\u044f \u0437\u0430\u0432\u0435\u0440\u0448\u0451\u043d\u043d\u044b\u0435 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b \u0438 manifest.")

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "line":
                    self._handle_line(str(payload))
                elif kind == "count":
                    self.recursive_note.set(str(payload))
                elif kind == "preview":
                    self._handle_preview(dict(payload))
                elif kind == "exit":
                    self._handle_exit(int(payload))
        except queue.Empty:
            pass
        if not (getattr(self, "_closing", False) and self._proc is None):
            self.after(100, self._drain_events)

    def _handle_line(self, text: str) -> None:
        match = PROGRESS_RE.search(text)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            rate, elapsed = float(match.group(3)), float(match.group(4))
            readers, procs, writers = (int(match.group(5)),
                                       int(match.group(6)),
                                       int(match.group(7)))
            self.progress.configure(maximum=max(1, total), value=done)
            self.status_var.set(f"\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 {done} \u0438\u0437 {total} \xb7 "
                                f"{rate:.1f} \u0444\u0430\u0439\u043b/\u0441")
            self.stats_var.set(f"\u0412\u0440\u0435\u043c\u044f: {elapsed:.1f} \u0441 \xb7 "
                               f"\u0413\u043e\u0442\u043e\u0432\u043e: {done} \u0438\u0437 {total}")
            self.view.metrics.set(f"{done} / {total} \u0444\u0430\u0439\u043b\u043e\u0432  \xb7  {rate:.1f} \u0444\u0430\u0439\u043b/\u0441  \xb7  {elapsed:.1f} \u0441")
            self.threads_var.set(f"\u041f\u043e\u0442\u043e\u043a\u0438: \u0447\u0442\u0435\u043d\u0438\u0435={readers}, "
                                 f"\u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430={procs}, \u0437\u0430\u043f\u0438\u0441\u044c={writers} "
                                 f"(\u0432\u0441\u0435\u0433\u043e {readers + procs + writers})")
            return
        self._log_line(text)

    def _handle_exit(self, code: int) -> None:
        elapsed = time.perf_counter() - self._start_time
        if self._mode == "checking":
            self.status_var.set("\u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u044f \u0437\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u0430" if code == 0 else
                                f"\u041e\u0448\u0438\u0431\u043a\u0430 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u044f (\u043a\u043e\u0434 {code})")
        elif self._stopping:
            self.status_var.set(f"\u041e\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u043e \u0447\u0435\u0440\u0435\u0437 {elapsed:.1f} \u0441")
        elif code == 0:
            self.status_var.set(f"\u0413\u043e\u0442\u043e\u0432\u043e \u0437\u0430 {elapsed:.1f} \u0441")
            self.progress.configure(value=self.progress["maximum"])
        else:
            self.status_var.set(f"\u0417\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u043e \u0441 \u043e\u0448\u0438\u0431\u043a\u0430\u043c\u0438 (\u043a\u043e\u0434 {code}) "
                                f"\u0447\u0435\u0440\u0435\u0437 {elapsed:.1f} \u0441")
        self._proc = None
        self.view.status_label.configure(foreground=("#A52C25" if code != 0 and not self._stopping
            else "#596981" if self._stopping else "#216040"))
        if self._mode == "checking":
            self.view.metrics.set("\u041e\u0442\u0447\u0451\u0442 \u043e\u0431 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u0438 \u2014 \u0432 \u0436\u0443\u0440\u043d\u0430\u043b\u0435")
        elif self._stopping:
            self.view.metrics.set("\u0421\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u044b \u0437\u0430\u0432\u0435\u0440\u0448\u0451\u043d\u043d\u044b\u0435 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b")
        elif code != 0:
            self.view.metrics.set("\u041f\u043e\u0434\u0440\u043e\u0431\u043d\u043e\u0441\u0442\u0438 \u043e\u0448\u0438\u0431\u043a\u0438 \u2014 \u0432 \u0436\u0443\u0440\u043d\u0430\u043b\u0435")
        else:
            self.view.metrics.set(self.stats_var.get() or "\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u0437\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u0430")
        self.view.set_running(False)
        if code != 0 and not self._stopping and not self.view.log_visible:
            self.view.toggle_log()
        if hasattr(self, "_cancel_dir"):
            self._cancel_dir.cleanup()
        if getattr(self, "_closing", False):
            self._stop_preview()
            self.destroy()
        else:
            self._resume_preview()

    def _on_close(self):
        if self._is_running():
            self._closing = True
            self._cancel()
            # Keep draining IPC until the child has saved its state.
            return
        self._stop_preview()
        self.destroy()


def main() -> int:
    app = BatchResizeUI()
    app.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
