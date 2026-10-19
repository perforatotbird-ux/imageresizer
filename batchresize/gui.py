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
from .preview import PreviewClient
from .profiles import (Profile, ProfileError, build_profile, builtin_profiles,
                       default_profiles_path, export_profile, find_profile,
                       import_into, load_profiles, read_profile_file,
                       save_profiles, unique_id, unique_name)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "input"
DEFAULT_OUTPUT = PROJECT_DIR / "output"

SIZING_MODES = [
    ("resolution", "По границам · ширина × высота"),
    ("percent", "В процентах от исходника"),
    ("megapixels", "По количеству мегапикселей"),
    ("longest", "По длинной стороне"),
    ("shortest", "По короткой стороне"),
]

QUALITY_CHOICES = [
    ("maximum", "Максимальное · q=95"),
    ("high", "Высокое · q=90"),
    ("balanced", "Баланс качества и размера · q=80"),
    ("small", "Компактный файл · q=60"),
    ("light", "Лёгкое сжатие · q=90"),
    ("medium", "Среднее сжатие · q=80"),
    ("strong", "Сильное сжатие · q=65"),
    ("extreme", "Максимальное сжатие · q=50"),
]

SUBSAMPLING_CHOICES = [
    ("auto", "Авто · зависит от качества JPEG"),
    ("444", "4:4:4 · полная цветовая детализация"),
    ("422", "4:2:2 · баланс детализации и размера"),
    ("420", "4:2:0 · меньший файл"),
]

PROGRESS_RE = re.compile(
    r"(\d+)/(\d+) files \| ([\d.]+) file/s \| ([\d.]+)s \| "
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
        self.title("imageresizer — обработка изображений")
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
        self.allow_upscale_var = tk.BooleanVar(value=False)
        self.filter_var = tk.StringVar(value="lanczos3")

        self.quality_var = tk.StringVar(value="balanced")
        self.jpeg_quality_var = tk.StringVar(value="80")
        self.subsampling_var = tk.StringVar(value="auto")
        self.bg_color_var = tk.StringVar(value="#FFFFFF")

        self.backend_var = tk.StringVar(value="auto")
        self.workers_var = tk.StringVar(value="8")
        self.batch_var = tk.StringVar(value="8")
        self.queue_var = tk.StringVar(value="32")
        self.vram_var = tk.StringVar(value="2.0")
        self.limit_var = tk.StringVar(value="")

        self.exif_var = tk.BooleanVar(value=True)
        self.icc_var = tk.BooleanVar(value=True)
        self.skip_existing_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=True)
        self.truncated_var = tk.BooleanVar(value=False)
        self.fallback_var = tk.BooleanVar(value=True)
        self.huffman_var = tk.BooleanVar(value=False)
        self.fsync_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="Готово к работе")
        self.stats_var = tk.StringVar(value="")
        self.threads_var = tk.StringVar(value="")

        self.profile_var = tk.StringVar(value="")
        self.profile_status_var = tk.StringVar(value="Профиль не выбран")
        self._profiles_path = default_profiles_path()
        self._user_profiles: list[Profile] = []
        self._all_profiles: list[Profile] = []
        self._active_profile_id: str | None = None
        self._applying_profile = False
        self._profile_errors: list[str] = []

        self.preview_source_var = tk.StringVar(value="")
        self.preview_mode_var = tk.StringVar(value="Результат")
        self.preview_view_var = tk.StringVar(value="fit")
        self.preview_status_var = tk.StringVar(value="Изображение не выбрано")
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
                    f"Профиль {profile.id!r} совпадает со встроенным и пропущен.")
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
            kind = "встроенный" if profile.builtin else "пользовательский"
            label = f"{profile.name} · {kind}"
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
                          if ALL_QUALITY_PRESETS[key] == quality), None)
            if match:
                self.quality_var.set(match[0])
                self.quality_combo.set(match[1])
            else:
                self.quality_var.set("custom")
                self.quality_combo.set("")
            sub_index = next((i for i, (key, _text) in enumerate(SUBSAMPLING_CHOICES)
                              if key == profile.subsampling), 0)
            self.subsampling_var.set(profile.subsampling)
            self.subsampling_combo.current(sub_index)
            self.bg_color_var.set(profile.bg_color)
            self.exif_var.set(profile.keep_exif)
            self.icc_var.set(profile.keep_icc)
        finally:
            self._applying_profile = False
        self._active_profile_id = profile.id
        self._set_profile_selection(profile.id)
        note = f"Профиль «{profile.name}» применён"
        if profile.variants:
            note += " · варианты появятся после этапа 3"
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
            self.profile_status_var.set(f"Профиль «{profile.name}» изменён")

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
        }

    def _profile_from_ui(self, name: str, profile_id: str | None = None) -> Profile:
        values = self._profile_values_from_ui()
        return build_profile(
            name, size=values["size"], mode=values["mode"],
            allow_upscale=values["allow_upscale"],
            filter_name=values["filter_name"], quality=values["quality"],
            subsampling=values["subsampling"], bg_color=values["bg_color"],
            keep_exif=values["keep_exif"], keep_icc=values["keep_icc"],
            profile_id=profile_id or unique_id(self._all_profiles, name))

    def _profile_from_profile(self, source: Profile, name: str,
                              profile_id: str | None = None) -> Profile:
        return build_profile(
            name, size=source.size, mode=source.mode,
            allow_upscale=source.allow_upscale, filter_name=source.filter,
            quality=source.quality, subsampling=source.subsampling,
            bg_color=source.bg_color, keep_exif=source.keep_exif,
            keep_icc=source.keep_icc,
            profile_id=profile_id or unique_id(self._all_profiles, name),
            description=source.description, variants=source.variants)

    def _save_profile_store(self) -> bool:
        try:
            save_profiles(self._profiles_path, self._user_profiles)
        except (OSError, ProfileError) as exc:
            messagebox.showerror("Профили", f"Не удалось сохранить профили:\n{exc}")
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
                initial = f"{active.name} (мой)" if active is not None else "Мой профиль"
                name = simpledialog.askstring("Сохранить профиль",
                                              "Название профиля:", initialvalue=initial,
                                              parent=self)
                if not name or not name.strip():
                    return
                name = name.strip()
                existing = find_profile(self._user_profiles, name)
                if existing is not None:
                    if not messagebox.askyesno(
                            "Профили", f"Профиль «{existing.name}» уже есть. Заменить?",
                            parent=self):
                        return
                    profile = self._profile_from_ui(name, profile_id=existing.id)
                    self._replace_user_profile(profile)
                else:
                    profile = self._profile_from_ui(name)
                    self._user_profiles.append(profile)
        except (ValueError, ProfileError) as exc:
            messagebox.showerror("Профили", str(exc), parent=self)
            return
        if self._save_profile_store():
            self._set_profile_selection(profile.id)
            self._apply_profile(profile)

    def _duplicate_profile(self) -> None:
        active = (find_profile(self._all_profiles, self._active_profile_id)
                  if self._active_profile_id else None)
        if active is None:
            messagebox.showinfo("Профили", "Сначала выберите профиль.", parent=self)
            return
        name = simpledialog.askstring("Дублировать профиль", "Название копии:",
                                      initialvalue=f"{active.name} (копия)",
                                      parent=self)
        if not name or not name.strip():
            return
        try:
            profile = self._profile_from_profile(active, name.strip())
        except (ValueError, ProfileError) as exc:
            messagebox.showerror("Профили", str(exc), parent=self)
            return
        self._user_profiles.append(profile)
        if self._save_profile_store():
            self._set_profile_selection(profile.id)
            self._apply_profile(profile)

    def _rename_profile(self) -> None:
        active = (find_profile(self._all_profiles, self._active_profile_id)
                  if self._active_profile_id else None)
        if active is None or active.builtin:
            messagebox.showinfo("Профили",
                                "Выберите пользовательский профиль для переименования.",
                                parent=self)
            return
        name = simpledialog.askstring("Переименовать профиль", "Новое название:",
                                      initialvalue=active.name, parent=self)
        if not name or not name.strip() or name.strip() == active.name:
            return
        try:
            profile = self._profile_from_profile(active, name.strip(),
                                                 profile_id=active.id)
        except (ValueError, ProfileError) as exc:
            messagebox.showerror("Профили", str(exc), parent=self)
            return
        self._replace_user_profile(profile)
        if self._save_profile_store():
            self._set_profile_selection(profile.id)
            self._apply_profile(profile)

    def _delete_profile(self) -> None:
        active = (find_profile(self._all_profiles, self._active_profile_id)
                  if self._active_profile_id else None)
        if active is None or active.builtin:
            messagebox.showinfo("Профили",
                                "Встроенный профиль нельзя удалить. Дублируйте его.",
                                parent=self)
            return
        if not messagebox.askyesno("Удалить профиль",
                                   f"Удалить профиль «{active.name}»?",
                                   parent=self):
            return
        self._user_profiles = [item for item in self._user_profiles
                               if item.id != active.id]
        self._active_profile_id = None
        self.profile_status_var.set("Профиль не выбран")
        if self._save_profile_store():
            self.profile_var.set("")

    def _import_profiles(self) -> None:
        path = filedialog.askopenfilename(
            title="Импорт профиля", parent=self,
            filetypes=[("JSON", "*.json"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            imported = read_profile_file(Path(path))
        except (OSError, ValueError, ProfileError) as exc:
            messagebox.showerror("Импорт профиля", str(exc), parent=self)
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
        messagebox.showinfo("Импорт профиля",
                            f"Импортировано профилей: {len(added)}", parent=self)

    def _export_profile(self) -> None:
        profile = (find_profile(self._all_profiles, self._active_profile_id)
                   if self._active_profile_id else None)
        if profile is None:
            name = simpledialog.askstring("Экспорт профиля", "Название профиля:",
                                          initialvalue="Мой профиль", parent=self)
            if not name or not name.strip():
                return
            try:
                profile = self._profile_from_ui(name.strip())
            except (ValueError, ProfileError) as exc:
                messagebox.showerror("Экспорт профиля", str(exc), parent=self)
                return
        path = filedialog.asksaveasfilename(
            title="Экспорт профиля", parent=self, defaultextension=".json",
            initialfile=f"{profile.id}.json",
            filetypes=[("JSON", "*.json"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            export_profile(Path(path), profile)
        except (OSError, ProfileError) as exc:
            messagebox.showerror("Экспорт профиля", str(exc), parent=self)
            return
        messagebox.showinfo("Экспорт профиля",
                            f"Профиль сохранён:\n{path}", parent=self)

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
            title="Изображение для превью",
            initialdir=initial_dir,
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
                       ("Все файлы", "*.*")])
        if path:
            self.preview_source_var.set(path)

    def _set_preview_view(self, view: str) -> None:
        if self.preview_mode_var.get() == "До / после":
            view = "fit"
        self.preview_view_var.set(view)
        self.view.set_preview_view(view)

    def _on_preview_mode_change(self, *_args) -> None:
        if self.preview_mode_var.get() == "До / после":
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
        if self._preview_has_result and self.preview_status_var.get().startswith("Готово"):
            self.preview_status_var.set("Параметры изменены — обновляем превью…")
        self._preview_after = self.after(320, self._request_preview)

    def _preview_box(self) -> tuple[int, int]:
        canvas = self.view.preview_canvas
        width = max(80, canvas.winfo_width())
        height = max(80, canvas.winfo_height())
        if width < 120:
            width, height = 292, 190
        if self.preview_mode_var.get() == "До / после":
            width = max(60, (width - 10) // 2)
        return width, height

    def _preview_params(self) -> dict:
        from .cli import parse_color
        from .validation import dimensions, positive

        width = int(self.width_var.get())
        height = int(self.height_var.get())
        quality = int(self.jpeg_quality_var.get())
        value = float(self.sizing_value_var.get())
        vram = float(self.vram_var.get())
        dimensions(width, height)
        positive(vram, "VRAM budget")
        sizing = self.sizing_var.get()
        if sizing != "resolution":
            positive(value, "Значение масштаба")
        if not 1 <= quality <= 100:
            raise ValueError("Качество должно быть от 1 до 100.")
        subsampling = {"auto": auto_subsampling(quality), "444": 0,
                       "422": 1, "420": 2}[self.subsampling_var.get()]
        return {
            "max_w": width,
            "max_h": height,
            "mode": self.mode_var.get(),
            "quality": quality,
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
        }

    def _request_preview(self) -> None:
        self._preview_after = None
        if self._preview_suspended or self._is_running():
            return
        source = Path(self.preview_source_var.get().strip()).expanduser()
        if not source.is_file():
            self.preview_status_var.set("Файл превью не найден")
            self.view.clear_preview("Файл превью не найден")
            return
        try:
            params = self._preview_params()
        except (ValueError, KeyError) as exc:
            self.preview_status_var.set(f"Проверьте параметры: {exc}")
            return
        if self._preview_client is None:
            self._preview_client = PreviewClient(
                self.worker, PROJECT_DIR, Path(self._preview_temp.name), self._events)
        width, height = self._preview_box()
        view = self.preview_view_var.get()
        if self.preview_mode_var.get() == "До / после":
            view = "fit"
        payload = {"source": str(source), "view": view,
                   "box_w": width, "box_h": height, "params": params}
        self._preview_request_id = self._preview_client.request(payload)
        self.preview_status_var.set("Загрузка превью…")

    def _handle_preview(self, payload: dict) -> None:
        kind = payload.get("kind")
        if kind == "log":
            self._log_line(f"preview: {payload.get('text', '')}")
            return
        if kind == "dead":
            self._log_line(f"preview: {payload.get('error', 'worker stopped')}")
            if not self._preview_suspended:
                self.preview_status_var.set(str(payload.get("error", "Превью недоступно")))
            return
        if kind == "invalidated":
            return
        if payload.get("id") != self._preview_request_id:
            return
        if not payload.get("ok"):
            error = str(payload.get("error", "неизвестная ошибка"))
            self.preview_status_var.set(f"Ошибка превью: {error}")
            self._log_line(f"preview: {error}")
            return
        if not self.view.show_preview(payload):
            self.preview_status_var.set("Не удалось показать превью")
            return
        self._preview_has_result = True
        source_w = int(payload.get("source_w", 0))
        source_h = int(payload.get("source_h", 0))
        result_w = int(payload.get("result_w", 0))
        result_h = int(payload.get("result_h", 0))
        kb = int(payload.get("jpeg_bytes", 0)) / 1024
        self.preview_info_var.set(
            f"{source_w} × {source_h} → {result_w} × {result_h} px  ·  {kb:.0f} КиБ")
        route = str(payload.get("route") or payload.get("backend") or "—")
        requested = str(payload.get("requested_backend") or "—")
        approximate = "  ·  приближённое" if payload.get("approximate") else ""
        self.preview_meta_var.set(f"Движок: {route}  ·  запрошен: {requested}{approximate}")
        self.preview_status_var.set("Готово" + (" · приближённое 100%" if payload.get("approximate") else ""))

    def _clear_preview(self) -> None:
        self._preview_has_result = False
        self.preview_status_var.set("Изображение не выбрано")
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
        self.preview_status_var.set("Превью приостановлено во время обработки")

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

    def _on_sizing_change(self, _event=None) -> None:
        idx = self.sizing_combo.current()
        self.sizing_var.set(SIZING_MODES[idx][0])

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
                value = ALL_QUALITY_PRESETS[key]
                self.jpeg_quality_var.set(str(value))
                self.quality_scale.set(value)
                return

    def _on_quality_scale(self, value) -> None:
        self.jpeg_quality_var.set(str(int(round(float(value)))))

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

    def _count_files(self):
        src = Path(self.input_var.get())
        dst = Path(self.output_var.get()).resolve()
        if not src.is_dir():
            self.recursive_note.set("Папка не найдена")
            return
        self.recursive_note.set("Подсчёт изображений…")
        def count():
            n = 0
            errors = []
            for base, dirs, files in os.walk(src, followlinks=False, onerror=errors.append):
                dirs[:] = [d for d in dirs if (Path(base) / d).resolve() != dst
                    and not (Path(base) / d).is_symlink()
                    and not getattr(Path(base) / d, "is_junction", lambda: False)()]
                n += sum(Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"} for f in files)
            self._events.put(("count", f"Изображений: {n} · Ошибок сканирования: {len(errors)}"))
        threading.Thread(target=count, daemon=True).start()

    def _open_output(self) -> None:
        path = Path(self.output_var.get()).expanduser()
        if not path.is_dir():
            messagebox.showinfo("imageresizer", "Папка результатов ещё не создана. Сначала запустите обработку.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("imageresizer", f"Не удалось открыть папку: {exc}")

    # -- process control -----------------------------------------------------
    def _base_command(self) -> list[str]:
        return [str(self.worker), "-u", "-m", "batchresize"]

    def _doctor(self) -> None:
        if self._is_running():
            return
        self._log_line("-" * 60)
        self._log_line("Проверка окружения")
        if not self.view.log_visible:
            self.view.toggle_log()
        self._run_process(self._base_command() + ["--doctor"], "checking")

    def _build_arguments(self) -> list[str] | None:
        src = Path(self.input_var.get())
        dst = Path(self.output_var.get())
        if not src.is_dir():
            messagebox.showerror("batchresize",
                                 f"Исходная папка не найдена:\n{src}")
            return None
        if src.resolve() == dst.resolve():
            messagebox.showerror("batchresize",
                                 "Исходная папка и папка результатов должны отличаться.")
            return None

        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            quality = int(self.jpeg_quality_var.get())
            value = float(self.sizing_value_var.get())
            workers = int(self.workers_var.get())
            batch = int(self.batch_var.get())
            depth = int(self.queue_var.get())
            vram = float(self.vram_var.get())
        except ValueError:
            messagebox.showerror("batchresize",
                                 "Числовые поля должны содержать числа.")
            return None

        try:
            from .validation import dimensions, positive
            from .cli import parse_color
            dimensions(width, height)
            positive(vram, "VRAM budget")
            for val, name, cap in [(workers, "Потоки", 64), (batch, "Batch", 128),
                                   (depth, "Очередь", 256)]:
                if not 1 <= val <= cap:
                    raise ValueError(f"{name}: допустимо от 1 до {cap}")
            if self.sizing_var.get() != "resolution":
                positive(value, "Значение масштаба")
            if not 1 <= quality <= 100:
                raise ValueError("Качество должно быть от 1 до 100.")
            parse_color(self.bg_color_var.get())
            if src.resolve().is_relative_to(dst.resolve()):
                raise ValueError("Папка результатов не может быть исходной или её родительской папкой.")
        except ValueError as exc:
            messagebox.showerror("imageresizer", str(exc))
            return None

        args = self._base_command()
        args += ["-i", str(src), "-o", str(dst)]
        args += ["--size", "custom", "--width", str(width),
                 "--height", str(height)]
        args += ["--sizing", self.sizing_var.get(),
                 "--sizing-value", str(value)]
        args += ["--mode", self.mode_var.get()]
        args += ["--jpeg-quality", str(quality)]
        args += ["--subsampling", self.subsampling_var.get()]
        args += ["--filter", self.filter_var.get()]
        args += ["--backend", self.backend_var.get()]
        args += ["--workers", str(workers)]
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
                                     "Лимит файлов должен быть положительным целым числом.")
                return None
            args += ["--limit", limit]

        return args

    def _start(self) -> None:
        if self._is_running():
            messagebox.showinfo("batchresize", "Обработка уже выполняется.")
            return
        args = self._build_arguments()
        if args is None:
            return

        self._log_line("=" * 62)
        self._log_line(" ".join(args[3:]))
        self.progress.configure(value=0, maximum=100)
        self.view.metrics.set("0 файлов  ·  Подготовка…")
        self.status_var.set("Подготовка к обработке…")
        self.stats_var.set("")
        self.threads_var.set("")
        self._run_process(args, "running")

    def _run_process(self, args: list[str], mode: str) -> None:
        if not Path(args[0]).exists():
            messagebox.showerror(
                "batchresize",
                f"Интерпретатор не найден:\n{args[0]}\n\n"
                "Сначала запустите install.bat.")
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
            messagebox.showerror("batchresize", f"Не удалось запустить рабочий процесс: {exc}")
            return

        self._mode = mode
        self.view.set_running(True)
        self.view.status_label.configure(foreground="#2361D1")
        if mode == "checking":
            self.view.metrics.set("Диагностика доступного окружения…")
        self.status_var.set("Проверка окружения…" if mode == "checking"
                            else "Обработка…")
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
        self.status_var.set("Завершаем текущие операции…")
        self.view.status_label.configure(foreground="#865310")
        assert self._proc is not None
        self._cancel_file.touch()
        self._log_line("Запрошена отмена. Сохраняются завершённые результаты и manifest.")

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
            self.status_var.set(f"Обработка {done} из {total} · "
                                f"{rate:.1f} файл/с")
            self.stats_var.set(f"Время: {elapsed:.1f} с · "
                               f"Готово: {done} из {total}")
            self.view.metrics.set(f"{done} / {total} файлов  ·  {rate:.1f} файл/с  ·  {elapsed:.1f} с")
            self.threads_var.set(f"Потоки: чтение={readers}, "
                                 f"обработка={procs}, запись={writers} "
                                 f"(всего {readers + procs + writers})")
            return
        self._log_line(text)

    def _handle_exit(self, code: int) -> None:
        elapsed = time.perf_counter() - self._start_time
        if self._mode == "checking":
            self.status_var.set("Проверка окружения завершена" if code == 0 else
                                f"Ошибка проверки окружения (код {code})")
        elif self._stopping:
            self.status_var.set(f"Остановлено через {elapsed:.1f} с")
        elif code == 0:
            self.status_var.set(f"Готово за {elapsed:.1f} с")
            self.progress.configure(value=self.progress["maximum"])
        else:
            self.status_var.set(f"Завершено с ошибками (код {code}) "
                                f"через {elapsed:.1f} с")
        self._proc = None
        self.view.status_label.configure(foreground=("#A52C25" if code != 0 and not self._stopping
            else "#596981" if self._stopping else "#216040"))
        if self._mode == "checking":
            self.view.metrics.set("Отчёт об окружении — в журнале")
        elif self._stopping:
            self.view.metrics.set("Сохранены завершённые результаты")
        elif code != 0:
            self.view.metrics.set("Подробности ошибки — в журнале")
        else:
            self.view.metrics.set(self.stats_var.get() or "Обработка завершена")
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
