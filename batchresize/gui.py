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
from tkinter import filedialog, messagebox, ttk

from .presets import ALL_QUALITY_PRESETS, SIZE_PRESETS, auto_subsampling

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

        self._build()
        self._on_quality_change()
        self._on_preset_change()
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
            self.destroy()

    def _on_close(self):
        if self._is_running():
            self._closing = True
            self._cancel()
            # Keep draining IPC until the child has saved its state.
            return
        self.destroy()


def main() -> int:
    app = BatchResizeUI()
    app.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
