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
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .presets import QUALITY_PRESETS, SIZE_PRESETS, auto_subsampling

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "input"
DEFAULT_OUTPUT = PROJECT_DIR / "output"

SIZING_MODES = [
    ("resolution", "Resolution - fit inside the box"),
    ("percent", "Percent - scale by %"),
    ("megapixels", "Megapixels - scale to MP"),
    ("longest", "Longest side - set pixels"),
    ("shortest", "Shortest side - set pixels"),
]

QUALITY_CHOICES = [
    ("maximum", "Maximum  q=95  visually lossless, large files"),
    ("high", "High  q=90  high quality, large files"),
    ("balanced", "Balanced  q=80  reasonable size"),
    ("small", "Small  q=60  space saving, visible loss"),
    ("light", "Light  q=90  almost no visible loss"),
    ("medium", "Medium  q=80  recommended"),
    ("strong", "Strong  q=65  noticeably smaller"),
    ("extreme", "Extreme  q=50  maximum compression"),
]

SUBSAMPLING_CHOICES = [
    ("auto", "Default - 4:4:4 up to q85, then matches the source"),
    ("444", "4:4:4 - full colour, highest quality"),
    ("422", "4:2:2 - less colour, significant saving"),
    ("420", "4:2:0 - colours may look washed out"),
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
        super().__init__()
        self.title("batchresize - GPU batch image resizer")
        self.geometry("940x780")
        self.minsize(860, 720)

        self._events: queue.Queue = queue.Queue()
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

        self.status_var = tk.StringVar(value="Ready")
        self.stats_var = tk.StringVar(value="")
        self.threads_var = tk.StringVar(value="")

        self._build()
        self._on_quality_change()
        self._on_preset_change()
        self.after(100, self._drain_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout --------------------------------------------------------------
    def _build(self) -> None:
        style = ttk.Style(self)
        for theme in ("vista", "winnative", "clam"):
            try:
                style.theme_use(theme)
                break
            except tk.TclError:
                continue

        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        self._build_input_tab()
        self._build_modify_tab()
        self._build_output_tab()
        self._build_advanced_tab()

        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=(10, 4))
        ttk.Button(bar, text="Start", width=14,
                   command=self._start).pack(side="left")
        ttk.Button(bar, text="Cancel", width=14,
                   command=self._cancel).pack(side="left", padx=6)
        ttk.Button(bar, text="Environment check", width=20,
                   command=self._doctor).pack(side="left")
        ttk.Button(bar, text="Open output", width=14,
                   command=self._open_output).pack(side="right")

        self.progress = ttk.Progressbar(outer, mode="determinate",
                                        maximum=100)
        self.progress.pack(fill="x", pady=(4, 4))

        ttk.Label(outer, textvariable=self.status_var,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w")
        ttk.Label(outer, textvariable=self.stats_var).pack(anchor="w")
        ttk.Label(outer, textvariable=self.threads_var).pack(anchor="w")

        log_frame = ttk.LabelFrame(outer, text="Log", padding=6)
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.log = tk.Text(log_frame, height=11, wrap="none", state="disabled",
                           font=("Consolas", 9))
        scroll = ttk.Scrollbar(log_frame, orient="vertical",
                               command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _build_input_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="Input")

        ttk.Label(tab, text="Source folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(tab, textvariable=self.input_var).grid(
            row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(tab, text="Browse...",
                   command=lambda: self._browse(self.input_var)).grid(
            row=1, column=1)

        ttk.Label(tab, text="Destination folder").grid(
            row=2, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(tab, textvariable=self.output_var).grid(
            row=3, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(tab, text="Browse...",
                   command=lambda: self._browse(self.output_var)).grid(
            row=3, column=1)

        ttk.Label(tab, text="Subfolders are always scanned and the folder "
                            "structure is kept.").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))

        ttk.Label(tab, text="Process only the first N files (blank = all)").grid(
            row=5, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(tab, textvariable=self.limit_var, width=12).grid(
            row=6, column=0, sticky="w")

        ttk.Label(tab, textvariable=self.recursive_note,
                  foreground="#444").grid(row=7, column=0, columnspan=2,
                                          sticky="w", pady=(12, 0))
        ttk.Button(tab, text="Count files",
                   command=self._count_files).grid(row=8, column=0,
                                                   sticky="w", pady=(8, 0))
        tab.columnconfigure(0, weight=1)

    def _build_modify_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="Modify")

        ttk.Label(tab, text="Downscaling method").grid(row=0, column=0,
                                                       sticky="w")
        self.sizing_combo = ttk.Combobox(
            tab, state="readonly", width=46,
            values=[label for _, label in SIZING_MODES])
        self.sizing_combo.current(0)
        self.sizing_combo.grid(row=1, column=0, sticky="w")
        self.sizing_combo.bind("<<ComboboxSelected>>",
                               self._on_sizing_change)

        ttk.Label(tab, text="Size preset").grid(row=2, column=0, sticky="w",
                                                pady=(12, 0))
        self.preset_combo = ttk.Combobox(
            tab, state="readonly", width=24,
            values=[*SIZE_PRESETS, "custom"])
        self.preset_combo.set("web_1024")
        self.preset_combo.grid(row=3, column=0, sticky="w")
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)

        dims = ttk.Frame(tab)
        dims.grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Label(dims, text="Max width").pack(side="left")
        ttk.Entry(dims, textvariable=self.width_var, width=9).pack(
            side="left", padx=(4, 14))
        ttk.Label(dims, text="Max height").pack(side="left")
        ttk.Entry(dims, textvariable=self.height_var, width=9).pack(
            side="left", padx=4)

        ttk.Label(tab, text="Value for percent / megapixels / pixels").grid(
            row=5, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(tab, textvariable=self.sizing_value_var, width=14).grid(
            row=6, column=0, sticky="w")

        ttk.Label(tab, text="Fit mode").grid(row=7, column=0, sticky="w",
                                             pady=(12, 0))
        modes = ttk.Frame(tab)
        modes.grid(row=8, column=0, sticky="w")
        ttk.Radiobutton(modes, text="Contain - fit inside, no crop",
                        variable=self.mode_var,
                        value="contain").pack(anchor="w")
        ttk.Radiobutton(modes, text="Cover - fill the box and crop the centre",
                        variable=self.mode_var, value="cover").pack(anchor="w")

        ttk.Checkbutton(tab, text="Allow enlarging small photos",
                        variable=self.allow_upscale_var).grid(
            row=9, column=0, sticky="w", pady=(10, 0))

        ttk.Label(tab, text="Resampling filter").grid(row=10, column=0,
                                                      sticky="w",
                                                      pady=(12, 0))
        ttk.Combobox(tab, state="readonly", width=24,
                     textvariable=self.filter_var,
                     values=["lanczos3", "bilinear", "box"]).grid(
            row=11, column=0, sticky="w")
        tab.columnconfigure(0, weight=1)

    def _build_output_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="Output")

        ttk.Label(tab, text="Quality preset (JPEG)").grid(row=0, column=0,
                                                          sticky="w")
        self.quality_combo = ttk.Combobox(
            tab, state="readonly", width=58,
            values=[label for _, label in QUALITY_CHOICES])
        self.quality_combo.set(QUALITY_CHOICES[2][1])
        self.quality_combo.grid(row=1, column=0, sticky="w")
        self.quality_combo.bind("<<ComboboxSelected>>",
                                self._on_quality_change)

        qrow = ttk.Frame(tab)
        qrow.grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Label(qrow, text="Quality").pack(side="left")
        self.quality_scale = ttk.Scale(qrow, from_=1, to=100,
                                       orient="horizontal", length=280,
                                       command=self._on_quality_scale)
        self.quality_scale.set(80)
        self.quality_scale.pack(side="left", padx=8)
        ttk.Label(qrow, textvariable=self.jpeg_quality_var,
                  width=4).pack(side="left")

        ttk.Label(tab, text="Chroma subsampling").grid(row=3, column=0,
                                                       sticky="w",
                                                       pady=(14, 0))
        self.subsampling_combo = ttk.Combobox(
            tab, state="readonly", width=58,
            values=[label for _, label in SUBSAMPLING_CHOICES])
        self.subsampling_combo.current(0)
        self.subsampling_combo.grid(row=4, column=0, sticky="w")
        self.subsampling_combo.bind("<<ComboboxSelected>>",
                                    self._on_subsampling_change)

        ttk.Label(tab, text="Background for transparency (#RRGGBB)").grid(
            row=5, column=0, sticky="w", pady=(14, 0))
        ttk.Entry(tab, textvariable=self.bg_color_var, width=14).grid(
            row=6, column=0, sticky="w")

        meta = ttk.LabelFrame(tab, text="Metadata", padding=8)
        meta.grid(row=7, column=0, sticky="w", pady=(14, 0))
        ttk.Checkbutton(meta, text="Copy EXIF",
                        variable=self.exif_var).pack(anchor="w")
        ttk.Checkbutton(meta, text="Copy ICC colour profile",
                        variable=self.icc_var).pack(anchor="w")

        files = ttk.LabelFrame(tab, text="Existing files", padding=8)
        files.grid(row=8, column=0, sticky="w", pady=(10, 0))
        ttk.Checkbutton(files, text="Skip files that already exist",
                        variable=self.skip_existing_var).pack(anchor="w")
        ttk.Checkbutton(files, text="Overwrite existing results",
                        variable=self.overwrite_var).pack(anchor="w")
        tab.columnconfigure(0, weight=1)

    def _build_advanced_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="Advanced")

        ttk.Label(tab, text="Backend").grid(row=0, column=0, sticky="w")
        ttk.Combobox(tab, state="readonly", width=20,
                     textvariable=self.backend_var,
                     values=["auto", "gpu", "hybrid", "cpu"]).grid(
            row=1, column=0, sticky="w")

        grid = ttk.Frame(tab)
        grid.grid(row=2, column=0, columnspan=2, sticky="w", pady=(14, 0))
        rows = [
            ("Threads (readers)", self.workers_var),
            ("GPU batch size", self.batch_var),
            ("Queue depth", self.queue_var),
            ("VRAM budget, GiB", self.vram_var),
        ]
        for i, (label, var) in enumerate(rows):
            ttk.Label(grid, text=label).grid(row=i, column=0, sticky="w",
                                             pady=3)
            ttk.Entry(grid, textvariable=var, width=10).grid(
                row=i, column=1, sticky="w", padx=8)

        opts = ttk.LabelFrame(tab, text="Robustness", padding=8)
        opts.grid(row=3, column=0, columnspan=2, sticky="w", pady=(14, 0))
        ttk.Checkbutton(opts, text="CPU fallback for files the GPU rejects",
                        variable=self.fallback_var).pack(anchor="w")
        ttk.Checkbutton(opts, text="Decode truncated JPEGs",
                        variable=self.truncated_var).pack(anchor="w")
        ttk.Checkbutton(opts, text="Optimize Huffman tables",
                        variable=self.huffman_var).pack(anchor="w")
        ttk.Checkbutton(opts, text="fsync before rename (safer, slower)",
                        variable=self.fsync_var).pack(anchor="w")

        ttk.Label(tab, text=f"Worker interpreter: {self.worker}",
                  foreground="#555").grid(row=4, column=0, columnspan=2,
                                          sticky="w", pady=(16, 0))
        ttk.Label(tab, text="Settings are applied when you press Start.",
                  foreground="#555").grid(row=5, column=0, sticky="w",
                                          pady=(6, 0))
        tab.columnconfigure(0, weight=1)

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
                value = QUALITY_PRESETS[key]
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
        self.log.see("end")
        self.log.configure(state="disabled")

    def _count_files(self) -> None:
        src = Path(self.input_var.get())
        if not src.is_dir():
            self.recursive_note.set("Folder does not exist")
            return
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        count = sum(1 for p in src.rglob("*")
                    if p.is_file() and p.suffix.lower() in exts)
        self.recursive_note.set(f"{count} image file(s) found")

    def _open_output(self) -> None:
        path = Path(self.output_var.get())
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])

    # -- process control -----------------------------------------------------
    def _base_command(self) -> list[str]:
        return [str(self.worker), "-u", "-m", "batchresize"]

    def _doctor(self) -> None:
        if self._is_running():
            return
        self._log_line("-" * 60)
        self._log_line("Environment check")
        self._run_process(self._base_command() + ["--doctor"], "checking")

    def _build_arguments(self) -> list[str] | None:
        src = Path(self.input_var.get())
        dst = Path(self.output_var.get())
        if not src.is_dir():
            messagebox.showerror("batchresize",
                                 f"Source folder not found:\n{src}")
            return None
        if src.resolve() == dst.resolve():
            messagebox.showerror("batchresize",
                                 "Source and destination must differ.")
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
                                 "Numeric fields must contain numbers.")
            return None

        if not 1 <= quality <= 100:
            messagebox.showerror("batchresize",
                                 "Quality must be between 1 and 100.")
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
                                     "The file limit must be a positive number.")
                return None
            args += ["--limit", limit]

        return args

    def _start(self) -> None:
        if self._is_running():
            messagebox.showinfo("batchresize", "A run is already in progress.")
            return
        args = self._build_arguments()
        if args is None:
            return

        self._log_line("=" * 62)
        self._log_line(" ".join(args[3:]))
        self.progress.configure(value=0, maximum=100)
        self.status_var.set("Starting...")
        self.stats_var.set("")
        self.threads_var.set("")
        self._run_process(args, "running")

    def _run_process(self, args: list[str], mode: str) -> None:
        if not Path(args[0]).exists():
            messagebox.showerror(
                "batchresize",
                f"Interpreter not found:\n{args[0]}\n\n"
                "Run install.bat first.")
            return
        self._start_time = time.perf_counter()
        self._stopping = False
        try:
            self._proc = subprocess.Popen(
                args, cwd=str(PROJECT_DIR),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            messagebox.showerror("batchresize", f"Cannot start worker: {exc}")
            return

        self._mode = mode
        self.status_var.set("Checking environment..." if mode == "checking"
                            else "Processing...")
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
        self._events.put(("exit", code))

    def _is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _cancel(self) -> None:
        if not self._is_running():
            return
        self._stopping = True
        self.status_var.set("Cancelling...")
        assert self._proc is not None
        self._proc.terminate()

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "line":
                    self._handle_line(str(payload))
                elif kind == "exit":
                    self._handle_exit(int(payload))
        except queue.Empty:
            pass
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
            self.status_var.set(f"Processing {done}/{total} - "
                                f"{rate:.1f} file/s")
            self.stats_var.set(f"Elapsed {elapsed:.1f}s | "
                               f"done {done} of {total}")
            self.threads_var.set(f"Threads: readers={readers}, "
                                 f"processors={procs}, writers={writers} "
                                 f"(total {readers + procs + writers})")
            return
        self._log_line(text)

    def _handle_exit(self, code: int) -> None:
        elapsed = time.perf_counter() - self._start_time
        if self._mode == "checking":
            self.status_var.set("Environment check finished")
            return
        if self._stopping:
            self.status_var.set(f"Cancelled after {elapsed:.1f}s")
        elif code == 0:
            self.status_var.set(f"Done in {elapsed:.1f}s")
            self.progress.configure(value=self.progress["maximum"])
        else:
            self.status_var.set(f"Finished with errors (exit code {code}) "
                                f"after {elapsed:.1f}s")
        self._proc = None

    def _on_close(self) -> None:
        if self._is_running():
            try:
                assert self._proc is not None
                self._proc.terminate()
            except Exception:
                pass
        self.destroy()


def main() -> int:
    app = BatchResizeUI()
    app.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
