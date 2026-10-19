"""Tkinter desktop UI for batchresize.

Layout follows XL Converter's mental model: an *Input* tab, a *Modify* tab
(downscaling) and an *Output* tab (quality / chroma), plus a live log and a
progress bar. Everything is standard library - no extra dependencies.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .cli import _select_engine, _write_report
from .cpu_backend import CpuEngine
from .geometry import compute_box, make_plan
from .pipeline import Pipeline, RunOptions, discover
from .presets import (ALL_QUALITY_PRESETS, QUALITY_LABELS, SIZE_PRESETS,
                      SUBSAMPLING_LABELS)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "input"
DEFAULT_OUTPUT = PROJECT_DIR / "output"

SIZING_LABELS = [
    ("resolution", "Resolution - fit inside W x H"),
    ("percent", "Percent - scale by %"),
    ("megapixels", "Megapixels - scale to MP"),
    ("longest", "Longest side - set px"),
    ("shortest", "Shortest side - set px"),
]

QUALITY_ORDER = ["maximum", "high", "balanced", "small",
                 "light", "medium", "strong", "extreme"]

SUBSAMPLING_ORDER = ["auto", "444", "422", "420"]


class BatchResizeUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("batchresize - GPU batch image resizer")
        self.geometry("900x760")
        self.minsize(820, 700)

        self._events: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._pipeline: Pipeline | None = None
        self._start_time = 0.0
        self._cancel = threading.Event()

        self.input_var = tk.StringVar(value=str(DEFAULT_INPUT))
        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT))
        self.recursive_var = tk.BooleanVar(value=True)

        self.quality_var = tk.StringVar(value="balanced")
        self.jpeg_quality_var = tk.IntVar(value=80)
        self.subsampling_var = tk.StringVar(value="auto")
        self.mode_var = tk.StringVar(value="contain")
        self.sizing_var = tk.StringVar(value="resolution")
        self.size_preset_var = tk.StringVar(value="web_1024")
        self.width_var = tk.IntVar(value=1024)
        self.height_var = tk.IntVar(value=1024)
        self.sizing_value_var = tk.DoubleVar(value=100.0)
        self.allow_upscale_var = tk.BooleanVar(value=False)
        self.filter_var = tk.StringVar(value="lanczos3")

        self.backend_var = tk.StringVar(value="auto")
        self.workers_var = tk.IntVar(value=8)
        self.batch_var = tk.IntVar(value=8)
        self.queue_var = tk.IntVar(value=32)
        self.vram_var = tk.DoubleVar(value=2.0)

        self.exif_var = tk.BooleanVar(value=True)
        self.icc_var = tk.BooleanVar(value=True)
        self.skip_existing_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=True)
        self.truncated_var = tk.BooleanVar(value=False)
        self.fallback_var = tk.BooleanVar(value=True)
        self.huffman_var = tk.BooleanVar(value=False)
        self.fsync_var = tk.BooleanVar(value=False)
        self.bg_color_var = tk.StringVar(value="#FFFFFF")

        self.status_var = tk.StringVar(value="Ready")
        self.stats_var = tk.StringVar(value="")
        self.threads_var = tk.StringVar(value="")

        self._build()
        self._sync_preset_fields()
        self.after(120, self._drain_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout --------------------------------------------------------------
    def _build(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

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
        ttk.Button(bar, text="Start", command=self._start,
                   width=14).pack(side="left")
        ttk.Button(bar, text="Cancel", command=self._cancel_run,
                   width=14).pack(side="left", padx=6)
        ttk.Button(bar, text="Environment check",
                   command=self._doctor, width=20).pack(side="left")
        ttk.Button(bar, text="Open output",
                   command=self._open_output, width=16).pack(side="right")

        self.progress = ttk.Progressbar(outer, mode="determinate")
        self.progress.pack(fill="x", pady=(4, 4))

        ttk.Label(outer, textvariable=self.status_var,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w")
        ttk.Label(outer, textvariable=self.stats_var).pack(anchor="w")
        ttk.Label(outer, textvariable=self.threads_var).pack(anchor="w")

        log_frame = ttk.LabelFrame(outer, text="Log", padding=6)
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.log = tk.Text(log_frame, height=10, wrap="none",
                           state="disabled", font=("Consolas", 9))
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
            row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(tab, textvariable=self.output_var).grid(
            row=3, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(tab, text="Browse...",
                   command=lambda: self._browse(self.output_var)).grid(
            row=3, column=1)

        ttk.Checkbutton(tab, text="Scan subfolders (keep folder structure)",
                        variable=self.recursive_var).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self.input_info = ttk.Label(tab, text="", foreground="#444")
        self.input_info.grid(row=5, column=0, columnspan=2, sticky="w",
                             pady=(10, 0))
        ttk.Button(tab, text="Count files",
                   command=self._count_files).grid(
            row=6, column=0, sticky="w", pady=(10, 0))

        tab.columnconfigure(0, weight=1)

    def _build_modify_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="Modify")

        ttk.Label(tab, text="Downscaling method").grid(
            row=0, column=0, sticky="w")
        self.sizing_combo = ttk.Combobox(
            tab, state="readonly", width=38,
            values=[label for _, label in SIZING_LABELS])
        self.sizing_combo.current(0)
        self.sizing_combo.grid(row=1, column=0, columnspan=2, sticky="w")
        self.sizing_combo.bind("<<ComboboxSelected>>", self._on_sizing_change)

        ttk.Label(tab, text="Size preset").grid(
            row=2, column=0, sticky="w", pady=(12, 0))
        self.preset_combo = ttk.Combobox(
            tab, state="readonly", width=24,
            values=[*SIZE_PRESETS, "custom"])
        self.preset_combo.set("web_1024")
        self.preset_combo.grid(row=3, column=0, sticky="w")
        self.preset_combo.bind("<<ComboboxSelected>>",
                               self._on_preset_change)

        dims = ttk.Frame(tab)
        dims.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(dims, text="Max width").pack(side="left")
        ttk.Spinbox(dims, from_=1, to=30000, width=8,
                    textvariable=self.width_var).pack(side="left", padx=(4, 12))
        ttk.Label(dims, text="Max height").pack(side="left")
        ttk.Spinbox(dims, from_=1, to=30000, width=8,
                    textvariable=self.height_var).pack(side="left", padx=4)

        ttk.Label(tab, text="Value for the selected method "
                            "(%, MP, px)").grid(
            row=5, column=0, sticky="w", pady=(12, 0))
        ttk.Spinbox(tab, from_=0.01, to=100000.0, increment=0.1, width=14,
                    textvariable=self.sizing_value_var).grid(
            row=6, column=0, sticky="w")

        ttk.Label(tab, text="Fit mode").grid(
            row=7, column=0, sticky="w", pady=(12, 0))
        modes = ttk.Frame(tab)
        modes.grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(modes, text="Contain - fit inside, no crop",
                        variable=self.mode_var, value="contain").pack(
            anchor="w")
        ttk.Radiobutton(modes, text="Cover - fill and crop the centre",
                        variable=self.mode_var, value="cover").pack(anchor="w")

        ttk.Checkbutton(tab, text="Allow enlarging small photos",
                        variable=self.allow_upscale_var).grid(
            row=9, column=0, sticky="w", pady=(10, 0))

        ttk.Label(tab, text="Resampling filter").grid(
            row=10, column=0, sticky="w", pady=(12, 0))
        ttk.Combobox(tab, state="readonly", width=24,
                     textvariable=self.filter_var,
                     values=["lanczos3", "bilinear", "box"]).grid(
            row=11, column=0, sticky="w")

        tab.columnconfigure(0, weight=1)

    def _build_output_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="Output")

        ttk.Label(tab, text="Quality preset (JPEG)").grid(
            row=0, column=0, sticky="w")
        self.quality_combo = ttk.Combobox(
            tab, state="readonly", width=52,
            values=[QUALITY_LABELS[q] for q in QUALITY_ORDER])
        self.quality_combo.set(QUALITY_LABELS["balanced"])
        self.quality_combo.grid(row=1, column=0, columnspan=2, sticky="w")
        self.quality_combo.bind("<<ComboboxSelected>>",
                               self._on_quality_change)

        qrow = ttk.Frame(tab)
        qrow.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(qrow, text="Custom quality 1..100").pack(side="left")
        self.quality_scale = ttk.Scale(
            qrow, from_=1, to=100, orient="horizontal", length=260,
            command=self._on_quality_scale)
        self.quality_scale.set(80)
        self.quality_scale.pack(side="left", padx=8)
        ttk.Label(qrow, textvariable=self.jpeg_quality_var,
                  width=4).pack(side="left")
        ttk.Button(qrow, text="Use preset",
                   command=self._reset_quality_from_preset).pack(
            side="left", padx=8)

        ttk.Label(tab, text="Chroma subsampling").grid(
            row=3, column=0, sticky="w", pady=(14, 0))
        self.subsampling_combo = ttk.Combobox(
            tab, state="readonly", width=52,
            values=[SUBSAMPLING_LABELS[s] for s in SUBSAMPLING_ORDER])
        self.subsampling_combo.grid(row=4, column=0, columnspan=2, sticky="w")
        self.subsampling_combo.set(SUBSAMPLING_LABELS["auto"])
        self.subsampling_combo.bind("<<ComboboxSelected>>",
                                    self._on_subsampling_change)

        ttk.Label(tab, text="Background for transparency (#RRGGBB)").grid(
            row=5, column=0, sticky="w", pady=(14, 0))
        ttk.Entry(tab, textvariable=self.bg_color_var, width=14).grid(
            row=6, column=0, sticky="w")

        meta = ttk.LabelFrame(tab, text="Metadata", padding=8)
        meta.grid(row=7, column=0, columnspan=2, sticky="w", pady=(14, 0))
        ttk.Checkbutton(meta, text="Copy EXIF",
                        variable=self.exif_var).pack(anchor="w")
        ttk.Checkbutton(meta, text="Copy ICC colour profile",
                        variable=self.icc_var).pack(anchor="w")

        files = ttk.LabelFrame(tab, text="Existing files", padding=8)
        files.grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 0))
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
            ("Threads (readers/processors)", self.workers_var, 1, 64),
            ("GPU batch size", self.batch_var, 1, 64),
            ("Queue depth between stages", self.queue_var, 1, 256),
            ("VRAM budget, GiB", self.vram_var, 0.05, 64.0),
        ]
        for i, (label, var, lo, hi) in enumerate(rows):
            ttk.Label(grid, text=label).grid(row=i, column=0, sticky="w",
                                             pady=3)
            ttk.Spinbox(grid, from_=lo, to=hi, width=10,
                        textvariable=var).grid(row=i, column=1, sticky="w",
                                               padx=8)

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

        ttk.Label(tab, text="Settings are applied when you press Start.",
                  foreground="#555").grid(
            row=4, column=0, sticky="w", pady=(14, 0))

        tab.columnconfigure(0, weight=1)

    # -- UI helpers ----------------------------------------------------------
    def _browse(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(initialdir=var.get() or str(PROJECT_DIR))
        if path:
            var.set(path)

    def _on_sizing_change(self, _event=None) -> None:
        idx = self.sizing_combo.current()
        self.sizing_var.set(SIZING_LABELS[idx][0])

    def _on_preset_change(self, _event=None) -> None:
        name = self.preset_combo.get()
        if name in SIZE_PRESETS:
            w, h = SIZE_PRESETS[name]
            self.width_var.set(w)
            self.height_var.set(h)

    def _on_quality_change(self, _event=None) -> None:
        label = self.quality_combo.get()
        for key, text in QUALITY_LABELS.items():
            if text == label:
                self.quality_var.set(key)
                self.jpeg_quality_var.set(ALL_QUALITY_PRESETS[key])
                self.quality_scale.set(ALL_QUALITY_PRESETS[key])
                break

    def _on_quality_scale(self, value) -> None:
        self.jpeg_quality_var.set(int(round(float(value))))

    def _reset_quality_from_preset(self) -> None:
        q = ALL_QUALITY_PRESETS[self.quality_var.get()]
        self.jpeg_quality_var.set(q)
        self.quality_scale.set(q)

    def _on_subsampling_change(self, _event=None) -> None:
        label = self.subsampling_combo.get()
        for key, text in SUBSAMPLING_LABELS.items():
            if text == label:
                self.subsampling_var.set(key)
                break

    def _sync_preset_fields(self) -> None:
        self.sizing_combo.set(SIZING_LABELS[0][1])
        self.sizing_var.set("resolution")
        self._on_quality_change()

    def _log_line(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _count_files(self) -> None:
        src = Path(self.input_var.get())
        if not src.exists():
            self.input_info.configure(text="Folder does not exist")
            return
        files = discover(src, Path(self.output_var.get()))
        self.input_info.configure(text=f"{len(files)} image file(s) found")

    def _open_output(self) -> None:
        path = Path(self.output_var.get())
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _doctor(self) -> None:
        from .env import doctor_report

        self._log_line("-" * 60)
        self._log_line("Environment check")
        try:
            doctor_report()
        except Exception as exc:
            self._log_line(f"doctor failed: {exc}")
        self._log_line("(full report printed to the console)")

    # -- run -----------------------------------------------------------------
    def _collect_options(self) -> RunOptions | None:
        src = Path(self.input_var.get())
        dst = Path(self.output_var.get())
        if not src.is_dir():
            messagebox.showerror("batchresize", f"Source folder not found:\n{src}")
            return None
        if src.resolve() == dst.resolve():
            messagebox.showerror("batchresize",
                                 "Source and destination must differ.")
            return None

        sizing = self.sizing_var.get()
        max_w = int(self.width_var.get())
        max_h = int(self.height_var.get())
        value = float(self.sizing_value_var.get())

        if sizing == "resolution" and self.preset_combo.get() in SIZE_PRESETS:
            max_w, max_h = SIZE_PRESETS[self.preset_combo.get()]

        quality = int(self.jpeg_quality_var.get())
        subsampling = {"auto": None, "444": 0, "422": 1, "420": 2}[
            self.subsampling_var.get()]
        if subsampling is None:
            from .presets import auto_subsampling

            subsampling = auto_subsampling(quality)

        bg = self._parse_bg(self.bg_color_var.get())
        if bg is None:
            return None

        return RunOptions(
            input_dir=src,
            output_dir=dst,
            max_w=max_w,
            max_h=max_h,
            mode=self.mode_var.get(),
            quality=quality,
            subsampling=subsampling,
            sizing=sizing,
            sizing_value=value,
            allow_upscale=self.allow_upscale_var.get(),
            workers=max(1, int(self.workers_var.get())),
            batch_size=max(1, int(self.batch_var.get())),
            queue_depth=max(1, int(self.queue_var.get())),
            skip_existing=self.skip_existing_var.get(),
            overwrite=self.overwrite_var.get(),
            keep_exif=self.exif_var.get(),
            keep_icc=self.icc_var.get(),
            bg_color=bg,
            vram_budget_gb=float(self.vram_var.get()),
            filter_name=self.filter_var.get(),
            fallback=self.fallback_var.get(),
            allow_truncated=self.truncated_var.get(),
            optimize_huffman=self.huffman_var.get(),
            fsync=self.fsync_var.get(),
        )

    def _parse_bg(self, value: str) -> tuple[int, int, int] | None:
        from .cli import parse_color

        try:
            return parse_color(value)
        except ValueError as exc:
            messagebox.showerror("batchresize", f"Invalid background colour: {exc}")
            return None

    def _start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("batchresize", "A run is already in progress.")
            return
        opts = self._collect_options()
        if opts is None:
            return

        self._log_line("=" * 60)
        self._log_line(f"Input  : {opts.input_dir}")
        self._log_line(f"Output : {opts.output_dir}")
        self._log_line(f"Quality: q={opts.quality}, chroma={opts.subsampling}, "
                       f"size={opts.max_w}x{opts.max_h} ({opts.sizing}), "
                       f"mode={opts.mode}")
        self._log_line(f"Threads: {opts.workers}, batch={opts.batch_size}, "
                       f"backend={self.backend_var.get()}")

        self._cancel.clear()
        self.progress.configure(value=0, maximum=100)
        self.status_var.set("Starting...")
        self.stats_var.set("")
        self.threads_var.set("")
        self._start_time = time.perf_counter()

        self._worker = threading.Thread(target=self._run_worker,
                                        args=(opts,), daemon=True)
        self._worker.start()

    def _run_worker(self, opts: RunOptions) -> None:
        try:
            engine, kind = _select_engine(self.backend_var.get())
            if kind == "fail" or engine is None:
                self._events.put(("error", "GPU backend requested but unavailable."))
                return
            self._events.put(("log", f"engine: {kind}"))
            pipe = Pipeline(opts, engine, kind, progress=False,
                            on_progress=self._on_progress)
            self._pipeline = pipe
            rep = pipe.run()
            rep["wall_sec"] = round(time.perf_counter() - self._start_time, 3)
            report_path = opts.output_dir / "report.json"
            _write_report(rep, report_path)
            self._events.put(("done", rep))
        except Exception as exc:
            self._events.put(("error", f"{type(exc).__name__}: {exc}"))

    def _on_progress(self, data: dict) -> None:
        self._events.put(("progress", data))

    def _cancel_run(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop.set()
        self._cancel.set()
        self.status_var.set("Cancelling...")

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "progress":
                    self._apply_progress(payload)
                elif kind == "log":
                    self._log_line(str(payload))
                elif kind == "done":
                    self._apply_done(payload)
                elif kind == "error":
                    self._log_line(f"ERROR: {payload}")
                    self.status_var.set("Failed")
                    messagebox.showerror("batchresize", str(payload))
        except queue.Empty:
            pass
        self.after(120, self._drain_events)

    def _apply_progress(self, data: dict) -> None:
        total = max(1, data["total"])
        done = data["done"]
        self.progress.configure(maximum=total, value=done)
        self.status_var.set(
            f"Processing {done}/{total} - {data['rate']:.1f} file/s")
        self.stats_var.set(
            f"Elapsed {data['elapsed']:.1f}s | processed {data['processed']} | "
            f"failed {data['failed']} | skipped {data['skipped']}")
        self.threads_var.set(
            f"Threads: {self.workers_var.get()} readers, "
            f"writers {max(2, int(self.workers_var.get()) // 2)}")

    def _apply_done(self, rep: dict) -> None:
        self.progress.configure(value=rep["processed"] + rep["failed"]
                                + rep["skipped"],
                                maximum=max(1, rep["discovered"]))
        th = rep["threads"]
        self.status_var.set(
            f"Done in {rep['elapsed_sec']:.2f}s "
            f"({rep['files_per_sec']:.1f} file/s)")
        self.stats_var.set(
            f"Processed {rep['processed']}, failed {rep['failed']}, "
            f"skipped {rep['skipped']}, "
            f"{rep['bytes_in'] / 2 ** 20:.1f} -> "
            f"{rep['bytes_out'] / 2 ** 20:.1f} MiB")
        self.threads_var.set(
            f"Threads: readers={th['readers']}, processors={th['processors']}, "
            f"writers={th['writers']} (total {th['total']})")
        self._log_line(
            f"Finished: {rep['processed']} processed, {rep['failed']} failed, "
            f"{rep['skipped']} skipped in {rep['elapsed_sec']:.2f}s "
            f"({rep['files_per_sec']:.1f} file/s)")
        if rep["errors"]:
            self._log_line(f"{len(rep['errors'])} error(s):")
            for path, err in rep["errors"][:20]:
                self._log_line(f"  {path}: {err}")

    def _on_close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop.set()
        self.destroy()


def main() -> int:
    app = BatchResizeUI()
    app.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
