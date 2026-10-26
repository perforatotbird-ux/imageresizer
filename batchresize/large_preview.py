"""Large before/after preview window.

The window is intentionally a thin Tk view over the same preview worker used by
the sidebar. It requests full-frame images at an explicit zoom, so panning and
zooming never load a second resize implementation into the GUI process.
"""

from __future__ import annotations

import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from .preview import PreviewClient, preview_summary

ZOOM_STEPS = (0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75,
              1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
MAX_PREVIEW_PIXELS = 24_000_000


class LargePreviewWindow:
    """Resizable two-pane preview with fit/100%/zoom and scrollbars."""

    def __init__(self, app, project_dir: Path) -> None:
        self.app = app
        self.project_dir = Path(project_dir)
        self.win = tk.Toplevel(app)
        self.win.title("Превью до/после — крупный просмотр")
        self.win.geometry("1280x860")
        self.win.minsize(900, 600)
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self._temp = tempfile.TemporaryDirectory(prefix="batchresize-large-preview-")
        self._client = PreviewClient(
            Path(app.worker), self.project_dir, Path(self._temp.name),
            app._events, name="preview-large")
        self._request_id = 0
        self._after = None
        self._paused = False
        self._closed = False
        self._zoom = None
        self._images: dict[str, tk.PhotoImage] = {}
        self._data = None
        self._canvases: dict[str, tk.Canvas] = {}
        self.zoom_var = tk.StringVar(master=self.win, value="Вписать")
        self.status_var = tk.StringVar(master=self.win, value="Загрузка…")
        self.info_var = tk.StringVar(master=self.win, value="")

        self._build()
        self.schedule(immediate=True)

    # -- layout -------------------------------------------------------------
    def _build(self) -> None:
        toolbar = ttk.Frame(self.win, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="−", width=3,
                   command=lambda: self.zoom_by(-1)).pack(side="left")
        ttk.Label(toolbar, textvariable=self.zoom_var, width=10,
                  anchor="center").pack(side="left", padx=4)
        ttk.Button(toolbar, text="+", width=3,
                   command=lambda: self.zoom_by(1)).pack(side="left")
        ttk.Button(toolbar, text="Вписать",
                   command=self.fit).pack(side="left", padx=(12, 0))
        ttk.Button(toolbar, text="100%",
                   command=self.actual).pack(side="left", padx=(6, 0))
        ttk.Label(toolbar, text="Колесо мыши — зум, перетаскивание — панорама",
                  style="Muted.TLabel").pack(side="left", padx=(18, 0))
        ttk.Label(toolbar, textvariable=self.status_var,
                  style="Muted.TLabel").pack(side="right")
        ttk.Label(self.win, textvariable=self.info_var, style="Muted.TLabel",
                  justify="left", padding=(12, 0, 12, 6)).pack(fill="x")

        body = ttk.Frame(self.win, padding=(8, 0, 8, 8))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        for column, (key, title) in enumerate(
                (("original", "Оригинал"), ("result", "Результат"))):
            pane = ttk.Frame(body)
            pane.grid(row=0, column=column, sticky="nsew",
                      padx=(0, 6) if column == 0 else (6, 0))
            pane.columnconfigure(0, weight=1)
            pane.rowconfigure(1, weight=1)
            ttk.Label(pane, text=title, style="Eyebrow.TLabel").grid(
                row=0, column=0, sticky="w", pady=(0, 4))

            canvas = tk.Canvas(pane, background="#101827",
                               highlightthickness=1,
                               highlightbackground="#26324A", bd=0)
            vbar = ttk.Scrollbar(pane, orient="vertical",
                                 command=canvas.yview)
            hbar = ttk.Scrollbar(pane, orient="horizontal",
                                 command=canvas.xview)
            canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
            canvas.grid(row=1, column=0, sticky="nsew")
            vbar.grid(row=1, column=1, sticky="ns")
            hbar.grid(row=2, column=0, sticky="ew")
            canvas.bind("<Configure>", self._on_configure)
            canvas.bind("<ButtonPress-1>",
                        lambda event, c=canvas: c.scan_mark(event.x, event.y))
            canvas.bind("<B1-Motion>",
                        lambda event, c=canvas: c.scan_dragto(event.x, event.y, gain=1))
            canvas.bind("<MouseWheel>", self._on_wheel)
            canvas.bind("<Button-4>", lambda _event: self.zoom_by(1))
            canvas.bind("<Button-5>", lambda _event: self.zoom_by(-1))
            self._canvases[key] = canvas

    # -- zoom state ---------------------------------------------------------
    def _current_zoom(self) -> float:
        if self._zoom is not None:
            return self._zoom
        data = self._data or {}
        width = int(((data.get("original") or {}).get("w")) or 0)
        source_w = int(data.get("source_w") or 0)
        if width and source_w:
            return width / float(source_w)
        return 0.25

    def _zoom_label(self) -> None:
        if self._zoom is None:
            self.zoom_var.set("Вписать")
        else:
            self.zoom_var.set(f"{round(self._zoom * 100)}%")

    def zoom_by(self, direction: int) -> None:
        current = self._current_zoom()
        if direction > 0:
            target = next((z for z in ZOOM_STEPS if z > current + 1e-6), None)
        else:
            target = next((z for z in reversed(ZOOM_STEPS)
                           if z < current - 1e-6), None)
        if target is None:
            return
        self._zoom = target
        self._zoom_label()
        self._request()

    def fit(self) -> None:
        self._zoom = None
        self._zoom_label()
        self._request()

    def actual(self) -> None:
        self._zoom = 1.0
        self._zoom_label()
        self._request()

    def _on_wheel(self, event) -> str:
        if getattr(event, "delta", 0):
            self.zoom_by(1 if event.delta > 0 else -1)
        return "break"

    # -- data flow ----------------------------------------------------------
    def schedule(self, immediate: bool = False) -> None:
        if self._closed or self._paused:
            return
        self._cancel_after()
        source = self.app.preview_source_var.get().strip()
        if not source or not Path(source).expanduser().is_file():
            self.status_var.set("Файл превью не выбран")
            self._clear_images()
            return
        self._after = self.win.after(10 if immediate else 420, self._request)

    def _cancel_after(self) -> None:
        if self._after is not None:
            try:
                self.win.after_cancel(self._after)
            except tk.TclError:
                pass
            self._after = None

    def _box(self) -> tuple[int, int]:
        canvas = self._canvases["original"]
        width = max(80, canvas.winfo_width())
        height = max(80, canvas.winfo_height())
        if width < 120 or height < 120:
            return 560, 680
        return width, height

    def _request(self) -> None:
        self._after = None
        if self._closed or self._paused:
            return
        if self.app._is_running() or getattr(self.app, "_preview_suspended", False):
            return
        source = Path(self.app.preview_source_var.get().strip()).expanduser()
        if not source.is_file():
            self.status_var.set("Файл превью не найден")
            self._clear_images()
            return
        try:
            params = self.app._preview_params()
        except (ValueError, KeyError) as exc:
            self.status_var.set(f"Проверьте параметры: {exc}")
            return
        width, height = self._box()
        payload = {
            "source": str(source),
            "view": "fit" if self._zoom is None else "zoom",
            "box_w": width,
            "box_h": height,
            "params": params,
        }
        if self._zoom is not None:
            payload["zoom"] = self._zoom
            payload["max_pixels"] = MAX_PREVIEW_PIXELS
        self._request_id = self._client.request(payload)
        self.status_var.set("Обновление…")

    def handle_event(self, payload: dict) -> None:
        kind = payload.get("kind")
        if kind == "log":
            self.app._log_line(f"large preview: {payload.get('text', '')}")
            return
        if kind == "dead":
            self.status_var.set(str(payload.get("error", "Превью недоступно")))
            return
        if kind == "invalidated":
            return
        if payload.get("id") != self._request_id:
            return
        if not payload.get("ok"):
            error = str(payload.get("error", "неизвестная ошибка"))
            self.status_var.set(f"Ошибка превью: {error}")
            self.app._log_line(f"large preview: {error}")
            return
        images: dict[str, tk.PhotoImage] = {}
        for key in ("original", "result"):
            path = (payload.get(key) or {}).get("path")
            if not path:
                self.status_var.set("Превью не содержит изображения")
                return
            try:
                images[key] = tk.PhotoImage(master=self.win, file=path)
            except tk.TclError:
                self.status_var.set("Не удалось загрузить изображение превью")
                return
        self._images = images
        self._data = payload
        if payload.get("view") == "fit":
            self._zoom = None
        self._zoom_label()
        short, detail = preview_summary(payload)
        self.info_var.set(short + (f"\n{detail}" if detail else ""))
        self.status_var.set("Готово" + (" · приближённо"
                                       if payload.get("approximate") else ""))
        self._draw_all()

    # -- painting -----------------------------------------------------------
    def _clear_images(self) -> None:
        self._images = {}
        self._data = None
        self.info_var.set("")
        self._draw_all()

    def _draw_all(self) -> None:
        for key in ("original", "result"):
            self._draw_pane(key)

    def _draw_pane(self, key: str) -> None:
        canvas = self._canvases[key]
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        image = self._images.get(key)
        if image is None:
            canvas.configure(scrollregion=(0, 0, width, height))
            canvas.create_text(width / 2, height / 2,
                               text="Нет изображения", fill="#93A4C3",
                               font=("Segoe UI", 10))
            return
        iw, ih = image.width(), image.height()
        x = max(0, (width - iw) // 2)
        y = max(0, (height - ih) // 2)
        canvas.create_image(x, y, image=image, anchor="nw")
        canvas.configure(scrollregion=(0, 0, max(width, iw), max(height, ih)))
        if iw > width:
            canvas.xview_moveto(max(0.0, (iw - width) / 2.0 / iw))
        else:
            canvas.xview_moveto(0.0)
        if ih > height:
            canvas.yview_moveto(max(0.0, (ih - height) / 2.0 / ih))
        else:
            canvas.yview_moveto(0.0)

    def _on_configure(self, _event=None) -> None:
        if self._data is not None:
            self._draw_all()
        if self._zoom is None and self._after is None:
            self.schedule()

    # -- lifecycle ----------------------------------------------------------
    def show(self) -> None:
        if self._closed:
            return
        self.win.deiconify()
        self.win.lift()
        self.win.focus_force()
        self.schedule()

    def pause(self) -> None:
        if self._closed:
            return
        self._paused = True
        self._cancel_after()
        self._client.cancel()
        self.status_var.set("Превью приостановлено во время обработки")

    def resume(self) -> None:
        if self._closed:
            return
        self._paused = False
        self.schedule()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel_after()
        self._client.close()
        try:
            self._temp.cleanup()
        except (OSError, PermissionError):
            pass
        if getattr(self.app, "_large_preview", None) is self:
            self.app._large_preview = None
        try:
            self.win.destroy()
        except tk.TclError:
            pass
