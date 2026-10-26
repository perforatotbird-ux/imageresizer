"""Long-lived preview worker.

The GUI never imports Pillow, NumPy or CUDA. It starts this module with the
project virtual environment and talks to it over newline-delimited JSON on
stdin/stdout.

Every preview goes through the real :class:`batchresize.pipeline.Pipeline` on a
single source file in a private temporary output tree. The preview therefore
uses exactly the same geometry, metadata handling, encoder and backend route as
a batch run; it is not a second resize implementation.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageOps

from .source_quality import QUALITY_SOURCE, SUBSAMPLING_SOURCE

MAX_PREVIEW_PIXELS = 12_000_000
MAX_ZOOM_PREVIEW_PIXELS = 24_000_000
MAX_CACHE_BYTES = 64 * 2**20
PROTOCOL = sys.stdout


def _send(payload: dict) -> None:
    PROTOCOL.write(json.dumps(payload, ensure_ascii=False) + "\n")
    PROTOCOL.flush()


def _fit_size(width: int, height: int, box_w: int, box_h: int) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError("image has no pixels")
    box_w = max(1, int(box_w))
    box_h = max(1, int(box_h))
    scale = min(1.0, box_w / width, box_h / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _actual_size(width: int, height: int) -> tuple[int, int, bool]:
    pixels = width * height
    if pixels <= MAX_PREVIEW_PIXELS:
        return width, height, False
    scale = (MAX_PREVIEW_PIXELS / pixels) ** 0.5
    return max(1, round(width * scale)), max(1, round(height * scale)), True


def _zoom_size(width: int, height: int, zoom: float,
               max_pixels: int = MAX_ZOOM_PREVIEW_PIXELS) -> tuple[int, int, bool]:
    """Scale to an explicit zoom, then cap the preview at max_pixels."""
    factor = max(0.01, float(zoom))
    w = max(1, round(width * factor))
    h = max(1, round(height * factor))
    approximate = False
    if w * h > max_pixels:
        factor = (max_pixels / float(w * h)) ** 0.5
        w = max(1, round(w * factor))
        h = max(1, round(h * factor))
        approximate = True
    return w, h, approximate


class PreviewService:
    """Processes one request at a time and caches only the latest JPEG result."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.root / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.display_dir = self.root / "display"
        self.display_dir.mkdir(exist_ok=True)
        self._engine = None
        self._engine_key = None
        self._engine_kind = None
        self._last_key = None
        self._last_jpeg = None
        self._last_meta = None
        self._last_displays: list[Path] = []
        self._retained_dir: Path | None = None
        self._display_source: Image.Image | None = None
        self._display_source_key = None
        self._display_result: Image.Image | None = None
        self._display_result_key = None

    # -- engine ------------------------------------------------------------
    def _engine_for(self, backend: str, filter_name: str, optimize: bool):
        key = (backend, filter_name, bool(optimize))
        if self._engine is not None and self._engine_key == key:
            return self._engine, self._engine_kind
        from .cli import _select_engine

        proxy = SimpleNamespace(filter_name=filter_name,
                                optimize_huffman=bool(optimize))
        with contextlib.redirect_stdout(sys.stderr):
            engine, kind = _select_engine(backend, proxy)
        if kind == "fail" or engine is None:
            raise RuntimeError(f"backend {backend!r} is unavailable")
        self._engine, self._engine_key, self._engine_kind = engine, key, kind
        return engine, kind

    # -- pipeline ----------------------------------------------------------
    def _render_jpeg(self, source: Path, params: dict) -> None:
        from .pipeline import Pipeline, RunOptions

        backend = str(params.pop("backend", "auto"))
        if self._retained_dir is not None:
            shutil.rmtree(self._retained_dir, ignore_errors=True)
            self._retained_dir = None
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(self.root)))
        out_dir = run_dir / "out"
        out_dir.mkdir()
        try:
            opts = RunOptions(
                input_dir=source.parent,
                output_dir=out_dir,
                max_w=int(params["max_w"]),
                max_h=int(params["max_h"]),
                mode=str(params.get("mode", "contain")),
                quality=(params["quality"] if params["quality"] == QUALITY_SOURCE
                         else int(params["quality"])),
                subsampling=(params["subsampling"]
                             if params["subsampling"] == SUBSAMPLING_SOURCE
                             else int(params["subsampling"])),
                sizing=str(params.get("sizing", "resolution")),
                sizing_value=float(params.get("sizing_value", 0.0)),
                allow_upscale=bool(params.get("allow_upscale", False)),
                workers=1,
                batch_size=1,
                queue_depth=2,
                skip_existing=False,
                overwrite=True,
                limit=None,
                keep_exif=bool(params.get("keep_exif", True)),
                keep_icc=bool(params.get("keep_icc", True)),
                bg_color=tuple(int(c) for c in params.get("bg_color", (255, 255, 255))),
                vram_budget_gb=float(params.get("vram_gb", 2.0)),
                filter_name=str(params.get("filter", "lanczos3")),
                fallback=bool(params.get("fallback", True)),
                allow_truncated=bool(params.get("allow_truncated", False)),
                optimize_huffman=bool(params.get("optimize_huffman", False)),
                fsync=False,
                profile_cuda=False,
                sharpen=params.get("sharpen"),
                color_policy=str(params.get("color_policy", "preserve_rgb")),
                linear_light=bool(params.get("linear_light", False)),
                strip_personal_data=bool(params.get("strip_personal_data", False)),
                keep_datetime=bool(params.get("keep_datetime", False)),
                crop_anchor=str(params.get("crop_anchor", "center")),
                crop_focus=params.get("crop_focus"),
                max_file_size=params.get("max_file_size"),
                min_quality=int(params.get("min_quality", 30)),
                source_fallback_quality=int(params.get("source_fallback_quality", 90)),
                allow_downscale_on_size_limit=bool(params.get("allow_downscale_on_size_limit", False)),
                fast_jpeg_decode=bool(params.get("fast_jpeg_decode", False)),
            )
            engine, kind = self._engine_for(
                backend, opts.filter_name, opts.optimize_huffman)
            pipe = Pipeline(opts, engine, kind, progress=False, files=[source])
            with contextlib.redirect_stdout(sys.stderr):
                report = pipe.run()
            if int(report.get("processed", 0)) != 1:
                errors = report.get("errors") or report.get("fatal_errors") or []
                if errors:
                    message = errors[0]
                    if isinstance(message, (list, tuple)):
                        message = ": ".join(str(part) for part in message)
                    raise RuntimeError(str(message))
                raise RuntimeError("the batch pipeline did not produce an output")
            target = out_dir / source.relative_to(source.parent).with_suffix(".jpg")
            if not target.is_file():
                candidates = list(out_dir.rglob("*.jpg"))
                if len(candidates) != 1:
                    raise RuntimeError("preview output was not published")
                target = candidates[0]

            size = target.stat().st_size
            with Image.open(target) as rendered:
                out_w, out_h = rendered.size
            with target.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if size <= MAX_CACHE_BYTES:
                cached = self.cache_dir / "result.jpg"
                temporary = self.cache_dir / "result.tmp"
                shutil.copyfile(target, temporary)
                os.replace(temporary, cached)
                self._last_jpeg = cached
            else:
                self._last_jpeg = target
                self._retained_dir = run_dir
                run_dir = None  # keep the oversized result for this response
            self._display_result = None
            self._display_result_key = None
            self._last_meta = {
                "backend": report.get("backend", kind),
                "requested_backend": backend,
                "route": ", ".join(sorted(report.get("backend_counts", {}))) or kind,
                "codec_backend": report.get("codec_backend", ""),
                "jpeg_bytes": size,
                "source_bytes": source.stat().st_size,
                "jpeg_sha256": digest,
                "out_w": int(out_w),
                "out_h": int(out_h),
                "actual_quality": report.get("actual_quality"),
                "target_size_achieved": report.get("target_size_achieved"),
            }
        finally:
            if run_dir is not None:
                shutil.rmtree(run_dir, ignore_errors=True)

    # -- display -----------------------------------------------------------
    def _source_image(self, source: Path) -> Image.Image:
        stat = source.stat()
        key = (str(source.resolve()), stat.st_mtime_ns, stat.st_size)
        if self._display_source is None or self._display_source_key != key:
            with Image.open(source) as original_file:
                self._display_source = ImageOps.exif_transpose(original_file).convert("RGB")
            self._display_source_key = key
        return self._display_source

    def _result_image(self) -> Image.Image:
        assert self._last_jpeg is not None
        path = Path(self._last_jpeg)
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if self._display_result is None or self._display_result_key != key:
            with Image.open(path) as result_file:
                self._display_result = result_file.convert("RGB")
            self._display_result_key = key
        return self._display_result

    @staticmethod
    def _viewport_frame(image: Image.Image, box_w: int, box_h: int,
                        zoom: float, center_x: float,
                        center_y: float) -> Image.Image:
        """Crop the visible viewport and scale it to at most box_w x box_h."""
        iw, ih = image.size
        scale = max(0.01, float(zoom))
        disp_w = iw * scale
        disp_h = ih * scale
        if disp_w <= box_w and disp_h <= box_h:
            return image.resize((max(1, round(disp_w)), max(1, round(disp_h))),
                                Image.Resampling.LANCZOS)
        crop_w = min(float(iw), box_w / scale)
        crop_h = min(float(ih), box_h / scale)
        cx = min(1.0, max(0.0, float(center_x))) * iw
        cy = min(1.0, max(0.0, float(center_y))) * ih
        left = min(max(0.0, cx - crop_w / 2.0), iw - crop_w)
        top = min(max(0.0, cy - crop_h / 2.0), ih - crop_h)
        out_w = max(1, round(crop_w * scale))
        out_h = max(1, round(crop_h * scale))
        return image.resize((out_w, out_h), Image.Resampling.LANCZOS,
                            box=(left, top, left + crop_w, top + crop_h))

    def _clear_displays(self) -> None:
        for path in self._last_displays:
            try:
                path.unlink()
            except OSError:
                pass
        self._last_displays = []

    def _write_display(self, image: Image.Image, name: str, width: int,
                       height: int) -> Path:
        if (width, height) != image.size:
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        path = self.display_dir / name
        image.save(path, "PNG", compress_level=1)
        self._last_displays.append(path)
        return path

    def _displays(self, source: Path, req: dict, result_w: int,
                  result_h: int) -> dict:
        view = str(req.get("view", "fit"))
        box_w = max(1, int(req.get("box_w", 300)))
        box_h = max(1, int(req.get("box_h", 200)))
        approximate = False
        original = self._source_image(source)
        source_w, source_h = original.size
        result = self._result_image()

        zoom = None
        if view == "viewport":
            zoom = float(req.get("zoom", 1.0))
            center_x = float(req.get("center_x", 0.5))
            center_y = float(req.get("center_y", 0.5))
            original_frame = self._viewport_frame(
                original, box_w, box_h, zoom, center_x, center_y)
            result_frame = self._viewport_frame(
                result, box_w, box_h, zoom, center_x, center_y)
        elif view == "zoom":
            zoom = float(req.get("zoom", 1.0))
            max_pixels = int(req.get("max_pixels", MAX_ZOOM_PREVIEW_PIXELS))
            max_pixels = max(1, min(max_pixels, 48_000_000))
            ow, oh, oa = _zoom_size(source_w, source_h, zoom, max_pixels)
            rw, rh, ra = _zoom_size(result_w, result_h, zoom, max_pixels)
            approximate = oa or ra
            original_frame = original.resize((ow, oh), Image.Resampling.LANCZOS)
            result_frame = result.resize((rw, rh), Image.Resampling.LANCZOS)
        elif view == "actual":
            ow, oh, oa = _actual_size(source_w, source_h)
            rw, rh, ra = _actual_size(result_w, result_h)
            approximate = oa or ra
            original_frame = original.resize((ow, oh), Image.Resampling.LANCZOS)
            result_frame = result.resize((rw, rh), Image.Resampling.LANCZOS)
        else:
            ow, oh = _fit_size(source_w, source_h, box_w, box_h)
            rw, rh = _fit_size(result_w, result_h, box_w, box_h)
            original_frame = original.resize((ow, oh), Image.Resampling.LANCZOS)
            result_frame = result.resize((rw, rh), Image.Resampling.LANCZOS)

        ow, oh = original_frame.size
        rw, rh = result_frame.size
        original_path = self._write_display(original_frame, "original.png", ow, oh)
        result_path = self._write_display(result_frame, "result.png", rw, rh)

        return {
            "original": {"path": str(original_path), "w": ow, "h": oh},
            "result": {"path": str(result_path), "w": rw, "h": rh},
            "source_w": source_w,
            "source_h": source_h,
            "result_w": result_w,
            "result_h": result_h,
            "view": view,
            "zoom": zoom,
            "approximate": approximate,
        }

    # -- request -----------------------------------------------------------
    def handle(self, request: dict) -> dict:
        request_id = request.get("id")
        started = time.perf_counter()
        source = Path(str(request.get("source", ""))).expanduser()
        if not source.is_file():
            raise ValueError(f"preview source not found: {source}")
        source = source.resolve()
        params = dict(request.get("params") or {})
        required = {"max_w", "max_h", "quality", "subsampling"}
        missing = required - set(params)
        if missing:
            raise ValueError("missing preview parameters: " + ", ".join(sorted(missing)))

        stat = source.stat()
        key_data = {
            "source": str(source),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "params": params,
        }
        key = hashlib.sha256(
            json.dumps(key_data, sort_keys=True, ensure_ascii=False).encode("utf8")
        ).hexdigest()

        if key != self._last_key or self._last_jpeg is None or not Path(self._last_jpeg).is_file():
            self._render_jpeg(source, dict(params))
            if self._last_meta is None:
                raise RuntimeError("preview metadata was not recorded")
            self._last_key = key
            reused = False
        else:
            reused = True

        meta = self._last_meta
        assert meta is not None
        self._clear_displays()
        displays = self._displays(source, request, meta["out_w"], meta["out_h"])
        return {
            "id": request_id,
            "ok": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "reused": reused,
            **meta,
            **displays,
        }

    def cleanup(self) -> None:
        self._clear_displays()
        if self._retained_dir is not None:
            shutil.rmtree(self._retained_dir, ignore_errors=True)
            self._retained_dir = None
        shutil.rmtree(self.root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m batchresize.preview_worker <temp-root>", file=sys.stderr)
        return 2
    service = PreviewService(Path(argv[0]))
    atexit.register(service.cleanup)
    # Library diagnostics must never corrupt the JSON protocol.
    sys.stdout = sys.stderr
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as exc:
            _send({"id": None, "ok": False, "error": f"invalid request: {exc}"})
            continue
        if request.get("cmd") == "quit":
            break
        try:
            _send(service.handle(request))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _send({"id": request.get("id"), "ok": False,
                   "error": f"{type(exc).__name__}: {exc}"})
    service.cleanup()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
