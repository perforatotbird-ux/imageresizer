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

MAX_PREVIEW_PIXELS = 12_000_000
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
                quality=int(params["quality"]),
                subsampling=int(params["subsampling"]),
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
            self._last_meta = {
                "backend": report.get("backend", kind),
                "requested_backend": backend,
                "route": ", ".join(sorted(report.get("backend_counts", {}))) or kind,
                "codec_backend": report.get("codec_backend", ""),
                "jpeg_bytes": size,
                "jpeg_sha256": digest,
                "out_w": int(out_w),
                "out_h": int(out_h),
            }
        finally:
            if run_dir is not None:
                shutil.rmtree(run_dir, ignore_errors=True)

    # -- display -----------------------------------------------------------
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
        box_w = int(req.get("box_w", 300))
        box_h = int(req.get("box_h", 200))
        approximate = False
        with Image.open(source) as original_file:
            original = ImageOps.exif_transpose(original_file).convert("RGB")
            source_w, source_h = original.size
            if view == "actual":
                ow, oh, oa = _actual_size(source_w, source_h)
                rw, rh, ra = _actual_size(result_w, result_h)
                approximate = oa or ra
            else:
                ow, oh = _fit_size(source_w, source_h, box_w, box_h)
                rw, rh = _fit_size(result_w, result_h, box_w, box_h)
            original_path = self._write_display(original, "original.png", ow, oh)

        assert self._last_jpeg is not None
        with Image.open(self._last_jpeg) as result_file:
            result = result_file.convert("RGB")
            result_path = self._write_display(result, "result.png", rw, rh)

        return {
            "original": {"path": str(original_path), "w": ow, "h": oh},
            "result": {"path": str(result_path), "w": rw, "h": rh},
            "source_w": source_w,
            "source_h": source_h,
            "result_w": result_w,
            "result_h": result_h,
            "view": view,
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
