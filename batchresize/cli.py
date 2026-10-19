"""Command line interface for batchresize."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from .cpu_backend import CpuEngine
from .env import doctor_report
from .gpu_backend import GpuEngine, GpuUnavailable
from .pipeline import Pipeline, RunOptions
from .presets import (ALL_QUALITY_PRESETS, SIZE_PRESETS, OutputSpec,
                      resolve_quality, resolve_size)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "input"
DEFAULT_OUTPUT = PROJECT_DIR / "output"


def parse_color(value: str) -> tuple[int, int, int]:
    """Accept ``#RRGGBB``, ``#RGB`` or ``R,G,B``."""
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 3:
            raise ValueError(f"invalid colour {value!r}")
        rgb = tuple(int(p) for p in parts)
    else:
        if len(text) == 3:
            text = "".join(c * 2 for c in text)
        if len(text) != 6:
            raise ValueError(f"invalid colour {value!r}")
        rgb = tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
    if any(c < 0 or c > 255 for c in rgb):
        raise ValueError(f"colour component out of range in {value!r}")
    return rgb  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="batchresize",
        description="Batch image resizer: GPU (nvJPEG + CUDA) or CPU (OpenCV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", default=str(DEFAULT_INPUT),
                   help="source folder (fixed to ./input, override only if needed)")
    p.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT),
                   help="destination folder (fixed to ./output)")

    p.add_argument("--size", default="web_1024",
                   choices=[*SIZE_PRESETS, "custom"],
                   help="target size preset")
    p.add_argument("--width", type=int, default=None,
                   help="max width for --size custom")
    p.add_argument("--height", type=int, default=None,
                   help="max height for --size custom")
    p.add_argument("--sizing", default="resolution",
                   choices=["resolution", "percent", "megapixels",
                            "longest", "shortest"],
                   help="downscaling method (resolution/percent/megapixels/"
                        "longest/shortest)")
    p.add_argument("--sizing-value", type=float, default=0.0,
                   help="value for --sizing: percent, megapixels or pixels")
    p.add_argument("--quality", "--compress", dest="quality", default="balanced",
                   choices=[*ALL_QUALITY_PRESETS, "custom"],
                   help="recompression preset: light=90, medium=80, "
                        "strong=65, extreme=50 (also compact/balanced/"
                        "high/very_high)")
    p.add_argument("--jpeg-quality", type=int, default=None,
                   help="explicit JPEG quality 1..100 (overrides --quality)")
    p.add_argument("--subsampling", default="auto",
                   choices=["auto", "444", "422", "420"],
                   help="chroma subsampling")
    p.add_argument("--mode", default="contain", choices=["contain", "cover"],
                   help="contain = fit inside, cover = fill and crop")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "gpu", "hybrid", "cpu"],
                   help="processing backend")
    p.add_argument("--workers", "--threads", dest="workers", type=int, default=8,
                   help="number of threads: readers/processors "
                        "(writers = max(2, workers//2))")
    p.add_argument("--batch-size", type=int, default=8,
                   help="images per GPU batch")
    p.add_argument("--queue-depth", type=int, default=32,
                   help="bounded queue size between stages")
    p.add_argument("--vram-budget", type=float, default=2.0,
                   help="max GiB of decoded pixels in flight")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N files")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip files that already exist in the output folder")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="overwrite existing results (default: on unless "
                        "--skip-existing is given)")
    p.add_argument("--allow-upscale", action="store_true",
                   help="allow enlarging small photos")
    p.add_argument("--filter", default="lanczos3",
                   choices=["lanczos3", "bilinear", "box"],
                   help="resampling filter")
    p.add_argument("--bg-color", default="#FFFFFF",
                   help="background for transparent images")
    p.add_argument("--no-exif", action="store_true",
                   help="do not copy EXIF metadata")
    p.add_argument("--no-icc", action="store_true",
                   help="do not copy the ICC profile")
    p.add_argument("--allow-truncated", action="store_true",
                   help="try to decode truncated JPEGs instead of skipping")
    p.add_argument("--no-fallback", action="store_true",
                   help="fail a file instead of decoding it on the CPU")
    p.add_argument("--fsync", action="store_true",
                   help="fsync before the atomic rename (slower)")
    p.add_argument("--optimize-huffman", action="store_true",
                   help="optimized Huffman tables in the GPU encoder")
    p.add_argument("--report", default=None,
                   help="path to the JSON report (default: <output>/report.json)")
    p.add_argument("--benchmark", action="store_true",
                   help="compare GPU and CPU backends on the same files")
    p.add_argument("--benchmark-repeats", type=int, default=3,
                   help="runs per backend in benchmark mode")
    p.add_argument("--quiet", action="store_true",
                   help="no live progress line")
    p.add_argument("--gui", action="store_true",
                   help="open the desktop UI")
    p.add_argument("--doctor", action="store_true",
                   help="print the environment report and exit")
    p.add_argument("--version", action="version",
                   version="batchresize 0.1.0")
    return p


def _resolve_paths(args) -> tuple[Path, Path]:
    src = Path(args.input).expanduser()
    dst = Path(args.output).expanduser()
    return src, dst


def _make_options(args, src: Path, dst: Path) -> RunOptions:
    if args.size == "custom" and (args.width is None or args.height is None):
        raise SystemExit("error: --size custom requires --width and --height")
    try:
        max_w, max_h = resolve_size(
            None if args.size == "custom" else args.size,
            args.width, args.height)
        quality = resolve_quality(
            None if args.quality == "custom" else args.quality,
            args.jpeg_quality)
        bg = parse_color(args.bg_color)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    subsampling = {"auto": None, "444": 0, "422": 1, "420": 2}[args.subsampling]
    if subsampling is None:
        from .presets import auto_subsampling

        subsampling = auto_subsampling(quality)

    if args.workers < 1:
        raise SystemExit("error: --workers must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("error: --batch-size must be >= 1")
    if args.queue_depth < 1:
        raise SystemExit("error: --queue-depth must be >= 1")

    overwrite = args.overwrite or not args.skip_existing
    return RunOptions(
        input_dir=src,
        output_dir=dst,
        max_w=max_w,
        max_h=max_h,
        mode=args.mode,
        quality=quality,
        subsampling=subsampling,
        sizing=args.sizing,
        sizing_value=args.sizing_value,
        allow_upscale=args.allow_upscale,
        workers=args.workers,
        batch_size=args.batch_size,
        queue_depth=args.queue_depth,
        skip_existing=args.skip_existing,
        overwrite=overwrite,
        limit=args.limit,
        keep_exif=not args.no_exif,
        keep_icc=not args.no_icc,
        bg_color=bg,
        vram_budget_gb=args.vram_budget,
        filter_name=args.filter,
        fallback=not args.no_fallback,
        allow_truncated=args.allow_truncated,
        optimize_huffman=args.optimize_huffman,
        fsync=args.fsync,
    )


def _select_engine(backend: str):
    """Return (engine, engine_kind); ``(None, "fail")`` for a hard GPU request."""
    if backend == "cpu":
        return CpuEngine(), "cpu"

    try:
        engine = GpuEngine(optimize_huffman=False)
        ok, msg = engine.self_test()
        if not ok:
            raise GpuUnavailable(f"GPU self-test failed: {msg}")
    except Exception as exc:
        if backend == "auto":
            print(f"gpu unavailable ({exc}); falling back to CPU")
            return CpuEngine(), "cpu"
        print(f"FATAL: GPU backend requested but unavailable: {exc}")
        return None, "fail"

    return engine, ("hybrid" if backend == "hybrid" else "gpu")


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)} min {rest:.1f} s"


def _print_summary(rep: dict) -> None:
    th = rep["threads"]
    print()
    print("-" * 68)
    print(f"Input        : {rep['input']}")
    print(f"Output       : {rep['output']}")
    print(f"Backend      : {rep['backend']}")
    print(f"Discovered   : {rep['discovered']} file(s)")
    print(f"Processed    : {rep['processed']}")
    print(f"Failed       : {rep['failed']}")
    print(f"Skipped      : {rep['skipped']}")
    if rep["fallback_files"]:
        print(f"CPU fallback : {rep['fallback_files']}")
    print(f"Bytes        : {rep['bytes_in'] / 2 ** 20:.1f} MiB -> "
          f"{rep['bytes_out'] / 2 ** 20:.1f} MiB")
    print(f"Time         : {_format_seconds(rep['elapsed_sec'])} "
          f"({rep['files_per_sec']:.1f} file/s, "
          f"{rep['mb_per_sec_in']:.1f} MiB/s)")
    print(f"Threads      : readers={th['readers']}, "
          f"processors={th['processors']}, writers={th['writers']} "
          f"(total {th['total']})")
    t = rep["timings_avg_ms"]
    print(f"Avg per file : read {t['read']:.1f} ms, decode {t['decode']:.1f} ms, "
          f"resize {t['resize']:.1f} ms, encode {t['encode']:.1f} ms, "
          f"write {t['write']:.1f} ms")
    if rep["errors"]:
        print(f"Errors       : {len(rep['errors'])} (first 5)")
        for path, err in rep["errors"][:5]:
            print(f"  {path}: {err}")
    print("-" * 68)


def _write_report(rep: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                           encoding="utf-8")


def _run_once(opts: RunOptions, engine, kind: str, quiet: bool) -> dict:
    pipe = Pipeline(opts, engine, kind, progress=not quiet)
    return pipe.run()


def _benchmark(opts: RunOptions, repeats: int, quiet: bool) -> int:
    results: dict[str, dict] = {}
    engines = []
    try:
        engines.append(("gpu", GpuEngine()))
    except Exception as exc:
        print(f"gpu unavailable for benchmark: {exc}")
    engines.append(("cpu", CpuEngine()))

    for kind, engine in engines:
        best = None
        runs = []
        for i in range(max(1, repeats)):
            o = RunOptions(**{**opts.__dict__, "output_dir":
                              opts.output_dir / f"_bench_{kind}_{i}"})
            rep = _run_once(o, engine, kind, quiet)
            runs.append(rep)
            if best is None or rep["elapsed_sec"] < best["elapsed_sec"]:
                best = rep
        assert best is not None
        results[kind] = {**best, "runs": runs}

    print()
    print("=" * 68)
    print(f"Benchmark ({opts.max_w}x{opts.max_h}, q={opts.quality})")
    print("=" * 68)
    print(f"{'backend':<8}{'file/s':>10}{'time, s':>12}{'MiB/s in':>12}")
    for kind, rep in results.items():
        print(f"{kind:<8}{rep['files_per_sec']:>10.1f}"
              f"{rep['elapsed_sec']:>12.2f}{rep['mb_per_sec_in']:>12.1f}")
    if "gpu" in results and "cpu" in results:
        speedup = (results["gpu"]["files_per_sec"]
                   / max(1e-9, results["cpu"]["files_per_sec"]))
        print(f"\nGPU speedup: {speedup:.2f}x")
    print("=" * 68)
    path = opts.output_dir / "benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"Raw results: {path}")
    return 0


def _tkinter_available() -> bool:
    try:
        import tkinter  # noqa: F401
    except Exception:
        return False
    return True


def _alternative_pythons() -> list[str]:
    """Other interpreters on this machine that may ship with tkinter."""
    import shutil

    found: list[str] = []
    seen: set[str] = set()

    def add(path: str) -> None:
        try:
            key = str(Path(path).resolve()).lower()
        except OSError:
            return
        if key in seen:
            return
        seen.add(key)
        found.append(path)

    me = str(Path(sys.executable).resolve()).lower()
    for name in ("python3", "python"):
        path = shutil.which(name)
        if path and str(Path(path).resolve()).lower() != me:
            add(path)
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
        if local.is_dir():
            for child in sorted(local.iterdir(), reverse=True):
                exe = child / "python.exe"
                if exe.exists() and str(exe.resolve()).lower() != me:
                    add(str(exe))
    return found


def _launch_gui_elsewhere() -> int:
    """Re-launch the UI with an interpreter that has tkinter."""
    import os
    import subprocess

    project = str(PROJECT_DIR)
    env = dict(os.environ)
    env["PYTHONPATH"] = project + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    for exe in _alternative_pythons():
        try:
            probe = subprocess.run(
                [exe, "-c", "import tkinter"], cwd=project, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=20)
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode != 0:
            continue
        print(f"UI: using {exe} (this Python has no tkinter)")
        return subprocess.call([exe, "-m", "batchresize", "--gui"],
                               cwd=project, env=env)

    print("error: the desktop UI needs tkinter, but no interpreter with "
          "tkinter was found.", file=sys.stderr)
    print("Install Python from python.org (it includes tkinter) or use the "
          "command line.", file=sys.stderr)
    return 2


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.gui:
        if not _tkinter_available():
            return _launch_gui_elsewhere()
        from .gui import main as gui_main

        return gui_main()

    if args.doctor:
        return doctor_report()

    src, dst = _resolve_paths(args)
    if not src.exists() or not src.is_dir():
        raise SystemExit(f"error: input folder not found: {src}")
    if src.resolve() == dst.resolve():
        raise SystemExit("error: output folder must differ from the input folder")

    opts = _make_options(args, src, dst)
    engine, kind = _select_engine(args.backend)
    if kind == "fail":
        return 2

    if args.benchmark:
        return _benchmark(opts, args.benchmark_repeats, args.quiet)

    spec = OutputSpec(opts.max_w, opts.max_h, opts.mode, opts.quality,
                      opts.subsampling, opts.allow_upscale)
    print(f"engine       : {kind}")
    print(f"output       : {spec.describe()}")
    print(f"input        : {src}")
    print(f"output dir   : {dst}")
    print(f"threads      : readers={opts.workers}, "
          f"processors={opts.workers if kind == 'cpu' else 1}, "
          f"writers={max(2, opts.workers // 2)}")

    t0 = time.perf_counter()
    rep = _run_once(opts, engine, kind, args.quiet)
    rep["wall_sec"] = round(time.perf_counter() - t0, 3)

    report_path = Path(args.report) if args.report else dst / "report.json"
    _write_report(rep, report_path)
    _print_summary(rep)
    print(f"Report       : {report_path}")

    if rep["interrupted"]:
        print("Interrupted: finished files were kept.")
        return 130
    return 1 if rep["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
