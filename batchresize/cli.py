"""Command line interface for batchresize."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Lightweight factories keep GUI/bootstrap usable without NumPy/CUDA installed.
def CpuEngine(*args, **kwargs):
    from .cpu_backend import CpuEngine as Engine
    return Engine(*args, **kwargs)

def GpuEngine(*args, **kwargs):
    from .gpu_backend import GpuEngine as Engine
    return Engine(*args, **kwargs)

def doctor_report():
    from .env import doctor_report as report
    return report()

from .presets import (ALL_QUALITY_PRESETS, SIZE_PRESETS, OutputSpec,
                      resolve_quality, resolve_size)
from .profiles import (Profile, builtin_profiles,
                       default_profiles_path, find_profile, load_profiles,
                       read_profile_file)

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
        description="Batch image resizer: nvImageCodec + CUDA or CPU (OpenCV/Pillow).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", default=str(DEFAULT_INPUT),
                   help="source folder (fixed to ./input, override only if needed)")
    p.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT),
                   help="destination folder (fixed to ./output)")

    profile_group = p.add_mutually_exclusive_group()
    profile_group.add_argument("--profile", default=None, metavar="NAME_OR_ID",
                               help="apply a saved or built-in task profile")
    profile_group.add_argument("--profile-file", default=None, metavar="PATH",
                               help="apply a single profile from a JSON file")
    p.add_argument("--list-profiles", action="store_true",
                   help="list built-in and saved profiles and exit")

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
                   help="soft application-array estimate in GiB, NOT a hard VRAM cap")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N files")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip only outputs verified by source/config/output hashes and JPEG decode")
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
                   help="disable application CPU fallback; codec-internal CPU routing may still occur")
    p.add_argument("--fsync", action="store_true",
                   help="fsync before the atomic rename (slower)")
    p.add_argument("--optimize-huffman", action="store_true",
                   help="optimized Huffman tables in the GPU encoder")
    p.add_argument("--report", default=None,
                   help="path to the JSON report (default: <output>/report.json)")
    p.add_argument("--profile-cuda", action="store_true",
                   help="synchronize GPU stages for service profiling (changes throughput)")
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
                   version="batchresize " + __import__("batchresize").__version__)
    return p


def _resolve_paths(args) -> tuple[Path, Path]:
    src = Path(args.input).expanduser()
    dst = Path(args.output).expanduser()
    return src, dst


def _provided_dests(argv: list[str], parser: argparse.ArgumentParser) -> set[str]:
    """Which options the user actually typed, for profile precedence."""
    provided: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("-") or token == "-":
            continue
        key = token.split("=", 1)[0]
        action = parser._option_string_actions.get(key)
        if action is not None:
            provided.add(action.dest)
    return provided


def _merged_profiles() -> tuple[list[Profile], list[str]]:
    user, errors = load_profiles(default_profiles_path())
    combined = builtin_profiles()
    seen = {profile.id for profile in combined}
    for profile in user:
        if profile.id in seen:
            errors.append(f"profile {profile.id!r} collides with a built-in profile")
            continue
        combined.append(profile)
        seen.add(profile.id)
    return combined, errors


def _list_profiles() -> int:
    profiles, errors = _merged_profiles()
    print(f"Profiles: {default_profiles_path()}")
    for profile in profiles:
        kind = "built-in" if profile.builtin else "user"
        print(f"  {profile.id:32} {profile.name}  [{kind}]")
    for error in errors:
        print(f"warning: {error}", file=sys.stderr)
    return 0


def _load_cli_profile(args) -> Profile | None:
    if args.profile_file:
        path = Path(args.profile_file).expanduser()
        profiles = read_profile_file(path)
        if len(profiles) != 1:
            raise SystemExit("error: --profile-file must contain exactly one "
                             f"profile (found {len(profiles)})")
        return profiles[0]
    if args.profile:
        profiles, _ = _merged_profiles()
        profile = find_profile(profiles, args.profile)
        if profile is None:
            names = ", ".join(profile.name for profile in profiles)
            raise SystemExit(f"error: profile not found: {args.profile}. "
                             f"Available: {names}")
        return profile
    return None


def _make_options(args, src: Path, dst: Path, profile: Profile | None = None,
                  provided: set[str] | None = None):
    from .pipeline import RunOptions
    provided = provided or set()
    if profile is not None and profile.variants:
        raise SystemExit("error: profiles with output variants require the "
                         "multi-variant stage and are not supported yet")

    size_explicit = bool({"size", "width", "height", "sizing",
                          "sizing_value"} & provided)
    if profile is not None and not size_explicit:
        if profile.sizing == "resolution":
            max_w, max_h = profile.width, profile.height
            sizing = "resolution"
            sizing_value = 0.0
        else:
            # The box is ignored for percent/megapixel modes; keep it a valid
            # positive placeholder for RunOptions validation.
            max_w = max_h = 10_000
            sizing = profile.sizing
            sizing_value = profile.sizing_value
    else:
        if args.size == "custom" and (args.width is None or args.height is None):
            raise SystemExit("error: --size custom requires --width and --height")
        max_w, max_h = resolve_size(
            None if args.size == "custom" else args.size,
            args.width, args.height)
        sizing = args.sizing
        sizing_value = args.sizing_value

    try:
        if profile is not None and not ({"quality", "jpeg_quality"} & provided):
            quality = profile.quality
        else:
            quality = resolve_quality(
                None if args.quality == "custom" else args.quality,
                args.jpeg_quality)
        bg_text = (profile.bg_color if profile is not None
                   and "bg_color" not in provided else args.bg_color)
        bg = parse_color(bg_text)
        mode = (profile.mode if profile is not None and "mode" not in provided
                else args.mode)
        filter_name = (profile.filter if profile is not None
                       and "filter" not in provided else args.filter)
        allow_upscale = (profile.allow_upscale if profile is not None
                         and "allow_upscale" not in provided
                         else args.allow_upscale)
        keep_exif = (profile.keep_exif if profile is not None
                     and "no_exif" not in provided else not args.no_exif)
        keep_icc = (profile.keep_icc if profile is not None
                    and "no_icc" not in provided else not args.no_icc)
        sub_text = (profile.subsampling if profile is not None
                    and "subsampling" not in provided else args.subsampling)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    subsampling = {"auto": None, "444": 0, "422": 1, "420": 2}[sub_text]
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
        mode=mode,
        quality=quality,
        subsampling=subsampling,
        sizing=sizing,
        sizing_value=sizing_value,
        allow_upscale=allow_upscale,
        workers=args.workers,
        batch_size=args.batch_size,
        queue_depth=args.queue_depth,
        skip_existing=args.skip_existing,
        overwrite=overwrite,
        limit=args.limit,
        keep_exif=keep_exif,
        keep_icc=keep_icc,
        bg_color=bg,
        vram_budget_gb=args.vram_budget,
        filter_name=filter_name,
        fallback=not args.no_fallback,
        allow_truncated=args.allow_truncated,
        optimize_huffman=args.optimize_huffman,
        fsync=args.fsync,
        profile_cuda=args.profile_cuda,
        cancel_file=Path(os.environ["BATCHRESIZE_CANCEL_FILE"]) if os.environ.get("BATCHRESIZE_CANCEL_FILE") else None,
    )


def _select_engine(backend: str, opts=None):
    """Return (engine, engine_kind); ``(None, "fail")`` for a hard GPU request."""
    if backend == "cpu":
        return CpuEngine(opts.filter_name if opts else "lanczos3"), "cpu"

    try:
        engine = GpuEngine(filter_name=opts.filter_name if opts else "lanczos3",
                           optimize_huffman=opts.optimize_huffman if opts else False)
        ok, msg = engine.self_test()
        if not ok:
            raise RuntimeError(f"GPU self-test failed: {msg}")
    except Exception as exc:
        if backend == "auto":
            print(f"gpu unavailable ({exc}); falling back to CPU")
            return CpuEngine(opts.filter_name if opts else "lanczos3"), "cpu"
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
    for category in ("fatal_errors", "discovery_errors", "warnings"):
        for message in rep.get(category, [])[:5]:
            print(f"{category}: {message}")
    if rep["errors"]:
        print(f"Errors       : {len(rep['errors'])} (first 5)")
        for path, err in rep["errors"][:5]:
            print(f"  {path}: {err}")
    print("-" * 68)


def _write_report(rep: dict, report_path: Path) -> None:
    from .safety import atomic_write, STATE_NAME, LOCK_NAME
    if report_path.suffix.lower() != ".json" or report_path.name in (STATE_NAME, LOCK_NAME):
        raise ValueError("report must be a separate .json file, not an image or internal state")
    atomic_write(report_path, json.dumps(rep, indent=2, ensure_ascii=False).encode("utf8"))


def _run_once(opts, engine, kind: str, quiet: bool, files=None) -> dict:
    from .pipeline import Pipeline
    pipe = Pipeline(opts, engine, kind, progress=not quiet, files=files)
    return pipe.run()


def _benchmark(opts, repeats: int, quiet: bool) -> int:
    from .pipeline import RunOptions, discover
    from .safety import atomic_write, check_roots
    check_roots(opts.input_dir, opts.output_dir.resolve())
    if repeats < 1:
        raise ValueError("benchmark repeats must be >= 1")
    files = discover(opts.input_dir, exclude=opts.output_dir, limit=opts.limit)
    if not files:
        raise ValueError("benchmark requires at least one input")
    import hashlib
    import tempfile
    # Explicit file lifetime avoids descriptor accumulation across repeats.
    def manifest():
        entries = []
        for p in files:
            with p.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            entries.append({"path": str(p.relative_to(opts.input_dir)), "sha256": digest})
        return entries
    initial_manifest = manifest()
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="benchmark-", dir=opts.output_dir))
    results: dict[str, dict] = {}
    engines = []
    try:
        gpu = GpuEngine(filter_name=opts.filter_name, optimize_huffman=opts.optimize_huffman)
        ok, msg = gpu.self_test()
        if not ok:
            raise RuntimeError(msg)
        engines.append(("gpu", gpu))
    except Exception as exc:
        print(f"gpu unavailable for benchmark: {exc}")
    engines.append(("cpu", CpuEngine(opts.filter_name)))

    for kind, engine in engines:
        best = None
        runs = []
        for i in range(max(1, repeats)):
            o = RunOptions(**{**opts.__dict__, "output_dir":
                              run_root / f"{kind}_{i}", "overwrite": True,
                              "skip_existing": False})
            before_unchanged = manifest() == initial_manifest
            rep = _run_once(o, engine, kind, quiet, files=files)
            rep["benchmark_input_unchanged"] = before_unchanged and manifest() == initial_manifest
            rep["benchmark_valid"] = (rep["complete"] and rep["processed"] == len(files)
                                      and rep["benchmark_input_unchanged"])
            runs.append(rep)
            if rep["benchmark_valid"] and (best is None or rep["elapsed_sec"] < best["elapsed_sec"]):
                best = rep
        import statistics
        results[kind] = {**(best or runs[-1]), "runs": runs,
            "valid": all(r["benchmark_valid"] for r in runs),
            "median_files_per_sec": statistics.median(r["files_per_sec"] for r in runs),
            "headline_selection": "best valid run; all repeats and median retained",
            "cache_state": "hash checks read all inputs outside timing; warm OS cache, not cold IO",
            "run_root": str(run_root), "input_manifest": initial_manifest}

    print()
    print("=" * 68)
    print(f"Benchmark ({opts.max_w}x{opts.max_h}, q={opts.quality})")
    print("=" * 68)
    print(f"{'backend':<8}{'file/s':>10}{'time, s':>12}{'MiB/s in':>12}")
    for kind, rep in results.items():
        print(f"{kind:<8}{rep['files_per_sec']:>10.1f}"
              f"{rep['elapsed_sec']:>12.2f}{rep['mb_per_sec_in']:>12.1f}")
    if all(k in results and results[k]["valid"] for k in ("gpu", "cpu")):
        speedup = (results["gpu"]["files_per_sec"]
                   / max(1e-9, results["cpu"]["files_per_sec"]))
        print(f"\nGPU speedup: {speedup:.2f}x")
    print("=" * 68)
    path = opts.output_dir / "benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(results, indent=2, ensure_ascii=False).encode("utf8"))
    print(f"Raw results: {path}")
    return 0 if all(r["valid"] for r in results.values()) else 1


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


def _main(argv: Optional[list[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    provided = _provided_dests(raw_argv, parser)

    if args.list_profiles:
        return _list_profiles()

    if args.gui:
        if not _tkinter_available():
            return _launch_gui_elsewhere()
        from .gui import main as gui_main

        return gui_main()

    if args.doctor:
        return doctor_report()

    profile = _load_cli_profile(args)
    src, dst = _resolve_paths(args)
    if not src.exists() or not src.is_dir():
        raise SystemExit(f"error: input folder not found: {src}")
    if src.resolve() == dst.resolve():
        raise SystemExit("error: output folder must differ from the input folder")

    opts = _make_options(args, src, dst, profile, provided)
    from .safety import check_roots, reject_links, STATE_NAME, LOCK_NAME
    check_roots(opts.input_dir, opts.output_dir.resolve())
    report_path = Path(args.report).expanduser().absolute() if args.report else opts.output_dir / "report.json"
    if report_path.suffix.lower() != ".json" or report_path.name in (STATE_NAME, LOCK_NAME):
        raise ValueError("report must be a separate .json file")
    if report_path.resolve().is_relative_to(opts.input_dir) and not report_path.absolute().is_relative_to(opts.output_dir):
        raise ValueError("report must not overwrite a file in the input tree")
    reject_links(report_path, opts.output_dir if report_path.is_relative_to(opts.output_dir) else report_path.parent)
    if args.benchmark:
        return _benchmark(opts, args.benchmark_repeats, args.quiet)
    engine, kind = _select_engine(args.backend, opts)
    if kind == "fail":
        return 2

    spec = OutputSpec(opts.max_w, opts.max_h, opts.mode, opts.quality,
                      opts.subsampling, opts.allow_upscale)
    print(f"engine       : {kind}")
    if profile is not None:
        print(f"profile      : {profile.name} ({profile.id})")
    print(f"output       : {spec.describe()}")
    print(f"input        : {src}")
    print(f"output dir   : {dst}")
    print(f"threads      : readers={opts.workers}, "
          f"processors={opts.workers if kind == 'cpu' else 1}, "
          f"writers={max(2, opts.workers // 2)}")

    t0 = time.perf_counter()
    rep = _run_once(opts, engine, kind, args.quiet)
    rep["wall_sec"] = round(time.perf_counter() - t0, 3)

    report_start = time.perf_counter()
    _write_report(rep, report_path)
    print(f"Report write : {time.perf_counter() - report_start:.4f} s (not in pipeline elapsed)")
    _print_summary(rep)
    print(f"Report       : {report_path}")

    if rep["interrupted"]:
        print("Interrupted: finished files were kept.")
        return 130
    return 0 if rep["complete"] else 1


def main(argv: Optional[list[str]] = None) -> int:
    try:
        return _main(argv)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
