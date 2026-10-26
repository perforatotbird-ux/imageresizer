"""Run reproducible BatchResize benchmarks and store median results."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _versions() -> dict:
    import cv2
    import numpy
    import PIL

    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "opencv": cv2.__version__, "pillow": PIL.__version__}


def _run_once(cmd: list[str], out: Path) -> tuple[int, float, dict]:
    if out.exists():
        shutil.rmtree(out)
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    wall = time.perf_counter() - started
    report_path = out / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    return proc.returncode, wall, report


def _safe_label(label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")
    return clean or "run"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--input", type=Path, default=ROOT / "benchmark" / "corpus")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--results", type=Path, default=ROOT / "benchmark" / "results")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="extra CLI arguments, after -- (e.g. -- --backend gpu --size web_1024)")
    args = ap.parse_args()

    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    if not args.input.is_dir():
        raise SystemExit(f"input directory not found: {args.input}")
    for flag in ("-i", "--input", "-o", "--output"):
        if flag in extra:
            raise SystemExit(f"do not pass {flag}; use --input and --results")

    label = _safe_label(args.label)
    run_root = args.results / "_runs" / label
    results_dir = args.results
    results_dir.mkdir(parents=True, exist_ok=True)

    base = [sys.executable, "-m", "batchresize", "-i", str(args.input),
            "-o", str(run_root), "--quiet", *extra]
    print("command:", " ".join(base))

    for i in range(args.warmup):
        rc, _, rep = _run_once(base, run_root)
        if rc != 0 or rep.get("failed", 1):
            raise SystemExit(f"warmup {i + 1} failed rc={rc} report={rep}")

    rows = []
    for i in range(args.repeats):
        rc, wall, rep = _run_once(base, run_root)
        row = {
            "run": i + 1,
            "wall_s": round(wall, 3),
            "elapsed_s": rep.get("elapsed_sec"),
            "files_per_s": rep.get("files_per_sec"),
            "processed": rep.get("processed"),
            "failed": rep.get("failed"),
            "timings_avg_ms": rep.get("timings_avg_ms"),
            "rc": rc,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    ok = [r for r in rows if r["rc"] == 0 and r["failed"] == 0]
    summary = {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "environment": _versions(),
        "input": str(args.input.resolve()),
        "command": base,
        "warmup": args.warmup,
        "medians": {
            "wall_s": round(statistics.median(r["wall_s"] for r in ok), 3) if ok else None,
            "elapsed_s": round(statistics.median(r["elapsed_s"] for r in ok), 3) if ok else None,
            "files_per_s": round(statistics.median(r["files_per_s"] for r in ok), 3) if ok else None,
        },
        "rows": rows,
    }
    (results_dir / f"{label}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {label}",
        "",
        f"- command: `{' '.join(base)}`",
        f"- input: `{args.input}`",
        f"- repeats: {args.repeats}, warmup: {args.warmup}",
        "",
        "| run | wall s | elapsed s | files/s | processed | failed | decode ms | resize ms | encode ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        t = r["timings_avg_ms"] or {}
        lines.append(
            f"| {r['run']} | {r['wall_s']} | {r['elapsed_s']} | {r['files_per_s']} | "
            f"{r['processed']} | {r['failed']} | {t.get('decode', '')} | "
            f"{t.get('resize', '')} | {t.get('encode', '')} |")
    lines += ["", f"Median: {summary['medians']}", ""]
    text = "\n".join(lines)
    (results_dir / f"{label}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
