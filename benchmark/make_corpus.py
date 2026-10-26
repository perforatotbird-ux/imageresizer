"""Generate a deterministic synthetic photo corpus for benchmark runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _photo(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = x / max(1, width - 1)
    yn = y / max(1, height - 1)
    phase = rng.uniform(0.0, math.tau, size=6)

    base = np.empty((height, width, 3), dtype=np.float32)
    base[..., 0] = 40 + 170 * xn + 25 * np.sin(yn * 9.0 + phase[0])
    base[..., 1] = 35 + 150 * yn + 30 * np.sin(xn * 7.0 + phase[1])
    base[..., 2] = 55 + 120 * (1.0 - xn) + 25 * np.cos((xn + yn) * 6.0 + phase[2])
    texture = (
        8.0 * np.sin(x * 0.31 + y * 0.17 + phase[3])
        + 6.0 * np.cos(x * 0.07 - y * 0.23 + phase[4])
        + rng.normal(0.0, 2.5, (height, width))
    )
    base += texture[..., None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")

    draw = ImageDraw.Draw(img)
    for i in range(12):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        rx = int(rng.integers(width // 40, max(width // 8, width // 40 + 1)))
        ry = int(rng.integers(height // 40, max(height // 8, height // 40 + 1)))
        color = tuple(int(v) for v in rng.integers(20, 236, size=3))
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=3)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("corpus"))
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--width", type=int, default=5760)
    ap.add_argument("--height", type=int, default=3840)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.out.glob("*.jpg"))
    if existing and not args.force:
        raise SystemExit(f"{args.out} already contains {len(existing)} JPEGs; use --force to regenerate")
    for old in existing:
        old.unlink()

    entries = []
    for i in range(args.count):
        width, height = (args.height, args.width) if i % 3 == 1 else (args.width, args.height)
        path = args.out / f"photo_{i:04d}.jpg"
        img = _photo(width, height, args.seed + i)
        img.save(path, "JPEG", quality=args.quality, subsampling=2, optimize=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"file": path.name, "width": width, "height": height,
                        "bytes": path.stat().st_size, "sha256": digest})
        print(f"{path.name} {width}x{height} {path.stat().st_size / 2**20:.1f} MiB")

    manifest = {
        "generator": "benchmark/make_corpus.py",
        "seed": args.seed,
        "count": args.count,
        "quality": args.quality,
        "files": entries,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
