"""Generate a deterministic test corpus covering formats, sizes and EXIF cases.

Usage: python tools/make_testset.py [target_dir]
"""

from __future__ import annotations

import io
import os
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("testdata/photos")


def gradient(w: int, h: int, seed: int = 0, alpha: bool = False) -> Image.Image:
    """Smooth, photo-like content (noise compresses badly, gradients look real)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (xx / max(w - 1, 1) * 255.0)
    g = (yy / max(h - 1, 1) * 255.0)
    b = ((np.sin(xx / 37.0) + np.cos(yy / 53.0) + 2.0) * 63.75)
    base = np.dstack([r, g, b])
    noise = rng.normal(0, 6, size=(h, w, 3))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    if alpha:
        a = np.clip(255 - (xx / max(w - 1, 1)) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(np.dstack([arr, a]), "RGBA")
    return Image.fromarray(arr, "RGB")


def with_orientation(img: Image.Image, orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    exif[0x010F] = "batchresize-test"
    exif[0x0110] = "synthetic"
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92, exif=exif)
    return buf.getvalue()


def with_icc(img: Image.Image, orientation: int = 1) -> bytes:
    # small but valid sRGB-ish ICC profile header is enough for round-trip tests
    icc = bytes.fromhex(
        "0000002461637370200000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000f6d6000100000000")
    exif = Image.Exif()
    exif[0x0112] = orientation
    exif[0x0131] = "batchresize"
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90, exif=exif, icc_profile=icc)
    return buf.getvalue()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "Family").mkdir(exist_ok=True)
    (ROOT / "Family" / "2024").mkdir(exist_ok=True)
    (ROOT / "Landscapes").mkdir(exist_ok=True)
    (ROOT / "Mixed").mkdir(exist_ok=True)

    n = 0
    # --- large JPEGs (the real workload) ---------------------------------
    for i, (w, h) in enumerate([(4000, 3000), (6000, 4000), (3000, 3000),
                                (5472, 3648)]):
        p = ROOT / "Landscapes" / f"big_{w}x{h}.jpg"
        gradient(w, h, seed=i).save(p, "JPEG", quality=93)
        n += 1

    # --- nested structure -------------------------------------------------
    gradient(2400, 1600, seed=11).save(ROOT / "Family" / "portrait.jpg", "JPEG", quality=92)
    gradient(1800, 2400, seed=12).save(ROOT / "Family" / "2024" / "group.jpg",
                                       "JPEG", quality=92)
    n += 2

    # --- other formats ----------------------------------------------------
    gradient(1600, 1200, seed=21).save(ROOT / "Mixed" / "photo.png")
    gradient(1200, 900, seed=22, alpha=True).save(ROOT / "Mixed" / "logo_alpha.png")
    gradient(1400, 1000, seed=23).save(ROOT / "Mixed" / "scan.bmp")
    gradient(2000, 1500, seed=24).save(ROOT / "Mixed" / "archive.tiff")
    gradient(1600, 1200, seed=25).save(ROOT / "Mixed" / "web.webp", quality=90)
    n += 5

    # --- colour modes -----------------------------------------------------
    gray = gradient(1200, 800, seed=51).convert("L")
    gray.save(ROOT / "Mixed" / "gray.jpg", "JPEG", quality=90)
    gray.save(ROOT / "Mixed" / "gray.png")
    gradient(1200, 800, seed=52).convert("CMYK").save(
        ROOT / "Mixed" / "cmyk.jpg", "JPEG", quality=90)
    rgba = gradient(900, 600, seed=53, alpha=True)
    rgba.save(ROOT / "Mixed" / "rgba.png")
    rgba.save(ROOT / "Mixed" / "rgba.tiff")
    gradient(1600, 1000, seed=54).save(
        ROOT / "Mixed" / "progressive.jpg", "JPEG", quality=90, progressive=True)
    n += 6

    # --- unicode and spaces in paths --------------------------------------
    unicode_dir = ROOT / "Семья 2024"
    unicode_dir.mkdir(exist_ok=True)
    gradient(1920, 1280, seed=61).save(
        unicode_dir / "Портрет 01.jpg", "JPEG", quality=92)
    n += 1

    # --- name clash: photo.png and photo.jpg in the same folder -----------
    gradient(900, 700, seed=31).save(ROOT / "Mixed" / "photo.jpg", "JPEG", quality=90)
    n += 1  # photo.png already counted above

    # --- tiny and odd sizes ----------------------------------------------
    gradient(120, 90, seed=41).save(ROOT / "Mixed" / "tiny.jpg", "JPEG", quality=90)
    gradient(1000, 400, seed=42).save(ROOT / "Mixed" / "panorama.jpg", "JPEG", quality=90)
    gradient(400, 1200, seed=43).save(ROOT / "Mixed" / "tower.jpg", "JPEG", quality=90)
    n += 3

    # --- EXIF orientations 1..8 ------------------------------------------
    for o in range(1, 9):
        img = gradient(640, 480, seed=100 + o)
        # asymmetric marker so a wrong rotation is detectable
        px = img.load()
        for y in range(0, 60):
            for x in range(0, 60):
                px[x, y] = (255, 0, 0)
        (ROOT / "Mixed" / f"orient_{o}.jpg").write_bytes(with_orientation(img, o))
        n += 1

    # --- EXIF + ICC -------------------------------------------------------
    (ROOT / "Mixed" / "with_icc.jpg").write_bytes(with_icc(gradient(800, 600, seed=201)))
    n += 1

    # --- corrupted files --------------------------------------------------
    good = (ROOT / "Mixed" / "photo.jpg").read_bytes()
    (ROOT / "Mixed" / "corrupt.jpg").write_bytes(good[: len(good) // 3])
    (ROOT / "Mixed" / "not_an_image.jpg").write_bytes(b"this is not a jpeg" * 10)
    (ROOT / "Mixed" / "truncated_tail.jpg").write_bytes(good[:-2])
    n += 3

    total = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    print(f"created {n} images in {ROOT} ({total / 2 ** 20:.1f} MiB)")


if __name__ == "__main__":
    main()
