"""End-to-end pipeline tests for Stage 5 (Color Policy, Linear Light, Metadata Sanitization)."""

from __future__ import annotations

import io
import json
from pathlib import Path
import piexif
import pytest
from PIL import Image

from batchresize.cli import main as cli_main
from batchresize.color_management import SRGB_ICC_BYTES
from batchresize.imageio_meta import verify_metadata_sanitized


def _create_photo_with_gps(path: Path) -> None:
    im = Image.new("RGB", (100, 100), (80, 140, 200))
    zeroth = {
        piexif.ImageIFD.Make: "TestCamera",
        piexif.ImageIFD.Artist: "Secret Author",
        piexif.ImageIFD.DateTime: "2026:01:01 10:00:00",
    }
    exif = {
        piexif.ExifIFD.DateTimeOriginal: "2026:01:01 10:00:00",
        piexif.ExifIFD.BodySerialNumber: "SECRET-SER-1234",
    }
    gps = {
        piexif.GPSIFD.GPSLatitude: ((55, 1), (45, 1), (0, 1)),
    }
    raw_exif = piexif.dump({"0th": zeroth, "Exif": exif, "GPS": gps, "1st": {}, "thumbnail": None})
    im.save(path, "JPEG", exif=raw_exif)


def test_cli_strip_personal_data_e2e(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    _create_photo_with_gps(src / "test_gps.jpg")

    # Run with --strip-personal-data
    code = cli_main([
        "-i", str(src),
        "-o", str(dst),
        "--size", "thumb_256",
        "--backend", "cpu",
        "--strip-personal-data",
    ])
    assert code == 0

    out_file = dst / "test_gps.jpg"
    assert out_file.is_file()
    verify_metadata_sanitized(out_file.read_bytes(), keep_datetime=False)

    # Check that Make is preserved but Artist and GPS are gone
    loaded = piexif.load(out_file.read_bytes())
    assert loaded["0th"][piexif.ImageIFD.Make] == b"TestCamera"
    assert piexif.ImageIFD.Artist not in loaded["0th"]
    assert len(loaded["GPS"]) == 0


def test_cli_strip_personal_keep_datetime_e2e(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    _create_photo_with_gps(src / "test_gps.jpg")

    code = cli_main([
        "-i", str(src),
        "-o", str(dst),
        "--size", "thumb_256",
        "--backend", "cpu",
        "--strip-personal-data",
        "--keep-datetime",
    ])
    assert code == 0

    out_file = dst / "test_gps.jpg"
    assert out_file.is_file()
    verify_metadata_sanitized(out_file.read_bytes(), keep_datetime=True)

    loaded = piexif.load(out_file.read_bytes())
    assert len(loaded["GPS"]) == 0
    assert piexif.ImageIFD.Artist not in loaded["0th"]
    assert loaded["0th"][piexif.ImageIFD.DateTime] == b"2026:01:01 10:00:00"
    assert loaded["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"2026:01:01 10:00:00"


def test_cli_web_srgb_policy_embeds_profile(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    # Create plain PNG without profile
    Image.new("RGB", (80, 80), (100, 150, 200)).save(src / "plain.png")

    code = cli_main([
        "-i", str(src),
        "-o", str(dst),
        "--size", "thumb_256",
        "--backend", "cpu",
        "--color-policy", "web_srgb",
    ])
    assert code == 0

    out_file = dst / "plain.jpg"
    assert out_file.is_file()
    with Image.open(out_file) as im:
        assert "icc_profile" in im.info
        assert im.info["icc_profile"] == SRGB_ICC_BYTES


def test_cli_linear_light_transparent_png(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"

    # Create RGBA PNG with transparent gradient
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    # Draw a circle / square with semi-transparent red
    for y in range(16, 48):
        for x in range(16, 48):
            im.putpixel((x, y), (255, 0, 0, 128))
    im.save(src / "trans.png")

    code = cli_main([
        "-i", str(src),
        "-o", str(dst),
        "--size", "thumb_256",
        "--backend", "cpu",
        "--linear-light",
        "--bg-color", "#FFFFFF",
    ])
    assert code == 0

    out_file = dst / "trans.jpg"
    assert out_file.is_file()
    with Image.open(out_file) as rendered:
        rgb = rendered.convert("RGB")
        w, h = rgb.size
        # Center should be pinkish (blended red over white)
        center_color = rgb.getpixel((w // 2, h // 2))
        assert center_color[0] > 200  # High red
        # Outer corner should be pure white
        corner_color = rgb.getpixel((0, 0))
        assert corner_color[0] >= 250 and corner_color[1] >= 250 and corner_color[2] >= 250


def test_resume_cache_invalidates_on_color_and_metadata_change(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    _create_photo_with_gps(src / "photo.jpg")

    # 1. Initial run with default options
    cli_main([
        "-i", str(src),
        "-o", str(dst),
        "--size", "thumb_256",
        "--backend", "cpu",
    ])
    report1 = json.loads((dst / "report.json").read_text(encoding="utf-8"))
    assert report1["processed"] == 1

    # 2. Re-run without changes using --skip-existing: should skip
    cli_main([
        "-i", str(src),
        "-o", str(dst),
        "--size", "thumb_256",
        "--backend", "cpu",
        "--skip-existing",
    ])
    report2 = json.loads((dst / "report.json").read_text(encoding="utf-8"))
    assert report2["skipped"] == 1
    assert report2["processed"] == 0

    # 3. Change --linear-light with --skip-existing: config digest mismatch, must fail with FileExistsError
    code3 = cli_main([
        "-i", str(src),
        "-o", str(dst),
        "--size", "thumb_256",
        "--backend", "cpu",
        "--linear-light",
        "--skip-existing",
    ])
    assert code3 != 0
    report_fail = json.loads((dst / "report.json").read_text(encoding="utf-8"))
    assert report_fail["failed"] == 1
    assert "configuration changed" in report_fail["errors"][0][1]

    # 4. With --overwrite and --strip-personal-data, it successfully updates
    cli_main([
        "-i", str(src),
        "-o", str(dst),
        "--size", "thumb_256",
        "--backend", "cpu",
        "--strip-personal-data",
        "--overwrite",
    ])
    report3 = json.loads((dst / "report.json").read_text(encoding="utf-8"))
    assert report3["processed"] == 1
    verify_metadata_sanitized((dst / "photo.jpg").read_bytes(), keep_datetime=False)
