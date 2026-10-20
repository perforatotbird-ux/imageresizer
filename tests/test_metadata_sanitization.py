"""Tests for personal metadata stripping, sanitization, and verification."""

from __future__ import annotations

import io
import pytest
from PIL import Image
import piexif

from batchresize.imageio_meta import (
    clean_exif,
    splice_metadata,
    verify_metadata_sanitized,
    _strip_app_segments,
)


def _make_rich_exif(include_gps=True, include_personal=True, include_datetime=True) -> bytes:
    zeroth = {
        piexif.ImageIFD.Make: "CameraCorp",
        piexif.ImageIFD.Model: "Model 9000",
        0x0112: 6,  # Orientation
    }
    if include_personal:
        zeroth[piexif.ImageIFD.Artist] = "Jane Doe"
        zeroth[piexif.ImageIFD.Copyright] = "2026 Jane Doe"
        zeroth[piexif.ImageIFD.Software] = "PhotoApp Pro"
        zeroth[0x010E] = "Personal holiday trip"
    if include_datetime:
        zeroth[piexif.ImageIFD.DateTime] = "2026:05:15 14:30:00"

    exif = {
        piexif.ExifIFD.FocalLength: (50, 1),
        piexif.ExifIFD.ExposureTime: (1, 125),
    }
    if include_personal:
        exif[piexif.ExifIFD.UserComment] = b"Private confidential notes"
        exif[piexif.ExifIFD.BodySerialNumber] = "SN-987654321"
        exif[piexif.ExifIFD.CameraOwnerName] = "Jane Doe"
    if include_datetime:
        exif[piexif.ExifIFD.DateTimeOriginal] = "2026:05:15 14:30:00"
        exif[piexif.ExifIFD.DateTimeDigitized] = "2026:05:15 14:30:00"

    gps = {}
    if include_gps:
        gps[piexif.GPSIFD.GPSLatitude] = ((45, 1), (10, 1), (30, 1))
        gps[piexif.GPSIFD.GPSLongitude] = ((9, 1), (20, 1), (40, 1))

    data = {"0th": zeroth, "Exif": exif, "GPS": gps, "1st": {}, "thumbnail": None}
    return piexif.dump(data)


def test_clean_exif_without_strip_keeps_metadata():
    raw_exif = _make_rich_exif()
    cleaned = clean_exif(raw_exif, dimensions=(800, 600), strip_personal=False)
    loaded = piexif.load(cleaned)

    # Orientation normalized to 1, dimensions updated
    assert loaded["0th"][0x0112] == 1
    assert loaded["0th"][256] == 800
    assert loaded["0th"][257] == 600

    # Personal tags still present
    assert piexif.ImageIFD.Artist in loaded["0th"]
    assert len(loaded["GPS"]) > 0
    assert piexif.ExifIFD.BodySerialNumber in loaded["Exif"]


def test_clean_exif_with_strip_personal():
    raw_exif = _make_rich_exif()
    # Strip personal data without keeping datetime
    cleaned = clean_exif(raw_exif, dimensions=(800, 600), strip_personal=True, keep_datetime=False)
    loaded = piexif.load(cleaned)

    # Orientation normalized to 1
    assert loaded["0th"][0x0112] == 1
    # GPS completely empty
    assert len(loaded["GPS"]) == 0
    assert loaded["thumbnail"] is None

    # Camera model and technical settings preserved
    assert loaded["0th"][piexif.ImageIFD.Make] == b"CameraCorp"
    assert loaded["0th"][piexif.ImageIFD.Model] == b"Model 9000"
    assert loaded["Exif"][piexif.ExifIFD.FocalLength] == (50, 1)

    # Personal and date tags stripped
    assert piexif.ImageIFD.Artist not in loaded["0th"]
    assert piexif.ImageIFD.Copyright not in loaded["0th"]
    assert piexif.ImageIFD.Software not in loaded["0th"]
    assert 0x010E not in loaded["0th"]
    assert piexif.ImageIFD.DateTime not in loaded["0th"]
    assert piexif.ExifIFD.BodySerialNumber not in loaded["Exif"]
    assert piexif.ExifIFD.UserComment not in loaded["Exif"]
    assert piexif.ExifIFD.DateTimeOriginal not in loaded["Exif"]


def test_clean_exif_strip_personal_with_keep_datetime():
    raw_exif = _make_rich_exif()
    cleaned = clean_exif(raw_exif, dimensions=(800, 600), strip_personal=True, keep_datetime=True)
    loaded = piexif.load(cleaned)

    # GPS stripped
    assert len(loaded["GPS"]) == 0
    # Personal stripped
    assert piexif.ImageIFD.Artist not in loaded["0th"]
    assert piexif.ExifIFD.BodySerialNumber not in loaded["Exif"]
    # DateTime preserved!
    assert loaded["0th"][piexif.ImageIFD.DateTime] == b"2026:05:15 14:30:00"
    assert loaded["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"2026:05:15 14:30:00"


def test_splice_and_verify_metadata_sanitized():
    raw_exif = _make_rich_exif()
    # Create plain JPEG
    buf = io.BytesIO()
    im = Image.new("RGB", (64, 48), (200, 100, 50))
    im.save(buf, "JPEG")
    plain_jpeg = buf.getvalue()

    # Splice with strip_personal
    sanitized_jpeg = splice_metadata(
        plain_jpeg,
        exif=raw_exif,
        icc=None,
        dimensions=(64, 48),
        strip_personal=True,
        keep_datetime=False,
    )

    # Verification passes without error
    verify_metadata_sanitized(sanitized_jpeg, keep_datetime=False)

    # Check that verification detects leaks if raw unsanitized EXIF was injected
    unsanitized_jpeg = splice_metadata(
        plain_jpeg,
        exif=raw_exif,
        icc=None,
        dimensions=(64, 48),
        strip_personal=False,
    )
    with pytest.raises(ValueError, match="personal metadata leak"):
        verify_metadata_sanitized(unsanitized_jpeg, keep_datetime=False)
