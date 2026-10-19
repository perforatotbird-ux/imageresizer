"""Header parsing, EXIF orientation and JPEG metadata splicing."""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Optional

from PIL import Image

ORIENTATION_TAG = 0x0112
_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


@dataclass(frozen=True)
class SourceMeta:
    width: int
    height: int
    orientation: int = 1
    exif: Optional[bytes] = None
    icc: Optional[bytes] = None

    @property
    def display_width(self) -> int:
        return self.height if self.orientation >= 5 else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.orientation >= 5 else self.height


def read_source_meta(data: bytes) -> SourceMeta:
    """Parse header-only metadata. Raises on undecodable data."""
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        orientation = 1
        exif = None
        try:
            exif_obj = im.getexif()
            orientation = int(exif_obj.get(ORIENTATION_TAG, 1) or 1)
            if orientation not in range(1, 9):
                orientation = 1
            if len(exif_obj):
                exif = exif_obj.tobytes()
        except Exception:
            orientation = 1
        icc = im.info.get("icc_profile")
        if isinstance(icc, str):  # pragma: no cover - defensive
            icc = icc.encode("latin-1", "ignore")
        return SourceMeta(int(w), int(h), orientation, exif, icc)


def is_truncated_jpeg(data: bytes) -> bool:
    """A JPEG that lost its EOI marker was cut off - decoding it yields garbage."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return True
    return data.rfind(b"\xff\xd9", max(0, len(data) - 4096)) == -1


def _strip_orientation(exif_bytes: bytes) -> bytes:
    """Return EXIF with Orientation reset to 1 (pixels are already rotated).

    The returned blob is a bare TIFF block (no ``Exif\\0\\0`` prefix) so the
    caller can wrap it into an APP1 segment.
    """
    try:
        import piexif

        exif_dict = piexif.load(exif_bytes)
        ifd0 = exif_dict.get("0th", {})
        ifd0[ORIENTATION_TAG] = 1
        # thumbnail is tied to the old orientation - drop it
        exif_dict["1st"] = {}
        exif_dict["thumbnail"] = None
        dumped = piexif.dump(exif_dict)
        if dumped.startswith(b"Exif\x00\x00"):
            dumped = dumped[6:]
        return dumped
    except Exception:
        if exif_bytes.startswith(b"Exif\x00\x00"):
            return exif_bytes[6:]
        return exif_bytes


def _icc_chunks(icc: bytes) -> list[bytes]:
    """Split an ICC profile into <=64KB APP2 payloads (ICC_PROFILE spec)."""
    limit = 65519
    chunks = [icc[i:i + limit] for i in range(0, len(icc), limit)]
    total = len(chunks)
    out = []
    for i, chunk in enumerate(chunks, start=1):
        header = b"ICC_PROFILE\x00" + bytes([i, total])
        out.append(header + chunk)
    return out


def _app_segment(marker: int, payload: bytes) -> bytes:
    if len(payload) + 2 > 0xFFFF:
        raise ValueError("APP segment too large")
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


def splice_metadata(jpeg: bytes, exif: Optional[bytes],
                    icc: Optional[bytes]) -> bytes:
    """Insert EXIF/ICC into an encoded JPEG right after SOI.

    Any APP1/APP2 segments already present in ``jpeg`` are removed first so
    the result has exactly one EXIF block and one ICC block.
    """
    if not exif and not icc:
        return jpeg
    if not jpeg.startswith(b"\xFF\xD8"):
        raise ValueError("not a JPEG stream")

    cleaned = _strip_app12(jpeg)

    segments = b""
    if exif:
        payload = b"Exif\x00\x00" + _strip_orientation(exif)
        segments += _app_segment(0xE1, payload)
    if icc:
        for chunk in _icc_chunks(icc):
            segments += _app_segment(0xE2, chunk)

    return cleaned[:2] + segments + cleaned[2:]


def _strip_app12(jpeg: bytes) -> bytes:
    """Remove APP1/APP2 segments that precede the first SOS."""
    out = bytearray(jpeg[:2])
    i = 2
    n = len(jpeg)
    while i + 4 <= n:
        if jpeg[i] != 0xFF:
            break
        marker = jpeg[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            out += jpeg[i:i + 2]
            i += 2
            continue
        if marker == 0xDA:  # start of scan - copy the rest verbatim
            out += jpeg[i:]
            return bytes(out)
        seg_len = struct.unpack(">H", jpeg[i + 2:i + 4])[0]
        seg = jpeg[i:i + 2 + seg_len]
        if marker not in (0xE1, 0xE2):
            out += seg
        i += 2 + seg_len
    out += jpeg[i:]
    return bytes(out)


def png_has_alpha(data: bytes) -> bool:
    """True if the PNG has an alpha channel or a tRNS transparency chunk."""
    if len(data) < 26 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return False
    color_type = data[25]
    if color_type in (4, 6):        # grey+alpha / truecolour+alpha
        return True
    if color_type == 3:             # palette - only transparent with tRNS
        return b"tRNS" in data[:4096]
    return False


def webp_has_alpha(data: bytes) -> bool:
    if len(data) < 21 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return False
    fourcc = data[12:16]
    if fourcc == b"VP8X":
        return bool(data[20] & 0x10)      # alpha flag
    if fourcc == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return bool((bits >> 28) & 1)
    return b"ALPH" in data[:64]


def has_alpha_hint(data: bytes, suffix: str) -> bool:
    s = suffix.lower()
    if s == ".png":
        return png_has_alpha(data)
    if s == ".webp":
        return webp_has_alpha(data)
    return False


def jpeg_dimensions(jpeg: bytes) -> tuple[int, int]:
    """Read SOF dimensions without decoding pixels."""
    i = 2
    n = len(jpeg)
    while i + 4 <= n:
        if jpeg[i] != 0xFF:
            i += 1
            continue
        marker = jpeg[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = struct.unpack(">H", jpeg[i + 2:i + 4])[0]
        if marker in _SOF_MARKERS:
            h, w = struct.unpack(">HH", jpeg[i + 5:i + 9])
            return int(w), int(h)
        i += 2 + seg_len
    raise ValueError("no SOF marker found")
