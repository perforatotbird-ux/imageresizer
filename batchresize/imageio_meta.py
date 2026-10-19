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
    mode: str = "RGB"
    has_alpha: bool = False

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
        except Exception as exc:
            raise ValueError("cannot parse source EXIF safely") from exc
        icc = im.info.get("icc_profile")
        if isinstance(icc, str):  # pragma: no cover - defensive
            icc = icc.encode("latin-1", "ignore")
        alpha = "A" in im.getbands() or "transparency" in im.info
        return SourceMeta(int(w), int(h), orientation, exif, icc, im.mode, alpha)


def is_truncated_jpeg(data: bytes) -> bool:
    """Marker-structure check only, NOT proof of entropy/pixel integrity.

    Skip APP payloads and stuffed entropy bytes, accepting arbitrary trailers.
    """
    if not data.startswith(b"\xff\xd8"):
        return True
    i = 2
    scan = False
    while i < len(data):
        if data[i] != 255:
            if not scan:
                return True
            i += 1
            continue
        i += 1
        while i < len(data) and data[i] == 255:
            i += 1
        if i >= len(data):
            return True
        marker = data[i]; i += 1
        if marker == 0 and scan or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xD9:
            return not scan
        if marker in (0xD8, 0x01):
            continue
        if i + 2 > len(data):
            return True
        n = int.from_bytes(data[i:i+2], "big")
        if n < 2 or i + n > len(data):
            return True
        scan = scan or marker == 0xDA
        i += n
    return True


def _strip_orientation(exif_bytes: bytes, dimensions=None) -> bytes:
    """Normalize EXIF or raise; never attach stale orientation on failure."""
    import piexif
    ex = piexif.load(exif_bytes)
    ex.setdefault("0th", {})[ORIENTATION_TAG] = 1
    ex["1st"] = {}; ex["thumbnail"] = None
    if dimensions is not None:
        w, h = dimensions
        ex["0th"][256] = w; ex["0th"][257] = h
        ex.setdefault("Exif", {})[40962] = w
        ex["Exif"][40963] = h
    blob = piexif.dump(ex)
    return blob[6:] if blob.startswith(b"Exif\x00\x00") else blob


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
                    icc: Optional[bytes], dimensions=None) -> bytes:
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
        payload = b"Exif\x00\x00" + _strip_orientation(exif, dimensions)
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


def has_alpha_hint(data: bytes, suffix: str = "") -> bool:
    # Content-based: handles TIFF, LA, RGB/P tRNS and mislabeled extensions.
    with Image.open(io.BytesIO(data)) as im:
        return "A" in im.getbands() or "transparency" in im.info


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
