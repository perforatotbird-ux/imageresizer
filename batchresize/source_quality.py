"""Estimate the JPEG encoding profile of a source file.

The "same as source" quality mode re-encodes every picture with the
quantisation tables and chroma layout of the original, so that only the
geometry changes and the picture quality stays where it was.

The module is intentionally dependency free (no numpy/PIL/OpenCV): the GUI
process may run on a bare Python installation.
"""

from __future__ import annotations

from typing import Optional

#: Sentinel for "keep the quality of the source file".
QUALITY_SOURCE = "source"
#: Sentinel for "keep the chroma subsampling of the source file".
SUBSAMPLING_SOURCE = "source"

#: Used when the source quality cannot be determined (not a JPEG or a
#: non-standard quantisation table).
FALLBACK_QUALITY = 90
#: Used when the source chroma layout cannot be determined.
FALLBACK_SUBSAMPLING = 2  # 4:2:0

# Annex K base quantisation tables in natural order.
_LUMA_BASE = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)
_CHROMA_BASE = (
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
)
# Natural index of every zig-zag position (the order used inside DQT).
_ZIGZAG = (
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
)

#: Maximum accepted table distance for a trustworthy quality estimate.
_MATCH_TOLERANCE = 96


def _scaled_table(base: tuple[int, ...], quality: int) -> list[int]:
    """libjpeg quality scaling of an Annex K base table (natural order)."""
    quality = max(1, min(100, int(quality)))
    scale = 5000 // quality if quality < 50 else 200 - quality * 2
    out = []
    for value in base:
        scaled = (value * scale + 50) // 100
        out.append(max(1, min(255, scaled)))
    return out


def _standard_tables(quality: int) -> tuple[list[int], list[int]]:
    """Standard tables in the zig-zag order used inside a DQT segment."""
    luma = _scaled_table(_LUMA_BASE, quality)
    chroma = _scaled_table(_CHROMA_BASE, quality)
    return ([luma[i] for i in _ZIGZAG], [chroma[i] for i in _ZIGZAG])


def _iter_segments(data: bytes):
    """Yield (marker, payload) for every JPEG segment of ``data``."""
    if not data.startswith(b"\xff\xd8"):
        return
    i = 2
    size = len(data)
    while i + 4 <= size:
        if data[i] != 0xFF:
            return
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            i += 2
            continue
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if length < 2 or i + 2 + length > size:
            return
        yield marker, data[i + 4:i + 2 + length]
        if marker == 0xDA:  # start of scan - entropy coded data follows
            return
        i += 2 + length


def read_quantisation_tables(data: bytes) -> list[list[int]]:
    """All DQT tables of a JPEG in the order they appear (zig-zag order)."""
    tables: list[list[int]] = []
    for marker, payload in _iter_segments(data):
        if marker != 0xDB or not payload:
            continue
        i = 0
        while i < len(payload):
            spec = payload[i]
            i += 1
            precision = spec >> 4
            table_id = spec & 15
            count = 64 * (2 if precision else 1)
            if i + count > len(payload):
                break
            if precision:
                values = [int.from_bytes(payload[i + 2 * k:i + 2 * k + 2], "big")
                          for k in range(64)]
            else:
                values = list(payload[i:i + 64])
            i += count
            if table_id == 0:
                tables.insert(0, values)
            else:
                while len(tables) <= table_id:
                    tables.append([])
                tables[table_id] = values
    return [t for t in tables if len(t) == 64]


def dezigzag(table: list[int]) -> list[int]:
    """Convert a table from file (zig-zag) order into natural order."""
    natural = [0] * 64
    for position, index in enumerate(_ZIGZAG):
        natural[index] = table[position]
    return natural


def source_quantisation_tables(data: bytes) -> Optional[tuple[list[int], list[int]]]:
    """Luma and chroma quantisation tables of a source in natural order.

    The tables are returned unchanged (no quality scaling) so that they can
    be written into the output verbatim. ``None`` is returned when the source
    is not a JPEG or carries no usable tables.
    """
    tables = read_quantisation_tables(data)
    if not tables:
        return None
    luma = dezigzag(tables[0])
    chroma = dezigzag(tables[1]) if len(tables) > 1 else luma
    return luma, chroma


def read_subsampling(data: bytes) -> Optional[int]:
    """Chroma subsampling of a JPEG as 0=4:4:4, 1=4:2:2, 2=4:2:0."""
    for marker, payload in _iter_segments(data):
        if marker not in (0xC0, 0xC1, 0xC2) or len(payload) < 6:
            continue
        count = payload[5]
        if count < 3 or len(payload) < 6 + count * 3:
            return None
        factors = []
        for c in range(count):
            off = 6 + c * 3
            factors.append((payload[off + 1] >> 4, payload[off + 1] & 15))
        luma_h, luma_v = factors[0]
        chroma_h, chroma_v = factors[1]
        if (chroma_h, chroma_v) == (luma_h, luma_v):
            return 0
        if (luma_h, luma_v) == (2, 2) and (chroma_h, chroma_v) == (1, 1):
            return 2
        if (luma_h, luma_v) == (2, 1) and (chroma_h, chroma_v) == (1, 1):
            return 1
        return None
    return None


def estimate_source_quality(data: bytes) -> Optional[int]:
    """Quality (1..100) whose standard tables match the source best.

    ``None`` is returned when the source has no usable quantisation tables
    or when they differ too much from every standard table (custom tables
    written by other encoders).
    """
    tables = read_quantisation_tables(data)
    if not tables:
        return None
    luma = tables[0]
    chroma = tables[1] if len(tables) > 1 else luma
    best_q: Optional[int] = None
    best_distance = None
    for quality in range(1, 101):
        std_luma, std_chroma = _standard_tables(quality)
        distance = sum(abs(a - b) for a, b in zip(luma, std_luma))
        distance += sum(abs(a - b) for a, b in zip(chroma, std_chroma))
        if best_distance is None or distance < best_distance:
            best_q, best_distance = quality, distance
    if best_q is None or best_distance > _MATCH_TOLERANCE:
        return None
    return best_q


def source_encoding_profile(data: bytes) -> tuple[int, int]:
    """(quality, subsampling) of a source file with safe fallbacks."""
    quality = estimate_source_quality(data)
    if quality is None:
        quality = FALLBACK_QUALITY
    subsampling = read_subsampling(data)
    if subsampling is None:
        subsampling = FALLBACK_SUBSAMPLING
    return quality, subsampling
