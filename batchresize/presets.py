"""Presets for output size and JPEG quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- size presets -----------------------------------------------------------
# Values are interpreted as the maximum width/height for `contain`
# and as the desired target box for `cover` (smaller if upscaling is off).
SIZE_PRESETS: dict[str, tuple[int, int]] = {
    "thumb_256": (256, 256),
    "small_512": (512, 512),
    "web_1024": (1024, 1024),
    "large_1600": (1600, 1600),
}

# --- JPEG quality presets ---------------------------------------------------
# Values follow XL Converter's JPEG guidance:
#   95 visually lossless / 90 high / 80 balanced / 60 space-saving.
QUALITY_PRESETS: dict[str, int] = {
    "maximum": 95,
    "high": 90,
    "balanced": 80,
    "small": 60,
    # Legacy names kept for backwards compatibility with older scripts.
    "very_high": 95,
    "compact": 65,
}

# Compatibility aliases (resolved before lookup).
QUALITY_ALIASES: dict[str, str] = {}

# Recompression strength presets (plain-language aliases).
COMPRESSION_PRESETS: dict[str, int] = {
    "light": 90,
    "medium": 80,
    "strong": 65,
    "extreme": 50,
}

ALL_QUALITY_PRESETS: dict[str, int] = {**QUALITY_PRESETS, **COMPRESSION_PRESETS}

# Human-readable labels for the UI.
QUALITY_LABELS: dict[str, str] = {
    "maximum": "Maximum (q=95) - visually lossless, large files",
    "high": "High (q=90) - high quality, large files",
    "balanced": "Balanced (q=80) - reasonable quality and size",
    "small": "Small (q=60) - space-saving, visible loss",
    "light": "Light (q=90) - almost no visible loss",
    "medium": "Medium (q=80) - recommended",
    "strong": "Strong (q=65) - noticeably smaller",
    "extreme": "Extreme (q=50) - maximum compression",
}

# XL Converter style: default chroma depends on quality and source.
SUBSAMPLING_LABELS: dict[str, str] = {
    "auto": "Auto - 4:2:0 to q85, 4:2:2 to q94, then 4:4:4",
    "444": "4:4:4 - full colour, highest quality and size",
    "422": "4:2:2 - less colour, significant space saving",
    "420": "4:2:0 - colours may appear washed out",
}

# Chroma subsampling: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0
SUBSAMPLING_NAMES = {
    "auto": None,
    "444": 0,
    "422": 1,
    "420": 2,
}


def auto_subsampling(quality: int) -> int:
    """Pick chroma subsampling the way mozjpeg/libjpeg-turbo tuning does.

    * q <= 85  -> 4:2:0 (smallest files, visible colour detail loss)
    * q <= 94  -> 4:2:2 (middle ground)
    * q >= 95  -> 4:4:4 (no chroma downsampling, largest files)
    """
    if quality <= 85:
        return 2
    if quality <= 94:
        return 1
    return 0


@dataclass(frozen=True)
class OutputSpec:
    """Fully resolved output configuration."""

    max_w: int
    max_h: int
    mode: str          # "contain" | "cover"
    quality: int
    subsampling: int   # 0,1,2
    allow_upscale: bool = False

    def describe(self) -> str:
        sub = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}[self.subsampling]
        return (
            f"{self.max_w}x{self.max_h} {self.mode}, JPEG q={self.quality}, "
            f"chroma={sub}, upscale={'on' if self.allow_upscale else 'off'}"
        )


def resolve_size(
    size_preset: Optional[str],
    width: Optional[int],
    height: Optional[int],
) -> tuple[int, int]:
    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("--width and --height must be given together")
        if width <= 0 or height <= 0:
            raise ValueError("--width/--height must be positive")
        return int(width), int(height)
    if size_preset is None:
        size_preset = "web_1024"
    if size_preset not in SIZE_PRESETS:
        raise ValueError(
            f"unknown size preset {size_preset!r}; "
            f"choose from {', '.join(SIZE_PRESETS)} or pass --width/--height"
        )
    return SIZE_PRESETS[size_preset]


def resolve_quality(
    quality_preset: Optional[str],
    jpeg_quality: Optional[int],
) -> int:
    if jpeg_quality is not None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("--jpeg-quality must be in 1..100")
        return int(jpeg_quality)
    if quality_preset is None:
        quality_preset = "balanced"
    quality_preset = QUALITY_ALIASES.get(quality_preset, quality_preset)
    if quality_preset not in ALL_QUALITY_PRESETS:
        raise ValueError(
            f"unknown quality preset {quality_preset!r}; "
            f"choose from {', '.join(ALL_QUALITY_PRESETS)} or pass --jpeg-quality"
        )
    return ALL_QUALITY_PRESETS[quality_preset]
