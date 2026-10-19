"""Output sharpening (Unsharp Mask) on luminance.

Performs Unsharp Mask on the Y (luminance) channel in YCrCb color space to
enhance edge sharpness without introducing color fringing (chromatic halos).
Thresholding prevents amplification of flat sensor noise, and hard clipping
prevents arithmetic overflow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

SHARPEN_PRESETS = {
    "off": {"enabled": False, "amount": 0.0, "radius": 0.0, "threshold": 0},
    "weak": {"enabled": True, "amount": 0.6, "radius": 0.8, "threshold": 2},
    "standard": {"enabled": True, "amount": 1.2, "radius": 1.0, "threshold": 2},
}


class SharpenError(ValueError):
    pass


@dataclass(frozen=True)
class SharpenConfig:
    enabled: bool = False
    preset: str = "off"
    amount: float = 0.0
    radius: float = 0.0
    threshold: int = 0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise SharpenError("sharpen.enabled must be a boolean")
        if self.preset not in ("off", "weak", "standard", "custom"):
            raise SharpenError(f"unknown sharpen preset {self.preset!r}")
        if not isinstance(self.amount, (int, float)) or isinstance(self.amount, bool) or self.amount < 0.0 or self.amount > 10.0:
            raise SharpenError("sharpen.amount must be between 0.0 and 10.0")
        if not isinstance(self.radius, (int, float)) or isinstance(self.radius, bool) or self.radius < 0.0 or self.radius > 20.0:
            raise SharpenError("sharpen.radius must be between 0.0 and 20.0")
        if not isinstance(self.threshold, int) or isinstance(self.threshold, bool) or self.threshold < 0 or self.threshold > 255:
            raise SharpenError("sharpen.threshold must be an integer between 0 and 255")

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "preset": self.preset,
            "amount": round(float(self.amount), 3),
            "radius": round(float(self.radius), 3),
            "threshold": int(self.threshold),
        }

    @classmethod
    def from_preset(cls, preset: str) -> "SharpenConfig":
        if preset not in SHARPEN_PRESETS:
            raise SharpenError(f"unknown sharpen preset {preset!r}; choose from {tuple(SHARPEN_PRESETS.keys())}")
        data = SHARPEN_PRESETS[preset]
        return cls(enabled=data["enabled"], preset=preset,
                   amount=data["amount"], radius=data["radius"],
                   threshold=data["threshold"])

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SharpenConfig":
        if not isinstance(data, (dict, Mapping)):
            raise SharpenError("sharpen config must be a dictionary")
        preset = str(data.get("preset", "custom"))
        if preset in SHARPEN_PRESETS and not any(k in data for k in ("amount", "radius", "threshold")):
            return cls.from_preset(preset)
        enabled = bool(data.get("enabled", False))
        amount = float(data.get("amount", 1.0 if enabled else 0.0))
        radius = float(data.get("radius", 1.0 if enabled else 0.0))
        threshold = int(data.get("threshold", 0))
        return cls(enabled=enabled, preset=preset if preset in ("off", "weak", "standard") else "custom",
                   amount=amount, radius=radius, threshold=threshold)


def parse_sharpen(spec: Any) -> SharpenConfig:
    if spec is None or spec is False or spec == "off":
        return SharpenConfig(enabled=False, preset="off")
    if spec is True:
        return SharpenConfig.from_preset("standard")
    if isinstance(spec, str):
        if spec in SHARPEN_PRESETS:
            return SharpenConfig.from_preset(spec)
        raise SharpenError(f"unknown sharpen preset {spec!r}; choose from {tuple(SHARPEN_PRESETS.keys())}")
    if isinstance(spec, SharpenConfig):
        return spec
    if isinstance(spec, (dict, Mapping)):
        return SharpenConfig.from_dict(spec)
    raise SharpenError(f"invalid sharpen specification {spec!r}")


def apply_sharpen(img: np.ndarray, config: SharpenConfig) -> np.ndarray:
    """Apply Unsharp Mask on the luminance (Y) channel of a BGR uint8 image.

    Returns the sharpened BGR uint8 image (or original if disabled).
    """
    if not config.enabled or config.amount <= 0.0 or config.radius <= 0.0:
        return img
    if cv2 is None:
        raise RuntimeError("opencv is required for output sharpening")
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError("apply_sharpen requires uint8 BGR image with three channels")

    # Convert to YCrCb (channel 0 is Luminance)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)

    # Compute Gaussian blur kernel size from sigma/radius
    ksize = int(math.ceil(config.radius * 3.0)) * 2 + 1
    blur = cv2.GaussianBlur(y, (ksize, ksize), config.radius, config.radius)

    # High-frequency difference
    diff = y - blur

    # Tonal difference threshold: suppress noise amplification in flat areas
    if config.threshold > 0:
        diff = np.where(np.abs(diff) >= float(config.threshold), diff, 0.0)

    # Apply sharpening and clip strictly to [0, 255]
    sharp_y = np.clip(y + config.amount * diff + 0.5, 0.0, 255.0).astype(np.uint8)

    # Recombine with untouched chrominance channels (zero color fringing)
    ycrcb[:, :, 0] = sharp_y
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
