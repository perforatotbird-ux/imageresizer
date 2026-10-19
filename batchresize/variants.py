"""Explicit output variants for one source image."""

from __future__ import annotations

import re
from dataclasses import dataclass

VARIANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
SIZING_MODES = ("resolution", "percent", "megapixels", "longest", "shortest")


class VariantError(ValueError):
    pass


def _fail(path: str, message: str) -> None:
    raise VariantError(f"{path}: {message}")


def validate_id(value) -> str:
    if not isinstance(value, str) or not VARIANT_ID_RE.fullmatch(value):
        _fail("variant.id", "must match [a-z0-9][a-z0-9._-]{0,63}")
    return value


def _integer(value, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "must be an integer")
    if not minimum <= value <= maximum:
        _fail(path, f"must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class OutputVariant:
    id: str
    name: str = ""
    max_w: int = 1024
    max_h: int = 1024
    mode: str = "contain"
    quality: int = 80
    subsampling: str = "auto"
    allow_upscale: bool = False
    sizing: str = "resolution"
    sizing_value: float = 0.0
    bg_color: str = "#FFFFFF"

    def __post_init__(self):
        validate_id(self.id)
        if not self.name:
            object.__setattr__(self, "name", self.id)
        if self.mode not in ("contain", "cover"):
            _fail("variant.mode", "must be contain or cover")
        _integer(self.quality, "variant.quality", 1, 100)
        if self.subsampling not in ("auto", "444", "422", "420"):
            _fail("variant.subsampling", "must be auto, 444, 422 or 420")
        if self.sizing not in SIZING_MODES:
            _fail("variant.sizing", "unknown sizing mode")
        if (not isinstance(self.bg_color, str) or
                not re.fullmatch(r"#[0-9A-Fa-f]{6}", self.bg_color)):
            _fail("variant.bg_color", "must be #RRGGBB")
        if self.sizing == "resolution":
            _integer(self.max_w, "variant.max_w", 1, 100_000)
            _integer(self.max_h, "variant.max_h", 1, 100_000)
        else:
            if not isinstance(self.sizing_value, (int, float)) or isinstance(
                    self.sizing_value, bool) or self.sizing_value <= 0:
                _fail("variant.sizing_value", "must be positive")

    def describe(self) -> str:
        if self.sizing == "resolution":
            size = f"{self.max_w}x{self.max_h}"
        else:
            size = f"{self.sizing}={self.sizing_value:g}"
        return (f"{self.id}: {size} {self.mode}, q={self.quality}, "
                f"chroma={self.subsampling}")

    def to_profile_dict(self) -> dict:
        if self.sizing == "resolution":
            size = {"mode": "resolution", "width": self.max_w,
                    "height": self.max_h, "value": None}
        else:
            size = {"mode": self.sizing, "width": None, "height": None,
                    "value": self.sizing_value}
        return {
            "id": self.id,
            "name": self.name,
            "size": size,
            "fit": {"mode": self.mode, "allow_upscale": self.allow_upscale,
                    "crop_position": "center"},
            "jpeg": {"quality": self.quality,
                     "subsampling": self.subsampling,
                     "bg_color": self.bg_color},
        }

    @classmethod
    def from_profile_dict(cls, data: dict) -> "OutputVariant":
        if not isinstance(data, dict):
            _fail("variant", "must be an object")
        size = data.get("size")
        fit = data.get("fit") or {}
        jpeg = data.get("jpeg") or {}
        if not isinstance(size, dict) or not isinstance(fit, dict) or not isinstance(jpeg, dict):
            _fail("variant", "size, fit and jpeg must be objects")
        mode = size.get("mode", "resolution")
        if mode == "resolution":
            max_w = size.get("width")
            max_h = size.get("height")
            sizing_value = 0.0
        else:
            max_w = max_h = 10_000
            sizing_value = float(size.get("value") or 0.0)
        return cls(
            id=data.get("id"),
            name=data.get("name") or data.get("id") or "",
            max_w=max_w,
            max_h=max_h,
            mode=fit.get("mode", "contain"),
            quality=jpeg.get("quality", 80),
            subsampling=jpeg.get("subsampling", "auto"),
            allow_upscale=bool(fit.get("allow_upscale", False)),
            sizing=mode,
            sizing_value=sizing_value,
            bg_color=jpeg.get("bg_color", "#FFFFFF"),
        )


def _tail_mode_quality(parts, mode_index, quality_index, sub_index):
    mode = parts[mode_index] if len(parts) > mode_index and parts[mode_index] else "contain"
    quality = 80
    if len(parts) > quality_index and parts[quality_index]:
        try:
            quality = int(parts[quality_index])
        except ValueError:
            _fail("variant.quality", "must be an integer")
    subsampling = parts[sub_index] if len(parts) > sub_index and parts[sub_index] else "auto"
    return mode, quality, subsampling


def parse_variant_spec(text: str) -> OutputVariant:
    """Parse ``thumb:320x320:cover:80:420``. Also accepts a bare number
    (``web:1280`` -> best ``longest:1280``) or a sizing mode plus value
    (``longest:320``, ``shortest:200``, ``percent:50``, ``megapixels:2``).
    Mode, quality and subsampling are optional, in that order after the value.
    """
    if not isinstance(text, str):
        _fail("variant", "must be a string")
    parts = text.strip().split(":")
    if len(parts) < 2:
        _fail("variant", "expected ID:SIZE[:MODE[:QUALITY[:SUBSAMPLING]]]")
    variant_id = parts[0]
    token = parts[1]
    if re.fullmatch(r"\d+[xX]\d+", token):
        w_text, h_text = token.lower().split("x", 1)
        mode, quality, subsampling = _tail_mode_quality(parts, 2, 3, 4)
        return OutputVariant(id=variant_id, max_w=int(w_text), max_h=int(h_text),
                             mode=mode, quality=quality, subsampling=subsampling,
                             sizing="resolution")
    if re.fullmatch(r"\d+", token):
        dim = int(token)
        mode, quality, subsampling = _tail_mode_quality(parts, 2, 3, 4)
        return OutputVariant(id=variant_id, max_w=dim, max_h=dim,
                             mode=mode, quality=quality, subsampling=subsampling,
                             sizing="resolution")
    if token in SIZING_MODES:
        if len(parts) < 3:
            _fail("variant.size", "expected SIZING:VALUE")
        try:
            value = float(parts[2])
        except ValueError:
            _fail("variant.size", "value must be a number")
        mode, quality, subsampling = _tail_mode_quality(parts, 3, 4, 5)
        return OutputVariant(id=variant_id, max_w=10_000, max_h=10_000,
                             mode=mode, quality=quality, subsampling=subsampling,
                             sizing=token, sizing_value=value)
    try:
        value = float(token)
    except ValueError:
        _fail("variant.size", "expected WIDTHxHEIGHT or SIZING:VALUE")
    mode, quality, subsampling = _tail_mode_quality(parts, 2, 3, 4)
    return OutputVariant(id=variant_id, max_w=10_000, max_h=10_000,
                         mode=mode, quality=quality, subsampling=subsampling,
                         sizing="longest", sizing_value=value)
