"""Geometry: how a source image maps onto the requested output box."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from .validation import dimensions, positive

ANCHORS: dict[str, tuple[float, float]] = {
    "center": (0.5, 0.5),
    "top": (0.5, 0.0),
    "bottom": (0.5, 1.0),
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
    "top_left": (0.0, 0.0),
    "top_right": (1.0, 0.0),
    "bottom_left": (0.0, 1.0),
    "bottom_right": (1.0, 1.0),
    "top-left": (0.0, 0.0),
    "top-right": (1.0, 0.0),
    "bottom-left": (0.0, 1.0),
    "bottom-right": (1.0, 1.0),
    "north": (0.5, 0.0),
    "south": (0.5, 1.0),
    "west": (0.0, 0.5),
    "east": (1.0, 0.5),
    "n": (0.5, 0.0),
    "s": (0.5, 1.0),
    "w": (0.0, 0.5),
    "e": (1.0, 0.5),
    "north_west": (0.0, 0.0),
    "north_east": (1.0, 0.0),
    "south_west": (0.0, 1.0),
    "south_east": (1.0, 1.0),
    "nw": (0.0, 0.0),
    "ne": (1.0, 0.0),
    "sw": (0.0, 1.0),
    "se": (1.0, 1.0),
}


@dataclass(frozen=True)
class Plan:
    """Resolved geometry for one image.

    The source region is ``[x0, x0+cw) x [y0, y0+ch)`` in *display*
    (EXIF-oriented) coordinates, resampled to ``out_w x out_h``.
    If ``canvas_w > 0``, the resampled image is placed onto a canvas of size
    ``canvas_w x canvas_h`` at offset ``(pad_left, pad_top)``.
    """

    src_w: int
    src_h: int
    out_w: int
    out_h: int
    x0: float
    y0: float
    cw: float
    ch: float
    canvas_w: int = 0
    canvas_h: int = 0
    pad_left: int = 0
    pad_top: int = 0

    @property
    def target_w(self) -> int:
        return self.canvas_w if self.canvas_w > 0 else self.out_w

    @property
    def target_h(self) -> int:
        return self.canvas_h if self.canvas_h > 0 else self.out_h

    @property
    def is_pad(self) -> bool:
        return (self.canvas_w > 0 and self.canvas_w != self.out_w) or (
            self.canvas_h > 0 and self.canvas_h != self.out_h
        )

    @property
    def scale_x(self) -> float:
        return self.out_w / self.cw

    @property
    def scale_y(self) -> float:
        return self.out_h / self.ch

    @property
    def is_crop(self) -> bool:
        return self.cw < self.src_w - 1e-6 or self.ch < self.src_h - 1e-6

    @property
    def is_identity(self) -> bool:
        return (
            self.out_w == self.src_w
            and self.out_h == self.src_h
            and not self.is_crop
            and not self.is_pad
        )


def parse_crop_focus(value) -> tuple[float, float]:
    """Parse normalized crop focus (x, y) with 0.0 <= x, y <= 1.0."""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 2:
            raise ValueError("crop focus must be 'X,Y' (e.g. '0.5,0.2')")
        try:
            fx, fy = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise ValueError(f"invalid crop focus coordinates {value!r}") from exc
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        fx, fy = float(value[0]), float(value[1])
    else:
        raise ValueError("crop focus must be a 2-tuple or 'X,Y' string")

    if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
        raise ValueError(f"crop focus coordinates must be in [0.0, 1.0], got ({fx}, {fy})")
    return fx, fy


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def plan_contain(src_w: int, src_h: int, max_w: int, max_h: int,
                 allow_upscale: bool = False) -> Plan:
    """Fit inside the box, keep aspect, never crop."""
    dimensions(src_w, src_h)
    dimensions(max_w, max_h)
    scale = min(max_w / src_w, max_h / src_h)
    if scale > 1.0 and not allow_upscale:
        scale = 1.0
    out_w = max(1, int(round(src_w * scale)))
    out_h = max(1, int(round(src_h * scale)))
    # rounding can push us one pixel over the requested bound
    if out_w > max_w and scale <= 1.0:
        out_w = max_w
    if out_h > max_h and scale <= 1.0:
        out_h = max_h
    return Plan(src_w, src_h, out_w, out_h, 0.0, 0.0, float(src_w), float(src_h))


def plan_cover(src_w: int, src_h: int, max_w: int, max_h: int,
               allow_upscale: bool = True,
               anchor: str = "center",
               focus: Optional[Tuple[float, float]] = None) -> Plan:
    """Fill the box, keep aspect, crop the overflow according to anchor or focus point.

    Coordinates are in display (post EXIF orientation) space.
    ``cover`` semantically requires filling the box, so small images are
    upscaled unless ``allow_upscale`` is explicitly disabled (then the
    result is the largest crop that does not upscale).
    """
    dimensions(src_w, src_h)
    dimensions(max_w, max_h)
    scale = max(max_w / src_w, max_h / src_h)
    if scale > 1.0 and not allow_upscale:
        scale = 1.0
    cw = min(float(src_w), max_w / scale)
    ch = min(float(src_h), max_h / scale)
    out_w = max(1, int(round(cw * scale)))
    out_h = max(1, int(round(ch * scale)))
    if not allow_upscale:
        out_w = min(out_w, max_w)
        out_h = min(out_h, max_h)

    # Determine normalized point of interest (fx, fy)
    if focus is not None:
        fx, fy = parse_crop_focus(focus)
    else:
        anchor_key = str(anchor).lower().strip()
        if anchor_key not in ANCHORS:
            raise ValueError(f"unknown crop anchor {anchor!r} (valid: {', '.join(sorted(ANCHORS))})")
        fx, fy = ANCHORS[anchor_key]

    if src_w > cw:
        x0 = _clamp(fx * src_w - cw / 2.0, 0.0, src_w - cw)
    else:
        x0 = 0.0

    if src_h > ch:
        y0 = _clamp(fy * src_h - ch / 2.0, 0.0, src_h - ch)
    else:
        y0 = 0.0

    return Plan(src_w, src_h, out_w, out_h, x0, y0, cw, ch)


def plan_pad(src_w: int, src_h: int, target_w: int, target_h: int,
             allow_upscale: bool = False,
             anchor: str = "center") -> Plan:
    """Fit inside target box without cropping, padding canvas to exact target dimensions."""
    dimensions(src_w, src_h)
    dimensions(target_w, target_h)
    scale = min(target_w / src_w, target_h / src_h)
    if scale > 1.0 and not allow_upscale:
        scale = 1.0
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))
    if scaled_w > target_w and scale <= 1.0:
        scaled_w = target_w
    if scaled_h > target_h and scale <= 1.0:
        scaled_h = target_h

    canvas_w = target_w
    canvas_h = target_h
    pad_w = canvas_w - scaled_w
    pad_h = canvas_h - scaled_h

    anchor_key = str(anchor).lower().strip()
    if anchor_key not in ANCHORS:
        raise ValueError(f"unknown crop anchor {anchor!r}")
    ax, ay = ANCHORS[anchor_key]

    pad_left = int(round(pad_w * ax))
    pad_top = int(round(pad_h * ay))
    pad_left = max(0, min(pad_left, pad_w))
    pad_top = max(0, min(pad_top, pad_h))

    return Plan(
        src_w=src_w,
        src_h=src_h,
        out_w=scaled_w,
        out_h=scaled_h,
        x0=0.0,
        y0=0.0,
        cw=float(src_w),
        ch=float(src_h),
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        pad_left=pad_left,
        pad_top=pad_top,
    )


def apply_pad(img: np.ndarray, plan: Plan, bg_color: tuple[int, int, int]) -> np.ndarray:
    """Place resampled image onto padded canvas according to plan."""
    if not plan.is_pad:
        return img
    canvas_h, canvas_w = plan.canvas_h, plan.canvas_w
    channels = img.shape[2]
    if img.dtype == np.uint8:
        # BGR uint8
        bg = np.array([bg_color[2], bg_color[1], bg_color[0]], dtype=np.uint8)
        out = np.empty((canvas_h, canvas_w, channels), dtype=np.uint8)
        out[:, :] = bg
    else:
        # float32 linear
        from .color_management import SRGB_TO_LINEAR_LUT
        bg_u8 = np.array([bg_color[0], bg_color[1], bg_color[2]], dtype=np.uint8)
        bg_lin = SRGB_TO_LINEAR_LUT[bg_u8]
        out = np.empty((canvas_h, canvas_w, channels), dtype=np.float32)
        out[:, :] = bg_lin

    out[plan.pad_top : plan.pad_top + plan.out_h, plan.pad_left : plan.pad_left + plan.out_w] = img
    return out


def make_plan(src_w: int, src_h: int, max_w: int, max_h: int,
              mode: str = "contain", allow_upscale: bool = False,
              anchor: str = "center",
              focus: Optional[Tuple[float, float]] = None) -> Plan:
    if mode == "contain":
        return plan_contain(src_w, src_h, max_w, max_h, allow_upscale)
    if mode == "cover":
        return plan_cover(src_w, src_h, max_w, max_h, allow_upscale, anchor=anchor, focus=focus)
    if mode == "pad":
        return plan_pad(src_w, src_h, max_w, max_h, allow_upscale, anchor=anchor)
    raise ValueError(f"unknown mode {mode!r}")


def compute_box(src_w: int, src_h: int, sizing: str, max_w: int, max_h: int,
                value: float = 0.0) -> tuple[int, int]:
    """Translate a downscaling mode into a concrete target box.

    Modes mirror XL Converter's downscaling tab:

    * ``resolution`` - clamp into ``max_w x max_h``
    * ``percent``    - scale both dimensions by ``value`` percent
    * ``megapixels`` - scale down to ``value`` megapixels
    * ``longest``    - make the longest side ``value`` pixels
    * ``shortest``   - make the shortest side ``value`` pixels
    """
    dimensions(src_w, src_h)
    dimensions(max_w, max_h)
    if sizing in ("", "resolution", None):
        return max_w, max_h
    positive(value, "sizing value")
    dimensions(src_w, src_h)
    dimensions(max_w, max_h)

    if sizing == "percent":
        scale = value / 100.0
    elif sizing == "megapixels":
        target_px = max(1.0, value * 1_000_000.0)
        scale = (target_px / float(src_w * src_h)) ** 0.5
    elif sizing == "longest":
        scale = max(1.0, value) / float(max(src_w, src_h))
    elif sizing == "shortest":
        scale = max(1.0, value) / float(min(src_w, src_h))
    else:
        raise ValueError(f"unknown sizing mode {sizing!r}")

    box_w = max(1, int(round(src_w * scale)))
    box_h = max(1, int(round(src_h * scale)))
    dimensions(box_w, box_h)
    return box_w, box_h
