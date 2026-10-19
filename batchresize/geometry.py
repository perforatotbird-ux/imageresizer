"""Geometry: how a source image maps onto the requested output box."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    """Resolved geometry for one image.

    The source region is ``[x0, x0+cw) x [y0, y0+ch)`` in *display*
    (EXIF-oriented) coordinates, resampled to ``out_w x out_h``.
    """

    src_w: int
    src_h: int
    out_w: int
    out_h: int
    x0: float
    y0: float
    cw: float
    ch: float

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
        )


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def plan_contain(src_w: int, src_h: int, max_w: int, max_h: int,
                 allow_upscale: bool = False) -> Plan:
    """Fit inside the box, keep aspect, never crop."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError("empty image")
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
               allow_upscale: bool = True) -> Plan:
    """Fill the box, keep aspect, crop the centre overflow.

    ``cover`` semantically requires filling the box, so small images are
    upscaled unless ``allow_upscale`` is explicitly disabled (then the
    result is the largest centred crop that does not upscale).
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError("empty image")
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
    x0 = _clamp((src_w - cw) / 2.0, 0.0, src_w - cw)
    y0 = _clamp((src_h - ch) / 2.0, 0.0, src_h - ch)
    return Plan(src_w, src_h, out_w, out_h, x0, y0, cw, ch)


def make_plan(src_w: int, src_h: int, max_w: int, max_h: int,
              mode: str = "contain", allow_upscale: bool = False) -> Plan:
    if mode == "contain":
        return plan_contain(src_w, src_h, max_w, max_h, allow_upscale)
    if mode == "cover":
        return plan_cover(src_w, src_h, max_w, max_h, allow_upscale)
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
    if sizing in ("", "resolution", None):
        return max_w, max_h
    if src_w <= 0 or src_h <= 0:
        raise ValueError("empty image")

    if sizing == "percent":
        scale = max(0.001, value / 100.0)
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
    return box_w, box_h
