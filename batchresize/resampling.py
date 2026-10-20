"""Resampling filters and CPU implementations for batchresize.

Supported filters:
- lanczos3: 3-lobe Lanczos kernel, high-detail standard for photography.
- mitchell: Mitchell-Netravali kernel (B=1/3, C=1/3), smooth cubic resampling with reduced ringing.
- area: Exact fractional area averaging for downscaling (upscaling rejected).
- bilinear: Linear interpolation with anti-aliasing downscaling support.
- nearest: Nearest-neighbour point sampling for pixel art and diagrams.
- box: Existing box filter (unit support).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from PIL import Image

from .geometry import Plan
from .validation import dimensions

SUPPORTED_FILTERS = ("lanczos3", "mitchell", "area", "bilinear", "nearest", "box")

PIL_FILTER_MAP = {
    "lanczos3": Image.Resampling.LANCZOS,
    "bilinear": Image.Resampling.BILINEAR,
    "box": Image.Resampling.BOX,
    "nearest": Image.Resampling.NEAREST,
}


def mitchell_kernel(dist: np.ndarray) -> np.ndarray:
    """Mitchell-Netravali kernel with B=1/3, C=1/3."""
    x = np.abs(dist)
    out = np.zeros_like(x, dtype=np.float32)
    m1 = x < 1.0
    m2 = (x >= 1.0) & (x < 2.0)
    # B=1/3, C=1/3:
    # |x| < 1: 1/6 * (7*|x|^3 - 12*|x|^2 + 16/3)
    out[m1] = (7.0 * x[m1] ** 3 - 12.0 * x[m1] ** 2 + 16.0 / 3.0) / 6.0
    # 1 <= |x| < 2: 1/6 * (-7/3*|x|^3 + 12*|x|^2 - 20*|x| + 32/3)
    out[m2] = (-7.0 / 3.0 * x[m2] ** 3 + 12.0 * x[m2] ** 2 - 20.0 * x[m2] + 32.0 / 3.0) / 6.0
    return out


def build_separable_weights(src_len: int, dst_len: int, scale: float, offset: float,
                            filt: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1D separable resampling indices and normalized weights.

    Coordinates:
    - Target pixel centers are at (j + 0.5) / scale - 0.5 + offset.
    - Source pixels cover [i - 0.5, i + 0.5].
    - Filter kernel is stretched when downscaling (scale < 1.0) for anti-aliasing.
    """
    if filt not in SUPPORTED_FILTERS:
        raise ValueError(f"unknown filter {filt!r}")

    if filt in ("box", "area"):
        if filt == "area" and scale > 1.0:
            raise ValueError(
                "area filter is only supported for downscaling (scale <= 1.0); "
                "use bilinear, lanczos3, or mitchell for upscaling"
            )
        support = 0.5 / min(scale, 1.0)
    elif filt == "bilinear":
        support = 1.0 / min(scale, 1.0)
    elif filt == "mitchell":
        support = 2.0 / min(scale, 1.0)
    elif filt == "lanczos3":
        support = 3.0 / min(scale, 1.0)
    elif filt == "nearest":
        support = 0.5
    else:
        support = 1.0

    taps = max(1, int(math.ceil(support * 2.0)) + 1)
    centers = (np.arange(dst_len, dtype=np.float64) + 0.5) / scale - 0.5 + offset
    first = np.floor(centers - support + 0.5).astype(np.int64)
    idx = first[:, None] + np.arange(taps, dtype=np.int64)[None, :]
    dist = centers[:, None] - idx

    if filt == "area":
        half = 0.5 / scale
        left = centers[:, None] - half
        right = centers[:, None] + half
        w = np.maximum(0.0, np.minimum(idx + 0.5, right) - np.maximum(idx - 0.5, left))
    elif filt == "box":
        w = (np.abs(dist * min(scale, 1.0)) <= 0.5).astype(np.float32)
    elif filt == "bilinear":
        d = np.abs(dist * min(scale, 1.0))
        w = np.clip(1.0 - d, 0.0, None).astype(np.float32)
    elif filt == "mitchell":
        d = dist * min(scale, 1.0)
        w = mitchell_kernel(d)
    elif filt == "lanczos3":
        d = np.abs(dist * min(scale, 1.0))
        w = np.zeros_like(d, dtype=np.float32)
        small = d < 3.0
        x = d[small]
        w[small] = np.sinc(x) * np.sinc(x / 3.0)
    elif filt == "nearest":
        # Nearest neighbor picks single closest pixel index: floor(center + 0.5)
        nearest_idx = np.clip(np.floor(centers + 0.5).astype(np.int64), 0, src_len - 1)
        idx = nearest_idx[:, None]
        w = np.ones((dst_len, 1), dtype=np.float32)
        return idx.astype(np.int32), w
    else:
        w = np.ones_like(dist, dtype=np.float32)

    idx = np.clip(idx, 0, src_len - 1).astype(np.int32)
    w = w.astype(np.float32)
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    w /= s
    return idx, w


def separable_resample(img: np.ndarray, plan: Plan, filter_name: str) -> np.ndarray:
    """CPU reference 2-pass separable resampling without intermediate uint8 rounding."""
    src_h, src_w, channels = img.shape
    scale_x = plan.out_w / plan.cw
    scale_y = plan.out_h / plan.ch

    idx_x, w_x = build_separable_weights(src_w, plan.out_w, scale_x, plan.x0, filter_name)
    idx_y, w_y = build_separable_weights(src_h, plan.out_h, scale_y, plan.y0, filter_name)

    # Horizontal pass: accumulate into float32 intermediate buffer
    inter = np.zeros((src_h, plan.out_w, channels), dtype=np.float32)
    for k in range(w_x.shape[1]):
        inter += img[:, idx_x[:, k], :].astype(np.float32) * w_x[None, :, k, None]

    # Vertical pass: accumulate into float32 output buffer
    out = np.zeros((plan.out_h, plan.out_w, channels), dtype=np.float32)
    for k in range(w_y.shape[1]):
        out += inter[idx_y[:, k], :, :] * w_y[:, k, None, None]

    return np.clip(out + 0.5, 0.0, 255.0).astype(np.uint8)


def separable_resample_float(img: np.ndarray, plan: Plan, filter_name: str) -> np.ndarray:
    """CPU reference 2-pass separable resampling on float32 image without uint8 clamping."""
    src_h, src_w, channels = img.shape
    scale_x = plan.out_w / plan.cw
    scale_y = plan.out_h / plan.ch

    idx_x, w_x = build_separable_weights(src_w, plan.out_w, scale_x, plan.x0, filter_name)
    idx_y, w_y = build_separable_weights(src_h, plan.out_h, scale_y, plan.y0, filter_name)

    inter = np.zeros((src_h, plan.out_w, channels), dtype=np.float32)
    img_f = img.astype(np.float32)
    for k in range(w_x.shape[1]):
        inter += img_f[:, idx_x[:, k], :] * w_x[None, :, k, None]

    out = np.zeros((plan.out_h, plan.out_w, channels), dtype=np.float32)
    for k in range(w_y.shape[1]):
        out += inter[idx_y[:, k], :, :] * w_y[:, k, None, None]

    return out


def resample_cpu(img: np.ndarray, plan: Plan, filter_name: str) -> np.ndarray:
    """Resample a BGR uint8 image according to plan.

    Uses Pillow for lanczos3, bilinear, box, nearest to ensure maximum speed and
    byte-compatibility with existing tests; uses separable reference for mitchell
    and area.
    """
    dimensions(plan.out_w, plan.out_h)
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError("resample requires uint8 BGR with three channels")
    if plan.is_identity:
        return np.ascontiguousarray(img)
    if filter_name not in SUPPORTED_FILTERS:
        raise ValueError(f"unknown filter {filter_name!r}")

    if filter_name in PIL_FILTER_MAP:
        rgb = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
        box = (plan.x0, plan.y0, plan.x0 + plan.cw, plan.y0 + plan.ch)
        out = rgb.resize((plan.out_w, plan.out_h), PIL_FILTER_MAP[filter_name], box=box)
        return np.ascontiguousarray(np.asarray(out)[:, :, ::-1])

    # mitchell and area filters:
    return separable_resample(img, plan, filter_name)
