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
import os
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

RESAMPLE_ENGINE = os.environ.get("BATCHRESIZE_RESAMPLE_ENGINE", "auto").strip().lower()

_CV_FILTER_MAP = {
    "lanczos3": "INTER_LANCZOS4",
    "mitchell": "INTER_CUBIC",
    "area": "INTER_AREA",
    "bilinear": "INTER_LINEAR",
    "box": "INTER_AREA",
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


def _pixel_region(plan: Plan, src_h: int, src_w: int):
    """Whole-pixel source box for a plan (sub-pixel crop is rounded).

    Plans may sit a fraction of a pixel outside the source because of float
    geometry. Such regions are clamped to the source; anything genuinely
    outside falls back to the reference engine.
    """
    raw_x0, raw_y0 = plan.x0, plan.y0
    raw_x1, raw_y1 = plan.x0 + plan.cw, plan.y0 + plan.ch
    if (raw_x0 < -0.5 or raw_y0 < -0.5
            or raw_x1 > src_w + 0.5 or raw_y1 > src_h + 0.5):
        return None
    x0 = max(0, min(src_w - 1, int(round(raw_x0))))
    y0 = max(0, min(src_h - 1, int(round(raw_y0))))
    x1 = max(x0 + 1, min(src_w, int(round(raw_x1))))
    y1 = max(y0 + 1, min(src_h, int(round(raw_y1))))
    return y0, y1, x0, x1


def _nearest_resample(img: np.ndarray, plan: Plan, region) -> np.ndarray:
    """Exact nearest-neighbour gather for the whole-pixel source region."""
    y0, y1, x0, x1 = region
    idx_x, _ = build_separable_weights(x1 - x0, plan.out_w,
                                       plan.out_w / (x1 - x0), 0.0, "nearest")
    idx_y, _ = build_separable_weights(y1 - y0, plan.out_h,
                                       plan.out_h / (y1 - y0), 0.0, "nearest")
    source = img[y0:y1, x0:x1]
    return np.ascontiguousarray(source[idx_y[:, 0][:, None], idx_x[:, 0][None, :]])


def _cv2_resample(img: np.ndarray, plan: Plan, filter_name: str):
    """Fast full-region resampling through OpenCV's specialized SIMD kernels.

    Returns ``None`` when the reference engine was requested explicitly;
    callers then use the separable reference implementation.
    """
    if RESAMPLE_ENGINE in ("reference", "pillow"):
        return None
    region = _pixel_region(plan, img.shape[0], img.shape[1])
    if region is None:
        return None
    y0, y1, x0, x1 = region
    scale_x = plan.out_w / (x1 - x0)
    scale_y = plan.out_h / (y1 - y0)
    if filter_name in ("area", "box") and (scale_x > 1.0 or scale_y > 1.0):
        return None
    if filter_name == "nearest":
        return _nearest_resample(img, plan, region)

    import cv2

    interpolation = getattr(cv2, _CV_FILTER_MAP[filter_name])
    source = img[y0:y1, x0:x1]
    return cv2.resize(source, (plan.out_w, plan.out_h),
                      interpolation=interpolation)


def _pillow_resample(img: np.ndarray, plan: Plan, filter_name: str) -> np.ndarray:
    """Previous Pillow engine, retained for numeric A/B comparison."""
    rgb = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
    box = (plan.x0, plan.y0, plan.x0 + plan.cw, plan.y0 + plan.ch)
    out = rgb.resize((plan.out_w, plan.out_h), PIL_FILTER_MAP[filter_name], box=box)
    return np.ascontiguousarray(np.asarray(out)[:, :, ::-1])


def resample_cpu_float(img: np.ndarray, plan: Plan, filter_name: str) -> np.ndarray:
    """Fast float32 resampling for the linear-light pipeline.

    OpenCV's float kernels operate on the same whole-pixel source region as
    the uint8 path. ``area``/``box`` upscaling and genuinely out-of-bounds
    plans fall back to the separable reference.
    """
    dimensions(plan.out_w, plan.out_h)
    if img.ndim != 3 or img.dtype != np.float32:
        raise ValueError("float resample requires float32 image")
    if plan.is_identity:
        return np.ascontiguousarray(img)
    if filter_name not in SUPPORTED_FILTERS:
        raise ValueError(f"unknown filter {filter_name!r}")
    if RESAMPLE_ENGINE not in ("reference", "pillow"):
        region = _pixel_region(plan, img.shape[0], img.shape[1])
        if region is not None:
            y0, y1, x0, x1 = region
            scale_x = plan.out_w / (x1 - x0)
            scale_y = plan.out_h / (y1 - y0)
            if filter_name not in ("area", "box") or (scale_x <= 1.0 and scale_y <= 1.0):
                if filter_name == "nearest":
                    return _nearest_resample(img, plan, region)
                import cv2

                filter_key = _CV_FILTER_MAP[filter_name]
                # Lanczos4 rings visibly on large linear-light reductions.
                # Area averaging is both closer to the reference result and
                # cheaper when each output pixel covers many source pixels.
                if filter_name in ("lanczos3", "mitchell") and max(scale_x, scale_y) <= 0.5:
                    filter_key = "INTER_AREA"
                source = np.ascontiguousarray(img[y0:y1, x0:x1])
                return cv2.resize(source, (plan.out_w, plan.out_h),
                                  interpolation=getattr(cv2, filter_key))
    return separable_resample_float(img, plan, filter_name)


def resample_cpu(img: np.ndarray, plan: Plan, filter_name: str) -> np.ndarray:
    """Resample a BGR uint8 image according to plan.

    OpenCV SIMD kernels handle the common full-frame (and integer-crop) case;
    the separable float reference remains available via
    ``BATCHRESIZE_RESAMPLE_ENGINE=reference`` and for sub-pixel crops.
    """
    dimensions(plan.out_w, plan.out_h)
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError("resample requires uint8 BGR with three channels")
    if plan.is_identity:
        return np.ascontiguousarray(img)
    if filter_name not in SUPPORTED_FILTERS:
        raise ValueError(f"unknown filter {filter_name!r}")

    if RESAMPLE_ENGINE == "pillow" and filter_name in PIL_FILTER_MAP:
        return _pillow_resample(img, plan, filter_name)
    fast = _cv2_resample(img, plan, filter_name)
    if fast is not None:
        return fast
    return separable_resample(img, plan, filter_name)
