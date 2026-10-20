"""Tests for color management, color policies, linear-light conversions, and alpha operations."""

from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image, ImageCms

from batchresize.color_management import (
    COLOR_POLICY_PRESERVE,
    COLOR_POLICY_SRGB,
    SRGB_ICC_BYTES,
    canonical_color_policy,
    composite_linear_bg,
    decode_and_manage_color,
    get_icc_color_space,
    is_valid_icc,
    linear_to_srgb,
    premultiply_alpha,
    srgb_to_linear,
    unpremultiply_alpha,
)


def test_canonical_color_policy():
    assert canonical_color_policy("preserve") == COLOR_POLICY_PRESERVE
    assert canonical_color_policy("preserve_rgb") == COLOR_POLICY_PRESERVE
    assert canonical_color_policy("PRESERVE_RGB") == COLOR_POLICY_PRESERVE
    assert canonical_color_policy("srgb") == COLOR_POLICY_SRGB
    assert canonical_color_policy("web_srgb") == COLOR_POLICY_SRGB
    assert canonical_color_policy("WEB_SRGB") == COLOR_POLICY_SRGB
    with pytest.raises(ValueError, match="unknown color policy"):
        canonical_color_policy("cmyk_direct")


def test_srgb_linear_roundtrip():
    # Test boundary values
    arr = np.array([0, 1, 10, 128, 200, 255], dtype=np.uint8)
    lin = srgb_to_linear(arr)
    assert lin[0] == 0.0
    assert abs(lin[-1] - 1.0) < 1e-6
    back = linear_to_srgb(lin)
    assert np.all(arr == back)

    # Test all 256 values: maximum difference due to float32 rounding must be <= 1
    all_uint8 = np.arange(256, dtype=np.uint8)
    all_lin = srgb_to_linear(all_uint8)
    all_back = linear_to_srgb(all_lin)
    assert np.all(np.abs(all_uint8.astype(int) - all_back.astype(int)) <= 1)


def test_srgb_icc_profile_validity():
    assert is_valid_icc(SRGB_ICC_BYTES)
    assert get_icc_color_space(SRGB_ICC_BYTES) == "RGB"
    assert not is_valid_icc(b"invalid_icc_garbage")
    assert not is_valid_icc(None)


def test_alpha_premultiply_and_unpremultiply():
    # 4 pixels:
    # 0: fully opaque Red
    # 1: semi-transparent Green (alpha = 0.5)
    # 2: fully transparent Blue (alpha = 0.0)
    # 3: low alpha (alpha = 0.01)
    rgba = np.array([
        [[1.0, 0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0, 0.5]],
        [[0.0, 0.0, 1.0, 0.0]],
        [[1.0, 1.0, 0.0, 0.01]],
    ], dtype=np.float32)

    premul = premultiply_alpha(rgba)
    assert premul[0, 0, 0] == 1.0 and premul[0, 0, 3] == 1.0
    assert abs(premul[1, 0, 1] - 0.5) < 1e-6 and premul[1, 0, 3] == 0.5
    assert premul[2, 0, 2] == 0.0 and premul[2, 0, 3] == 0.0

    # Unpremultiply
    unpremul = unpremultiply_alpha(premul)
    assert np.allclose(unpremul[0, 0, :3], [1.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(unpremul[1, 0, :3], [0.0, 1.0, 0.0], atol=1e-5)
    # Zero alpha pixel does not produce NaN or Inf
    assert not np.any(np.isnan(unpremul))
    assert not np.any(np.isinf(unpremul))
    assert np.allclose(unpremul[2, 0, :3], [0.0, 0.0, 0.0], atol=1e-5)


def test_composite_linear_bg_no_dark_fringes():
    # Semi-transparent pure red pixel (RGB=[1.0, 0.0, 0.0], alpha=0.5 in linear space)
    rgba = np.array([[[1.0, 0.0, 0.0, 0.5]]], dtype=np.float32)
    premul = premultiply_alpha(rgba)

    # Composite on pure white background (255, 255, 255)
    comp_white = composite_linear_bg(premul, (255, 255, 255), is_premultiplied=True)
    # Red channel: 1.0 * 0.5 + 1.0 * 0.5 = 1.0
    # Green/Blue channels: 0.0 * 0.5 + 1.0 * 0.5 = 0.5
    assert abs(comp_white[0, 0, 0] - 1.0) < 1e-5
    assert abs(comp_white[0, 0, 1] - 0.5) < 1e-5
    assert abs(comp_white[0, 0, 2] - 0.5) < 1e-5

    # Composite on pure black background (0, 0, 0)
    comp_black = composite_linear_bg(premul, (0, 0, 0), is_premultiplied=True)
    assert abs(comp_black[0, 0, 0] - 0.5) < 1e-5
    assert abs(comp_black[0, 0, 1] - 0.0) < 1e-5
    assert abs(comp_black[0, 0, 2] - 0.0) < 1e-5

    # Fully transparent pixel
    rgba_trans = np.array([[[0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    comp_trans_white = composite_linear_bg(rgba_trans, (255, 255, 255), is_premultiplied=True)
    assert np.allclose(comp_trans_white[0, 0], [1.0, 1.0, 1.0], atol=1e-5)


def test_decode_and_manage_color_srgb_policy():
    # Create image with RGB and sRGB profile
    buf = io.BytesIO()
    im = Image.new("RGB", (32, 32), (120, 60, 200))
    im.save(buf, "JPEG", icc_profile=SRGB_ICC_BYTES)
    data = buf.getvalue()

    arr, has_alpha, out_icc = decode_and_manage_color(data, policy="web_srgb", keep_icc=True)
    assert arr.shape == (32, 32, 3)
    assert not has_alpha
    assert out_icc == SRGB_ICC_BYTES

    # When keep_icc is False
    _, _, out_icc_no = decode_and_manage_color(data, policy="web_srgb", keep_icc=False)
    assert out_icc_no is None


def test_decode_and_manage_color_preserve_rgb():
    # Plain RGB without ICC profile
    buf = io.BytesIO()
    im = Image.new("RGB", (20, 20), (50, 100, 150))
    im.save(buf, "PNG")
    data = buf.getvalue()

    arr, has_alpha, out_icc = decode_and_manage_color(data, policy="preserve_rgb", keep_icc=True)
    assert arr.shape == (20, 20, 3)
    assert not has_alpha
    # In preserve_rgb, if source had no ICC, we do not attach a fake profile
    assert out_icc is None


def test_decode_and_manage_color_cmyk_converts_to_srgb():
    buf = io.BytesIO()
    im = Image.new("CMYK", (24, 24), (0, 100, 100, 0))  # red-ish in CMYK
    im.save(buf, "JPEG")
    data = buf.getvalue()

    # Under preserve_rgb: non-RGB must be converted to RGB and given sRGB profile
    arr, has_alpha, out_icc = decode_and_manage_color(data, policy="preserve_rgb", keep_icc=True)
    assert arr.shape == (24, 24, 3)
    assert not has_alpha
    assert out_icc == SRGB_ICC_BYTES

    # Under web_srgb: same
    arr2, has_alpha2, out_icc2 = decode_and_manage_color(data, policy="web_srgb", keep_icc=True)
    assert arr2.shape == (24, 24, 3)
    assert not has_alpha2
    assert out_icc2 == SRGB_ICC_BYTES


def test_decode_and_manage_color_corrupt_icc_rejected():
    buf = io.BytesIO()
    im = Image.new("RGB", (16, 16), (10, 20, 30))
    im.save(buf, "JPEG", icc_profile=b"corrupted_icc_bytes_that_fail")
    data = buf.getvalue()

    with pytest.raises(ValueError, match="cannot parse source ICC profile safely"):
        decode_and_manage_color(data, policy="web_srgb", keep_icc=True)

    with pytest.raises(ValueError, match="cannot parse source ICC profile safely"):
        decode_and_manage_color(data, policy="preserve_rgb", keep_icc=True)
