"""Tests for resampling filters: lanczos3, mitchell, area, bilinear, nearest, box."""

import numpy as np
import pytest

from batchresize.geometry import Plan
from batchresize.resampling import (
    SUPPORTED_FILTERS,
    build_separable_weights,
    mitchell_kernel,
    resample_cpu,
    resample_cpu_float,
    separable_resample,
    separable_resample_float,
)


def make_plan(src_w, src_h, out_w, out_h, x0=0.0, y0=0.0, cw=None, ch=None):
    return Plan(
        src_w=src_w,
        src_h=src_h,
        out_w=out_w,
        out_h=out_h,
        x0=x0,
        y0=y0,
        cw=float(cw or src_w),
        ch=float(ch or src_h),
    )


class TestFilterKernels:
    def test_mitchell_exact_values(self):
        # Mitchell-Netravali (B=1/3, C=1/3):
        # k(0) = 8/9, k(1) = 1/18, k(2) = 0
        assert np.isclose(mitchell_kernel(np.array([0.0]))[0], 8.0 / 9.0)
        assert np.isclose(mitchell_kernel(np.array([1.0]))[0], 1.0 / 18.0)
        assert np.isclose(mitchell_kernel(np.array([2.0]))[0], 0.0)
        assert np.isclose(mitchell_kernel(np.array([3.0]))[0], 0.0)

    def test_mitchell_symmetry(self):
        pts = np.linspace(0.0, 2.5, 50)
        assert np.allclose(mitchell_kernel(pts), mitchell_kernel(-pts))

    def test_mitchell_continuity_at_boundaries(self):
        eps = 1e-6
        # Near 1.0
        assert np.isclose(mitchell_kernel(np.array([1.0 - eps]))[0],
                          mitchell_kernel(np.array([1.0 + eps]))[0], atol=1e-5)
        # Near 2.0
        assert np.isclose(mitchell_kernel(np.array([2.0 - eps]))[0],
                          mitchell_kernel(np.array([2.0 + eps]))[0], atol=1e-5)

    def test_mitchell_dc_gain_is_one(self):
        """Guard: unit DC gain, otherwise resizing shifts image brightness.

        k(0) + 2*k(1) + 2*k(2) == 1 keeps a 1:1 resample brightness-neutral.
        The mis-transcribed form (7x^3-12x^2+5)/6 yields 0.944 (~5.6% darker).
        """
        k = mitchell_kernel(np.array([0.0, 1.0, 2.0]))
        assert np.isclose(k[0] + 2.0 * k[1] + 2.0 * k[2], 1.0, atol=1e-6)


class TestSeparableWeights:
    @pytest.mark.parametrize("filt", SUPPORTED_FILTERS)
    def test_partition_of_unity(self, filt):
        """Weights must sum to 1.0 for each target pixel to conserve energy."""
        idx, w = build_separable_weights(src_len=100, dst_len=40, scale=0.4,
                                         offset=0.0, filt=filt)
        sums = w.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_area_rejects_upscaling(self):
        with pytest.raises(ValueError, match="only supported for downscaling"):
            build_separable_weights(src_len=50, dst_len=100, scale=2.0,
                                    offset=0.0, filt="area")

    def test_nearest_indices_are_exact_nearest_pixels(self):
        idx, w = build_separable_weights(src_len=10, dst_len=5, scale=0.5,
                                         offset=0.0, filt="nearest")
        assert idx.shape == (5, 1)
        assert np.all(w == 1.0)
        # For scale=0.5, centers are at (j + 0.5)/0.5 - 0.5 = 2*j + 0.5.
        # round(2*j + 0.5) gives 1, 3, 5, 7, 9
        expected = np.array([[1], [3], [5], [7], [9]], dtype=np.int32)
        assert np.array_equal(idx, expected)

    def test_unknown_filter_rejected(self):
        with pytest.raises(ValueError, match="unknown filter"):
            build_separable_weights(10, 5, 0.5, 0.0, "unknown_xyz")


class TestResampleCPU:
    @pytest.mark.parametrize("filt", SUPPORTED_FILTERS)
    def test_constant_field_preservation(self, filt):
        """Downscaling a uniform field must preserve the exact value without shift."""
        val = 140
        src = np.full((64, 64, 3), val, dtype=np.uint8)
        plan = make_plan(64, 64, 32, 32)
        out = resample_cpu(src, plan, filt)
        assert out.shape == (32, 32, 3)
        assert np.max(np.abs(out.astype(int) - val)) <= 1

    @pytest.mark.parametrize("filt", ["lanczos3", "mitchell", "bilinear", "nearest"])
    def test_constant_field_upscaling_preservation(self, filt):
        val = 180
        src = np.full((32, 32, 3), val, dtype=np.uint8)
        plan = make_plan(32, 32, 64, 64)
        out = resample_cpu(src, plan, filt)
        assert out.shape == (64, 64, 3)
        assert np.max(np.abs(out.astype(int) - val)) <= 1

    def test_step_boundary(self):
        src = np.zeros((40, 40, 3), dtype=np.uint8)
        src[:, 20:] = 255
        plan = make_plan(40, 40, 20, 20)
        out = resample_cpu(src, plan, "mitchell")
        assert out.shape == (20, 20, 3)
        # Left side is near 0, right side near 255
        assert np.all(out[:, :8] < 20)
        assert np.all(out[:, 12:] > 235)

    def test_gradient_preservation(self):
        x = np.linspace(0, 255, 64, dtype=np.uint8)
        src = np.tile(x[None, :, None], (64, 1, 3))
        plan = make_plan(64, 64, 32, 32)
        for filt in ("lanczos3", "mitchell", "area", "bilinear"):
            out = resample_cpu(src, plan, filt)
            # Monotonic increase along horizontal axis
            diffs = np.diff(out[16, :, 0].astype(int))
            assert np.all(diffs >= 0)

    def test_checkerboard_downscaling(self):
        # 1-pixel checkerboard: average is ~127.5
        cb = np.indices((64, 64)).sum(axis=0) % 2 * 255
        src = np.repeat(cb[:, :, None], 3, axis=2).astype(np.uint8)
        plan = make_plan(64, 64, 16, 16)
        out = resample_cpu(src, plan, "area")
        # Area filter downscales 4x4 blocks of checkerboard -> exactly 127 or 128
        assert np.all(np.abs(out.astype(int) - 128) <= 2)

    def test_extreme_aspect_ratios_1xN_and_Nx1(self):
        # 1 x 64 image
        src1 = np.full((1, 64, 3), 100, dtype=np.uint8)
        plan1 = make_plan(64, 1, 32, 1)
        out1 = resample_cpu(src1, plan1, "mitchell")
        assert out1.shape == (1, 32, 3)

        # 64 x 1 image
        src2 = np.full((64, 1, 3), 100, dtype=np.uint8)
        plan2 = make_plan(1, 64, 1, 32)
        out2 = resample_cpu(src2, plan2, "mitchell")
        assert out2.shape == (32, 1, 3)

    def test_fractional_crop(self):
        src = np.zeros((100, 100, 3), dtype=np.uint8)
        src[20:60, 20:60] = 200
        plan = make_plan(100, 100, 40, 40, x0=20.0, y0=20.0, cw=40.0, ch=40.0)
        out = resample_cpu(src, plan, "mitchell")
        assert out.shape == (40, 40, 3)
        assert np.all(out > 170)

    def test_float_fast_path_stays_close_to_reference(self):
        y, x = np.mgrid[0:192, 0:256].astype(np.float32)
        src = np.stack([
            x / 255.0,
            y / 191.0,
            (x + y) / 446.0,
        ], axis=2).astype(np.float32)
        plan = make_plan(256, 192, 128, 96)
        fast = resample_cpu_float(src, plan, "lanczos3")
        ref = separable_resample_float(src, plan, "lanczos3")
        mse = float(np.mean((fast - ref) ** 2))
        psnr = 10.0 * np.log10(1.0 / mse) if mse else 99.0
        assert fast.shape == (96, 128, 3)
        assert fast.dtype == np.float32
        assert psnr > 35.0

    def test_identity_plan_returns_contiguous_copy(self):
        src = np.full((20, 20, 3), 50, dtype=np.uint8)
        plan = make_plan(20, 20, 20, 20)
        out = resample_cpu(src, plan, "mitchell")
        assert np.array_equal(src, out)
