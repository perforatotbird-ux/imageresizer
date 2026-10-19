"""Tests for output sharpening (luminance Unsharp Mask)."""

import cv2
import numpy as np
import pytest

from batchresize.sharpen import (
    SharpenConfig,
    SharpenError,
    apply_sharpen,
    parse_sharpen,
)


class TestSharpenConfig:
    def test_presets(self):
        off = SharpenConfig.from_preset("off")
        assert not off.enabled

        weak = SharpenConfig.from_preset("weak")
        assert weak.enabled
        assert weak.amount == 0.6
        assert weak.radius == 0.8
        assert weak.threshold == 2

        std = SharpenConfig.from_preset("standard")
        assert std.enabled
        assert std.amount == 1.2
        assert std.radius == 1.0
        assert std.threshold == 2

    def test_parse_sharpen(self):
        assert not parse_sharpen("off").enabled
        assert parse_sharpen("weak").amount == 0.6
        assert parse_sharpen("standard").amount == 1.2
        assert parse_sharpen(None).enabled is False
        assert parse_sharpen(False).enabled is False
        assert parse_sharpen(True).enabled is True

        custom = parse_sharpen({"enabled": True, "amount": 1.5, "radius": 1.2, "threshold": 4})
        assert custom.enabled
        assert custom.amount == 1.5
        assert custom.radius == 1.2
        assert custom.threshold == 4
        assert custom.preset == "custom"

    def test_validation_errors(self):
        with pytest.raises(SharpenError):
            SharpenConfig(amount=-1.0)
        with pytest.raises(SharpenError):
            SharpenConfig(radius=-0.5)
        with pytest.raises(SharpenError):
            SharpenConfig(threshold=-1)
        with pytest.raises(SharpenError):
            SharpenConfig(threshold=300)
        with pytest.raises(SharpenError):
            parse_sharpen("non_existent_preset")


class TestApplySharpen:
    def test_disabled_returns_original(self):
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        cfg = SharpenConfig(enabled=False)
        out = apply_sharpen(img, cfg)
        assert np.array_equal(img, out)

    def test_constant_field_preservation(self):
        """Sharpening a uniform area must not change pixel values."""
        val = 150
        img = np.full((50, 50, 3), val, dtype=np.uint8)
        cfg = SharpenConfig(enabled=True, amount=2.0, radius=1.5, threshold=0)
        out = apply_sharpen(img, cfg)
        assert np.max(np.abs(out.astype(int) - val)) == 0

    def test_edge_enhancement(self):
        """Sharpening must increase the contrast across a step boundary."""
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:, 20:] = 180
        cfg = SharpenConfig(enabled=True, amount=1.0, radius=1.0, threshold=0)
        out = apply_sharpen(img, cfg)
        # Pixel before edge (darker side) drops; pixel after edge (brighter side) rises
        assert out[20, 19, 0] <= img[20, 19, 0]
        assert out[20, 20, 0] >= img[20, 20, 0]

    def test_threshold_suppresses_low_amplitude_noise(self):
        """Differences below threshold must not be amplified."""
        # Baseline with flat tone + 1-level noise
        base = np.full((40, 40, 3), 120, dtype=np.uint8)
        base[20, 20] = 121  # 1 level difference
        cfg = SharpenConfig(enabled=True, amount=2.0, radius=1.0, threshold=3)
        out = apply_sharpen(base, cfg)
        # Since diff < 3, output remains unchanged
        assert np.array_equal(base, out)

    def test_no_color_fringing_chroma_preserved(self):
        """Chroma channels (Cr, Cb) must not be altered, preventing color halos."""
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:, :, 0] = 50   # B
        img[:, :, 1] = 120  # G
        img[:, :, 2] = 200  # R
        img[:, 20:, :] = np.clip(img[:, 20:, :] + 40, 0, 255)
        cfg = SharpenConfig(enabled=True, amount=1.5, radius=1.0, threshold=0)
        out = apply_sharpen(img, cfg)

        # Convert both to YCrCb and compare chroma
        ycrcb_in = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb_out = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)

        # Cr and Cb channels must be preserved within standard integer quantization tolerance (<= 2)
        assert np.all(np.abs(ycrcb_in[:, :, 1].astype(int) - ycrcb_out[:, :, 1].astype(int)) <= 2)
        assert np.all(np.abs(ycrcb_in[:, :, 2].astype(int) - ycrcb_out[:, :, 2].astype(int)) <= 2)

    def test_anti_overflow_clipping(self):
        """Values must never wrap around uint8 boundaries."""
        img = np.zeros((30, 30, 3), dtype=np.uint8)
        img[:, 15:] = 250
        # High amount to push past 255 and below 0
        cfg = SharpenConfig(enabled=True, amount=5.0, radius=2.0, threshold=0)
        out = apply_sharpen(img, cfg)
        assert out.dtype == np.uint8
        assert np.all(out >= 0) and np.all(out <= 255)
