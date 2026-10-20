"""Tests for Stage 6 geometry enhancements: pad mode, crop anchors, and crop focus."""

import numpy as np
import pytest
from PIL import Image

from batchresize.geometry import (
    ANCHORS,
    Plan,
    apply_pad,
    make_plan,
    parse_crop_focus,
    plan_cover,
    plan_pad,
)


class TestCropFocusParsing:
    def test_parse_valid_string(self):
        assert parse_crop_focus("0.5, 0.5") == (0.5, 0.5)
        assert parse_crop_focus("0, 1") == (0.0, 1.0)
        assert parse_crop_focus("0.25,0.75") == (0.25, 0.75)

    def test_parse_valid_tuple(self):
        assert parse_crop_focus((0.1, 0.9)) == (0.1, 0.9)
        assert parse_crop_focus([0.0, 1.0]) == (0.0, 1.0)

    @pytest.mark.parametrize("invalid", [
        "0.5",
        "0.5, 0.5, 0.5",
        "foo,bar",
        (-0.1, 0.5),
        (0.5, 1.1),
        "1.2, 0.5",
        None,
        123,
    ])
    def test_parse_invalid_rejected(self, invalid):
        with pytest.raises(ValueError):
            parse_crop_focus(invalid)


class TestPlanCoverAnchorsAndFocus:
    def test_all_standard_anchors_recognized(self):
        expected_keys = {
            "center", "top", "bottom", "left", "right",
            "top-left", "top-right", "bottom-left", "bottom-right",
            "n", "s", "e", "w", "nw", "ne", "sw", "se",
        }
        assert expected_keys.issubset(set(ANCHORS.keys()))

    def test_landscape_crop_horizontal_anchors(self):
        # 200x100 into 100x100 box -> cover scale is 1.0 (max(100/200, 100/100) = 1.0)
        # cw=100, ch=100; overflow on width (src_w=200 > cw=100)
        plan_c = plan_cover(200, 100, 100, 100, anchor="center")
        assert plan_c.x0 == 50.0
        assert plan_c.y0 == 0.0

        plan_l = plan_cover(200, 100, 100, 100, anchor="left")
        assert plan_l.x0 == 0.0
        assert plan_l.y0 == 0.0

        plan_r = plan_cover(200, 100, 100, 100, anchor="right")
        assert plan_r.x0 == 100.0
        assert plan_r.y0 == 0.0

    def test_portrait_crop_vertical_anchors(self):
        # 100x200 into 100x100 box -> cover scale 1.0; cw=100, ch=100
        plan_c = plan_cover(100, 200, 100, 100, anchor="center")
        assert plan_c.x0 == 0.0
        assert plan_c.y0 == 50.0

        plan_t = plan_cover(100, 200, 100, 100, anchor="top")
        assert plan_t.x0 == 0.0
        assert plan_t.y0 == 0.0

        plan_b = plan_cover(100, 200, 100, 100, anchor="bottom")
        assert plan_b.x0 == 0.0
        assert plan_b.y0 == 100.0

    def test_focus_point_placement_and_clamping(self):
        # 400x100 into 100x100 -> cw=100, ch=100
        # focus=(0.25, 0.5): fx*src_w - cw/2 = 0.25*400 - 50 = 50.0
        plan_f = plan_cover(400, 100, 100, 100, focus=(0.25, 0.5))
        assert plan_f.x0 == 50.0

        # Clamping at left boundary
        plan_clamp_left = plan_cover(400, 100, 100, 100, focus=(0.05, 0.5))
        assert plan_clamp_left.x0 == 0.0

        # Clamping at right boundary
        plan_clamp_right = plan_cover(400, 100, 100, 100, focus=(0.95, 0.5))
        assert plan_clamp_right.x0 == 300.0

    def test_cover_upscale_disabled_contract(self):
        # Small image (50x25) into target (100x100) with allow_upscale=False
        # Contract: scale capped at 1.0; output cannot exceed target box; result can be smaller than target
        plan = plan_cover(50, 25, 100, 100, allow_upscale=False)
        assert plan.out_w <= 100
        assert plan.out_h <= 100
        assert plan.out_w == 50
        assert plan.out_h == 25
        assert plan.cw == 50.0
        assert plan.ch == 25.0


class TestPlanPad:
    def test_pad_canvas_exact_size(self):
        # 100x50 into 100x100 -> fits at 100x50 (scale 1.0)
        # canvas is 100x100, pad_h = 50, pad_top = 25
        plan = plan_pad(100, 50, 100, 100, allow_upscale=False, anchor="center")
        assert plan.is_pad is True
        assert plan.out_w == 100
        assert plan.out_h == 50
        assert plan.canvas_w == 100
        assert plan.canvas_h == 100
        assert plan.target_w == 100
        assert plan.target_h == 100
        assert plan.pad_left == 0
        assert plan.pad_top == 25

    def test_pad_odd_margin_rounding(self):
        # 100x51 into 100x100 -> pad_h = 49.
        # Center anchor (ay=0.5): round(49 * 0.5) = 24 or 25
        plan = plan_pad(100, 51, 100, 100, anchor="center")
        assert plan.canvas_w == 100
        assert plan.canvas_h == 100
        assert plan.pad_top in (24, 25)
        # Verify that content + margins fits perfectly inside canvas
        assert plan.pad_top + plan.out_h <= plan.canvas_h

    def test_pad_anchors(self):
        # 100x50 into 100x100: pad_w=0, pad_h=50
        plan_top = plan_pad(100, 50, 100, 100, anchor="top")
        assert plan_top.pad_top == 0

        plan_bot = plan_pad(100, 50, 100, 100, anchor="bottom")
        assert plan_bot.pad_top == 50

        # 50x100 into 100x100: pad_w=50, pad_h=0
        plan_left = plan_pad(50, 100, 100, 100, anchor="left")
        assert plan_left.pad_left == 0

        plan_right = plan_pad(50, 100, 100, 100, anchor="right")
        assert plan_right.pad_left == 50


class TestApplyPad:
    def test_apply_pad_uint8(self):
        plan = plan_pad(4, 2, 6, 4, anchor="center")  # pad_left=1, pad_top=1
        img = np.full((2, 4, 3), 100, dtype=np.uint8)
        bg_color = (10, 20, 30)  # RGB
        padded = apply_pad(img, plan, bg_color)
        assert padded.shape == (4, 6, 3)
        assert padded.dtype == np.uint8
        # Corners must be background in BGR (30, 20, 10)
        assert tuple(padded[0, 0]) == (30, 20, 10)
        # Center must be image content
        assert tuple(padded[1, 1]) == (100, 100, 100)

    def test_apply_pad_float32(self):
        plan = plan_pad(4, 2, 6, 4, anchor="top-left")
        img = np.full((2, 4, 3), 0.5, dtype=np.float32)
        bg_color = (255, 255, 255)
        padded = apply_pad(img, plan, bg_color)
        assert padded.shape == (4, 6, 3)
        assert padded.dtype == np.float32
        # Top-left has image
        assert np.allclose(padded[0, 0], [0.5, 0.5, 0.5])
        # Bottom-right has white background (1.0)
        assert np.allclose(padded[3, 5], [1.0, 1.0, 1.0])

    def test_make_plan_pad_integration(self):
        plan = make_plan(200, 100, 100, 100, mode="pad", anchor="center")
        assert plan.is_pad is True
        assert plan.target_w == 100
        assert plan.target_h == 100
        assert plan.out_w == 100
        assert plan.out_h == 50
