import pytest

from batchresize.geometry import make_plan, plan_contain, plan_cover


def test_contain_keeps_aspect_and_fits():
    p = plan_contain(4000, 3000, 1024, 1024)
    assert (p.out_w, p.out_h) == (1024, 768)
    assert p.out_w <= 1024 and p.out_h <= 1024
    assert abs(p.out_w / p.out_h - 4000 / 3000) < 0.01


def test_contain_portrait():
    p = plan_contain(3000, 4000, 1024, 1024)
    assert (p.out_w, p.out_h) == (768, 1024)


def test_contain_no_upscale_by_default():
    p = plan_contain(120, 90, 1024, 1024)
    assert (p.out_w, p.out_h) == (120, 90)
    assert p.is_identity


def test_contain_upscale_when_allowed():
    p = plan_contain(120, 90, 1024, 1024, allow_upscale=True)
    assert (p.out_w, p.out_h) == (1024, 768)


def test_contain_wide_image_leaves_letterbox():
    p = plan_contain(1000, 400, 512, 512)
    assert (p.out_w, p.out_h) == (512, 205)
    assert p.out_h < 512


def test_cover_exact_size_and_centered_crop():
    p = plan_cover(4000, 3000, 1024, 1024)
    assert (p.out_w, p.out_h) == (1024, 1024)
    assert p.cw == pytest.approx(3000.0)
    assert p.x0 == pytest.approx(500.0)
    assert p.y0 == pytest.approx(0.0)


def test_cover_upscales_small_image():
    p = plan_cover(100, 50, 200, 200)
    assert (p.out_w, p.out_h) == (200, 200)


def test_presets_are_contained():
    for w, h in [(4032, 3024), (6000, 4000), (1080, 1920), (500, 500)]:
        p = make_plan(w, h, 512, 512)
        assert p.out_w <= 512 and p.out_h <= 512
        assert max(p.out_w, p.out_h) == 512 or (w < 512 and h < 512)


def test_invalid_mode():
    with pytest.raises(ValueError):
        make_plan(10, 10, 5, 5, mode="stretch")


def test_zero_size_rejected():
    with pytest.raises(ValueError):
        plan_contain(0, 10, 100, 100)
