import io

from PIL import Image

from batchresize.imageio_meta import (has_alpha_hint, is_truncated_jpeg,
                                      jpeg_dimensions, read_source_meta,
                                      splice_metadata)


def _jpeg(w=64, h=48, orientation=None, icc=None) -> bytes:
    img = Image.new("RGB", (w, h), (10, 120, 200))
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation
    exif[0x0131] = "unit-test"
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90, exif=exif, icc_profile=icc)
    return buf.getvalue()


def test_read_meta_orientation_and_display_dims():
    data = _jpeg(640, 480, orientation=6)
    m = read_source_meta(data)
    assert m.orientation == 6
    assert (m.width, m.height) == (640, 480)
    assert (m.display_width, m.display_height) == (480, 640)


def test_read_meta_defaults_to_one():
    m = read_source_meta(_jpeg())
    assert m.orientation == 1


def test_truncated_jpeg_detected():
    data = _jpeg(200, 200)
    assert not is_truncated_jpeg(data)
    assert is_truncated_jpeg(data[: len(data) // 2])
    assert is_truncated_jpeg(b"not a jpeg at all")


def test_png_alpha_detection():
    rgb = io.BytesIO()
    Image.new("RGB", (10, 10)).save(rgb, "PNG")
    assert not has_alpha_hint(rgb.getvalue(), ".png")

    rgba = io.BytesIO()
    Image.new("RGBA", (10, 10)).save(rgba, "PNG")
    assert has_alpha_hint(rgba.getvalue(), ".png")

    # palette PNG without tRNS has no alpha
    pal = io.BytesIO()
    Image.new("P", (10, 10)).save(pal, "PNG")
    assert not has_alpha_hint(pal.getvalue(), ".png")


def test_splice_sets_orientation_one_and_keeps_icc():
    src = _jpeg(100, 80, orientation=8, icc=b"\x00" * 40)
    meta = read_source_meta(src)
    plain = _jpeg(50, 40)  # encoder output without metadata

    out = splice_metadata(plain, meta.exif, meta.icc)
    with Image.open(io.BytesIO(out)) as im:
        assert im.getexif().get(0x0112) == 1
        assert im.info.get("icc_profile") == b"\x00" * 40


def test_splice_is_idempotent():
    src = _jpeg(100, 80, orientation=3)
    meta = read_source_meta(src)
    once = splice_metadata(_jpeg(20, 20), meta.exif, meta.icc)
    twice = splice_metadata(once, meta.exif, meta.icc)
    assert once.count(b"Exif\x00\x00") == 1
    assert twice.count(b"Exif\x00\x00") == 1


def test_jpeg_dimensions_reader():
    data = _jpeg(321, 123)
    assert jpeg_dimensions(data) == (321, 123)


def test_splice_without_metadata_is_noop():
    data = _jpeg(30, 30)
    assert splice_metadata(data, None, None) == data
