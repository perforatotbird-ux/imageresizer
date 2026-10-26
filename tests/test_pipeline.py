import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from conftest import make_rgb, pil_reference, psnr, read_rgb, run_cli, save_jpeg

BACKENDS = ["cpu", "gpu"]


@pytest.fixture(params=BACKENDS)
def backend(request, gpu_available):
    if request.param == "gpu" and not gpu_available:
        pytest.skip("GPU backend unavailable")
    return request.param


def test_contain_dimensions(tmp_path, sample_tree, backend):
    out = tmp_path / "out"
    code, rep = run_cli(sample_tree, out, "--size", "web_1024", "--backend", backend)
    assert code == 0
    assert rep["processed"] == 3
    # smaller than the box -> untouched by default
    with Image.open(out / "top.jpg") as im:
        assert im.size == (400, 300)
    with Image.open(out / "a" / "mid.jpg") as im:
        assert im.size == (300, 400)


def test_contain_dimensions_with_upscale(tmp_path, sample_tree, backend):
    out = tmp_path / "out"
    run_cli(sample_tree, out, "--size", "web_1024", "--allow-upscale",
            "--backend", backend)
    with Image.open(out / "top.jpg") as im:
        assert im.size == (1024, 768)
    with Image.open(out / "a" / "mid.jpg") as im:
        assert im.size == (768, 1024)


def test_no_upscale_by_default(tmp_path, sample_tree, backend):
    out = tmp_path / "out"
    run_cli(sample_tree, out, "--size", "large_1600", "--backend", backend)
    with Image.open(out / "a" / "b" / "deep.jpg") as im:
        assert im.size == (256, 256)


def test_allow_upscale(tmp_path, sample_tree, backend):
    out = tmp_path / "out"
    run_cli(sample_tree, out, "--size", "small_512", "--allow-upscale",
            "--backend", backend)
    with Image.open(out / "a" / "b" / "deep.jpg") as im:
        assert im.size == (512, 512)


def test_structure_and_extension_preserved(tmp_path, backend):
    src = tmp_path / "photos"
    (src / "x" / "y").mkdir(parents=True)
    make_rgb(300, 200).save(src / "x" / "y" / "photo.PNG")
    make_rgb(300, 200).save(src / "x" / "pic.bmp")
    out = tmp_path / "resized"
    code, rep = run_cli(src, out, "--size", "thumb_256", "--backend", backend)
    assert code == 0
    assert (out / "x" / "y" / "photo.jpg").exists()
    assert (out / "x" / "pic.jpg").exists()
    assert not list(out.rglob("*.png"))
    assert not list(out.rglob("*.bmp"))


def test_cover_is_exact_and_centered(tmp_path, backend):
    src = tmp_path / "photos"
    src.mkdir()
    img = Image.new("RGB", (600, 200), (255, 0, 0))
    # green 200x200 block exactly in the centre
    for x in range(200, 400):
        for y in range(0, 200):
            img.putpixel((x, y), (0, 255, 0))
    img.save(src / "wide.jpg", "JPEG", quality=98)

    out = tmp_path / "out"
    code, _ = run_cli(src, out, "--size", "custom", "--width", "200",
                      "--height", "200", "--mode", "cover", "--quality",
                      "very_high", "--backend", backend)
    assert code == 0
    arr = read_rgb(out / "wide.jpg")
    assert arr.shape[:2] == (200, 200)
    centre = arr[80:120, 80:120].reshape(-1, 3).mean(axis=0)
    assert centre[1] > 180 and centre[0] < 80, centre


def test_contain_matches_reference(tmp_path, backend):
    src = tmp_path / "photos"
    src.mkdir()
    save_jpeg(src / "p.jpg", make_rgb(1200, 900, 5), quality=96)
    out = tmp_path / "out"
    run_cli(src, out, "--size", "small_512", "--quality", "very_high",
            "--backend", backend)
    ref = pil_reference((src / "p.jpg").read_bytes(), 512, 512, "contain")
    got = read_rgb(out / "p.jpg")
    assert got.shape == ref.shape
    assert psnr(got, ref) > 28


def test_cover_matches_reference(tmp_path, backend):
    src = tmp_path / "photos"
    src.mkdir()
    save_jpeg(src / "p.jpg", make_rgb(1200, 800, 6), quality=96)
    out = tmp_path / "out"
    run_cli(src, out, "--size", "custom", "--width", "300", "--height", "300",
            "--mode", "cover", "--quality", "very_high", "--backend", backend)
    ref = pil_reference((src / "p.jpg").read_bytes(), 300, 300, "cover")
    got = read_rgb(out / "p.jpg")
    assert got.shape == ref.shape
    assert psnr(got, ref) > 27


def test_name_collision_does_not_overwrite(tmp_path, backend):
    src = tmp_path / "photos"
    src.mkdir()
    Image.new("RGB", (300, 200), (250, 10, 10)).save(src / "photo.png")
    Image.new("RGB", (300, 200), (10, 10, 250)).save(src / "photo.jpg", "JPEG",
                                                    quality=95)
    out = tmp_path / "out"
    code, rep = run_cli(src, out, "--size", "small_512", "--backend", backend)
    assert code == 0
    files = sorted(p.name for p in out.glob("*.jpg"))
    assert files == ["photo.jpg", "photo__2.jpg"]
    a = read_rgb(out / "photo.jpg")
    b = read_rgb(out / "photo__2.jpg")
    assert psnr(a, b) < 30, "the two sources must not have collapsed into one result"


def test_corrupt_files_are_skipped(tmp_path, backend):
    src = tmp_path / "photos"
    src.mkdir()
    save_jpeg(src / "good.jpg", make_rgb(400, 300))
    good = (src / "good.jpg").read_bytes()
    (src / "truncated.jpg").write_bytes(good[: len(good) // 2])
    (src / "garbage.jpg").write_bytes(b"nonsense" * 50)
    out = tmp_path / "out"
    code, rep = run_cli(src, out, "--size", "thumb_256", "--backend", backend)
    assert rep["processed"] == 1
    assert rep["failed"] == 2
    assert code == 1
    assert (out / "good.jpg").exists()


def test_skip_existing(tmp_path, sample_tree, backend):
    out = tmp_path / "out"
    run_cli(sample_tree, out, "--size", "thumb_256", "--backend", backend)
    target = out / "top.jpg"
    mtime = target.stat().st_mtime_ns
    code, rep = run_cli(sample_tree, out, "--size", "thumb_256",
                        "--skip-existing", "--backend", backend)
    assert rep["skipped"] == 3
    assert rep["processed"] == 0
    assert target.stat().st_mtime_ns == mtime


def test_overwrite(tmp_path, sample_tree, backend):
    out = tmp_path / "out"
    run_cli(sample_tree, out, "--size", "thumb_256", "--backend", backend)
    target = out / "top.jpg"
    target.write_bytes(b"stale")
    code, rep = run_cli(sample_tree, out, "--size", "thumb_256",
                        "--skip-existing", "--overwrite", "--backend", backend)
    assert rep["processed"] == 3
    assert target.read_bytes()[:2] == b"\xff\xd8"


def test_output_inside_input_is_not_rescanned(tmp_path, sample_tree, backend):
    out = sample_tree / "resized"
    run_cli(sample_tree, out, "--size", "thumb_256", "--backend", backend)
    code, rep = run_cli(sample_tree, out, "--size", "thumb_256",
                        "--skip-existing", "--backend", backend)
    assert rep["discovered"] == 3, rep
    assert not (out / "resized").exists()


def test_no_temp_files_left(tmp_path, sample_tree, backend):
    out = tmp_path / "out"
    run_cli(sample_tree, out, "--size", "thumb_256", "--backend", backend)
    assert not list(out.rglob("*.tmp"))


def test_quality_presets_change_size(tmp_path, backend):
    src = tmp_path / "photos"
    src.mkdir()
    save_jpeg(src / "p.jpg", make_rgb(800, 600, 9), quality=98)
    sizes = {}
    for preset in ("compact", "high", "very_high"):
        out = tmp_path / f"out_{preset}"
        run_cli(src, out, "--size", "small_512", "--quality", preset,
                "--backend", backend)
        sizes[preset] = (out / "p.jpg").stat().st_size
    assert sizes["compact"] < sizes["high"] < sizes["very_high"]


def test_subsampling_changes_size(tmp_path, backend):
    src = tmp_path / "photos"
    src.mkdir()
    save_jpeg(src / "p.jpg", make_rgb(900, 700, 10), quality=98)
    out420 = tmp_path / "a"
    out444 = tmp_path / "b"
    run_cli(src, out420, "--size", "small_512", "--quality", "high",
            "--subsampling", "420", "--backend", backend)
    run_cli(src, out444, "--size", "small_512", "--quality", "high",
            "--subsampling", "444", "--backend", backend)
    assert (out420 / "p.jpg").stat().st_size < (out444 / "p.jpg").stat().st_size


def test_exif_orientation_all_eight(tmp_path, backend):
    src = tmp_path / "photos"
    src.mkdir()
    for o in range(1, 9):
        save_jpeg(src / f"o{o}.jpg", make_rgb(400, 300, 30 + o),
                  quality=98, orientation=o)
    out = tmp_path / "out"
    code, rep = run_cli(src, out, "--size", "custom", "--width", "400",
                        "--height", "300", "--quality", "very_high",
                        "--subsampling", "444", "--backend", backend)
    assert rep["processed"] == 8
    for o in range(1, 9):
        ref = pil_reference((src / f"o{o}.jpg").read_bytes(), 400, 300, "contain")
        got = read_rgb(out / f"o{o}.jpg")
        assert got.shape == ref.shape, f"orientation {o}: {got.shape} vs {ref.shape}"
        assert psnr(got, ref) > 28, f"orientation {o} psnr too low"
        with Image.open(out / f"o{o}.jpg") as im:
            assert im.getexif().get(0x0112) == 1


def test_exif_and_icc_kept_or_dropped(tmp_path, backend):
    src = tmp_path / "photos"
    src.mkdir()
    img = make_rgb(300, 200)
    exif = Image.Exif()
    exif[0x0131] = "kept-software"
    img.save(src / "p.jpg", "JPEG", quality=92, exif=exif,
             icc_profile=b"\x00" * 32)

    out = tmp_path / "keep"
    run_cli(src, out, "--size", "thumb_256", "--backend", backend)
    with Image.open(out / "p.jpg") as im:
        assert im.getexif().get(0x0131) == "kept-software"
        assert len(im.info.get("icc_profile", b"")) == 32

    out2 = tmp_path / "drop"
    run_cli(src, out2, "--size", "thumb_256", "--no-exif", "--no-icc",
            "--backend", backend)
    with Image.open(out2 / "p.jpg") as im:
        assert im.getexif().get(0x0131) is None
        assert "icc_profile" not in im.info


def test_alpha_flattened_to_background(tmp_path, backend):
    src = tmp_path / "photos"
    src.mkdir()
    img = Image.new("RGBA", (200, 200), (255, 0, 0, 0))
    for x in range(100, 200):
        for y in range(200):
            img.putpixel((x, y), (0, 0, 255, 255))
    img.save(src / "alpha.png")
    out = tmp_path / "out"
    code, rep = run_cli(src, out, "--size", "custom", "--width", "200",
                        "--height", "200", "--bg-color", "#FFFF00",
                        "--quality", "very_high", "--backend", backend)
    assert rep["processed"] == 1
    arr = read_rgb(out / "alpha.jpg")
    bg = arr[20, 20].astype(int)                       # transparent -> yellow
    assert abs(bg[0] - 255) <= 4 and abs(bg[1] - 255) <= 4 and bg[2] <= 4, bg
    assert arr[20, 180][2] > 200                       # opaque blue kept


def test_limit(tmp_path, sample_tree, backend):
    out = tmp_path / "out"
    code, rep = run_cli(sample_tree, out, "--size", "thumb_256",
                        "--limit", "1", "--backend", backend)
    assert rep["processed"] == 1
    assert rep["discovered"] == 1


def test_gpu_cpu_outputs_are_close(tmp_path, gpu_available):
    if not gpu_available:
        pytest.skip("GPU backend unavailable")
    src = tmp_path / "photos"
    src.mkdir()
    save_jpeg(src / "p.jpg", make_rgb(1600, 1200, 77), quality=97)
    out_gpu = tmp_path / "gpu"
    out_cpu = tmp_path / "cpu"
    run_cli(src, out_gpu, "--size", "small_512", "--quality", "high",
            "--backend", "gpu")
    run_cli(src, out_cpu, "--size", "small_512", "--quality", "high",
            "--backend", "cpu")
    a = read_rgb(out_gpu / "p.jpg")
    b = read_rgb(out_cpu / "p.jpg")
    assert a.shape == b.shape
    assert psnr(a, b) > 27


def test_actual_route_flags_maps_backend_labels():
    """AUD-001 regression: route strings must follow the recorded labels.

    A run whose results were produced by the GPU engine used to be reported
    as CPU because the derivation looked for the literal labels 'gpu'/'hybrid'
    while ``_finish`` wrote 'nvimgcodec-auto'.
    """
    from batchresize.pipeline import actual_route_flags

    decode, compute = actual_route_flags({"cpu": 8})
    assert decode is False and compute is False

    for label in ("gpu", "nvimgcodec-auto"):
        decode, compute = actual_route_flags({label: 8})
        assert decode is True and compute is True

    decode, compute = actual_route_flags({"hybrid": 8})
    assert decode is False and compute is True

    decode, compute = actual_route_flags({"cpu": 3, "nvimgcodec-auto": 1})
    assert decode is True and compute is True

    assert actual_route_flags({}) == (False, False)


def test_report_actual_routes_follow_backend(tmp_path, sample_tree, backend):
    """The published route strings must describe what actually ran."""
    out = tmp_path / "out"
    code, rep = run_cli(sample_tree, out, "--size", "small_512", "--backend", backend)
    assert code == 0
    labels = set(rep["backend_counts"])
    routes = rep["actual_routes"]
    gpu_decode = bool(labels & {"gpu", "nvimgcodec-auto"})
    gpu_compute = gpu_decode or "hybrid" in labels
    assert (routes["decode"] == "nvImageCodec automatic plugin") == gpu_decode
    assert (routes["resize"] == "CUDA separable kernel") == gpu_compute
    assert (routes["encode"] == "nvJPEG GPU") == gpu_compute
