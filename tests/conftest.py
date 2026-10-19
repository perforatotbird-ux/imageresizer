import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batchresize.cli import main as cli_main  # noqa: E402


def make_rgb(w: int, h: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = xx / max(w - 1, 1) * 255.0
    g = yy / max(h - 1, 1) * 255.0
    b = (np.sin(xx / 23.0) * 60 + 120).astype(np.float32)
    arr = np.dstack([r, g, b]) + rng.normal(0, 3, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def save_jpeg(path: Path, img: Image.Image, quality: int = 92,
              orientation: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if orientation:
        exif = Image.Exif()
        exif[0x0112] = orientation
        img.save(path, "JPEG", quality=quality, exif=exif)
    else:
        img.save(path, "JPEG", quality=quality)


def run_cli(src: Path, out: Path, *args: str) -> tuple[int, dict]:
    argv = ["-i", str(src), "-o", str(out), *args]
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli_main(argv)
    rep_path = out / "report.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
    return code, rep


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def pil_reference(data: bytes, max_w: int, max_h: int, mode: str,
                  allow_upscale: bool = False) -> np.ndarray:
    from PIL import ImageOps

    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        if mode == "contain":
            s = min(max_w / w, max_h / h)
            if s > 1 and not allow_upscale:
                s = 1.0
            out = (max(1, round(w * s)), max(1, round(h * s)))
            crop = im
        else:
            s = max(max_w / w, max_h / h)
            cw, ch = max_w / s, max_h / s
            x0, y0 = (w - cw) / 2, (h - ch) / 2
            crop = im.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch)))
            out = (max_w, max_h)
        return np.asarray(crop.resize(out, Image.LANCZOS))


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    try:
        from batchresize.gpu_backend import GpuEngine

        eng = GpuEngine()
        ok, _ = eng.self_test()
        return ok
    except Exception:
        return False


@pytest.fixture()
def sample_tree(tmp_path: Path) -> Path:
    src = tmp_path / "photos"
    (src / "a" / "b").mkdir(parents=True)
    save_jpeg(src / "top.jpg", make_rgb(400, 300, 1))
    save_jpeg(src / "a" / "mid.jpg", make_rgb(300, 400, 2))
    save_jpeg(src / "a" / "b" / "deep.jpg", make_rgb(256, 256, 3))
    return src
