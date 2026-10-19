"""CPU backends.

``CpuEngine``      - the reference/fallback path: libjpeg-turbo decode + SIMD
                     resize + libjpeg-turbo encode (all inside OpenCV).
``decode_to_rgb``  - robust host-side decode used by the *hybrid* GPU mode
                     (Pillow covers exotic PNG/TIFF/WebP variants).
"""

from __future__ import annotations

import io

import numpy as np

from .geometry import Plan

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class CpuEngine:
    name = "cpu"

    def __init__(self, filter_name: str = "lanczos3"):
        if cv2 is None:
            raise RuntimeError("opencv-python-headless is required for the CPU backend")
        self.filter = filter_name

    # -- decode --------------------------------------------------------------
    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        """Decode to BGR uint8 (alpha composited later by the caller)."""
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise ValueError("cv2.imdecode returned None (corrupt or unsupported)")
        if img.dtype == np.uint16:
            img = (img / 257.0).round().astype(np.uint8)
        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            pass  # BGRA, handled by composite_bg()
        elif img.shape[2] == 3:
            pass
        else:
            raise ValueError(f"unsupported channel count {img.shape[2]}")
        return img

    @staticmethod
    def composite_bg(img: np.ndarray, bg_rgb: tuple[int, int, int]) -> np.ndarray:
        if img.shape[2] != 4:
            return img
        b, g, r = bg_rgb[2], bg_rgb[1], bg_rgb[0]
        alpha = img[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
        bg_arr = np.array([b, g, r], dtype=np.float32)
        out = img[:, :, :3].astype(np.float32) * alpha + bg_arr * (1.0 - alpha)
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
        if orientation == 1:
            return img
        if orientation == 2:
            return cv2.flip(img, 1)
        if orientation == 3:
            return cv2.rotate(img, cv2.ROTATE_180)
        if orientation == 4:
            return cv2.flip(img, 0)
        if orientation == 5:
            return cv2.transpose(img)
        if orientation == 6:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if orientation == 7:
            return cv2.flip(cv2.transpose(img), -1)
        if orientation == 8:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    # -- resize --------------------------------------------------------------
    def resize(self, img: np.ndarray, plan: Plan) -> np.ndarray:
        x0 = int(round(plan.x0))
        y0 = int(round(plan.y0))
        x1 = int(round(plan.x0 + plan.cw))
        y1 = int(round(plan.y0 + plan.ch))
        x1 = min(x1, img.shape[1])
        y1 = min(y1, img.shape[0])
        region = img[y0:y1, x0:x1]
        if (plan.out_w, plan.out_h) == (region.shape[1], region.shape[0]):
            return np.ascontiguousarray(region)
        down = plan.scale_x < 1.0 or plan.scale_y < 1.0
        interp = cv2.INTER_AREA if down else cv2.INTER_LINEAR
        return cv2.resize(region, (plan.out_w, plan.out_h), interpolation=interp)

    # -- encode --------------------------------------------------------------
    @staticmethod
    def encode(img: np.ndarray, quality: int, subsampling: int) -> bytes:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        factor = {
            0: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
            1: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422,
            2: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
        }[subsampling]
        params += [int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), int(factor)]
        ok, buf = cv2.imencode(".jpg", img, params)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()

    # -- full path -----------------------------------------------------------
    def process(self, data: bytes, plan: Plan, quality: int, subsampling: int,
                orientation: int = 1, bg: tuple[int, int, int] = (255, 255, 255),
                has_alpha: bool = False) -> bytes:
        img = self.decode(data)
        if img.shape[2] == 4 or has_alpha:
            img = self.composite_bg(img, bg)
        img = self.apply_orientation(img, orientation)
        small = self.resize(img, plan)
        return self.encode(small, quality, subsampling)


def decode_to_rgb(data: bytes, bg: tuple[int, int, int] = (255, 255, 255)
                  ) -> tuple[np.ndarray, bool]:
    """Host decode for hybrid mode. Returns (RGB/RGBA uint8, has_alpha)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        has_alpha = im.mode in ("RGBA", "LA", "PA") or (
            im.mode == "P" and "transparency" in im.info
        )
        if im.mode == "CMYK":
            im = im.convert("RGB")
        elif im.mode not in ("RGB", "RGBA", "L", "LA", "I;16", "I", "F"):
            im = im.convert("RGBA" if has_alpha else "RGB")
        if im.mode in ("I;16", "I", "F"):
            arr = np.asarray(im, dtype=np.float32)
            arr = np.clip(arr / 257.0 if im.mode == "I;16" else arr, 0, 255)
            rgb = np.repeat(arr[:, :, None].astype(np.uint8), 3, axis=2)
            return rgb, False
        arr = np.asarray(im)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        return np.ascontiguousarray(arr), has_alpha
