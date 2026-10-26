"""CPU backends.

``CpuEngine``      - OpenCV decode/encode with Pillow fractional-box resampling.
                     Alpha and managed non-RGB inputs use Pillow host decoding.
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

    def __init__(self, filter_name: str = "lanczos3", optimize_huffman: bool = False):
        if cv2 is None:
            raise RuntimeError("opencv-python-headless is required for the CPU backend")
        self.filter = filter_name
        self.optimize_huffman = bool(optimize_huffman)

    # -- decode --------------------------------------------------------------
    @staticmethod
    def decode(data: bytes, reduce_scale: int = 1) -> np.ndarray:
        """Decode to BGR uint8 (alpha composited later by the caller)."""
        buf = np.frombuffer(data, dtype=np.uint8)
        flags = cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION
        if reduce_scale == 2:
            flags = cv2.IMREAD_REDUCED_COLOR_2 | cv2.IMREAD_IGNORE_ORIENTATION
        elif reduce_scale == 4:
            flags = cv2.IMREAD_REDUCED_COLOR_4 | cv2.IMREAD_IGNORE_ORIENTATION
        elif reduce_scale >= 8:
            flags = cv2.IMREAD_REDUCED_COLOR_8 | cv2.IMREAD_IGNORE_ORIENTATION
        img = cv2.imdecode(buf, flags)
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
        try:
            from PIL import Image

            b, g, r = bg_rgb[2], bg_rgb[1], bg_rgb[0]
            fg = Image.fromarray(np.ascontiguousarray(img), "RGBA")
            canvas = Image.new("RGBA", fg.size, (b, g, r, 255))
            return np.ascontiguousarray(
                np.asarray(Image.alpha_composite(canvas, fg).convert("RGB")))
        except Exception:
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
        from .resampling import resample_cpu
        return resample_cpu(img, plan, self.filter)

    # -- encode --------------------------------------------------------------
    def encode(self, img: np.ndarray, quality: int, subsampling: int,
               qtables: tuple[list[int], list[int]] | None = None) -> bytes:
        if qtables is not None:
            return self._encode_with_tables(img, subsampling, qtables)
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        factor = {
            0: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
            1: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422,
            2: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
        }[subsampling]
        params += [int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), int(factor)]
        if self.optimize_huffman:
            # Optimized Huffman tables: ~20-25% smaller files at the same
            # quality (mirrors the GPU backend's `optimized_huffman` flag).
            params += [int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
        ok, buf = cv2.imencode(".jpg", img, params)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()

    @staticmethod
    def _encode_with_tables(img: np.ndarray, subsampling: int,
                            qtables: tuple[list[int], list[int]]) -> bytes:
        """Encode with the quantisation tables of the source file.

        OpenCV cannot write custom quantisation tables, so this path uses
        Pillow. The quality number is irrelevant here - the tables define the
        compression completely.
        """
        import io
        from PIL import Image

        rgb = np.ascontiguousarray(img[:, :, ::-1])
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(
            buffer, "JPEG", qtables=[list(qtables[0]), list(qtables[1])],
            subsampling={0: 0, 1: 1, 2: 2}[subsampling], optimize=True)
        return buffer.getvalue()

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

    from PIL import ImageCms
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        alpha = im.convert("RGBA").getchannel("A") if (
            "A" in im.getbands() or "transparency" in im.info) else None
        if im.info.get("icc_profile") and im.mode not in ("RGB", "RGBA", "P"):
            profile = ImageCms.getOpenProfile(io.BytesIO(im.info["icc_profile"]))
            colour = im.convert("L") if im.mode == "LA" else im
            im = ImageCms.profileToProfile(colour, profile,
                    ImageCms.createProfile("sRGB"), outputMode="RGB")
        elif im.mode.startswith("I;16"):
            arr = np.asarray(im, dtype=np.float32)
            arr = np.clip(arr / 257.0, 0, 255).round().astype(np.uint8)
            return np.repeat(arr[:, :, None], 3, axis=2), False
        elif im.mode in ("I", "F"):
            arr = np.clip(np.asarray(im, dtype=np.float32), 0, 255).astype(np.uint8)
            return np.repeat(arr[:, :, None], 3, axis=2), False
        else:
            im = im.convert("RGB")
        if alpha is not None:
            im.putalpha(alpha)
        return np.ascontiguousarray(np.asarray(im)), alpha is not None
