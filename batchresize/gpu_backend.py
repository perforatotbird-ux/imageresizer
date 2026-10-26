"""GPU backend: nvImageCodec (nvJPEG) decode/encode + CuPy resampling.

The whole hot path stays in device memory:

    file bytes --(nvjpeg decode)--> VRAM uint8 --(CUDA resample)--> VRAM uint8
              --(nvjpeg encode)--> compressed bytes (host, ~10-50 KB)

Only the compressed bitstream ever crosses PCIe after the initial upload.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .geometry import Plan
from .presets import auto_subsampling

_CUDA_SRC = r"""
extern "C" __global__
void rs_h(const unsigned char* __restrict__ src,
          float* __restrict__ dst,
          const int* __restrict__ idx,
          const float* __restrict__ w,
          int src_w, int src_h, int dst_w, int taps)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= src_h) return;
    const unsigned char* row = src + (long long)y * src_w * 3;
    float r = 0.f, g = 0.f, b = 0.f;
    const int* ip = idx + x * taps;
    const float* wp = w + x * taps;
    for (int k = 0; k < taps; ++k) {
        const unsigned char* p = row + (long long)ip[k] * 3;
        const float ww = wp[k];
        r += p[0] * ww; g += p[1] * ww; b += p[2] * ww;
    }
    float* o = dst + ((long long)y * dst_w + x) * 3;
    o[0] = r; o[1] = g; o[2] = b;
}

extern "C" __global__
void rs_v(const float* __restrict__ src,
          unsigned char* __restrict__ dst,
          const int* __restrict__ idx,
          const float* __restrict__ w,
          int src_h, int dst_w, int dst_h, int taps)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;
    float r = 0.f, g = 0.f, b = 0.f;
    const int* ip = idx + y * taps;
    const float* wp = w + y * taps;
    for (int k = 0; k < taps; ++k) {
        const float* p = src + ((long long)ip[k] * dst_w + x) * 3;
        const float ww = wp[k];
        r += p[0] * ww; g += p[1] * ww; b += p[2] * ww;
    }
    unsigned char* o = dst + ((long long)y * dst_w + x) * 3;
    o[0] = (unsigned char)fminf(fmaxf(r + 0.5f, 0.f), 255.f);
    o[1] = (unsigned char)fminf(fmaxf(g + 0.5f, 0.f), 255.f);
    o[2] = (unsigned char)fminf(fmaxf(b + 0.5f, 0.f), 255.f);
}

extern "C" __global__
void box_reduce(const unsigned char* __restrict__ src,
                unsigned char* __restrict__ dst,
                int src_w, int src_h, int dst_w, int dst_h, int k)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;
    const int x0 = x * k, y0 = y * k;
    const int x1 = min(x0 + k, src_w), y1 = min(y0 + k, src_h);
    int r = 0, g = 0, b = 0, n = 0;
    for (int yy = y0; yy < y1; ++yy) {
        const unsigned char* row = src + (long long)yy * src_w * 3;
        for (int xx = x0; xx < x1; ++xx) {
            const unsigned char* p = row + (long long)xx * 3;
            r += p[0]; g += p[1]; b += p[2]; ++n;
        }
    }
    const float inv = 1.0f / (float)n;
    unsigned char* o = dst + ((long long)y * dst_w + x) * 3;
    o[0] = (unsigned char)(r * inv + 0.5f);
    o[1] = (unsigned char)(g * inv + 0.5f);
    o[2] = (unsigned char)(b * inv + 0.5f);
}

extern "C" __global__
void alpha_over(const unsigned char* __restrict__ src,
                unsigned char* __restrict__ dst,
                long long npix,
                float br, float bg, float bb)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix) return;
    const float a = src[i * 4 + 3] * (1.0f / 255.0f);
    const float ia = 1.0f - a;
    const unsigned char* p = src + i * 4;
    unsigned char* o = dst + i * 3;
    o[0] = (unsigned char)fminf(fmaxf(p[0] * a + br * ia + 0.5f, 0.f), 255.f);
    o[1] = (unsigned char)fminf(fmaxf(p[1] * a + bg * ia + 0.5f, 0.f), 255.f);
    o[2] = (unsigned char)fminf(fmaxf(p[2] * a + bb * ia + 0.5f, 0.f), 255.f);
}
"""


class GpuUnavailable(RuntimeError):
    """Raised when the GPU stack cannot be used at all."""


@dataclass
class GpuInfo:
    device_name: str
    compute_capability: str
    total_vram_gb: float
    driver_version: str
    nvimgcodec_version: str
    cuda_runtime: str
    free_vram_gb: float


def _kernel_value(dist: np.ndarray, name: str) -> np.ndarray:
    if name == "box":
        return (np.abs(dist) <= 0.5).astype(np.float32)
    if name == "bilinear":
        d = np.abs(dist)
        return np.clip(1.0 - d, 0.0, None).astype(np.float32)
    # lanczos3
    d = np.abs(dist)
    out = np.zeros_like(d, dtype=np.float32)
    small = d < 3.0
    x = d[small]
    out[small] = np.sinc(x) * np.sinc(x / 3.0)
    return out


def build_weights(src_len: int, dst_len: int, scale: float, offset: float,
                  filt: str) -> tuple[np.ndarray, np.ndarray]:
    """Precompute a separable resampling table (indices + normalised weights).

    ``scale`` is dst/src, ``offset`` is the source-space origin of the region
    being resampled (used by the ``cover`` crop).
    """
    from .resampling import build_separable_weights
    return build_separable_weights(src_len, dst_len, scale, offset, filt)


class GpuEngine:
    """Batched decode -> resize -> encode on one CUDA context."""

    name = "gpu"

    def __init__(self, filter_name: str = "lanczos3", strict: bool = False,
                 max_table_cache: int = 256, optimize_huffman: bool = False):
        if strict:
            raise GpuUnavailable("strict GPU-only codec attestation is unsupported; "
                                 "nvImageCodec automatic routing can use CPU plugins")
        try:
            import cupy as cp
            import nvidia.nvimgcodec as nvimg
        except Exception as exc:  # pragma: no cover - environment dependent
            raise GpuUnavailable(f"CUDA python stack not importable: {exc}") from exc

        self.cp = cp
        self.nvimg = nvimg
        self.filter = filter_name
        self.strict = strict
        self.backend_policy = "nvImageCodec automatic (actual plugin unverified)"
        self.optimize_huffman = optimize_huffman

        try:
            cp.cuda.Device(0).use()
            props = cp.cuda.runtime.getDeviceProperties(0)
        except Exception as exc:
            raise GpuUnavailable(f"no usable CUDA device: {exc}") from exc

        self._decoder = nvimg.Decoder()
        self._encoder = nvimg.Encoder()
        # nvImageCodec applies the EXIF orientation itself (verified against
        # PIL for all 8 cases), so no extra device copy is needed here.
        self._decode_rgb = nvimg.DecodeParams()
        self._decode_rgb.color_spec = nvimg.SRGB
        self._decode_rgb.sample_format = nvimg.I_RGB
        self._decode_rgb.apply_exif_orientation = True
        self._decode_rgba = nvimg.DecodeParams()
        self._decode_rgba.color_spec = nvimg.SRGB
        self._decode_rgba.sample_format = nvimg.I_RGBA
        self._decode_rgba.apply_exif_orientation = True

        self._rs_h = cp.RawKernel(_CUDA_SRC, "rs_h")
        self._rs_v = cp.RawKernel(_CUDA_SRC, "rs_v")
        self._alpha = cp.RawKernel(_CUDA_SRC, "alpha_over")
        self._box = cp.RawKernel(_CUDA_SRC, "box_reduce")
        self._block = (32, 8)
        self._table_cache: dict[tuple, tuple] = {}
        self._cache_lock = threading.Lock()
        self._max_cache = max_table_cache
        self._cache_bytes = 0
        self._cache_limit = 64 * 2**20

        free, total = cp.cuda.runtime.memGetInfo()
        drv = cp.cuda.runtime.driverGetVersion()
        nv = getattr(nvimg, "__version__", "?")
        rt = cp.cuda.runtime.runtimeGetVersion()
        self.info = GpuInfo(
            device_name=props["name"].decode() if isinstance(props["name"], bytes)
            else str(props["name"]),
            compute_capability=f"{props['major']}.{props['minor']}",
            total_vram_gb=total / 2 ** 30,
            free_vram_gb=free / 2 ** 30,
            driver_version=f"{drv // 1000}.{(drv % 1000) // 10}",
            nvimgcodec_version=str(nv),
            cuda_runtime=f"{rt // 1000}.{(rt % 1000) // 10}",
        )

    def activate(self):
        """Select the engine device in the worker, not just in its creator."""
        self.cp.cuda.Device(0).use()
        self.cp.cuda.Stream.null.use()

    def synchronize(self):
        self.cp.cuda.Stream.null.synchronize()

    def clear_cache(self):
        with self._cache_lock:
            self._table_cache.clear()
            self._cache_bytes = 0

    # -- capabilities --------------------------------------------------------
    def self_test(self) -> tuple[bool, str]:
        """End-to-end smoke test of the accelerated path."""
        import io

        from PIL import Image

        rng = np.random.default_rng(7)
        arr = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "JPEG", quality=92)
        try:
            plan = Plan(640, 480, 320, 240, 0.0, 0.0, 640.0, 480.0)
            out = self.process_bytes(buf.getvalue(), plan, 90, 2)
            w, h = self._jpeg_size(out)
            if (w, h) != (320, 240):
                return False, f"unexpected output size {w}x{h}"
            return True, "decode+resize+encode OK"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _jpeg_size(jpeg: bytes) -> tuple[int, int]:
        from .imageio_meta import jpeg_dimensions

        return jpeg_dimensions(jpeg)

    # -- tables --------------------------------------------------------------
    def _table(self, src_len: int, dst_len: int, scale: float, offset: float):
        key = (src_len, dst_len, round(scale, 6), round(offset, 3), self.filter)
        with self._cache_lock:
            hit = self._table_cache.get(key)
        if hit is not None:
            return hit
        idx, w = build_weights(src_len, dst_len, scale, offset, self.filter)
        dev = (self.cp.asarray(idx), self.cp.asarray(w))
        with self._cache_lock:
            size = int(idx.nbytes + w.nbytes)
            while self._table_cache and (len(self._table_cache) >= self._max_cache or
                                        self._cache_bytes + size > self._cache_limit):
                old = self._table_cache.pop(next(iter(self._table_cache)))
                self._cache_bytes -= sum(int(x.nbytes) for x in old)
            if size <= self._cache_limit:
                self._table_cache[key] = dev
                self._cache_bytes += size
        return dev

    # -- device helpers ------------------------------------------------------
    def to_device(self, rgb: np.ndarray):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError("upload requires packed uint8 RGB/RGBA")
        return self.cp.asarray(np.ascontiguousarray(rgb))

    def apply_orientation(self, arr, orientation: int):
        cp = self.cp
        if orientation == 1:
            return arr
        if orientation == 2:
            return cp.ascontiguousarray(arr[:, ::-1])
        if orientation == 3:
            return cp.ascontiguousarray(arr[::-1, ::-1])
        if orientation == 4:
            return cp.ascontiguousarray(arr[::-1])
        if orientation == 5:
            return cp.ascontiguousarray(arr.transpose(1, 0, 2))
        if orientation == 6:
            return cp.ascontiguousarray(cp.rot90(arr, -1))
        if orientation == 7:
            return cp.ascontiguousarray(arr.transpose(1, 0, 2)[::-1, ::-1])
        if orientation == 8:
            return cp.ascontiguousarray(cp.rot90(arr, 1))
        return arr

    def alpha_to_rgb(self, arr, bg: tuple[int, int, int]):
        cp = self.cp
        if arr.dtype != cp.uint8 or arr.ndim != 3 or int(arr.shape[2]) != 4:
            raise ValueError("alpha composite requires uint8 RGBA")
        arr = cp.ascontiguousarray(arr)
        h, w = arr.shape[0], arr.shape[1]
        out = cp.empty((h, w, 3), dtype=cp.uint8)
        n = h * w
        block = 256
        grid = ((n + block - 1) // block,)
        self._alpha(grid, (block,),
                    (arr, out, np.int64(n),
                     np.float32(bg[0]), np.float32(bg[1]), np.float32(bg[2])))
        return out

    def _prefilter_factor(self, plan: Plan) -> int:
        """Integer box-reduction factor applied before the resample.

        Large downscales need dozens of lanczos taps per output pixel; a
        coalesced k x k box pass first cuts the gather cost by ~k^2 and also
        acts as a proper antialiasing prefilter (same idea as ``PIL.reduce``).
        """
        if self.filter == "box":
            return 1
        fx = 1.0 / plan.scale_x if plan.scale_x > 0 else 1.0
        fy = 1.0 / plan.scale_y if plan.scale_y > 0 else 1.0
        f = min(fx, fy)
        if f < 4.0:
            return 1
        return max(1, min(int(f // 2), 8))

    def box_reduce(self, src, k: int):
        cp = self.cp
        src_h, src_w = int(src.shape[0]), int(src.shape[1])
        dst_w = (src_w + k - 1) // k
        dst_h = (src_h + k - 1) // k
        out = cp.empty((dst_h, dst_w, 3), dtype=cp.uint8)
        bx, by = self._block
        self._box(((dst_w + bx - 1) // bx, (dst_h + by - 1) // by), (bx, by),
                  (src, out, src_w, src_h, dst_w, dst_h, np.int32(k)))
        return out

    def resize(self, src, plan: Plan):
        cp = self.cp
        if src.dtype != cp.uint8 or src.ndim != 3 or int(src.shape[2]) != 3:
            raise ValueError("CUDA resize requires uint8 HWC RGB (C3)")
        src = cp.ascontiguousarray(src)
        if tuple(src.shape[:2]) != (plan.src_h, plan.src_w):
            raise ValueError("decoded size differs from oriented geometry")
        if plan.is_identity:
            return cp.ascontiguousarray(src)

        k = self._prefilter_factor(plan)
        if k > 1:
            src = self.box_reduce(src, k)
            inv = 1.0 / k
            plan = Plan(
                src_w=max(1, int(round(plan.src_w * inv))),
                src_h=max(1, int(round(plan.src_h * inv))),
                out_w=plan.out_w, out_h=plan.out_h,
                x0=plan.x0 * inv, y0=plan.y0 * inv,
                cw=plan.cw * inv, ch=plan.ch * inv,
            )

        src_h, src_w = int(src.shape[0]), int(src.shape[1])
        ix, wx = self._table(src_w, plan.out_w, plan.scale_x, plan.x0)
        iy, wy = self._table(src_h, plan.out_h, plan.scale_y, plan.y0)
        taps_x = int(wx.shape[1])
        taps_y = int(wy.shape[1])

        tmp = cp.empty((src_h, plan.out_w, 3), dtype=cp.float32)
        bx, by = self._block
        self._rs_h(((plan.out_w + bx - 1) // bx, (src_h + by - 1) // by),
                   (bx, by), (src, tmp, ix, wx, src_w, src_h, plan.out_w, taps_x))

        out = cp.empty((plan.out_h, plan.out_w, 3), dtype=cp.uint8)
        self._rs_v(((plan.out_w + bx - 1) // bx, (plan.out_h + by - 1) // by),
                   (bx, by), (tmp, out, iy, wy, src_h, plan.out_w, plan.out_h, taps_y))
        return out

    def _encode_params(self, quality: int, subsampling: int):
        nvimg = self.nvimg
        ep = nvimg.EncodeParams()
        ep.quality_type = nvimg.QUALITY
        ep.quality_value = int(quality)
        ep.chroma_subsampling = {
            0: nvimg.CSS_444, 1: nvimg.CSS_422, 2: nvimg.CSS_420
        }[subsampling]
        if self.optimize_huffman:
            jp = ep.jpeg_params
            jp.optimized_huffman = True
        return ep

    def encode(self, arr, quality: int, subsampling: int,
               qtables: tuple[list[int], list[int]] | None = None) -> bytes:
        """Encode a JPEG frame.

        ``qtables`` (the quantisation tables of the source file) is accepted
        for interface symmetry with the CPU engine, but nvJPEG cannot write
        custom quantisation tables - the quality number is used instead.
        """
        if arr.dtype != self.cp.uint8 or arr.ndim != 3 or int(arr.shape[2]) != 3:
            raise ValueError("JPEG encoder requires uint8 RGB")
        arr = self.cp.ascontiguousarray(arr)
        cs = self._encoder.encode(arr, "jpeg",
                                  params=self._encode_params(quality, subsampling))
        if cs is None:
            raise RuntimeError("nvJPEG encode failed")
        return bytes(cs)

    # -- public API ----------------------------------------------------------
    def _recycle_decoder_if_needed(self) -> None:
        """Best-effort memory-pressure mitigation, not proof of a library leak.

        Opaque codec allocations and other processes are outside our budget.
        Rebuilding can stall and is not a hard VRAM bound.
        """
        cp = self.cp
        free, total = cp.cuda.runtime.memGetInfo()
        if free < total * 0.30:
            self._decoder = self.nvimg.Decoder()
            cp.get_default_memory_pool().free_all_blocks()

    def decode_batch(self, datas: Sequence[bytes], want_alpha: Sequence[bool]):
        """Decode a batch in one nvJPEG call per colour layout."""
        nvimg = self.nvimg
        self._recycle_decoder_if_needed()
        streams: list[Optional[object]] = []
        for d in datas:
            try:
                streams.append(nvimg.CodeStream(d))
            except Exception:
                streams.append(None)  # header unreadable -> treated as failure

        rgb_idx = [i for i, a in enumerate(want_alpha)
                   if not a and streams[i] is not None]
        rgba_idx = [i for i, a in enumerate(want_alpha)
                    if a and streams[i] is not None]
        result: list[Optional[object]] = [None] * len(datas)
        for idxs, params in ((rgb_idx, self._decode_rgb),
                             (rgba_idx, self._decode_rgba)):
            if not idxs:
                continue
            # one batched nvJPEG call per colour layout
            imgs = self._decoder.decode([streams[i] for i in idxs], params=params)
            for j, i in enumerate(idxs):
                img = imgs[j]
                if img is None:
                    continue
                result[i] = self.cp.asarray(img)

        # Do not retry alpha as RGB: callers must use alpha-aware host fallback.
        self.cp.cuda.Stream.null.synchronize()
        return result

    def process_array(self, arr, plan: Plan, quality: int, subsampling: int,
                      bg: tuple[int, int, int] = (255, 255, 255),
                      qtables: tuple[list[int], list[int]] | None = None) -> bytes:
        if int(arr.shape[2]) == 4:
            arr = self.alpha_to_rgb(arr, bg)
        small = self.resize(arr, plan)
        return self.encode(small, quality, subsampling, qtables)

    def process_bytes(self, data: bytes, plan: Plan, quality: int,
                      subsampling: int) -> bytes:
        arr = self.decode_batch([data], [False])[0]
        if arr is None:
            raise RuntimeError("decode failed")
        return self.process_array(arr, plan, quality, subsampling)


def resolve_subsampling(name: str, quality: int) -> int:
    from .presets import SUBSAMPLING_NAMES

    v = SUBSAMPLING_NAMES[name]
    return auto_subsampling(quality) if v is None else v
