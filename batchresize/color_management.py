"""Color management, color policies, linear-light conversions, and alpha operations."""

from __future__ import annotations

import io
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageCms

COLOR_POLICY_PRESERVE = "preserve_rgb"
COLOR_POLICY_SRGB = "web_srgb"
SUPPORTED_COLOR_POLICIES = {
    "preserve_rgb",
    "preserve",
    "web_srgb",
    "srgb",
}

# Pre-created sRGB ICC profile bytes (standard IEC 61966-2-1)
_srgb_profile_obj = ImageCms.createProfile("sRGB")
SRGB_ICC_BYTES: bytes = ImageCms.ImageCmsProfile(_srgb_profile_obj).tobytes()

# Precalculated LUT for sRGB uint8 [0..255] -> linear float32 [0.0..1.0]
_c = np.arange(256, dtype=np.float32) / 255.0
SRGB_TO_LINEAR_LUT: np.ndarray = np.where(
    _c <= 0.04045,
    _c / 12.92,
    np.power((_c + 0.055) / 1.055, 2.4),
).astype(np.float32)


def canonical_color_policy(policy: str) -> str:
    """Normalize color policy name to canonical form."""
    p = str(policy).strip().lower()
    if p in ("preserve", "preserve_rgb"):
        return COLOR_POLICY_PRESERVE
    if p in ("srgb", "web_srgb"):
        return COLOR_POLICY_SRGB
    raise ValueError(f"unknown color policy: {policy!r} (must be 'preserve_rgb' or 'web_srgb')")


def is_valid_icc(icc_bytes: Optional[bytes]) -> bool:
    """Check if the provided bytes represent a valid, parseable ICC profile."""
    if not icc_bytes:
        return False
    try:
        ImageCms.getOpenProfile(io.BytesIO(icc_bytes))
        return True
    except Exception:
        return False


def get_icc_color_space(icc_bytes: bytes) -> str:
    """Get color space name from ICC profile (e.g. 'RGB', 'CMYK', 'GRAY')."""
    profile = ImageCms.getOpenProfile(io.BytesIO(icc_bytes))
    return str(profile.profile.xcolor_space).strip()


def srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    """Convert sRGB array (uint8 or float32 [0.0..1.0]) to linear float32 [0.0..1.0]."""
    if arr.dtype == np.uint8:
        return SRGB_TO_LINEAR_LUT[arr]
    arr_f = np.clip(arr.astype(np.float32), 0.0, 1.0)
    return np.where(
        arr_f <= 0.04045,
        arr_f / 12.92,
        np.power((arr_f + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_to_srgb(arr: np.ndarray) -> np.ndarray:
    """Convert linear float32 [0.0..1.0] to sRGB uint8 [0..255]."""
    clipped = np.clip(arr, 0.0, 1.0).astype(np.float32)
    low = clipped <= 0.0031308
    out = np.empty_like(clipped, dtype=np.float32)
    out[low] = clipped[low] * 12.92
    out[~low] = 1.055 * np.power(clipped[~low], 1.0 / 2.4) - 0.055
    return np.clip(np.round(out * 255.0), 0.0, 255.0).astype(np.uint8)


def premultiply_alpha(rgba_float: np.ndarray) -> np.ndarray:
    """Premultiply RGB by alpha in float32 space [0.0..1.0]."""
    out = np.empty_like(rgba_float, dtype=np.float32)
    alpha = rgba_float[:, :, 3:4]
    out[:, :, :3] = rgba_float[:, :, :3] * alpha
    out[:, :, 3:4] = alpha
    return out


def unpremultiply_alpha(rgba_premul: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Safely unpremultiply RGB from alpha."""
    alpha = np.clip(rgba_premul[:, :, 3:4], 0.0, 1.0)
    safe_alpha = np.where(alpha > eps, alpha, 1.0)
    rgb = np.where(alpha > eps, rgba_premul[:, :, :3] / safe_alpha, 0.0)
    out = np.empty_like(rgba_premul, dtype=np.float32)
    out[:, :, :3] = np.clip(rgb, 0.0, 1.0)
    out[:, :, 3:4] = alpha
    return out


def composite_linear_bg(
    rgba_premul: np.ndarray,
    bg_rgb_uint8: tuple[int, int, int],
    is_premultiplied: bool = True,
) -> np.ndarray:
    """Composite RGBA onto background in linear light space.
    
    Returns RGB float32 in [0.0..1.0].
    """
    bg_arr = np.array([bg_rgb_uint8[0], bg_rgb_uint8[1], bg_rgb_uint8[2]], dtype=np.uint8)
    bg_lin = SRGB_TO_LINEAR_LUT[bg_arr]
    alpha = np.clip(rgba_premul[:, :, 3:4], 0.0, 1.0)
    if is_premultiplied:
        rgb_term = rgba_premul[:, :, :3]
    else:
        rgb_term = rgba_premul[:, :, :3] * alpha
    out = rgb_term + bg_lin[None, None, :] * (1.0 - alpha)
    return np.clip(out, 0.0, 1.0)


def decode_and_manage_color(
    data: bytes,
    policy: str = COLOR_POLICY_PRESERVE,
    keep_icc: bool = True,
) -> Tuple[np.ndarray, bool, Optional[bytes]]:
    """Decode image data and manage color profile according to policy.

    Returns:
        (image_rgb_or_rgba_uint8, has_alpha, output_icc_bytes)
    """
    canonical_policy = canonical_color_policy(policy)

    with Image.open(io.BytesIO(data)) as im:
        im.load()
        has_alpha = "A" in im.getbands() or "transparency" in im.info
        alpha = None
        if has_alpha:
            alpha = im.convert("RGBA").getchannel("A")

        icc_raw = im.info.get("icc_profile")
        if isinstance(icc_raw, str):
            icc_raw = icc_raw.encode("latin-1", "ignore")

        src_profile = None
        if icc_raw:
            try:
                src_profile = ImageCms.getOpenProfile(io.BytesIO(icc_raw))
            except Exception as exc:
                raise ValueError(
                    f"cannot parse source ICC profile safely: corrupt profile ({exc})"
                ) from exc

        srgb_profile = ImageCms.createProfile("sRGB")

        if canonical_policy == COLOR_POLICY_SRGB:
            if src_profile is not None:
                # Convert from source ICC to sRGB
                colour = im.convert("L") if im.mode == "LA" else (
                    im.convert("RGB") if im.mode in ("RGB", "RGBA") else im
                )
                im = ImageCms.profileToProfile(
                    colour, src_profile, srgb_profile, outputMode="RGB"
                )
            elif im.mode in ("CMYK", "YCbCr"):
                im = im.convert("RGB")
            elif im.mode.startswith("I;16"):
                arr = np.asarray(im, dtype=np.float32)
                arr = np.clip(arr / 257.0, 0, 255).round().astype(np.uint8)
                im = Image.fromarray(np.repeat(arr[:, :, None], 3, axis=2))
            elif im.mode in ("I", "F"):
                arr = np.clip(np.asarray(im, dtype=np.float32), 0, 255).astype(np.uint8)
                im = Image.fromarray(np.repeat(arr[:, :, None], 3, axis=2))
            else:
                im = im.convert("RGB")

            output_icc = SRGB_ICC_BYTES if keep_icc else None

        else:
            # COLOR_POLICY_PRESERVE
            if src_profile is not None:
                space = str(src_profile.profile.xcolor_space).strip()
                if space == "RGB" and im.mode in ("RGB", "RGBA", "P"):
                    # Preserve original RGB color and profile
                    im = im.convert("RGB")
                    output_icc = icc_raw if keep_icc else None
                else:
                    # Non-RGB space with profile (e.g. CMYK): convert to sRGB
                    colour = im.convert("L") if im.mode == "LA" else im
                    im = ImageCms.profileToProfile(
                        colour, src_profile, srgb_profile, outputMode="RGB"
                    )
                    # When converted to another space, never keep the non-RGB ICC
                    output_icc = SRGB_ICC_BYTES if keep_icc else None
            elif im.mode in ("RGB", "RGBA", "P"):
                im = im.convert("RGB")
                # Do not assign fake sRGB if no ICC profile was present on RGB
                output_icc = None
            elif im.mode.startswith("I;16"):
                arr = np.asarray(im, dtype=np.float32)
                arr = np.clip(arr / 257.0, 0, 255).round().astype(np.uint8)
                im = Image.fromarray(np.repeat(arr[:, :, None], 3, axis=2))
                output_icc = SRGB_ICC_BYTES if keep_icc else None
            elif im.mode in ("I", "F"):
                arr = np.clip(np.asarray(im, dtype=np.float32), 0, 255).astype(np.uint8)
                im = Image.fromarray(np.repeat(arr[:, :, None], 3, axis=2))
                output_icc = SRGB_ICC_BYTES if keep_icc else None
            else:
                # CMYK or other non-RGB without ICC
                im = im.convert("RGB")
                output_icc = SRGB_ICC_BYTES if keep_icc else None

        if alpha is not None:
            im.putalpha(alpha)

        arr_out = np.ascontiguousarray(np.asarray(im))
        return arr_out, alpha is not None, output_icc
