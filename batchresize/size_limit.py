"""Target file size optimization and byte unit parsing."""

from __future__ import annotations

import re
import time
from typing import Callable, Optional, Tuple
import numpy as np

SIZE_PATTERN = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)$")

UNIT_MULTIPLIERS = {
    "": 1,
    "b": 1,
    "byte": 1,
    "bytes": 1,
    # Decimal SI units (1 KB = 1000 bytes)
    "kb": 1_000,
    "k": 1_000,
    "mb": 1_000_000,
    "m": 1_000_000,
    "gb": 1_000_000_000,
    "g": 1_000_000_000,
    # Binary IEC units (1 KiB = 1024 bytes)
    "kib": 1_024,
    "mib": 1_048_576,
    "gib": 1_073_741_824,
}


def parse_size_bytes(value: str | int | float) -> int:
    """Parse size string with SI/IEC unit or integer into exact bytes count.

    Examples:
        '500KB' -> 500_000
        '500KiB' -> 512_000
        '1.5MB' -> 1_500_000
        '1.5MiB' -> 1_572_864
        '250000' -> 250_000
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        import math
        if value <= 0 or not math.isfinite(value):
            raise ValueError(f"size limit must be positive finite number, got {value}")
        return int(round(value))

    text = str(value).strip().lower()
    match = SIZE_PATTERN.match(text)
    if not match:
        raise ValueError(f"cannot parse size string {value!r} (expected e.g. '500KB', '500KiB', '2MB')")

    num_str, unit_str = match.groups()
    num = float(num_str)
    if num <= 0:
        raise ValueError(f"size limit must be positive, got {value!r}")

    if unit_str not in UNIT_MULTIPLIERS:
        raise ValueError(
            f"unknown size unit {unit_str!r} in {value!r} "
            f"(supported: KB, KiB, MB, MiB, B)"
        )

    multiplier = UNIT_MULTIPLIERS[unit_str]
    total_bytes = int(round(num * multiplier))
    if total_bytes < 1:
        raise ValueError(f"size limit too small: {value!r}")
    return total_bytes


def optimize_jpeg_quality(
    encode_fn: Callable[[np.ndarray, int], bytes],
    img: np.ndarray,
    initial_quality: int,
    min_quality: int,
    max_bytes: int,
    splice_fn: Optional[Callable[[bytes], bytes]] = None,
    max_trials: int = 7,
    max_seconds: float = 2.0,
) -> Tuple[bytes, int, bool]:
    """Find the highest JPEG quality in [min_quality, initial_quality] that fits under max_bytes.

    Includes final spliced metadata in byte count measurement.
    Does not rely on strict monotonicity: tracks best observed valid JPEG.

    Returns:
        (jpeg_bytes, actual_quality, target_achieved)
    """
    if splice_fn is None:
        splice_fn = lambda j: j

    min_quality = max(1, min(int(min_quality), int(initial_quality)))
    initial_quality = max(1, min(100, int(initial_quality)))

    # Trial 1: initial quality
    trial_jpeg = splice_fn(encode_fn(img, initial_quality))
    if len(trial_jpeg) <= max_bytes:
        return trial_jpeg, initial_quality, True

    if initial_quality <= min_quality:
        return trial_jpeg, initial_quality, False

    t0 = time.perf_counter()
    trials = 1

    # Check minimum quality
    min_jpeg = splice_fn(encode_fn(img, min_quality))
    trials += 1
    if len(min_jpeg) > max_bytes:
        # Unreachable even at minimum allowed quality
        return min_jpeg, min_quality, False

    best_jpeg = min_jpeg
    best_q = min_quality

    # Binary search over [min_quality + 1, initial_quality - 1]
    low = min_quality + 1
    high = initial_quality - 1

    while low <= high and trials < max_trials and (time.perf_counter() - t0) < max_seconds:
        mid = (low + high) // 2
        mid_jpeg = splice_fn(encode_fn(img, mid))
        trials += 1
        size = len(mid_jpeg)

        if size <= max_bytes:
            best_jpeg = mid_jpeg
            best_q = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_jpeg, best_q, True
