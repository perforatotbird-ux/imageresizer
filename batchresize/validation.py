"""Lightweight validation shared by entrypoints, geometry and pipeline."""
import math

def positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value

def dimensions(w, h, max_pixels=100_000_000):
    positive(w, "width"); positive(h, "height")
    if int(w) != w or int(h) != h:
        raise ValueError("dimensions must be integers")
    if w * h > max_pixels:
        raise ValueError(f"image exceeds {max_pixels} pixel limit")
