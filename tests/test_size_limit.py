"""Unit tests for size limit parsing and JPEG quality optimization (Stage 6)."""

import io
import time
import cv2
import numpy as np
import pytest
from PIL import Image

from batchresize.size_limit import (
    optimize_jpeg_quality,
    parse_size_bytes,
)


class TestParseSizeBytes:
    def test_decimal_si_units(self):
        assert parse_size_bytes("1KB") == 1_000
        assert parse_size_bytes("500KB") == 500_000
        assert parse_size_bytes("1.5MB") == 1_500_000
        assert parse_size_bytes("2GB") == 2_000_000_000
        assert parse_size_bytes("100kb") == 100_000
        assert parse_size_bytes("10K") == 10_000
        assert parse_size_bytes("2M") == 2_000_000

    def test_binary_iec_units(self):
        assert parse_size_bytes("1KiB") == 1_024
        assert parse_size_bytes("500KiB") == 512_000
        assert parse_size_bytes("1.5MiB") == 1_572_864
        assert parse_size_bytes("1GiB") == 1_073_741_824
        assert parse_size_bytes("64kib") == 65_536

    def test_raw_integers_and_bytes(self):
        assert parse_size_bytes(5000) == 5000
        assert parse_size_bytes(100_000.0) == 100_000
        assert parse_size_bytes("5000") == 5000
        assert parse_size_bytes("5000B") == 5000
        assert parse_size_bytes("5000 byte") == 5000
        assert parse_size_bytes("5000 bytes") == 5000

    def test_whitespace_handling(self):
        assert parse_size_bytes("  500 KB  ") == 500_000
        assert parse_size_bytes("  2.5   MiB ") == 2_621_440

    @pytest.mark.parametrize("invalid", [
        0,
        -100,
        -1.5,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        "0KB",
        "-50KB",
        "foo",
        "100TB",
        "100XYZ",
        "",
        "   ",
    ])
    def test_invalid_sizes_rejected(self, invalid):
        with pytest.raises(ValueError):
            parse_size_bytes(invalid)


class TestOptimizeJpegQuality:
    @pytest.fixture
    def test_image(self):
        # A 256x256 RGB image with high-entropy gradient
        np.random.seed(42)
        return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    def test_already_fits_at_initial_quality(self, test_image):
        def encode(img, q):
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            return bytes(buf)

        # Huge limit: fits on trial 1
        jpeg, q, achieved = optimize_jpeg_quality(
            encode, test_image, initial_quality=85, min_quality=30, max_bytes=10_000_000
        )
        assert achieved is True
        assert q == 85
        assert len(jpeg) <= 10_000_000

    def test_binary_search_finds_valid_quality(self, test_image):
        trials_count = 0

        def encode(img, q):
            nonlocal trials_count
            trials_count += 1
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            return bytes(buf)

        # Target between min_quality (e.g. ~30KB) and initial_quality (e.g. ~70KB)
        # Check actual sizes first
        size_min = len(encode(test_image, 30))
        size_max = len(encode(test_image, 90))
        target_limit = (size_min + size_max) // 2

        trials_count = 0
        jpeg, q, achieved = optimize_jpeg_quality(
            encode,
            test_image,
            initial_quality=90,
            min_quality=30,
            max_bytes=target_limit,
            max_trials=7,
        )

        assert achieved is True
        assert 30 <= q <= 90
        assert len(jpeg) <= target_limit
        assert trials_count <= 7

    def test_unreachable_target_reports_failure(self, test_image):
        def encode(img, q):
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            return bytes(buf)

        # Impossible target (50 bytes)
        jpeg, q, achieved = optimize_jpeg_quality(
            encode, test_image, initial_quality=85, min_quality=30, max_bytes=50
        )
        assert achieved is False
        assert q == 30
        assert len(jpeg) > 50

    def test_splice_fn_metadata_accounted(self, test_image):
        # Splice adds a 5000-byte header
        metadata_overhead = b"M" * 5000

        def encode(img, q):
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            return bytes(buf)

        def splice(j):
            return metadata_overhead + j

        raw_size = len(encode(test_image, 60))
        # Set limit such that raw_size fits, but spliced (raw_size + 5000) does NOT fit
        limit = raw_size + 2000

        jpeg, q, achieved = optimize_jpeg_quality(
            encode,
            test_image,
            initial_quality=85,
            min_quality=20,
            max_bytes=limit,
            splice_fn=splice,
        )

        # Spliced result must fit under limit
        assert len(jpeg) <= limit
        assert jpeg.startswith(metadata_overhead)
        assert achieved is True
        # Actual quality must be lower than 60 because 60 would be > limit
        assert q < 60

    def test_non_monotonic_behavior_tracking(self):
        # Mock encoder where quality 50 produces 2000 bytes, but quality 60 produces 1500 bytes (anomaly)
        mock_sizes = {
            80: 5000,
            30: 1000,
            55: 3000,
            67: 4000,
            42: 1200,
            48: 1800,
        }

        def encode(img, q):
            size = mock_sizes.get(q, q * 50)
            return b"X" * size

        fake_img = np.zeros((10, 10, 3), dtype=np.uint8)
        jpeg, q, achieved = optimize_jpeg_quality(
            encode,
            fake_img,
            initial_quality=80,
            min_quality=30,
            max_bytes=2500,
            max_trials=7,
        )
        assert achieved is True
        assert len(jpeg) <= 2500
