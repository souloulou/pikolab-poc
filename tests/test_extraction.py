"""Tests for color extraction functions."""

import numpy as np
import pytest
from app import compute_skin_stats, extract_iris_dominant, extract_pixels, pixels_to_lab


class TestExtractPixels:
    def test_extracts_masked_pixels(self, skin_tone_image, full_mask_100):
        pixels = extract_pixels(skin_tone_image, full_mask_100)
        assert pixels.shape == (10000, 3)  # 100x100 pixels
        np.testing.assert_array_equal(pixels[0], [185, 150, 130])

    def test_empty_mask_returns_empty(self, skin_tone_image, empty_mask_100):
        pixels = extract_pixels(skin_tone_image, empty_mask_100)
        assert len(pixels) == 0
        assert pixels.shape == (0, 3)

    def test_partial_mask(self, skin_tone_image):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # 20x20 region
        pixels = extract_pixels(skin_tone_image, mask)
        assert pixels.shape == (400, 3)


class TestPixelsToLab:
    def test_white_pixel(self):
        white = np.array([[255, 255, 255]], dtype=np.uint8)
        lab = pixels_to_lab(white)
        assert lab.shape == (1, 3)
        assert lab[0, 0] > 95  # L* should be close to 100

    def test_black_pixel(self):
        black = np.array([[0, 0, 0]], dtype=np.uint8)
        lab = pixels_to_lab(black)
        assert lab[0, 0] < 5  # L* close to 0

    def test_red_pixel_has_positive_a(self):
        red = np.array([[255, 0, 0]], dtype=np.uint8)
        lab = pixels_to_lab(red)
        assert lab[0, 1] > 30  # a* should be strongly positive

    def test_blue_pixel_has_negative_b(self):
        blue = np.array([[0, 0, 255]], dtype=np.uint8)
        lab = pixels_to_lab(blue)
        assert lab[0, 2] < -30  # b* should be strongly negative

    def test_empty_input(self):
        empty = np.array([]).reshape(0, 3)
        lab = pixels_to_lab(empty)
        assert lab.shape == (0, 3)

    def test_skin_tone_range(self):
        skin = np.array([[185, 150, 130]], dtype=np.uint8)
        lab = pixels_to_lab(skin)
        L, a, b = lab[0]
        assert 40 < L < 80, f"Skin L*={L} out of expected range"
        assert 0 < a < 30, f"Skin a*={a} out of expected range"
        assert 0 < b < 35, f"Skin b*={b} out of expected range"

    def test_batch_conversion(self):
        pixels = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        lab = pixels_to_lab(pixels)
        assert lab.shape == (3, 3)


class TestComputeSkinStats:
    def test_uniform_pixels(self):
        # All same color
        lab = np.array([[55.0, 12.0, 20.0]] * 100)
        stats = compute_skin_stats(lab)
        assert abs(stats["L"] - 55.0) < 0.01
        assert abs(stats["a"] - 12.0) < 0.01
        assert abs(stats["b"] - 20.0) < 0.01
        expected_c = np.sqrt(12**2 + 20**2)
        assert abs(stats["C"] - expected_c) < 0.01

    def test_empty_returns_zeros(self):
        lab = np.array([]).reshape(0, 3)
        stats = compute_skin_stats(lab)
        assert stats["L"] == 0.0
        assert stats["C"] == 0.0

    def test_chroma_is_nonnegative(self):
        lab = np.array([[50.0, -10.0, -15.0]] * 50)
        stats = compute_skin_stats(lab)
        assert stats["C"] > 0

    def test_mixed_pixels(self):
        lab = np.array([
            [60.0, 10.0, 20.0],
            [40.0, 14.0, 24.0],
        ])
        stats = compute_skin_stats(lab)
        assert abs(stats["L"] - 50.0) < 0.01
        assert abs(stats["a"] - 12.0) < 0.01
        assert abs(stats["b"] - 22.0) < 0.01


class TestExtractIrisDominant:
    def test_returns_none_for_few_pixels(self):
        pixels = np.array([[100, 80, 60]] * 5, dtype=np.uint8)
        assert extract_iris_dominant(pixels) is None

    def test_returns_dict_for_enough_pixels(self):
        rng = np.random.RandomState(42)
        # Simulate brown iris
        pixels = rng.randint(80, 150, size=(200, 3)).astype(np.uint8)
        pixels[:, 0] = np.clip(pixels[:, 0] + 30, 0, 255)  # More red
        result = extract_iris_dominant(pixels)
        assert result is not None
        assert "L" in result
        assert "a" in result
        assert "b" in result
        assert "C" in result
        assert "rgb" in result

    def test_dominant_color_is_not_darkest(self):
        # Mix: 100 dark pixels (pupil), 100 medium (iris), 50 bright (reflection)
        dark = np.full((100, 3), [20, 15, 10], dtype=np.uint8)
        medium = np.full((100, 3), [120, 100, 80], dtype=np.uint8)
        bright = np.full((50, 3), [240, 240, 240], dtype=np.uint8)
        pixels = np.vstack([dark, medium, bright])
        result = extract_iris_dominant(pixels)
        # Dominant should be the medium cluster, not dark or bright
        assert result["L"] > 20, "Should not pick the darkest (pupil)"
        assert result["L"] < 90, "Should not pick the brightest (reflection)"

    def test_rgb_field_is_integer(self):
        pixels = np.full((50, 3), [100, 90, 80], dtype=np.uint8)
        result = extract_iris_dominant(pixels)
        if result:
            assert result["rgb"].dtype in [np.int32, np.int64, np.intp]
