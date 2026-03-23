"""Tests for color correction functions."""

import numpy as np
import pytest
from app import correct_exposure, correct_wb_with_reference, detect_white_region


class TestDetectWhiteRegion:
    def test_finds_white_sheet(self, white_sheet_image):
        ref = detect_white_region(white_sheet_image)
        assert ref is not None, "Should detect the white rectangle"
        assert ref.shape == (3,)
        # All channels should be close to 250
        assert all(c > 230 for c in ref), f"White region mean should be bright, got {ref}"

    def test_returns_none_for_no_white(self):
        dark = np.full((100, 100, 3), [50, 40, 30], dtype=np.uint8)
        assert detect_white_region(dark) is None

    def test_returns_none_for_small_white_spot(self):
        img = np.full((200, 200, 3), [80, 70, 60], dtype=np.uint8)
        # Tiny white spot (< 2% of image)
        img[95:105, 95:105] = [250, 250, 250]
        assert detect_white_region(img) is None

    def test_ignores_colored_bright_regions(self):
        img = np.full((200, 200, 3), [80, 70, 60], dtype=np.uint8)
        # Bright but saturated (not white)
        img[20:180, 20:180] = [255, 100, 100]
        assert detect_white_region(img) is None


class TestCorrectWbWithReference:
    def test_neutral_reference_no_change(self):
        img = np.full((50, 50, 3), [128, 128, 128], dtype=np.uint8)
        ref = np.array([255.0, 255.0, 255.0])
        corrected = correct_wb_with_reference(img, ref)
        np.testing.assert_array_equal(corrected, img)

    def test_warm_cast_corrected(self):
        img = np.full((50, 50, 3), [200, 180, 150], dtype=np.uint8)
        # Reference shows warm cast: R and G are fine, B is low
        ref = np.array([250.0, 250.0, 200.0])
        corrected = correct_wb_with_reference(img, ref)
        # Blue channel should be boosted relative to others
        mean_orig_b = img[:, :, 2].mean()
        mean_corr_b = corrected[:, :, 2].mean()
        mean_orig_r = img[:, :, 0].mean()
        mean_corr_r = corrected[:, :, 0].mean()
        # B/R ratio should increase
        assert (mean_corr_b / mean_corr_r) > (mean_orig_b / mean_orig_r)

    def test_output_clipped_to_255(self):
        img = np.full((10, 10, 3), [200, 200, 200], dtype=np.uint8)
        ref = np.array([100.0, 100.0, 100.0])
        corrected = correct_wb_with_reference(img, ref)
        assert corrected.max() <= 255
        assert corrected.dtype == np.uint8

    def test_does_not_crash_with_zero_reference(self):
        img = np.full((10, 10, 3), [128, 128, 128], dtype=np.uint8)
        ref = np.array([0.0, 0.0, 0.0])
        corrected = correct_wb_with_reference(img, ref)
        assert corrected.shape == img.shape


class TestCorrectExposure:
    def test_brightens_severely_dark_image(self):
        # Very underexposed: median L will be < 40 in OpenCV LAB
        img = np.full((50, 50, 3), [15, 10, 8], dtype=np.uint8)
        mask = np.full((50, 50), 255, dtype=np.uint8)
        corrected = correct_exposure(img, mask)
        assert corrected.mean() > img.mean(), "Severely dark image should be brightened"

    def test_darkens_severely_bright_image(self):
        # Very overexposed: median L > 230
        img = np.full((50, 50, 3), [250, 248, 245], dtype=np.uint8)
        mask = np.full((50, 50), 255, dtype=np.uint8)
        corrected = correct_exposure(img, mask)
        assert corrected.mean() < img.mean(), "Severely bright image should be darkened"

    def test_no_change_for_normal_exposure(self):
        # Mid-tone image — should NOT be corrected
        img = np.full((50, 50, 3), [140, 130, 120], dtype=np.uint8)
        mask = np.full((50, 50), 255, dtype=np.uint8)
        corrected = correct_exposure(img, mask)
        np.testing.assert_array_equal(corrected, img, "Normal exposure should not change")

    def test_no_change_for_dark_skin(self):
        # Dark skin (L* ~35) is NATURAL, not underexposed
        img = np.full((50, 50, 3), [80, 60, 50], dtype=np.uint8)
        mask = np.full((50, 50), 255, dtype=np.uint8)
        corrected = correct_exposure(img, mask)
        diff = abs(float(corrected.mean()) - float(img.mean()))
        assert diff < 5, f"Dark skin should NOT be lightened aggressively, diff={diff}"

    def test_no_change_for_light_skin(self):
        # Light skin (L* ~70) is NATURAL, not overexposed
        img = np.full((50, 50, 3), [200, 180, 170], dtype=np.uint8)
        mask = np.full((50, 50), 255, dtype=np.uint8)
        corrected = correct_exposure(img, mask)
        diff = abs(float(corrected.mean()) - float(img.mean()))
        assert diff < 5, f"Light skin should NOT be darkened, diff={diff}"

    def test_empty_mask_returns_original(self):
        img = np.full((50, 50, 3), [100, 90, 80], dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        corrected = correct_exposure(img, mask)
        np.testing.assert_array_equal(corrected, img)

    def test_output_dtype_and_range(self):
        img = np.full((50, 50, 3), [15, 10, 8], dtype=np.uint8)
        mask = np.full(img.shape[:2], 255, dtype=np.uint8)
        corrected = correct_exposure(img, mask)
        assert corrected.dtype == np.uint8
        assert corrected.max() <= 255
        assert corrected.min() >= 0

    def test_conservative_correction_range(self):
        # Severely underexposed: correction should be mild, not extreme
        img = np.full((50, 50, 3), [12, 8, 5], dtype=np.uint8)
        mask = np.full((50, 50), 255, dtype=np.uint8)
        corrected = correct_exposure(img, mask)
        # Should brighten but NOT to mid-range (that would be too aggressive)
        assert corrected.mean() < 120, "Correction should be conservative, not normalize to mid-tone"
