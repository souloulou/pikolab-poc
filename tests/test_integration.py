"""Integration tests — full pipeline end-to-end."""

import numpy as np
import pytest
from app import (
    DEFAULTS,
    SEASON_PALETTES,
    classify_season,
    compute_confidence,
    compute_scores,
    compute_skin_stats,
    correct_exposure,
    correct_wb_with_reference,
    create_iris_mask,
    create_skin_mask,
    detect_face,
    detect_white_region,
    extract_iris_dominant,
    extract_pixels,
    pixels_to_lab,
    render_face_overlay,
    render_lab_histograms,
    render_palette,
    render_radar_chart,
)

DEFAULT_PARAMS = {
    "temp_center": DEFAULTS["temp_center"],
    "temp_scale": DEFAULTS["temp_scale"],
    "value_center": DEFAULTS["value_center"],
    "sat_center": DEFAULTS["sat_center"],
    "sat_scale": DEFAULTS["sat_scale"],
}


class TestFullPipeline:
    """Run the complete analysis pipeline on a real face image."""

    def test_pipeline_produces_valid_season(self, test_face_rgb):
        # 1. Detect
        landmarks = detect_face(test_face_rgb)
        if landmarks is None:
            pytest.skip("No face detected in test image")

        # 2. Masks
        skin_mask = create_skin_mask(test_face_rgb.shape, landmarks)
        iris_mask = create_iris_mask(test_face_rgb.shape, landmarks)
        assert np.count_nonzero(skin_mask) > 100

        # 3. Correct exposure
        corrected = correct_exposure(test_face_rgb, skin_mask)
        assert corrected.shape == test_face_rgb.shape

        # 4. Extract
        skin_px = extract_pixels(corrected, skin_mask)
        skin_lab = pixels_to_lab(skin_px)
        skin_stats = compute_skin_stats(skin_lab)

        iris_px = extract_pixels(corrected, iris_mask)
        iris_stats = extract_iris_dominant(iris_px) if len(iris_px) > 0 else None

        # 5. Classify
        scores = compute_scores(skin_stats, iris_stats, DEFAULT_PARAMS)
        season = classify_season(scores, DEFAULTS["dominance_thresh"])
        confidence = compute_confidence(scores)

        # Assertions
        assert season in SEASON_PALETTES, f"Unknown season: {season}"
        assert 0 <= confidence <= 1
        assert -1 <= scores["temperature"] <= 1
        assert -1 <= scores["value"] <= 1
        assert -1 <= scores["saturation"] <= 1

    def test_pipeline_with_wb_correction(self, test_face_rgb, white_sheet_image):
        """Test pipeline with white balance correction path."""
        ref = detect_white_region(white_sheet_image)
        assert ref is not None

        corrected = correct_wb_with_reference(test_face_rgb, ref)
        landmarks = detect_face(corrected)
        if landmarks is None:
            pytest.skip("No face after WB correction")

        skin_mask = create_skin_mask(corrected.shape, landmarks)
        skin_px = extract_pixels(corrected, skin_mask)
        skin_lab = pixels_to_lab(skin_px)
        skin_stats = compute_skin_stats(skin_lab)

        scores = compute_scores(skin_stats, None, DEFAULT_PARAMS)
        season = classify_season(scores, DEFAULTS["dominance_thresh"])
        assert season in SEASON_PALETTES

    def test_pipeline_deterministic(self, test_face_rgb):
        """Same image should always give same season."""
        results = []
        for _ in range(3):
            landmarks = detect_face(test_face_rgb)
            if landmarks is None:
                pytest.skip("No face detected")
            skin_mask = create_skin_mask(test_face_rgb.shape, landmarks)
            corrected = correct_exposure(test_face_rgb, skin_mask)
            skin_px = extract_pixels(corrected, skin_mask)
            skin_lab = pixels_to_lab(skin_px)
            skin_stats = compute_skin_stats(skin_lab)
            scores = compute_scores(skin_stats, None, DEFAULT_PARAMS)
            season = classify_season(scores, DEFAULTS["dominance_thresh"])
            results.append(season)

        assert len(set(results)) == 1, f"Non-deterministic: {results}"


class TestVisualizationRendering:
    """Verify visualization functions don't crash and produce output."""

    def test_render_face_overlay(self, test_face_rgb):
        skin_mask = np.zeros(test_face_rgb.shape[:2], dtype=np.uint8)
        skin_mask[100:200, 100:200] = 255
        iris_mask = np.zeros_like(skin_mask)

        overlay = render_face_overlay(test_face_rgb, skin_mask, iris_mask)
        assert overlay.shape == test_face_rgb.shape
        assert overlay.dtype == np.uint8

    def test_render_radar_chart(self):
        scores = {"temperature": 0.5, "value": -0.3, "saturation": 0.7}
        fig = render_radar_chart(scores, "Bright Spring")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_palette_valid_season(self):
        fig = render_palette("True Winter")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_palette_unknown_season(self):
        fig = render_palette("NonExistent Season")
        assert fig is None

    def test_render_lab_histograms(self):
        lab = np.random.RandomState(42).randn(500, 3) * 10 + [55, 12, 18]
        fig = render_lab_histograms(lab, "Test")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_lab_histograms_empty(self):
        lab = np.array([]).reshape(0, 3)
        fig = render_lab_histograms(lab, "Empty")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestEdgeCases:
    """Pipeline behavior on edge-case inputs."""

    def test_very_small_face(self):
        """A tiny 32x32 skin-colored image — face detection should fail gracefully."""
        tiny = np.full((32, 32, 3), [180, 150, 130], dtype=np.uint8)
        landmarks = detect_face(tiny)
        # Either no detection or too few skin pixels — both are acceptable
        if landmarks:
            mask = create_skin_mask(tiny.shape, landmarks)
            # Might have very few pixels, which is fine
            assert mask.shape == (32, 32)

    def test_grayscale_input(self):
        """Grayscale image converted to 3-channel should not crash."""
        gray = np.full((200, 200), 128, dtype=np.uint8)
        rgb = np.stack([gray] * 3, axis=-1)
        landmarks = detect_face(rgb)
        # May or may not detect face, but should not crash

    def test_high_res_image(self):
        """Large image should work (may be slow)."""
        big = np.full((1920, 1080, 3), [170, 140, 120], dtype=np.uint8)
        # Should not crash
        landmarks = detect_face(big)
        # No face in uniform image, that's expected
        assert landmarks is None
