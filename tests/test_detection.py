"""Tests for face detection and mask creation."""

import numpy as np
import pytest
from app import (
    create_iris_mask,
    create_polygon_mask,
    create_skin_mask,
    detect_face,
)


class TestDetectFace:
    def test_detects_real_face(self, test_face_rgb):
        landmarks = detect_face(test_face_rgb)
        assert landmarks is not None, "Should detect at least one face"
        assert len(landmarks) >= 468

    def test_returns_478_with_iris_refinement(self, test_face_rgb):
        landmarks = detect_face(test_face_rgb)
        if landmarks:
            assert len(landmarks) == 478, "refine_landmarks=True should yield 478 landmarks"

    def test_returns_none_for_blank(self):
        blank = np.full((200, 200, 3), 128, dtype=np.uint8)
        assert detect_face(blank) is None

    def test_returns_none_for_noise(self):
        rng = np.random.RandomState(0)
        noise = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        assert detect_face(noise) is None

    def test_landmarks_within_bounds(self, test_face_rgb):
        landmarks = detect_face(test_face_rgb)
        if landmarks is None:
            pytest.skip("No face detected in test image")
        h, w = test_face_rgb.shape[:2]
        for x, y in landmarks:
            assert -5 <= x <= w + 5, f"x={x} out of bounds"
            assert -5 <= y <= h + 5, f"y={y} out of bounds"

    def test_deterministic(self, test_face_rgb):
        r1 = detect_face(test_face_rgb)
        r2 = detect_face(test_face_rgb)
        if r1 and r2:
            assert r1 == r2, "Same image should give same landmarks"


class TestCreatePolygonMask:
    def test_creates_filled_polygon(self):
        mask = create_polygon_mask((100, 100, 3), [(10, 10), (90, 10), (90, 90), (10, 90)], [0, 1, 2, 3])
        assert mask.shape == (100, 100)
        assert np.count_nonzero(mask) > 1000

    def test_returns_empty_for_2_points(self):
        mask = create_polygon_mask((100, 100, 3), [(10, 10), (90, 90)], [0, 1])
        assert np.count_nonzero(mask) == 0

    def test_returns_empty_for_1_point(self):
        mask = create_polygon_mask((50, 50, 3), [(25, 25)], [0])
        assert np.count_nonzero(mask) == 0

    def test_handles_out_of_range_indices(self):
        landmarks = [(10, 10), (50, 10), (50, 50)]
        mask = create_polygon_mask((100, 100, 3), landmarks, [0, 1, 2, 999])
        assert mask.shape == (100, 100)

    def test_mask_dtype_is_uint8(self):
        mask = create_polygon_mask((50, 50, 3), [(5, 5), (45, 5), (45, 45)], [0, 1, 2])
        assert mask.dtype == np.uint8

    def test_mask_values_are_0_or_255(self):
        mask = create_polygon_mask((50, 50, 3), [(5, 5), (45, 5), (45, 45), (5, 45)], [0, 1, 2, 3])
        unique = np.unique(mask)
        assert set(unique).issubset({0, 255})


class TestCreateSkinMask:
    def test_nonzero_with_fake_landmarks(self, fake_landmarks_478):
        mask = create_skin_mask((512, 512, 3), fake_landmarks_478)
        assert np.count_nonzero(mask) > 50

    def test_shape_matches_image(self, fake_landmarks_478):
        mask = create_skin_mask((480, 640, 3), fake_landmarks_478)
        assert mask.shape == (480, 640)

    def test_with_real_face(self, test_face_rgb):
        landmarks = detect_face(test_face_rgb)
        if landmarks is None:
            pytest.skip("No face detected")
        mask = create_skin_mask(test_face_rgb.shape, landmarks)
        assert np.count_nonzero(mask) > 100, "Real face should have substantial skin pixels"

    def test_covers_both_cheeks(self, fake_landmarks_478):
        from app import LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX, create_polygon_mask

        left = create_polygon_mask((512, 512, 3), fake_landmarks_478, LEFT_CHEEK_IDX)
        right = create_polygon_mask((512, 512, 3), fake_landmarks_478, RIGHT_CHEEK_IDX)
        assert np.count_nonzero(left) > 0
        assert np.count_nonzero(right) > 0


class TestCreateIrisMask:
    def test_nonzero_with_iris_landmarks(self, fake_landmarks_478):
        mask = create_iris_mask((512, 512, 3), fake_landmarks_478)
        assert np.count_nonzero(mask) > 0

    def test_empty_without_iris_landmarks(self):
        short_lm = [(50, 50)] * 468
        mask = create_iris_mask((100, 100, 3), short_lm)
        assert np.count_nonzero(mask) == 0

    def test_pupil_excluded(self, fake_landmarks_478):
        mask = create_iris_mask((512, 512, 3), fake_landmarks_478)
        # Center of left iris (landmark 468) should be 0 (pupil hole)
        cx, cy = fake_landmarks_478[468]
        assert mask[cy, cx] == 0, "Pupil center should be excluded"

    def test_iris_ring_has_pixels(self, fake_landmarks_478):
        mask = create_iris_mask((512, 512, 3), fake_landmarks_478)
        # Edge of left iris (landmark 470) should be 255
        ex, ey = fake_landmarks_478[470]
        assert mask[ey, ex] == 255, "Iris edge should be included"

    def test_with_real_face(self, test_face_rgb):
        landmarks = detect_face(test_face_rgb)
        if landmarks is None or len(landmarks) < 478:
            pytest.skip("No iris landmarks")
        mask = create_iris_mask(test_face_rgb.shape, landmarks)
        assert np.count_nonzero(mask) > 5, "Should detect iris pixels"
