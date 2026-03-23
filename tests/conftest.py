"""Shared fixtures for the PikoLab test suite."""

import os
import sys
import urllib.request

import cv2
import numpy as np
import pytest

# Make app importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TEST_FACE_PATH = os.path.join(FIXTURES_DIR, "test_face.jpg")


def _download_test_face():
    """Download a real face image and cache it to disk."""
    if os.path.exists(TEST_FACE_PATH):
        return
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    try:
        req = urllib.request.Request(
            "https://thispersondoesnotexist.com",
            headers={"User-Agent": "PikoLab-Test/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        # Validate it's a decodable image
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Downloaded data is not a valid image")
        cv2.imwrite(TEST_FACE_PATH, img)
    except Exception as exc:
        print(f"WARNING: could not download test face ({exc}), generating synthetic one")
        _generate_synthetic_face()


def _generate_synthetic_face():
    """Fallback: generate a crude face-shaped image for smoke tests."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    img = np.ones((512, 512, 3), dtype=np.uint8) * 200
    # Face oval
    cv2.ellipse(img, (256, 256), (130, 170), 0, 0, 360, (185, 155, 135), -1)
    # Eyes (dark circles)
    cv2.circle(img, (210, 220), 18, (60, 45, 30), -1)
    cv2.circle(img, (302, 220), 18, (60, 45, 30), -1)
    # Iris (lighter center)
    cv2.circle(img, (210, 220), 8, (110, 90, 70), -1)
    cv2.circle(img, (302, 220), 8, (110, 90, 70), -1)
    # Nose
    cv2.line(img, (256, 240), (248, 280), (165, 135, 115), 2)
    # Mouth
    cv2.ellipse(img, (256, 310), (35, 12), 0, 0, 180, (160, 110, 110), 2)
    cv2.imwrite(TEST_FACE_PATH, img)


# ---- Session-scoped fixtures ----

@pytest.fixture(scope="session", autouse=True)
def ensure_test_face():
    _download_test_face()


@pytest.fixture(scope="session")
def test_face_path():
    return TEST_FACE_PATH


@pytest.fixture
def test_face_rgb():
    """Real (or synthetic) face image as RGB numpy array."""
    img = cv2.imread(TEST_FACE_PATH)
    assert img is not None, f"Could not read {TEST_FACE_PATH}"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---- Synthetic fixtures for unit tests ----

@pytest.fixture
def skin_tone_image():
    """100x100 uniform skin-colored image."""
    return np.full((100, 100, 3), [185, 150, 130], dtype=np.uint8)


@pytest.fixture
def white_sheet_image():
    """Image with a big white rectangle (simulates white paper)."""
    img = np.full((300, 300, 3), [90, 75, 60], dtype=np.uint8)
    img[30:270, 30:270] = [250, 248, 246]
    return img


@pytest.fixture
def dark_image():
    """Very underexposed image."""
    return np.full((100, 100, 3), [30, 20, 15], dtype=np.uint8)


@pytest.fixture
def bright_image():
    """Very overexposed image."""
    return np.full((100, 100, 3), [245, 240, 238], dtype=np.uint8)


@pytest.fixture
def fake_landmarks_478():
    """478 fake landmarks for a 512x512 image with plausible cheek & iris positions."""
    rng = np.random.RandomState(42)
    base = [(int(256 + 80 * np.cos(i * 0.5)), int(256 + 80 * np.sin(i * 0.5))) for i in range(478)]

    # Left cheek polygon (compact cluster on left cheek area)
    lc = [(180 + i * 6, 250 + (i % 3) * 8) for i in range(10)]
    for k, idx in enumerate([234, 93, 132, 58, 172, 136, 150, 149, 176, 148]):
        base[idx] = lc[k]

    # Right cheek polygon
    rc = [(310 + i * 6, 250 + (i % 3) * 8) for i in range(10)]
    for k, idx in enumerate([454, 323, 361, 288, 397, 365, 379, 378, 400, 377]):
        base[idx] = rc[k]

    # Left iris
    base[468] = (210, 220)   # center
    base[469] = (210, 212)   # top
    base[470] = (218, 220)   # right
    base[471] = (210, 228)   # bottom
    base[472] = (202, 220)   # left

    # Right iris
    base[473] = (302, 220)
    base[474] = (302, 212)
    base[475] = (310, 220)
    base[476] = (302, 228)
    base[477] = (294, 220)

    return base


@pytest.fixture
def full_mask_100():
    """100x100 mask that is entirely 255."""
    return np.full((100, 100), 255, dtype=np.uint8)


@pytest.fixture
def empty_mask_100():
    """100x100 mask that is entirely 0."""
    return np.zeros((100, 100), dtype=np.uint8)
