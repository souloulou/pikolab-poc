"""
PikoLab PoC — Analyse Colorimetrique Saisonniere
Determine la palette saisonniere (16 saisons) a partir d'une photo de visage.
Pipeline : MediaPipe Face Mesh -> masquage peau/iris -> correction couleur -> CIELab -> classification
"""

import os
import urllib.request

import streamlit as st
import google.generativeai as genai
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
import cv2
import numpy as np
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 150
import matplotlib.patches as mpatches
from collections import OrderedDict

from season_advice import SEASON_ADVICE

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")


def ensure_model():
    """Download the MediaPipe face landmarker model if not already cached."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ============================================================
# CONSTANTS
# ============================================================

LEFT_CHEEK_IDX = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
RIGHT_CHEEK_IDX = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]
LEFT_IRIS_IDX = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]
PUPIL_RATIO = 0.3

# Forehead top boundary (for hair sampling above)
FOREHEAD_TOP_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 109, 67, 103, 54, 21, 162, 127]

# Lip contour (outer upper + outer lower, forms closed polygon)
UPPER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_IDX = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

# Eyebrow landmarks
LEFT_EYEBROW_IDX = [70, 63, 105, 66, 107]
RIGHT_EYEBROW_IDX = [300, 293, 334, 296, 336]

SKIN_WEIGHT = 0.7
IRIS_WEIGHT = 0.3

DEFAULTS = {
    "temp_center": 17.0,
    "temp_scale": 12.0,
    "value_center": 55.0,
    "sat_center": 20.0,
    "sat_scale": 15.0,
    "dominance_thresh": 0.3,
}

# ============================================================
# 16 SEASON DATA
# ============================================================

SEASON_PALETTES = OrderedDict({
    "Light Spring":  ["#FFFDD0", "#FADADD", "#FCD5CE", "#FFE5B4", "#C9E4CA", "#FFDAB9", "#B5EAD7", "#FFB7B2"],
    "Warm Spring":   ["#E07A5F", "#F2CC8F", "#81B29A", "#F4A261", "#D4A373", "#CCD5AE", "#E9C46A", "#FAEDCD"],
    "Bright Spring": ["#FF6B35", "#FFD166", "#06D6A0", "#EF476F", "#118AB2", "#FFE66D", "#F77F00", "#43AA8B"],
    "True Spring":   ["#FF5733", "#FFC300", "#28B463", "#FF69B4", "#00B4D8", "#DAF7A6", "#FF8C00", "#2EC4B6"],
    "Light Summer":  ["#CDB4DB", "#BDE0FE", "#FFAFCC", "#A2D2FF", "#D0D1FF", "#FFC8DD", "#B8C0FF", "#BBD0FF"],
    "Cool Summer":   ["#7B9EA8", "#9A8C98", "#C9ADA7", "#4A4E69", "#9CADB7", "#84A98C", "#B5838D", "#6D6875"],
    "Soft Summer":   ["#B5838D", "#A39BA8", "#8D99AE", "#C9CBA3", "#A8A4CE", "#CEB5A7", "#95B8D1", "#B8B8D1"],
    "True Summer":   ["#DB7093", "#2A6F97", "#E8E8E8", "#7678ED", "#C77DFF", "#66CDAA", "#1D3557", "#BA68C8"],
    "Soft Autumn":   ["#C2B280", "#8B8C7A", "#C9ADA7", "#5F7A61", "#A0937D", "#7F6B5D", "#8DAA9D", "#9B8E7E"],
    "Warm Autumn":   ["#BC6C25", "#DDA15E", "#606C38", "#E76F51", "#6B4423", "#CC5803", "#2D6A4F", "#D4A017"],
    "Deep Autumn":   ["#6B2737", "#4A5240", "#8B4513", "#5D4037", "#1B5E20", "#8B6914", "#722F37", "#2E4600"],
    "True Autumn":   ["#D35400", "#6B8E23", "#B7410E", "#795548", "#B8860B", "#A0522D", "#008080", "#C7962A"],
    "Deep Winter":   ["#1A1A2E", "#E8E8E8", "#CC0000", "#1A5276", "#0B6623", "#FF1493", "#6A0DAD", "#C0C0C0"],
    "Cool Winter":   ["#B0E0E6", "#DC143C", "#1A1A2E", "#F0F0F0", "#4B0082", "#C71585", "#36454F", "#01796F"],
    "Bright Winter": ["#0066FF", "#FF1493", "#F8F8FF", "#FFF700", "#8B00FF", "#FF0000", "#00CC44", "#1A1A2E"],
    "True Winter":   ["#CC0000", "#0000CD", "#1A1A2E", "#F5F5F5", "#00563F", "#FF69B4", "#C0C0C0", "#301934"],
})

SUBSEASON_RULES = {
    "Spring": [
        ("Light Spring",  "value",       "high"),
        ("Warm Spring",   "temperature", "high"),
        ("Bright Spring", "saturation",  "high"),
        ("True Spring",   None,          None),
    ],
    "Summer": [
        ("Light Summer",  "value",       "high"),
        ("Cool Summer",   "temperature", "low"),
        ("Soft Summer",   "saturation",  "low"),
        ("True Summer",   None,          None),
    ],
    "Autumn": [
        ("Soft Autumn",   "saturation",  "low"),
        ("Warm Autumn",   "temperature", "high"),
        ("Deep Autumn",   "value",       "low"),
        ("True Autumn",   None,          None),
    ],
    "Winter": [
        ("Deep Winter",   "value",       "low"),
        ("Cool Winter",   "temperature", "low"),
        ("Bright Winter", "saturation",  "high"),
        ("True Winter",   None,          None),
    ],
}

# Centroid scores for each season (temperature, value, saturation)
SEASON_CENTROIDS = {
    "Light Spring":  ( 0.3,  0.8,  0.3),
    "Warm Spring":   ( 0.8,  0.3,  0.3),
    "Bright Spring": ( 0.4,  0.4,  0.8),
    "True Spring":   ( 0.5,  0.5,  0.5),
    "Light Summer":  (-0.3,  0.8, -0.2),
    "Cool Summer":   (-0.8,  0.3, -0.2),
    "Soft Summer":   (-0.3,  0.3, -0.7),
    "True Summer":   (-0.5,  0.5, -0.3),
    "Soft Autumn":   ( 0.3, -0.3, -0.7),
    "Warm Autumn":   ( 0.8, -0.3,  0.2),
    "Deep Autumn":   ( 0.3, -0.8,  0.0),
    "True Autumn":   ( 0.5, -0.5,  0.0),
    "Deep Winter":   (-0.3, -0.8,  0.2),
    "Cool Winter":   (-0.8, -0.3,  0.2),
    "Bright Winter": (-0.3, -0.3,  0.8),
    "True Winter":   (-0.5, -0.5,  0.3),
}


# ============================================================
# FACE DETECTION
# ============================================================

_landmarker_instance = None


def get_face_landmarker():
    global _landmarker_instance
    if _landmarker_instance is None:
        ensure_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        _landmarker_instance = FaceLandmarker.create_from_options(options)
    return _landmarker_instance


def detect_face(image_rgb):
    landmarker = get_face_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    h, w = image_rgb.shape[:2]
    return [(int(lm.x * w), int(lm.y * h)) for lm in result.face_landmarks[0]]


# ============================================================
# MASK CREATION
# ============================================================

def create_polygon_mask(shape, landmarks, indices):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    valid = [i for i in indices if i < len(landmarks)]
    if len(valid) < 3:
        return mask
    pts = np.array([landmarks[i] for i in valid], dtype=np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def create_skin_mask(shape, landmarks):
    left = create_polygon_mask(shape, landmarks, LEFT_CHEEK_IDX)
    right = create_polygon_mask(shape, landmarks, RIGHT_CHEEK_IDX)
    return cv2.bitwise_or(left, right)


def create_iris_mask(shape, landmarks):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(landmarks) < 478:
        return mask
    for iris_idx in [LEFT_IRIS_IDX, RIGHT_IRIS_IDX]:
        center = landmarks[iris_idx[0]]
        edges = [landmarks[i] for i in iris_idx[1:]]
        radius = int(np.mean([
            np.sqrt((e[0] - center[0]) ** 2 + (e[1] - center[1]) ** 2)
            for e in edges
        ]))
        if radius < 2:
            continue
        cv2.circle(mask, center, radius, 255, -1)
        pupil_r = max(1, int(radius * PUPIL_RATIO))
        cv2.circle(mask, center, pupil_r, 0, -1)
    return mask


def create_hair_mask(shape, landmarks):
    """Sample hair by taking a strip above the forehead landmarks."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    top_pts = [landmarks[i] for i in FOREHEAD_TOP_IDX if i < len(landmarks)]
    if len(top_pts) < 5:
        return mask
    min_y = min(p[1] for p in top_pts)
    xs = [p[0] for p in top_pts]
    left_x = max(0, min(xs))
    right_x = min(w, max(xs))
    hair_h = max(15, int(h * 0.06))
    top_y = max(0, min_y - hair_h)
    if right_x - left_x < 20 or min_y - top_y < 5:
        return mask
    mask[top_y:min_y, left_x:right_x] = 255
    return mask


def create_lip_mask(shape, landmarks):
    """Mask for the lip area using outer lip contour."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    upper = [landmarks[i] for i in UPPER_LIP_IDX if i < len(landmarks)]
    lower = [landmarks[i] for i in LOWER_LIP_IDX if i < len(landmarks)]
    if len(upper) < 5 or len(lower) < 5:
        return mask
    polygon = upper + lower[::-1]
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def create_eyebrow_mask(shape, landmarks):
    """Thin mask along eyebrows for natural hair color estimation."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for brow_idx in [LEFT_EYEBROW_IDX, RIGHT_EYEBROW_IDX]:
        pts = [landmarks[i] for i in brow_idx if i < len(landmarks)]
        if len(pts) < 3:
            continue
        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(mask, [arr], False, 255, 3)
    return mask


def classify_hair_color(lab_stats):
    """Classify hair into descriptive categories from CIELab stats."""
    if not lab_stats or lab_stats["L"] == 0:
        return {"color": "inconnu", "warmth": "neutre", "depth": "moyen"}
    L, a, b = lab_stats["L"], lab_stats["a"], lab_stats["b"]
    # Depth
    if L > 65:
        depth = "clair"
    elif L > 40:
        depth = "moyen"
    elif L > 25:
        depth = "fonce"
    else:
        depth = "tres fonce"
    # Warmth
    if b > 15 and a > 5:
        warmth = "chaud"
    elif b < 5:
        warmth = "froid"
    else:
        warmth = "neutre"
    # Color name
    if L > 65 and b > 15:
        color = "blond dore"
    elif L > 65 and b <= 15:
        color = "blond cendre"
    elif L > 40 and b > 15 and a > 10:
        color = "auburn/roux"
    elif L > 40 and b > 10:
        color = "chatain chaud"
    elif L > 40:
        color = "chatain froid"
    elif L > 25 and b > 10:
        color = "brun chaud"
    elif L > 25:
        color = "brun froid"
    else:
        color = "noir"
    return {"color": color, "warmth": warmth, "depth": depth}


def classify_lip_undertone(lab_stats):
    """Determine lip undertone from Lab values."""
    if not lab_stats or lab_stats["L"] == 0:
        return "inconnu"
    a, b = lab_stats["a"], lab_stats["b"]
    if a > 15 and b > 10:
        return "chaud (peche/corail)"
    elif a > 15 and b <= 10:
        return "froid (rose/berry)"
    elif a > 8:
        return "neutre-chaud"
    else:
        return "neutre"


def generate_personal_diagnostic(skin_stats, iris_stats, hair_info, lip_undertone,
                                  profile, season, advice, contrast):
    """Generate feature-by-feature personalized diagnostic."""
    diag = []
    season_is_warm = profile["raw_undertone"] > 0

    # --- SKIN ---
    t = profile["raw_undertone"]
    if t > 0.4:
        diag.append({
            "feature": "Peau",
            "icon": "🟢",
            "title": f"Sous-ton chaud marque (b*={skin_stats['b']:.0f})",
            "detail": "Les bijoux or, les couleurs terreuses et les tons peche vous illuminent naturellement.",
        })
    elif t > 0.1:
        diag.append({
            "feature": "Peau",
            "icon": "🟢",
            "title": f"Sous-ton neutre-chaud (b*={skin_stats['b']:.0f})",
            "detail": "Vous avez de la flexibilite mais les tons chauds restent vos meilleurs allies.",
        })
    elif t > -0.1:
        diag.append({
            "feature": "Peau",
            "icon": "🟡",
            "title": f"Sous-ton neutre (b*={skin_stats['b']:.0f})",
            "detail": "Sous-ton equilibre — evitez les extremes (ni trop dore, ni trop rose). Testez les deux metaux pres du visage.",
        })
    elif t > -0.4:
        diag.append({
            "feature": "Peau",
            "icon": "🟢",
            "title": f"Sous-ton neutre-froid (b*={skin_stats['b']:.0f})",
            "detail": "L'argent et les couleurs froides douces sont vos allies naturels.",
        })
    else:
        diag.append({
            "feature": "Peau",
            "icon": "🟢",
            "title": f"Sous-ton froid marque (b*={skin_stats['b']:.0f})",
            "detail": "L'argent, le platine et les couleurs froides pures vous subliment.",
        })

    # --- EYES ---
    if iris_stats:
        eye_warm = iris_stats["b"] > 10
        eye_light = iris_stats["L"] > 40
        eye_desc = []
        if eye_light and eye_warm:
            eye_desc = ["noisette/chauds", "Les fards bronze et cuivre feront ressortir leur chaleur."]
        elif eye_light and not eye_warm:
            eye_desc = ["clairs/froids", "Les fards pervenche, argent et taupe subliment vos yeux."]
        elif not eye_light and eye_warm:
            eye_desc = ["bruns chauds", "Les fards bronze fonce, olive et cuivre sont vos alliés."]
        else:
            eye_desc = ["fonces/froids", "Les fards gris charbon, bleu nuit et prune font ressortir leur profondeur."]

        harmony = "en harmonie" if (eye_warm == season_is_warm) else "en contraste"
        diag.append({
            "feature": "Yeux",
            "icon": "🟢" if eye_warm == season_is_warm else "🟡",
            "title": f"Yeux {eye_desc[0]} (L*={iris_stats['L']:.0f})",
            "detail": f"{eye_desc[1]} Vos yeux sont {harmony} avec votre sous-ton de peau.",
        })
    else:
        diag.append({
            "feature": "Yeux",
            "icon": "⚪",
            "title": "Iris non detecte",
            "detail": "L'analyse se base uniquement sur la peau. Pour plus de precision, assurez-vous que vos yeux sont bien ouverts et visibles.",
        })

    # --- HAIR ---
    if hair_info and hair_info["color"] != "inconnu":
        hair_warm = hair_info["warmth"] == "chaud"
        hair_match = hair_warm == season_is_warm
        ideal_colors = advice.get("hair", {}).get("ideal", [])

        if hair_match:
            diag.append({
                "feature": "Cheveux",
                "icon": "🟢",
                "title": f"Cheveux {hair_info['color']}",
                "detail": f"Vos cheveux sont en harmonie avec votre saison. Couleurs ideales : {', '.join(ideal_colors[:3])}.",
            })
        else:
            avoid_colors = advice.get("hair", {}).get("avoid", [])
            diag.append({
                "feature": "Cheveux",
                "icon": "🟠",
                "title": f"Cheveux {hair_info['color']} (decalage detecte)",
                "detail": (
                    f"La temperature de vos cheveux ne correspond pas a votre saison. "
                    f"{'Ils sont peut-etre teints. ' if hair_info['warmth'] != ('chaud' if season_is_warm else 'froid') else ''}"
                    f"Couleurs recommandees : {', '.join(ideal_colors[:3])}. "
                    f"A eviter : {', '.join(avoid_colors[:2])}."
                ),
            })
    else:
        diag.append({
            "feature": "Cheveux",
            "icon": "⚪",
            "title": "Cheveux non detectes",
            "detail": "La zone capillaire n'a pas pu etre analysee (cheveux couverts, image coupee, ou crane rase).",
        })

    # --- LIPS ---
    if lip_undertone and lip_undertone != "inconnu":
        lip_warm = "chaud" in lip_undertone
        lip_match = lip_warm == season_is_warm
        diag.append({
            "feature": "Levres",
            "icon": "🟢" if lip_match else "🟡",
            "title": f"Pigmentation {lip_undertone}",
            "detail": (
                "Vos levres confirment votre sous-ton." if lip_match
                else "La pigmentation de vos levres est legerement differente de votre sous-ton dominant — les rouges a levres recommandes dans votre palette corrigeront cet ecart."
            ),
        })

    # --- CONTRAST ---
    c = profile["raw_contrast"]
    if c > 0.5:
        diag.append({
            "feature": "Contraste",
            "icon": "🟢",
            "title": "Contraste eleve",
            "detail": "Vous pouvez porter des motifs graphiques, du color-blocking et des contrastes forts (noir/blanc, couleurs vives sur fond sombre).",
        })
    elif c > 0.25:
        diag.append({
            "feature": "Contraste",
            "icon": "🟢",
            "title": "Contraste moyen",
            "detail": "Les motifs moyens et les harmonies ton sur ton avec quelques accents de couleur sont ideaux.",
        })
    else:
        diag.append({
            "feature": "Contraste",
            "icon": "🟢",
            "title": "Contraste bas",
            "detail": "Privilegiez les harmonies douces, le ton sur ton et les camaieux. Evitez les ruptures brutales de couleur.",
        })

    # --- OVERALL HARMONY ---
    features_in_harmony = sum(1 for d in diag if d["icon"] == "🟢")
    total_features = sum(1 for d in diag if d["icon"] != "⚪")
    if total_features > 0:
        harmony_pct = features_in_harmony / total_features
        if harmony_pct >= 0.8:
            diag.append({
                "feature": "Harmonie globale",
                "icon": "✅",
                "title": "Excellente coherence",
                "detail": f"Vos traits (peau, yeux, cheveux) sont en harmonie. Votre classification {season} est fiable.",
            })
        elif harmony_pct >= 0.5:
            diag.append({
                "feature": "Harmonie globale",
                "icon": "🟡",
                "title": "Coherence partielle",
                "detail": f"Certains traits sont en leger decalage (cheveux teints ?). Votre saison {season} est indicative — consultez un coloriste pour affiner.",
            })
        else:
            diag.append({
                "feature": "Harmonie globale",
                "icon": "🟠",
                "title": "Profil mixte",
                "detail": f"Vos traits montrent des signaux mixtes. Vous etes peut-etre entre deux saisons. Consultez le top 3 saisons et essayez les palettes des 2 premieres.",
            })

    return diag


# ============================================================
# COLOR CORRECTION
# ============================================================

def detect_white_region(image_rgb):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0]
    a_ch = lab[:, :, 1].astype(int)
    b_ch = lab[:, :, 2].astype(int)
    white_mask = (
        (l_ch > 200) & (np.abs(a_ch - 128) < 15) & (np.abs(b_ch - 128) < 15)
    ).astype(np.uint8) * 255
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    min_area = image_rgb.shape[0] * image_rgb.shape[1] * 0.02
    large = [c for c in contours if cv2.contourArea(c) > min_area]
    if not large:
        return None
    biggest = max(large, key=cv2.contourArea)
    region_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(region_mask, [biggest], -1, 255, -1)
    return np.array(cv2.mean(image_rgb, mask=region_mask)[:3])


def correct_wb_with_reference(image_rgb, reference_rgb):
    gains = 255.0 / (np.array(reference_rgb) + 1e-6)
    gains = gains / gains.max()
    corrected = image_rgb.astype(np.float32)
    for c in range(3):
        corrected[:, :, c] *= gains[c]
    return np.clip(corrected, 0, 255).astype(np.uint8)


def correct_exposure(image_rgb, skin_mask):
    """Conservative exposure correction that preserves natural skin tone.

    Only corrects when the image shows clear signs of bad exposure
    (histogram clipping). Does NOT normalize skin lightness to a fixed
    target — dark skin stays dark, light skin stays light.
    """
    if skin_mask.sum() == 0:
        return image_rgb

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)
    skin_l = l_ch[skin_mask > 0]

    median_l = np.median(skin_l)
    if median_l < 5:
        return image_rgb

    # Detect exposure problems via histogram clipping
    pct_clipped_dark = np.mean(skin_l < 15) * 100   # % near black
    pct_clipped_bright = np.mean(skin_l > 245) * 100  # % near white

    severely_underexposed = pct_clipped_dark > 20 or median_l < 40
    severely_overexposed = pct_clipped_bright > 15 or median_l > 230

    if not severely_underexposed and not severely_overexposed:
        # Exposure is acceptable — no correction needed
        return image_rgb

    # Conservative correction: shift SLIGHTLY toward usable range
    # NOT toward a fixed target — preserve the natural skin tone
    if severely_underexposed:
        # Nudge up by 15-25% (mild gamma < 1)
        target_l = min(median_l * 1.4, median_l + 40)
    else:
        # Nudge down by 15-25% (mild gamma > 1)
        target_l = max(median_l * 0.7, median_l - 40)

    gamma = np.log(target_l / 255.0) / (np.log(median_l / 255.0) + 1e-6)
    gamma = np.clip(gamma, 0.7, 1.5)  # Much tighter range than before

    l_corrected = 255.0 * ((l_ch / 255.0) ** gamma)
    lab[:, :, 0] = np.clip(l_corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ============================================================
# COLOR EXTRACTION
# ============================================================

def extract_pixels(image_rgb, mask):
    if mask.sum() == 0:
        return np.array([]).reshape(0, 3)
    return image_rgb[mask > 0].reshape(-1, 3)


def pixels_to_lab(pixels_rgb):
    if len(pixels_rgb) == 0:
        return np.array([]).reshape(0, 3)
    pixels_f = pixels_rgb.astype(np.float64) / 255.0
    return rgb2lab(pixels_f.reshape(1, -1, 3)).reshape(-1, 3)


def compute_skin_stats(lab_pixels):
    if len(lab_pixels) == 0:
        return {"L": 0.0, "a": 0.0, "b": 0.0, "C": 0.0}
    return {
        "L": float(np.mean(lab_pixels[:, 0])),
        "a": float(np.mean(lab_pixels[:, 1])),
        "b": float(np.mean(lab_pixels[:, 2])),
        "C": float(np.mean(np.sqrt(lab_pixels[:, 1] ** 2 + lab_pixels[:, 2] ** 2))),
    }


def extract_iris_dominant(pixels_rgb):
    if len(pixels_rgb) < 20:
        return None
    k = min(3, max(2, len(pixels_rgb) // 10))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels_rgb)
    centers_lab = rgb2lab(
        kmeans.cluster_centers_.reshape(1, -1, 3).astype(np.float64) / 255.0
    ).reshape(-1, 3)
    sorted_idx = np.argsort(centers_lab[:, 0])
    dominant_idx = sorted_idx[1] if k >= 3 else sorted_idx[-1]
    return {
        "L": float(centers_lab[dominant_idx, 0]),
        "a": float(centers_lab[dominant_idx, 1]),
        "b": float(centers_lab[dominant_idx, 2]),
        "C": float(np.sqrt(centers_lab[dominant_idx, 1] ** 2 + centers_lab[dominant_idx, 2] ** 2)),
        "rgb": kmeans.cluster_centers_[dominant_idx].astype(int),
    }


# ============================================================
# PROFESSIONAL SCORING & CLASSIFICATION
# ============================================================

def compute_scores(skin_stats, iris_stats, params):
    """Normalized temperature / value / saturation scores in [-1, 1]."""
    s_temp = (skin_stats["b"] - params["temp_center"]) / params["temp_scale"]
    s_val = (skin_stats["L"] - params["value_center"]) / 50.0
    s_sat = (skin_stats["C"] - params["sat_center"]) / params["sat_scale"]

    if iris_stats:
        i_temp = iris_stats["b"] / 30.0
        i_val = (iris_stats["L"] - 40.0) / 40.0
        i_sat = (iris_stats["C"] - 20.0) / 20.0
        temperature = SKIN_WEIGHT * s_temp + IRIS_WEIGHT * i_temp
        value = SKIN_WEIGHT * s_val + IRIS_WEIGHT * i_val
        saturation = SKIN_WEIGHT * s_sat + IRIS_WEIGHT * i_sat
    else:
        temperature = s_temp
        value = s_val
        saturation = s_sat

    return {
        "temperature": float(np.clip(temperature, -1, 1)),
        "value": float(np.clip(value, -1, 1)),
        "saturation": float(np.clip(saturation, -1, 1)),
    }


def compute_contrast(skin_stats, iris_stats):
    """Contrast level between skin and iris (0-1 scale)."""
    if iris_stats is None:
        return 0.5  # Unknown, assume medium
    l_diff = abs(skin_stats["L"] - iris_stats["L"])
    c_diff = abs(skin_stats["C"] - iris_stats["C"])
    return float(np.clip((l_diff / 50.0 + c_diff / 30.0) / 2.0, 0, 1))


def compute_professional_profile(scores, contrast):
    """Human-readable 4-dimension profile for stylists."""
    t = scores["temperature"]
    v = scores["value"]
    s = scores["saturation"]

    # Undertone: 5 levels
    if t > 0.4:
        undertone = "Chaud"
    elif t > 0.1:
        undertone = "Neutre-chaud"
    elif t > -0.1:
        undertone = "Neutre"
    elif t > -0.4:
        undertone = "Neutre-froid"
    else:
        undertone = "Froid"

    # Value: 5 levels
    if v > 0.5:
        depth = "Tres clair"
    elif v > 0.15:
        depth = "Clair"
    elif v > -0.15:
        depth = "Medium"
    elif v > -0.5:
        depth = "Fonce"
    else:
        depth = "Tres fonce"

    # Chroma: 3 levels
    if s > 0.3:
        chroma = "Vif"
    elif s > -0.3:
        chroma = "Modere"
    else:
        chroma = "Doux"

    # Contrast: 3 levels
    if contrast > 0.5:
        contrast_label = "Eleve"
    elif contrast > 0.25:
        contrast_label = "Moyen"
    else:
        contrast_label = "Bas"

    return {
        "undertone": undertone,
        "depth": depth,
        "chroma": chroma,
        "contrast": contrast_label,
        "raw_undertone": t,
        "raw_depth": v,
        "raw_chroma": s,
        "raw_contrast": contrast,
    }


def classify_season(scores, dominance_threshold):
    """Classify into one of 16 seasons."""
    temp = scores["temperature"]
    val = scores["value"]

    if temp > 0 and val > 0:
        base = "Spring"
    elif temp <= 0 and val > 0:
        base = "Summer"
    elif temp > 0 and val <= 0:
        base = "Autumn"
    else:
        base = "Winter"

    rules = SUBSEASON_RULES[base]
    best_match = None
    best_strength = -1.0

    for sub_name, axis, direction in rules:
        if axis is None:
            continue
        score_val = scores[axis]
        strength = score_val if direction == "high" else -score_val
        if strength > best_strength:
            best_strength = strength
            best_match = sub_name

    if best_strength > dominance_threshold:
        return best_match

    for sub_name, axis, _ in rules:
        if axis is None:
            return sub_name
    return best_match


def classify_top3(scores):
    """Return top 3 seasons by distance to centroids, with match percentages."""
    point = np.array([scores["temperature"], scores["value"], scores["saturation"]])
    distances = {}
    for name, centroid in SEASON_CENTROIDS.items():
        dist = np.linalg.norm(point - np.array(centroid))
        distances[name] = dist

    sorted_seasons = sorted(distances.items(), key=lambda x: x[1])

    # Convert distances to match percentages (inverse, normalized)
    max_dist = max(d for _, d in sorted_seasons[:5]) + 0.01
    top3 = []
    total_score = 0
    for name, dist in sorted_seasons[:3]:
        score = max(0, (max_dist - dist) / max_dist)
        top3.append((name, score, dist))
        total_score += score

    # Normalize to percentages
    result = []
    for name, score, dist in top3:
        pct = (score / total_score * 100) if total_score > 0 else 33.3
        result.append({"season": name, "match_pct": round(pct, 1), "distance": round(dist, 3)})

    return result


def compute_confidence(scores):
    temp_dist = abs(scores["temperature"])
    value_dist = abs(scores["value"])
    return round(min(1.0, (temp_dist + value_dist) / 2.0 + 0.3), 2)


# ============================================================
# VISUALIZATION
# ============================================================

def render_face_overlay(image_rgb, skin_mask, iris_mask):
    overlay = image_rgb.copy().astype(np.float32)
    skin_region = skin_mask > 0
    overlay[skin_region] = overlay[skin_region] * 0.6 + np.array([0, 200, 0], dtype=np.float32) * 0.4
    iris_region = iris_mask > 0
    overlay[iris_region] = overlay[iris_region] * 0.5 + np.array([100, 100, 255], dtype=np.float32) * 0.5
    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_radar_chart(scores, season_name):
    labels = [
        "Temperature\n(froid | chaud)",
        "Valeur\n(sombre | clair)",
        "Saturation\n(doux | vif)",
    ]
    values = [scores["temperature"], scores["value"], scores["saturation"]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_plot = values + values[:1]
    angles_plot = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles_plot, values_plot, "o-", linewidth=2.5, color="#E07A5F", markersize=8)
    ax.fill(angles_plot, values_plot, alpha=0.25, color="#E07A5F")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(-1, 1)
    ax.set_title(season_name, fontsize=16, fontweight="bold", pad=20)
    ax.axhline(y=0, color="grey", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    return fig


def render_categorized_palette(season_name, advice):
    """Render 3 rows: neutrals, accents, to avoid."""
    rows = [
        ("Neutres (base)", advice.get("palette_neutrals", []), "#2d6a4f"),
        ("Accents", advice.get("palette_accents", []), "#e07a5f"),
        ("A eviter", advice.get("palette_avoid", []), "#d62828"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7, 4.5))
    for ax, (label, colors, title_color) in zip(axes, rows):
        if not colors:
            ax.axis("off")
            continue
        for i, color in enumerate(colors):
            rect = mpatches.FancyBboxPatch(
                (i * 1.15, 0), 1.05, 1.05,
                boxstyle="round,pad=0.06",
                facecolor=color,
                edgecolor="#333333",
                linewidth=0.5,
            )
            ax.add_patch(rect)
        ax.set_xlim(-0.3, max(len(colors), 1) * 1.15 + 0.2)
        ax.set_ylim(-0.3, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label, fontsize=12, fontweight="bold", color=title_color, loc="left")
    plt.tight_layout()
    return fig


def render_palette(season_name):
    colors = SEASON_PALETTES.get(season_name, [])
    if not colors:
        return None
    fig, ax = plt.subplots(figsize=(len(colors) * 1.2, 1.8))
    for i, color in enumerate(colors):
        rect = mpatches.FancyBboxPatch(
            (i * 1.1, 0), 1, 1,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.5,
        )
        ax.add_patch(rect)
        ax.text(i * 1.1 + 0.5, -0.3, color, ha="center", va="top", fontsize=7, color="#555555")
    ax.set_xlim(-0.2, len(colors) * 1.1 + 0.1)
    ax.set_ylim(-0.6, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Palette — {season_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def render_lab_histograms(lab_pixels, title="Distribution CIELab"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    channels = [
        ("L*", 0, (0, 100), "#555555"),
        ("a*", 1, (-50, 50), "#E07A5F"),
        ("b*", 2, (-50, 50), "#E9C46A"),
    ]
    for ax, (name, idx, rng, color) in zip(axes, channels):
        data = lab_pixels[:, idx] if len(lab_pixels) > 0 else []
        ax.hist(data, bins=50, range=rng, color=color, alpha=0.7, edgecolor="white")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Valeur")
        ax.set_ylabel("Pixels")
        if len(lab_pixels) > 0:
            mean_val = np.mean(lab_pixels[:, idx])
            ax.axvline(x=mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Moy={mean_val:.1f}")
            ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ============================================================
# VIRTUAL DRAPING
# ============================================================

# Jawline landmarks (ear to ear via chin)
JAWLINE_IDX = [234, 127, 162, 21, 54, 103, 67, 109, 10,
               338, 297, 332, 284, 251, 389, 356, 454,
               323, 361, 288, 397, 365, 379, 378, 400, 377,
               152, 148, 176, 149, 150, 136, 172, 58, 132, 93]


def render_drape(image_rgb, landmarks, color_hex):
    """Render a color drape below the jawline, simulating a fabric swatch."""
    h, w = image_rgb.shape[:2]
    result = image_rgb.copy()

    # Get jawline points
    jaw_pts = [landmarks[i] for i in JAWLINE_IDX if i < len(landmarks)]
    if len(jaw_pts) < 10:
        return result

    # Find chin (lowest point) and jaw sides
    chin_y = max(p[1] for p in jaw_pts)
    left_x = min(p[0] for p in jaw_pts)
    right_x = max(p[0] for p in jaw_pts)

    # Drape zone: from jawline down to bottom of image (or chin + 40% of face height)
    face_top = min(p[1] for p in jaw_pts)
    face_height = chin_y - face_top
    drape_bottom = min(h, chin_y + int(face_height * 0.6))

    # Create drape polygon: jawline bottom + rectangle extending down
    drape_top_pts = [(p[0], p[1]) for p in jaw_pts if p[1] > chin_y - int(face_height * 0.25)]
    if not drape_top_pts:
        drape_top_pts = [(left_x, chin_y), (right_x, chin_y)]

    # Sort by x for clean polygon
    drape_top_pts.sort(key=lambda p: p[0])

    # Build polygon: top contour + bottom rectangle
    margin = int((right_x - left_x) * 0.15)
    polygon = (
        drape_top_pts
        + [(right_x + margin, drape_bottom), (left_x - margin, drape_bottom)]
    )
    pts = np.array(polygon, dtype=np.int32)

    # Parse hex color
    color_rgb = tuple(int(color_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))

    # Draw filled drape with soft alpha blending
    overlay = result.copy()
    cv2.fillPoly(overlay, [pts], color_rgb)
    cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)

    return result


def render_draping_grid(image_rgb, landmarks, good_colors, bad_colors):
    """Render a 2-row grid: top=good colors, bottom=bad colors."""
    cols = max(len(good_colors), len(bad_colors), 1)
    cell_h, cell_w = image_rgb.shape[0], image_rgb.shape[1]

    # Scale down for grid
    scale = min(1.0, 400 / max(cell_h, cell_w))
    thumb_h = int(cell_h * scale)
    thumb_w = int(cell_w * scale)

    # Resize source once
    thumb = cv2.resize(image_rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    thumb_landmarks = [(int(x * scale), int(y * scale)) for x, y in landmarks]

    rows = []
    for colors, label in [(good_colors, "A porter"), (bad_colors, "A eviter")]:
        row_images = []
        for color_hex in colors[:3]:
            draped = render_drape(thumb, thumb_landmarks, color_hex)
            # Add color label at bottom
            cv2.rectangle(draped, (0, thumb_h - 25), (thumb_w, thumb_h), (40, 40, 40), -1)
            cv2.putText(draped, color_hex, (5, thumb_h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            row_images.append(draped)

        # Pad if fewer than 3
        while len(row_images) < 3:
            row_images.append(np.full((thumb_h, thumb_w, 3), 240, dtype=np.uint8))

        row = np.hstack(row_images)
        # Add row label
        label_bar = np.full((30, row.shape[1], 3), 255, dtype=np.uint8)
        color_text = (0, 150, 0) if "porter" in label else (200, 0, 0)
        cv2.putText(label_bar, label, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)
        rows.append(np.vstack([label_bar, row]))

    return np.vstack(rows)


# ============================================================
# COLOR COMPATIBILITY SCORE
# ============================================================

def compute_color_compatibility(color_hex, season_palettes_all, season_name):
    """Score a color against a season's palette. Returns (score 0-100, nearest match hex, suggestion)."""
    # Parse input color
    r, g, b = (int(color_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    input_lab = rgb2lab(np.array([[[r / 255, g / 255, b / 255]]], dtype=np.float64))[0, 0]

    # Get season palette (neutrals + accents)
    palette = season_palettes_all.get(season_name, [])
    if not palette:
        return 0, "#000000", "Palette non disponible"

    # Compute distance to each palette color
    best_dist = float("inf")
    best_hex = palette[0]
    for phex in palette:
        pr, pg, pb = (int(phex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        p_lab = rgb2lab(np.array([[[pr / 255, pg / 255, pb / 255]]], dtype=np.float64))[0, 0]
        dist = np.sqrt(np.sum((input_lab - p_lab) ** 2))
        if dist < best_dist:
            best_dist = dist
            best_hex = phex

    # Convert distance to score (0=far, 100=perfect match)
    # Delta E < 5 = near identical, < 15 = good, < 30 = noticeable, > 30 = very different
    score = max(0, min(100, int(100 - best_dist * 2.5)))

    if score >= 75:
        suggestion = "Excellente couleur pour vous !"
    elif score >= 50:
        suggestion = f"Acceptable. Pour un meilleur resultat, essayez {best_hex}"
    elif score >= 25:
        suggestion = f"Pas ideal. Preferez {best_hex} qui est dans votre palette."
    else:
        suggestion = f"A eviter. La couleur la plus proche dans votre palette est {best_hex}"

    return score, best_hex, suggestion


# ============================================================
# STORY IMAGE EXPORT
# ============================================================

def generate_story_image(image_rgb, season, tagline, palette_colors, profile):
    """Generate a 1080x1920 story image for sharing."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1080, 1920
    story = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(story)

    # Parse primary color
    pc = palette_colors[0] if palette_colors else "#E07A5F"

    # Background gradient strip at top
    for y in range(300):
        alpha = y / 300
        r = int(int(pc[1:3], 16) * (1 - alpha) + 255 * alpha)
        g = int(int(pc[3:5], 16) * (1 - alpha) + 255 * alpha)
        b = int(int(pc[5:7], 16) * (1 - alpha) + 255 * alpha)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # Photo (centered, circular crop)
    photo = Image.fromarray(image_rgb)
    size = 500
    photo = photo.resize((size, size), Image.LANCZOS)

    # Circular mask
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse([0, 0, size, size], fill=255)

    # Paste photo centered
    x_offset = (W - size) // 2
    y_offset = 180
    story.paste(photo, (x_offset, y_offset), mask)

    # Circle border
    draw.ellipse(
        [x_offset - 3, y_offset - 3, x_offset + size + 3, y_offset + size + 3],
        outline=pc, width=4,
    )

    # Season name
    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except (OSError, IOError):
        font_big = ImageFont.load_default()
        font_med = font_big
        font_sm = font_big

    y_text = y_offset + size + 40
    bbox = draw.textbbox((0, 0), season, font=font_big)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, y_text), season, fill=pc, font=font_big)

    # Tagline
    y_text += 80
    bbox = draw.textbbox((0, 0), tagline, font=font_med)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, y_text), tagline, fill="#666666", font=font_med)

    # Profile summary
    y_text += 70
    profile_text = (
        f"Sous-ton: {profile['undertone']}  |  Valeur: {profile['depth']}  |  "
        f"Chroma: {profile['chroma']}  |  Contraste: {profile['contrast']}"
    )
    bbox = draw.textbbox((0, 0), profile_text, font=font_sm)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, y_text), profile_text, fill="#888888", font=font_sm)

    # Palette swatches
    y_palette = y_text + 80
    swatch_size = 90
    gap = 15
    n = min(len(palette_colors), 8)
    total_w = n * swatch_size + (n - 1) * gap
    x_start = (W - total_w) // 2

    draw.text((x_start, y_palette - 5), "Votre palette", fill="#333333", font=font_sm)
    y_palette += 40

    for i, hex_color in enumerate(palette_colors[:n]):
        x = x_start + i * (swatch_size + gap)
        draw.rounded_rectangle(
            [x, y_palette, x + swatch_size, y_palette + swatch_size],
            radius=10, fill=hex_color, outline="#CCCCCC",
        )

    # Footer
    draw.text((W // 2 - 80, H - 80), "PikoLab", fill="#CCCCCC", font=font_med)

    # Convert to bytes
    import io
    buf = io.BytesIO()
    story.save(buf, format="PNG", quality=95)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# COACH IA (Gemini)
# ============================================================

def build_coach_system_prompt(season, advice, profile, diagnostic, hair_info, lip_undertone, quiz_data):
    """Build a comprehensive system prompt with all client context."""
    diag_text = "\n".join(
        f"- {d['feature']}: {d['title']} — {d['detail']}" for d in diagnostic
    )

    makeup = advice.get("makeup", {})
    clothing = advice.get("clothing", {})
    hair = advice.get("hair", {})
    acc = advice.get("accessories", {})
    expert = advice.get("expert", {})

    quiz_text = ""
    if quiz_data:
        quiz_text = f"""
Reponses questionnaire client :
- Cheveux teints : {quiz_data.get('hair_dyed', 'non renseigne')}
- Couleur naturelle : {quiz_data.get('natural_hair', 'non renseigne')}
- Style : {quiz_data.get('style', 'non renseigne')}
- Environnement travail : {quiz_data.get('work', 'non renseigne')}
- Couleurs portees : {quiz_data.get('current_colors', 'non renseigne')}
- Interet principal : {quiz_data.get('interest', 'non renseigne')}
"""

    return f"""Tu es un coach en image et styliste professionnel specialise en colorimetrie saisonniere.
Tu aides un(e) client(e) a comprendre et appliquer sa palette de couleurs au quotidien.

PROFIL DU CLIENT :
- Saison : {season}
- Description : {advice.get('description', '')}
- Sous-ton : {profile['undertone']} (score: {profile['raw_undertone']:.2f})
- Valeur : {profile['depth']}
- Chroma : {profile['chroma']}
- Contraste : {profile['contrast']}
- Cheveux detectes : {hair_info.get('color', 'inconnu')} ({hair_info.get('warmth', '')})
- Levres : {lip_undertone}

DIAGNOSTIC :
{diag_text}
{quiz_text}
PALETTE RECOMMANDEE :
- A porter : {advice.get('best_summary', '')}
- A eviter : {advice.get('avoid_summary', '')}
- Alternative au noir : {advice.get('black_alt', '')}
- Alternative au blanc : {advice.get('white_alt', '')}
- Metaux : {advice.get('metals', '')}

MAQUILLAGE :
- Fond de teint : {makeup.get('foundation', '')}
- Levres : {', '.join(makeup.get('lips', []))}
- Yeux : {', '.join(makeup.get('eyes', []))}
- Look naturel : {makeup.get('look_naturel', '')}
- Look soiree : {makeup.get('look_soiree', '')}
- Look bureau : {makeup.get('look_pro', '')}

VETEMENTS :
- Combinaisons : {', '.join(clothing.get('best_combinations', []))}
- Motifs : {clothing.get('patterns', '')}
- Tissus : {clothing.get('fabrics', '')}
- Capsule : {', '.join(clothing.get('capsule', []))}

CHEVEUX :
- Ideaux : {', '.join(hair.get('ideal', []))}
- A eviter : {', '.join(hair.get('avoid', []))}
- Conseil : {hair.get('tips', '')}

ACCESSOIRES :
- Lunettes : {acc.get('glasses', '')}
- Bijoux : {acc.get('jewelry', '')}
- Ongles : {acc.get('nails', '')}

REGLES :
1. Reponds TOUJOURS en francais
2. Base tes conseils UNIQUEMENT sur la saison et le profil du client ci-dessus
3. Sois precis et actionnable (noms de couleurs, types de vetements, marques si pertinent)
4. Si le client demande quelque chose hors de ton expertise, dis-le honnetement
5. Sois chaleureux, encourageant mais professionnel
6. Reponds de maniere concise (3-5 phrases max sauf si on te demande du detail)
7. Si le client mentionne un vetement ou une couleur specifique, dis si ca lui va ou pas et POURQUOI
"""


def get_gemini_model(api_key, system_prompt):
    """Initialize Gemini model with system prompt."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=system_prompt,
    )


def stream_coach_response(model, history, user_message):
    """Stream a response from Gemini, yielding chunks for st.write_stream."""
    # Convert history to Gemini format (role: user/model)
    gemini_history = []
    for m in history:
        role = "model" if m["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [m["content"]]})

    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(user_message, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


def render_gauge(value, vmin, vmax, label_left, label_right, title):
    """Horizontal gauge bar for profile dimensions. Mobile-friendly sizing."""
    fig, ax = plt.subplots(figsize=(6, 1.0))
    normalized = (value - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    # Background bar
    ax.barh(0, 1, height=0.5, color="#E8E8E8", left=0)
    # Gradient fill up to value
    gradient_color = plt.cm.RdYlBu_r(normalized)
    ax.barh(0, normalized, height=0.5, color=gradient_color, left=0)
    # Marker
    ax.plot(normalized, 0, "v", color="#333", markersize=16, zorder=5)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([label_left, "", label_right], fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================
# STREAMLIT APP
# ============================================================

MAX_DIMENSION = 1280  # Resize large images to speed up processing


def load_image(uploaded_file):
    """Load image from upload, handle large files, return RGB numpy array."""
    data = uploaded_file.getvalue()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if too large (iPhone photos can be 4000+ px)
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return img


MOBILE_CSS = "<style>\n" \
    "@media (max-width: 768px) {\n" \
    "  [data-testid='stHorizontalBlock'] { flex-direction: column !important; gap: 0.5rem !important; }\n" \
    "  [data-testid='stHorizontalBlock'] > div { width: 100% !important; flex: 1 1 100% !important; }\n" \
    "  button, [data-testid='stFileUploader'] { min-height: 48px !important; }\n" \
    "  h1 { font-size: 1.5rem !important; }\n" \
    "  h2 { font-size: 1.25rem !important; }\n" \
    "  h3 { font-size: 1.1rem !important; }\n" \
    "  [data-testid='stTabs'] [role='tablist'] { overflow-x: auto !important; flex-wrap: nowrap !important; -webkit-overflow-scrolling: touch; }\n" \
    "  [data-testid='stTabs'] [role='tab'] { white-space: nowrap !important; font-size: 0.85rem !important; padding: 0.5rem 0.75rem !important; }\n" \
    "  [data-testid='stSidebar'] { min-width: 0 !important; }\n" \
    "  [data-testid='stCameraInput'] { width: 100% !important; }\n" \
    "  [data-testid='stMetric'] { padding: 0.25rem !important; }\n" \
    "}\n" \
    "[data-testid='stCameraInput'] video { max-height: 50vh; object-fit: contain; }\n" \
    "</style>"


def main():
    st.set_page_config(
        page_title="PikoLab — Analyse Colorimetrique",
        page_icon="🎨",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)

    # ---- Sidebar ----
    st.sidebar.header("Mode d'affichage")
    view_mode = st.sidebar.radio(
        "Niveau de detail",
        ["Client", "Professionnel", "Avance"],
        index=0,
        help="Client = resultats simples, Pro = coaching styliste, Avance = debug technique",
    )

    # Classification params (Avance only)
    if view_mode == "Avance":
        with st.sidebar.expander("Seuils de classification", expanded=False):
            temp_center = st.slider("Temperature neutre (b*)", 5.0, 30.0, DEFAULTS["temp_center"], 0.5)
            temp_scale = st.slider("Echelle temperature", 5.0, 25.0, DEFAULTS["temp_scale"], 0.5)
            value_center = st.slider("Valeur mediane (L*)", 30.0, 70.0, DEFAULTS["value_center"], 1.0)
            sat_center = st.slider("Saturation mediane (C*)", 5.0, 35.0, DEFAULTS["sat_center"], 0.5)
            sat_scale = st.slider("Echelle saturation", 5.0, 25.0, DEFAULTS["sat_scale"], 0.5)
            dominance_thresh = st.slider("Seuil de dominance", 0.05, 0.60, DEFAULTS["dominance_thresh"], 0.05)
    else:
        temp_center = DEFAULTS["temp_center"]
        temp_scale = DEFAULTS["temp_scale"]
        value_center = DEFAULTS["value_center"]
        sat_center = DEFAULTS["sat_center"]
        sat_scale = DEFAULTS["sat_scale"]
        dominance_thresh = DEFAULTS["dominance_thresh"]

    params = {
        "temp_center": temp_center,
        "temp_scale": temp_scale,
        "value_center": value_center,
        "sat_center": sat_center,
        "sat_scale": sat_scale,
    }

    # ---- Acquisition ----
    image_rgb = None

    GUIDE_PHOTO = (
        "**Comment prendre votre photo ?**\n\n"
        "Pour de meilleurs resultats, tenez une **feuille A4 blanche** "
        "a cote de votre visage. Elle sera detectee automatiquement.\n\n"
        "1. Lumiere naturelle (fenetre, pas de soleil direct)\n"
        "2. Pas de maquillage, pas de lunettes\n"
        "3. Desactivez les filtres et le mode beaute\n"
        "4. Visage de face, cheveux attaches\n"
        "5. Feuille blanche bien eclairee, sans ombre\n\n"
        "*La feuille est optionnelle mais recommandee.*"
    )

    # ---- Hero section when no image yet ----
    if "analysis_done" not in st.session_state:
        st.markdown("""
        ## Decouvrez votre palette de couleurs
        Prenez un selfie et obtenez instantanement votre **saison colorimetrique**
        avec des conseils personnalises : maquillage, vetements, cheveux et accessoires.
        """)

    # Mode selector in main content (not hidden in sidebar)
    mode = st.radio(
        "Comment fournir votre photo ?",
        ["Appareil photo", "Galerie", "Demo"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Appareil photo":
        first_visit = "guide_shown" not in st.session_state
        with st.expander("Comment prendre une bonne photo", expanded=first_visit):
            st.markdown(GUIDE_PHOTO)
            st.session_state["guide_shown"] = True

        camera_photo = st.camera_input("Prenez votre selfie")
        if camera_photo:
            image_rgb = load_image(camera_photo)

    elif mode == "Galerie":
        uploaded = st.file_uploader(
            "Choisissez une photo",
            type=["jpg", "jpeg", "png", "webp", "heic", "heif"],
        )
        if uploaded:
            image_rgb = load_image(uploaded)
            if image_rgb is None:
                st.error(
                    "Format non reconnu. Sur iPhone, allez dans "
                    "Reglages > Appareil photo > Formats > Le plus compatible."
                )

    else:  # Demo
        if st.button("Generer un visage aleatoire", type="primary", use_container_width=True):
            with st.spinner("Telechargement..."):
                try:
                    req = urllib.request.Request(
                        "https://thispersondoesnotexist.com",
                        headers={"User-Agent": "PikoLab/1.0"},
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        data = resp.read()
                    arr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    st.session_state["demo_image"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception as exc:
                    st.error(f"Echec du telechargement : {exc}")

        if "demo_image" in st.session_state:
            image_rgb = st.session_state["demo_image"]

    if image_rgb is None:
        st.caption("Prenez un selfie ou choisissez une photo pour decouvrir votre palette.")
        return

    # ---- Auto-detect white sheet (skip for Demo mode) ----
    if mode == "Demo":
        wb_reference = None
    else:
        wb_reference = detect_white_region(image_rgb)
        if wb_reference is not None:
            st.success("Feuille blanche detectee — calibration couleur activee.")
        else:
            st.caption("Pas de feuille blanche detectee — resultats approximatifs.")

    # ---- Pipeline with progress ----
    progress = st.progress(0, text="Detection du visage...")
    landmarks = detect_face(image_rgb)
    if landmarks is None:
        progress.empty()
        st.error("Aucun visage detecte. Essayez avec une photo plus nette, de face, bien eclairee.")
        return

    progress.progress(15, text="Extraction des zones de peau et iris...")
    has_iris = len(landmarks) >= 478
    skin_mask = create_skin_mask(image_rgb.shape, landmarks)
    iris_mask = create_iris_mask(image_rgb.shape, landmarks) if has_iris else np.zeros(
        image_rgb.shape[:2], dtype=np.uint8
    )
    skin_px = int(np.count_nonzero(skin_mask))
    iris_px = int(np.count_nonzero(iris_mask))

    if skin_px < 100:
        progress.empty()
        st.error("Visage trop petit ou mal cadre. Rapprochez-vous de la camera.")
        return

    progress.progress(25, text="Detection cheveux, levres, sourcils...")
    hair_mask = create_hair_mask(image_rgb.shape, landmarks)
    lip_mask = create_lip_mask(image_rgb.shape, landmarks)
    eyebrow_mask = create_eyebrow_mask(image_rgb.shape, landmarks)
    hair_px = int(np.count_nonzero(hair_mask))
    lip_px = int(np.count_nonzero(lip_mask))

    progress.progress(40, text="Correction des couleurs...")
    if wb_reference is not None:
        corrected = correct_wb_with_reference(image_rgb, wb_reference)
        correction_method = "Feuille blanche"
    else:
        corrected = correct_exposure(image_rgb, skin_mask)
        correction_method = "Auto"

    progress.progress(55, text="Analyse colorimetrique peau et iris...")
    skin_pixels_rgb = extract_pixels(corrected, skin_mask)
    skin_lab = pixels_to_lab(skin_pixels_rgb)
    skin_stats = compute_skin_stats(skin_lab)

    iris_pixels_rgb = extract_pixels(corrected, iris_mask)
    iris_stats = extract_iris_dominant(iris_pixels_rgb) if len(iris_pixels_rgb) > 0 else None

    progress.progress(70, text="Analyse cheveux et levres...")
    hair_pixels = extract_pixels(corrected, hair_mask)
    hair_lab = pixels_to_lab(hair_pixels)
    hair_stats = compute_skin_stats(hair_lab) if len(hair_lab) > 0 else None
    hair_info = classify_hair_color(hair_stats)

    lip_pixels = extract_pixels(corrected, lip_mask)
    lip_lab = pixels_to_lab(lip_pixels)
    lip_stats = compute_skin_stats(lip_lab) if len(lip_lab) > 0 else None
    lip_undertone = classify_lip_undertone(lip_stats)

    # Eyebrow color as proxy for natural hair
    eyebrow_pixels = extract_pixels(corrected, eyebrow_mask)
    eyebrow_lab = pixels_to_lab(eyebrow_pixels)
    eyebrow_stats = compute_skin_stats(eyebrow_lab) if len(eyebrow_lab) > 0 else None
    eyebrow_info = classify_hair_color(eyebrow_stats)

    progress.progress(85, text="Classification saisonniere...")
    scores = compute_scores(skin_stats, iris_stats, params)
    contrast = compute_contrast(skin_stats, iris_stats)
    profile = compute_professional_profile(scores, contrast)
    season = classify_season(scores, dominance_thresh)
    top3 = classify_top3(scores)
    confidence = compute_confidence(scores)
    advice = SEASON_ADVICE.get(season, {})

    progress.progress(95, text="Diagnostic personnalise...")
    diagnostic = generate_personal_diagnostic(
        skin_stats, iris_stats, hair_info, lip_undertone,
        profile, season, advice, contrast,
    )

    progress.progress(100, text="Analyse terminee !")
    st.session_state["analysis_done"] = True
    progress.empty()

    # ---- Season result card ----
    season_colors = SEASON_PALETTES.get(season, ["#E07A5F", "#F2CC8F"])
    c1 = season_colors[0]
    c2 = season_colors[1] if len(season_colors) > 1 else c1

    if confidence >= 0.7:
        conf_text = "Resultat fiable"
    elif confidence >= 0.5:
        conf_text = "Resultat indicatif"
    else:
        conf_text = "Resultat a confirmer — reprenez la photo en lumiere naturelle"

    tagline = advice.get("tagline", "")
    desc = advice.get("description", "")

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {c1}22, {c2}22);
        border-left: 4px solid {c1};
        padding: 1.2rem 1.5rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0 1rem 0;
    ">
        <p style="margin:0; color: #888; font-size: 0.9rem;">Votre saison</p>
        <h1 style="margin: 0.15rem 0; color: {c1}; font-size: 2rem;">{season}</h1>
        <p style="margin:0 0 0.3rem 0; color: {c1}; font-size: 1rem; font-weight: 600;">{tagline}</p>
        <p style="margin:0; color: #555; font-style: italic; font-size: 0.95rem;">{desc}</p>
        <p style="margin-top: 0.5rem; color: #888; font-size: 0.85rem;">{conf_text}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Questionnaire (optional, after results) ----
    with st.expander("Affinez vos resultats (30 secondes)", expanded=False):
        st.caption("Repondez a ces questions pour des conseils encore plus personnalises.")

        q_hair_dyed = st.radio(
            "Vos cheveux sont-ils teints ?",
            ["Non, couleur naturelle", "Oui, ils sont teints"],
            horizontal=True, key="q_hair",
        )
        q_natural_hair = None
        if q_hair_dyed == "Oui, ils sont teints":
            q_natural_hair = st.selectbox(
                "Quelle est votre couleur naturelle ?",
                ["Blond", "Chatain clair", "Chatain fonce", "Brun", "Noir", "Roux"],
                key="q_nat_hair",
            )

        q_style = st.radio(
            "Votre style vestimentaire ?",
            ["Casual", "Classique", "Creatif", "Sportif"],
            horizontal=True, key="q_style",
        )

        q_work = st.radio(
            "Votre environnement de travail ?",
            ["Formel", "Smart casual", "Libre", "Pas concerne(e)"],
            horizontal=True, key="q_work",
        )

        q_current_colors = st.multiselect(
            "Quelles couleurs portez-vous le plus ?",
            ["Noir", "Bleu", "Blanc", "Beige/Camel", "Gris", "Couleurs vives", "Tons terre"],
            key="q_colors",
        )

        q_interest = st.radio(
            "Quel conseil vous interesse le plus ?",
            ["Garde-robe", "Maquillage", "Cheveux", "Tout"],
            horizontal=True, key="q_interest",
        )

    # Personalized alerts based on questionnaire
    has_quiz = len(q_current_colors) > 0
    if has_quiz:
        # Alert if wearing colors that don't match their season
        avoid_colors = advice.get("palette_avoid", [])
        avoid_summary = advice.get("avoid_summary", "")
        if "Noir" in q_current_colors and season_colors[0] not in ["#000000", "#1A1A2E", "#0D0D0D", "#1C1C1C"]:
            black_alt = advice.get("black_alt", "")
            if black_alt and "noir" not in black_alt.lower()[:20]:
                st.warning(f"Vous portez souvent du noir, mais votre saison prefere : **{black_alt}**")
        if "Beige/Camel" in q_current_colors and any("Spring" not in season and "Autumn" not in season for _ in [1]):
            if "Summer" in season or "Winter" in season:
                st.warning(f"Le beige/camel n'est pas ideal pour {season}. Preferez : **{advice.get('white_alt', 'blanc froid')}** ou **gris**")

        if q_hair_dyed == "Oui, ils sont teints":
            st.info("Vos cheveux sont teints — les conseils colorimetriques se basent sur votre couleur NATURELLE pour plus de precision. Votre styliste peut adapter.")

    # ---- Tabs by view mode ----
    # Check if Gemini API key is available
    has_ai = False
    try:
        ai_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
        has_ai = bool(ai_key)
    except Exception:
        ai_key = os.environ.get("GEMINI_API_KEY", "")
        has_ai = bool(ai_key)

    if view_mode == "Client":
        tab_labels = ["Profil", "Essayage", "Conseils"]
        if has_ai:
            tab_labels.append("Coach IA")
        tab_labels.append("Photo")
    elif view_mode == "Professionnel":
        tab_labels = ["Profil", "Essayage", "Conseils", "Coaching"]
        if has_ai:
            tab_labels.append("Coach IA")
        tab_labels.append("Photo")
    else:  # Avance
        tab_labels = ["Profil", "Essayage", "Conseils", "Coaching"]
        if has_ai:
            tab_labels.append("Coach IA")
        tab_labels += ["Detection", "Debug"]

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # ---- TAB: Profil ----
    with tabs[tab_idx]:
        st.markdown("### Votre profil en 4 dimensions")
        for label, raw, vmin, vmax, left, right in [
            (f"Sous-ton : {profile['undertone']}", profile["raw_undertone"], -1, 1, "Froid", "Chaud"),
            (f"Valeur : {profile['depth']}", profile["raw_depth"], -1, 1, "Fonce", "Clair"),
            (f"Chroma : {profile['chroma']}", profile["raw_chroma"], -1, 1, "Doux", "Vif"),
            (f"Contraste : {profile['contrast']}", profile["raw_contrast"], 0, 1, "Bas", "Eleve"),
        ]:
            fig = render_gauge(raw, vmin, vmax, left, right, label)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        st.markdown("### Correspondance saisons")
        for i, entry in enumerate(top3):
            icon = ["🥇", "🥈", "🥉"][i]
            st.markdown(f"{icon} **{entry['season']}** — {entry['match_pct']:.0f}%")
            st.progress(entry["match_pct"] / 100.0)

        # Quick summary
        if advice:
            st.markdown("---")
            st.success(f"**A porter** : {advice.get('best_summary', '')}")
            st.error(f"**A eviter** : {advice.get('avoid_summary', '')}")

            icons = advice.get("icons", [])
            if icons:
                st.caption(f"Meme palette que : {', '.join(icons)}")

        # Personal diagnostic
        st.markdown("---")
        st.markdown("### Votre diagnostic personnalise")
        for d in diagnostic:
            st.markdown(f"{d['icon']} **{d['feature']}** — {d['title']}")
            st.caption(d["detail"])

        # Hair & eyebrow info
        if hair_info["color"] != "inconnu":
            st.markdown("---")
            st.markdown(f"**Cheveux detectes** : {hair_info['color']} ({hair_info['warmth']}, {hair_info['depth']})")
        if eyebrow_info["color"] != "inconnu" and eyebrow_info["color"] != hair_info["color"]:
            st.caption(f"Sourcils : {eyebrow_info['color']} — si differents des cheveux, c'est souvent un indice de votre couleur naturelle.")
        if lip_undertone != "inconnu":
            st.caption(f"Levres : pigmentation {lip_undertone}")

        # Share button
        st.markdown("---")
        palette_for_story = SEASON_PALETTES.get(season, [])
        try:
            story_bytes = generate_story_image(
                image_rgb, season, tagline, palette_for_story, profile,
            )
            st.download_button(
                "Telecharger mon profil (image)",
                data=story_bytes,
                file_name=f"pikolab_{season.lower().replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception:
            pass  # Pillow font issue etc — graceful fallback
    tab_idx += 1

    # ---- TAB: Essayage (virtual draping) ----
    with tabs[tab_idx]:
        st.markdown("### Essayage virtuel")
        st.caption("Voyez sur votre propre visage quelles couleurs vous illuminent — et lesquelles vous ternissent.")

        good_hex = advice.get("palette_accents", SEASON_PALETTES.get(season, []))[:3]
        bad_hex = advice.get("palette_avoid", [])[:3]

        if good_hex and bad_hex:
            grid = render_draping_grid(image_rgb, landmarks, good_hex, bad_hex)
            st.image(grid, use_container_width=True)
        else:
            st.info("Pas assez de couleurs pour generer la comparaison.")

        # Free draping tool (all modes)
        st.markdown("---")
        st.markdown("### Testez une couleur")
        test_color = st.color_picker("Choisissez une couleur a essayer", "#E07A5F", key="drape_picker")
        draped = render_drape(image_rgb, landmarks, test_color)
        st.image(draped, caption=f"Draping {test_color}", use_container_width=True)

        # Compatibility score
        combined_palette = list(SEASON_PALETTES.get(season, []))
        combined_palette += advice.get("palette_neutrals", [])
        score, nearest, suggestion = compute_color_compatibility(test_color, {season: combined_palette}, season)

        if score >= 75:
            st.success(f"**{score}% match** — {suggestion}")
        elif score >= 50:
            st.warning(f"**{score}% match** — {suggestion}")
        else:
            st.error(f"**{score}% match** — {suggestion}")
    tab_idx += 1

    # ---- TAB: Conseils ----
    with tabs[tab_idx]:
        if not advice:
            st.warning(f"Pas de conseils disponibles pour {season}.")
        else:
            # Palette
            st.markdown("### Palette")
            fig_pal = render_categorized_palette(season, advice)
            st.pyplot(fig_pal)
            plt.close(fig_pal)
            st.markdown(f"**Metaux** : {advice.get('metals', '')}")
            black_alt = advice.get("black_alt", "")
            white_alt = advice.get("white_alt", "")
            if black_alt:
                st.markdown(f"**Alternative au noir** : {black_alt}")
            if white_alt:
                st.markdown(f"**Alternative au blanc** : {white_alt}")

            st.markdown("---")

            # Maquillage
            makeup = advice.get("makeup", {})
            show_makeup = (not has_quiz) or q_interest in ["Maquillage", "Tout"]
            with st.expander("Maquillage", expanded=show_makeup):
                st.markdown(f"**Fond de teint** : {makeup.get('foundation', '')}")
                lips = makeup.get("lips", [])
                if lips:
                    st.markdown(f"**Levres** : {', '.join(lips)}")
                eyes = makeup.get("eyes", [])
                if eyes:
                    st.markdown(f"**Yeux** : {', '.join(eyes)}")
                blush = makeup.get("blush", [])
                if blush:
                    st.markdown(f"**Blush** : {', '.join(blush)}")
                st.markdown(f"**Sourcils** : {makeup.get('eyebrows', '')}")

                # Makeup looks
                st.markdown("---")
                st.markdown("**Looks cles :**")
                for look_key, look_label in [("look_naturel", "Naturel"), ("look_soiree", "Soiree"), ("look_pro", "Bureau")]:
                    look = makeup.get(look_key, "")
                    if look:
                        st.markdown(f"- **{look_label}** : {look}")

            # Vetements
            clothing = advice.get("clothing", {})
            show_clothing = (not has_quiz) or q_interest in ["Garde-robe", "Tout"]
            with st.expander("Vetements", expanded=show_clothing):
                combos = clothing.get("best_combinations", [])
                if combos:
                    st.markdown("**Combinaisons gagnantes :**")
                    for combo in combos:
                        st.markdown(f"- {combo}")
                st.markdown(f"**Motifs** : {clothing.get('patterns', '')}")
                st.markdown(f"**Echelle motifs** : {clothing.get('pattern_scale', '')}")
                st.markdown(f"**Tissus** : {clothing.get('fabrics', '')}")
                st.info(f"Contraste : {clothing.get('contrast_tip', '')}")

                # Capsule wardrobe
                capsule = clothing.get("capsule", [])
                if capsule:
                    st.markdown("---")
                    st.markdown("**Capsule garde-robe :**")
                    for item in capsule:
                        st.markdown(f"- {item}")

                tip = clothing.get("shopping_tip", "")
                if tip:
                    st.markdown("---")
                    st.caption(f"Astuce shopping : {tip}")

            # Cheveux
            hair = advice.get("hair", {})
            show_hair = (not has_quiz) or q_interest in ["Cheveux", "Tout"]
            with st.expander("Cheveux", expanded=show_hair):
                for h in hair.get("ideal", []):
                    st.markdown(f"- ✅ {h}")
                for h in hair.get("avoid", []):
                    st.markdown(f"- ❌ {h}")
                tips = hair.get("tips", "")
                if tips:
                    st.caption(tips)

            # Accessoires
            acc = advice.get("accessories", {})
            with st.expander("Accessoires", expanded=False):
                st.markdown(f"**Lunettes** : {acc.get('glasses', '')}")
                st.markdown(f"**Bijoux** : {acc.get('jewelry', '')}")
                st.markdown(f"**Sacs & chaussures** : {acc.get('bags_shoes', '')}")
                scarves = acc.get("scarves", "")
                if scarves:
                    st.markdown(f"**Foulards** : {scarves}")
                nails = acc.get("nails", "")
                if nails:
                    st.markdown(f"**Ongles** : {nails}")
    tab_idx += 1

    # ---- TAB: Coaching (Pro + Avance) ----
    if view_mode in ["Professionnel", "Avance"]:
        with tabs[tab_idx]:
            expert = advice.get("expert", {})
            if not expert:
                st.info("Pas de donnees coaching pour cette saison.")
            else:
                st.markdown("### Draping — couleurs de test")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Confirment cette saison :**")
                    for c in expert.get("draping_confirm", []):
                        st.markdown(f"- ✅ {c}")
                with col2:
                    st.markdown("**Rejettent cette saison :**")
                    for c in expert.get("draping_reject", []):
                        st.markdown(f"- ❌ {c}")

                st.markdown("---")
                st.markdown("### Differentiation")
                st.markdown(f"**Saison cle** : {expert.get('key_differentiator', '')}")
                st.markdown(f"**Confondue avec** : {expert.get('confused_with', '')}")
                st.markdown(f"**Variations de carnation** : {expert.get('skin_variations', '')}")

                st.markdown("---")
                st.markdown("### Notes de coaching")
                st.info(expert.get("coaching_notes", ""))

                mistakes = expert.get("common_mistakes", [])
                if mistakes:
                    st.markdown("**Erreurs frequentes du client :**")
                    for m in mistakes:
                        st.markdown(f"- {m}")
        tab_idx += 1

    # ---- TAB: Coach IA ----
    if has_ai:
        with tabs[tab_idx]:
            st.markdown("### Votre coach colorimetrie personnel")
            st.caption(
                "Posez vos questions : tenue pour une occasion, achat en magasin, "
                "couleur de cheveux, maquillage... Le coach connait votre profil complet."
            )

            # Build quiz data for context
            quiz_data = {}
            if "q_hair" in st.session_state:
                quiz_data["hair_dyed"] = st.session_state.get("q_hair", "")
                quiz_data["natural_hair"] = st.session_state.get("q_nat_hair", "")
                quiz_data["style"] = st.session_state.get("q_style", "")
                quiz_data["work"] = st.session_state.get("q_work", "")
                quiz_data["current_colors"] = ", ".join(st.session_state.get("q_colors", []))
                quiz_data["interest"] = st.session_state.get("q_interest", "")

            system_prompt = build_coach_system_prompt(
                season, advice, profile, diagnostic, hair_info, lip_undertone, quiz_data,
            )

            # Init chat history
            if "coach_messages" not in st.session_state:
                st.session_state.coach_messages = []

            # Display history
            for msg in st.session_state.coach_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Chat input
            if prompt := st.chat_input("Ex: Je vais a un mariage, qu'est-ce que je porte ?"):
                st.session_state.coach_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        model = get_gemini_model(ai_key, system_prompt)
                        response = st.write_stream(
                            stream_coach_response(model, st.session_state.coach_messages[:-1], prompt)
                        )
                        st.session_state.coach_messages.append({"role": "assistant", "content": response})
                    except Exception as exc:
                        err_msg = f"Erreur du coach : {exc}"
                        st.error(err_msg)
                        st.session_state.coach_messages.append({"role": "assistant", "content": err_msg})
        tab_idx += 1

    # ---- TAB: Photo (Client + Pro) ----
    if view_mode in ["Client", "Professionnel"]:
        with tabs[tab_idx]:
            st.image(image_rgb, caption="Originale", use_container_width=True)
            st.image(corrected, caption=f"Apres correction ({correction_method})", use_container_width=True)
        tab_idx += 1

    # ---- TABs: Detection + Debug (Avance only) ----
    if view_mode == "Avance":
        with tabs[tab_idx]:  # Detection
            # Build extended overlay with all masks
            overlay = corrected.copy().astype(np.float32)
            for mask, color in [
                (skin_mask, [0, 200, 0]),       # green = skin
                (iris_mask, [100, 100, 255]),    # blue = iris
                (hair_mask, [255, 165, 0]),      # orange = hair
                (lip_mask, [255, 50, 100]),      # pink = lips
                (eyebrow_mask, [180, 120, 60]),  # brown = eyebrows
            ]:
                region = mask > 0
                if region.any():
                    overlay[region] = overlay[region] * 0.5 + np.array(color, dtype=np.float32) * 0.5
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            st.image(overlay, caption="Zones detectees", use_container_width=True)
            st.caption("Vert=peau, Bleu=iris, Orange=cheveux, Rose=levres, Brun=sourcils")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Peau", f"{skin_px:,} px")
            col2.metric("Iris", f"{iris_px:,} px")
            col3.metric("Cheveux", f"{hair_px:,} px")
            col4.metric("Levres", f"{lip_px:,} px")

            st.image(image_rgb, caption="Originale", use_container_width=True)
            st.image(corrected, caption=f"Correction ({correction_method})", use_container_width=True)
        tab_idx += 1

        with tabs[tab_idx]:  # Debug
            st.subheader("CIELab — Peau")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("L*", f"{skin_stats['L']:.1f}")
            col2.metric("a*", f"{skin_stats['a']:.1f}")
            col3.metric("b*", f"{skin_stats['b']:.1f}")
            col4.metric("C*", f"{skin_stats['C']:.1f}")

            fig_hist = render_lab_histograms(skin_lab, "Distribution CIELab — Peau")
            st.pyplot(fig_hist)
            plt.close(fig_hist)

            if iris_stats:
                st.subheader("Iris")
                iris_rgb = iris_stats["rgb"]
                st.markdown(
                    f"L*={iris_stats['L']:.1f}, a*={iris_stats['a']:.1f}, "
                    f"b*={iris_stats['b']:.1f}, C*={iris_stats['C']:.1f} — "
                    f"RGB({iris_rgb[0]}, {iris_rgb[1]}, {iris_rgb[2]})"
                )

            st.subheader("Radar")
            fig_radar = render_radar_chart(scores, season)
            st.pyplot(fig_radar)
            plt.close(fig_radar)

            st.markdown(
                f"Temperature: {scores['temperature']:+.3f} | "
                f"Valeur: {scores['value']:+.3f} | "
                f"Saturation: {scores['saturation']:+.3f} | "
                f"Contraste: {contrast:.3f}"
            )

            st.subheader("Distances 16 saisons")
            point = np.array([scores["temperature"], scores["value"], scores["saturation"]])
            all_distances = []
            for name, centroid in SEASON_CENTROIDS.items():
                dist = np.linalg.norm(point - np.array(centroid))
                all_distances.append({"Saison": name, "Distance": round(dist, 3)})
            all_distances.sort(key=lambda x: x["Distance"])
            st.dataframe(all_distances, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
