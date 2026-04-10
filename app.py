"""
PikoLab PoC — Analyse Colorimetrique Saisonniere
Determine la palette saisonniere (16 saisons) a partir d'une photo de visage.
Pipeline : MediaPipe Face Mesh -> masquage peau/iris -> correction couleur -> CIELab -> classification
"""

import os
import urllib.request
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from google import genai
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 150
import matplotlib.patches as mpatches
from collections import OrderedDict

from season_advice import SEASON_ADVICE
from multi_agent import run_consensus_analysis

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")


@st.cache_resource
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
    "temp_center": 13.0,  # Peau neutre réelle ≈ b* 12-14 en CIELab standard (était 17 → trop haut)
    "temp_scale": 8.0,   # Plage réaliste b*: 5 (froid) → 21 (chaud), scale réduit de 12 (était trop large)
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
    # Ordre : sous-saison la plus distinctive en premier, True en dernier (fallback)
    "Spring": [
        ("Warm Spring",   "temperature", "high"),  # clairement chaud avant tout
        ("Light Spring",  "value",       "high"),
        ("Bright Spring", "saturation",  "high"),
        ("True Spring",   None,          None),
    ],
    "Summer": [
        ("Cool Summer",   "temperature", "low"),   # clairement froid avant tout
        ("Light Summer",  "value",       "high"),
        ("Soft Summer",   "saturation",  "low"),
        ("True Summer",   None,          None),
    ],
    "Autumn": [
        ("Warm Autumn",   "temperature", "high"),  # clairement chaud avant tout
        ("Deep Autumn",   "value",       "low"),   # sombre ET chaud (bornes strictes)
        ("Soft Autumn",   "saturation",  "low"),
        ("True Autumn",   None,          None),
    ],
    "Winter": [
        ("Cool Winter",   "temperature", "low"),   # clairement froid avant tout
        ("Deep Winter",   "value",       "low"),
        ("Bright Winter", "saturation",  "high"),
        ("True Winter",   None,          None),
    ],
}

# Centroïdes professionnels (température, valeur, saturation) — Sci/ART / Zyla / Color Alliance
# Recalibrés pour temp_center=13, temp_scale=8 (b*=13 → score 0, b*=21 → score +1)
SEASON_CENTROIDS = {
    "Light Spring":  ( 0.35,  0.72,  0.22),   # chaud, très clair, modéré
    "Warm Spring":   ( 0.72,  0.32,  0.22),   # clairement chaud, moyen-clair
    "Bright Spring": ( 0.32,  0.38,  0.78),   # chaud, vif
    "True Spring":   ( 0.35,  0.52,  0.38),   # neutre-chaud équilibré, clair
    "Light Summer":  (-0.38,  0.75, -0.28),   # froid, très clair, doux
    "Cool Summer":   (-0.75,  0.22, -0.25),   # clairement froid, moyen
    "Soft Summer":   (-0.30,  0.20, -0.68),   # froid à neutre, très doux
    "True Summer":   (-0.50,  0.45, -0.32),   # froid équilibré, moyen-clair
    "Soft Autumn":   ( 0.20, -0.22, -0.68),   # légèrement chaud, moyen, très doux
    "Warm Autumn":   ( 0.72, -0.25,  0.18),   # clairement chaud, moyen-sombre
    "Deep Autumn":   ( 0.55, -0.72,  0.05),   # chaud ET sombre (pas neutre-chaud!)
    "True Autumn":   ( 0.35, -0.45, -0.12),   # neutre-chaud à chaud, moyen-sombre
    "Deep Winter":   (-0.32, -0.75,  0.22),   # froid à neutre, sombre
    "Cool Winter":   (-0.75, -0.22,  0.18),   # clairement froid, moyen
    "Bright Winter": (-0.28, -0.25,  0.78),   # froid à neutre, vif
    "True Winter":   (-0.52, -0.50,  0.25),   # froid équilibré, sombre
}

# ============================================================
# Intervalles professionnels par saison (temp_min, temp_max, val_min, val_max)
#
# Règles fondamentales (Sci/ART, Zyla, Color Me Beautiful) :
#   - "Neutre-chaud" (0.10 ≤ temp ≤ 0.35) → True Autumn / Soft Autumn / True Spring
#     JAMAIS Deep Autumn ni Warm Autumn ni Warm Spring
#   - Deep Autumn exige temp ≥ 0.38 ET val ≤ -0.38 (chaud ET sombre, pas juste sombre)
#   - Warm Spring / Warm Autumn exigent temp ≥ 0.45 (teint clairement doré-bronze)
#   - Cool Summer / Cool Winter exigent temp ≤ -0.42 (teint clairement rosé-bleuté)
#   - Soft Autumn / Soft Summer : zone neutre admise, chroma très faible seulement
#   - Light Spring / Light Summer : valeur élevée obligatoire (teint clair-pâle)
# ============================================================
SEASON_BOUNDS = {
    # (temp_min, temp_max, val_min, val_max)

    # PRINTEMPS — chaud, clair à moyen, vif à modéré
    "Light Spring":  ( 0.12,  0.65,  0.32,  1.00),  # chaud léger + clair
    "Warm Spring":   ( 0.45,  1.00, -0.12,  0.78),  # clairement chaud (doré-bronzé)
    "Bright Spring": ( 0.10,  0.70, -0.08,  0.85),  # chaud + très vif
    "True Spring":   ( 0.10,  0.65,  0.05,  0.85),  # neutre-chaud équilibré

    # ÉTÉ — froid, clair à moyen, doux
    "Light Summer":  (-0.70,  0.00,  0.30,  1.00),  # froid + clair
    "Cool Summer":   (-1.00, -0.42, -0.22,  0.72),  # clairement froid (rosé-bleuté)
    "Soft Summer":   (-0.65,  0.05, -0.35,  0.62),  # froid à neutre, très doux
    "True Summer":   (-0.78, -0.10,  0.00,  0.78),  # froid équilibré

    # AUTOMNE — chaud, moyen à sombre, doux à modéré
    "Soft Autumn":   ( 0.00,  0.42, -0.58,  0.22),  # neutre à légèrement chaud, doux
    "Warm Autumn":   ( 0.45,  1.00, -0.68,  0.22),  # clairement chaud (terre-dorée)
    "Deep Autumn":   ( 0.38,  0.85, -1.00, -0.38),  # chaud (≥0.38) ET sombre (≤-0.38)
    "True Autumn":   ( 0.12,  0.62, -0.78,  0.05),  # neutre-chaud à chaud, moyen-sombre

    # HIVER — froid, moyen à sombre, vif ou profond
    "Deep Winter":   (-0.62,  0.10, -1.00, -0.38),  # froid à neutre + sombre
    "Cool Winter":   (-1.00, -0.42, -0.58,  0.38),  # clairement froid
    "Bright Winter": (-0.62,  0.12, -0.48,  0.42),  # froid à neutre + très vif
    "True Winter":   (-0.80, -0.10, -0.82,  0.00),  # froid équilibré + sombre
}


# ============================================================
# FACE DETECTION
# ============================================================

@st.cache_resource(show_spinner="Chargement du modèle de détection visage...")
def get_face_landmarker():
    ensure_model()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def detect_face(image_rgb):
    landmarker = get_face_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    h, w = image_rgb.shape[:2]
    return [(int(lm.x * w), int(lm.y * h)) for lm in result.face_landmarks[0]]


def validate_face(landmarks, image_shape):
    """Validate face quality. Returns (ok, error_message)."""
    h, w = image_shape[:2]
    if not landmarks:
        return False, "Aucun visage detecte."

    # --- Face size: bounding box must be >12% of image area ---
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    face_w = max(xs) - min(xs)
    face_h = max(ys) - min(ys)
    face_area = face_w * face_h
    img_area = h * w
    if face_area < img_area * 0.04:
        return False, "Visage trop petit. Rapprochez-vous de la camera ou recadrez la photo."

    # --- Frontal check: left eye and right eye should be roughly symmetric ---
    # Landmark 33 = left eye outer, 263 = right eye outer
    # Landmark 1 = nose tip
    if len(landmarks) > 263:
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose = landmarks[1]

        # Distance from nose to each eye
        dist_left = abs(nose[0] - left_eye[0])
        dist_right = abs(nose[0] - right_eye[0])

        if min(dist_left, dist_right) > 0:
            symmetry = max(dist_left, dist_right) / min(dist_left, dist_right)
            if symmetry > 2.5:
                return False, "Visage trop de profil. Regardez la camera de face."

    # --- Eyes open check (for iris detection quality) ---
    if len(landmarks) >= 478:
        # Left eye: top lid ~159, bottom lid ~145, should have vertical distance
        top_lid = landmarks[159]
        bot_lid = landmarks[145]
        eye_opening = abs(top_lid[1] - bot_lid[1])
        if eye_opening < face_h * 0.02:
            return False, "Vos yeux semblent fermes. Ouvrez les yeux et reprenez la photo."

    return True, ""


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


def create_neck_mask(shape, landmarks):
    """
    Masque de la zone cou/decollete sous le menton.
    Utilise la position du menton (via JAWLINE_IDX) et descend de ~22 % de la hauteur du visage.
    Zone trapezoidale, retrecissant vers le bas pour eviter les bords cheveux/vetements.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    jaw_pts = [landmarks[i] for i in JAWLINE_IDX if i < len(landmarks)]
    if len(jaw_pts) < 5:
        return mask

    face_top = min(p[1] for p in jaw_pts)
    chin_y = max(p[1] for p in jaw_pts)
    face_height = max(1, chin_y - face_top)

    neck_top = chin_y
    neck_bottom = min(h - 5, chin_y + int(face_height * 0.22))
    if neck_bottom <= neck_top + 10:
        return mask

    left_x = min(p[0] for p in jaw_pts)
    right_x = max(p[0] for p in jaw_pts)
    jaw_width = max(1, right_x - left_x)
    # Inset les bords pour eviter les cheveux et le col du vetement
    inset_top = int(jaw_width * 0.18)
    inset_bot = int(jaw_width * 0.28)

    pts = np.array([
        [left_x + inset_top,  neck_top],
        [right_x - inset_top, neck_top],
        [right_x - inset_bot, neck_bottom],
        [left_x + inset_bot,  neck_bottom],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


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
    """Detect a white paper sheet. Strict criteria to avoid false positives.

    Requirements:
    - Large bright neutral area (>5% of image)
    - Roughly rectangular shape (solidity > 0.7)
    - Very neutral color (low chroma in LAB)
    - Mean brightness very high (L > 220 in OpenCV LAB)
    """
    h, w = image_rgb.shape[:2]
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0]
    a_ch = lab[:, :, 1].astype(int)
    b_ch = lab[:, :, 2].astype(int)

    # Thresholds: bright AND sufficiently neutral
    # Slightly relaxed to handle folded A4 (smaller area, possible fold shadow)
    white_mask = (
        (l_ch > 175)
        & (np.abs(a_ch - 128) < 22)
        & (np.abs(b_ch - 128) < 22)
    ).astype(np.uint8) * 255

    # Morphological cleanup to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = h * w * 0.02  # Folded A4 is half the area → lower threshold
    img_area = h * w

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            break

        # Max 40% of image (a wall is too big to be a sheet of paper)
        if area > img_area * 0.40:
            continue

        # Rectangularity check: solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.55:
            continue

        # Aspect ratio check: paper is roughly rectangular, not a thin strip
        x, y, rw, rh = cv2.boundingRect(contour)
        aspect = max(rw, rh) / (min(rw, rh) + 1)
        if aspect > 5:
            continue  # Too elongated (probably a wall edge or ceiling)

        # Verify mean color is truly white (very high L, very neutral a/b)
        region_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, -1)
        mean_l = cv2.mean(l_ch, mask=region_mask)[0]
        mean_a = cv2.mean(a_ch.astype(np.uint8), mask=region_mask)[0]
        mean_b = cv2.mean(b_ch.astype(np.uint8), mask=region_mask)[0]

        if mean_l < 185:
            continue  # Not bright enough
        if abs(mean_a - 128) > 18 or abs(mean_b - 128) > 18:
            continue  # Not neutral enough

        return np.array(cv2.mean(image_rgb, mask=region_mask)[:3])

    return None


def correct_wb_with_reference(image_rgb, reference_rgb):
    """Full RGB white balance correction (for visual output and iris/hair analysis)."""
    gains = 255.0 / (np.array(reference_rgb) + 1e-6)
    gains = gains / gains.max()
    corrected = image_rgb.astype(np.float32)
    for c in range(3):
        corrected[:, :, c] *= gains[c]
    return np.clip(corrected, 0, 255).astype(np.uint8)


def compute_wb_lab_cast(reference_rgb):
    """Measure the chromatic cast (a*, b*) of the white reference in Lab space.

    A true white paper under D65 has a*≈0, b*≈2 (slight warm due to optical
    brighteners). Under tungsten it might read b*≈18. The cast is the deviation
    from the neutral-paper baseline.
    Returns (a_cast, b_cast) in skimage Lab units.
    """
    ref = np.array([reference_rgb[:3]], dtype=np.uint8)
    ref_lab = pixels_to_lab(ref)[0]
    # Only correct warming casts: if reference already reads neutral or cool,
    # don't add warmth by overcorrecting (clamp to 0).
    return float(ref_lab[1]), max(0.0, float(ref_lab[2] - 2.0))


def apply_lab_cast_correction(lab_pixels, a_cast, b_cast, strength=1.0):
    """Subtract the illuminant chromatic cast from Lab pixels.

    Works directly in Lab space (linear and predictable — no RGB non-linearity).
    strength=1.0 removes the full measured cast; lower values remove less.
    """
    if len(lab_pixels) == 0:
        return lab_pixels
    corrected = lab_pixels.copy()
    corrected[:, 1] -= a_cast * strength
    corrected[:, 2] -= b_cast * strength
    return corrected


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


def compute_skin_stats_robust(lab_pixels):
    """Stats robustes : filtre les pixels haute chroma (imperfections, rougeurs)
    et utilise la médiane pour b* afin de résister aux pics d'overtone.

    - On conserve les 75% de pixels les moins saturés (joues nettes, cou propre)
    - b* en médiane : insensible aux taches jaunes/rouges isolées
    - C* en moyenne globale : représente la saturation réelle de la zone
    """
    if len(lab_pixels) == 0:
        return {"L": 0.0, "a": 0.0, "b": 0.0, "C": 0.0}
    C = np.sqrt(lab_pixels[:, 1] ** 2 + lab_pixels[:, 2] ** 2)
    threshold = np.percentile(C, 75)
    clean = lab_pixels[C <= threshold]
    if len(clean) < 20:
        clean = lab_pixels
    return {
        "L": float(np.mean(clean[:, 0])),
        "a": float(np.mean(clean[:, 1])),
        "b": float(np.median(clean[:, 2])),  # médiane : résiste aux overtones isolés
        "C": float(np.mean(C)),
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


def compute_professional_profile(scores, contrast, skin_temp=None):
    """Human-readable 4-dimension profile for stylists.

    skin_temp: temperature score computed from skin pixels only (not iris-blended).
    Used for the undertone display so that cool eye color doesn't make a warm
    skin person appear cold on the gauge.
    """
    # Undertone gauge: skin-only temperature when available
    t = float(np.clip(skin_temp, -1, 1)) if skin_temp is not None else scores["temperature"]
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


def _in_bounds(season_name, scores):
    """Vérifie que les scores respectent les intervalles professionnels de la saison."""
    b = SEASON_BOUNDS.get(season_name)
    if b is None:
        return True
    t_min, t_max, v_min, v_max = b
    return t_min <= scores["temperature"] <= t_max and v_min <= scores["value"] <= v_max


def classify_season(scores, dominance_threshold):
    """
    Classifie en l'une des 16 saisons.
    Utilise les intervalles professionnels (SEASON_BOUNDS) comme filtre dur :
    une saison hors de ses bornes temp/value ne peut pas être sélectionnée.
    Si aucune sous-saison n'est dans les bornes, on retombe sur la 'True' de la base.
    """
    temp = scores["temperature"]
    val  = scores["value"]

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
        if not _in_bounds(sub_name, scores):
            continue  # hors intervalles professionnels → exclu
        sv = scores[axis]
        strength = sv if direction == "high" else -sv
        if strength > best_strength:
            best_strength = strength
            best_match = sub_name

    if best_strength > dominance_threshold and best_match is not None:
        return best_match

    # Fallback : sous-saison la plus proche par centroïde (bornes respectées en priorité)
    point = np.array([scores["temperature"], scores["value"], scores["saturation"]])
    best_name, best_dist = None, float("inf")
    for sub_name, _, _ in rules:
        centroid = SEASON_CENTROIDS.get(sub_name)
        if centroid is None:
            continue
        penalty = 1.0 if _in_bounds(sub_name, scores) else 2.0
        d = np.linalg.norm(point - np.array(centroid)) * penalty
        if d < best_dist:
            best_dist = d
            best_name = sub_name
    return best_name if best_name else f"True {base}"


def classify_season_in_base(scores, base):
    """Retourne la meilleure sous-saison dans une base donnée (Spring/Summer/Autumn/Winter).
    Utilisé pour appliquer un override de base issu du consensus multi-agents
    tout en gardant les scores pixel intacts (évite désynchro jauge/saison).
    """
    rules = SUBSEASON_RULES.get(base, [])
    best_match = None
    best_strength = -1.0

    for sub_name, axis, direction in rules:
        if axis is None:
            continue
        if not _in_bounds(sub_name, scores):
            continue
        sv = scores[axis]
        strength = sv if direction == "high" else -sv
        if strength > best_strength:
            best_strength = strength
            best_match = sub_name

    if best_match:
        return best_match

    # Fallback : centroïde le plus proche dans cette base
    point = np.array([scores["temperature"], scores["value"], scores["saturation"]])
    best_name, best_dist = None, float("inf")
    for sub_name, _, _ in rules:
        centroid = SEASON_CENTROIDS.get(sub_name)
        if centroid is None:
            continue
        penalty = 1.0 if _in_bounds(sub_name, scores) else 2.0
        d = np.linalg.norm(point - np.array(centroid)) * penalty
        if d < best_dist:
            best_dist = d
            best_name = sub_name
    return best_name if best_name else f"True {base}"


def classify_top3(scores):
    """
    Top 3 saisons par distance aux centroides, avec filtre par intervalles professionnels.
    Les saisons hors de leurs bornes reçoivent une pénalité de distance × 2.5
    pour ne pas apparaître en tête si le profil ne correspond pas.
    """
    point = np.array([scores["temperature"], scores["value"], scores["saturation"]])
    distances = {}
    for name, centroid in SEASON_CENTROIDS.items():
        dist = np.linalg.norm(point - np.array(centroid))
        if not _in_bounds(name, scores):
            dist *= 2.5  # pénalité hors intervalles professionnels
        distances[name] = dist

    sorted_seasons = sorted(distances.items(), key=lambda x: x[1])

    max_dist = max(d for _, d in sorted_seasons[:5]) + 0.01
    top3 = []
    total_score = 0
    for name, dist in sorted_seasons[:3]:
        score = max(0, (max_dist - dist) / max_dist)
        top3.append((name, score, dist))
        total_score += score

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
    import io
    import textwrap

    W, H = 1080, 1920
    MARGIN = 60
    story = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(story)

    pc = palette_colors[0] if palette_colors else "#E07A5F"
    pc_rgb = tuple(int(pc.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))

    # Fonts
    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
        font_xs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except (OSError, IOError):
        font_big = ImageFont.load_default()
        font_med = font_big
        font_sm = font_big
        font_xs = font_big

    # --- Gradient header ---
    for y in range(350):
        a = y / 350
        r = int(pc_rgb[0] * (1 - a) + 255 * a)
        g = int(pc_rgb[1] * (1 - a) + 255 * a)
        b = int(pc_rgb[2] * (1 - a) + 255 * a)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # --- Photo circle ---
    photo = Image.fromarray(image_rgb)
    size = 420
    photo = photo.resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, size, size], fill=255)
    x_photo = (W - size) // 2
    y_photo = 200
    story.paste(photo, (x_photo, y_photo), mask)
    draw.ellipse(
        [x_photo - 4, y_photo - 4, x_photo + size + 4, y_photo + size + 4],
        outline=pc, width=5,
    )

    # --- Season name ---
    y_cur = y_photo + size + 50
    _draw_centered(draw, season, y_cur, W, font_big, pc)

    # --- Tagline (wrapped) ---
    y_cur += 90
    for line in textwrap.wrap(tagline, width=35):
        _draw_centered(draw, line, y_cur, W, font_med, "#666666")
        y_cur += 42

    # --- Profile: 2x2 grid ---
    y_cur += 30
    _draw_centered(draw, "Votre profil", y_cur, W, font_sm, "#333333")
    y_cur += 45

    items = [
        (f"Sous-ton: {profile['undertone']}", f"Valeur: {profile['depth']}"),
        (f"Chroma: {profile['chroma']}", f"Contraste: {profile['contrast']}"),
    ]
    for left_text, right_text in items:
        mid = W // 2
        # Left
        bbox = draw.textbbox((0, 0), left_text, font=font_sm)
        lw = bbox[2] - bbox[0]
        draw.text((mid - lw - 20, y_cur), left_text, fill="#888888", font=font_sm)
        # Right
        draw.text((mid + 20, y_cur), right_text, fill="#888888", font=font_sm)
        y_cur += 40

    # --- Palette: 2 rows of 4 ---
    y_cur += 40
    _draw_centered(draw, "Votre palette", y_cur, W, font_sm, "#333333")
    y_cur += 45

    swatch = 105
    gap = 18
    n = min(len(palette_colors), 8)
    cols = 4
    for row_start in range(0, n, cols):
        row_colors = palette_colors[row_start:row_start + cols]
        row_w = len(row_colors) * swatch + (len(row_colors) - 1) * gap
        x_start = (W - row_w) // 2
        for j, hex_c in enumerate(row_colors):
            x = x_start + j * (swatch + gap)
            draw.rounded_rectangle(
                [x, y_cur, x + swatch, y_cur + swatch],
                radius=12, fill=hex_c, outline="#DDDDDD", width=1,
            )
        y_cur += swatch + gap

    # --- Best/Avoid summary ---
    y_cur += 20
    # These come from advice but we keep it simple
    _draw_centered(draw, "pikolab.app", y_cur + 80, W, font_xs, "#CCCCCC")

    # --- Bottom gradient ---
    for y in range(H - 100, H):
        a = (y - (H - 100)) / 100
        r = int(255 * (1 - a) + pc_rgb[0] * 0.3 * a)
        g = int(255 * (1 - a) + pc_rgb[1] * 0.3 * a)
        b = int(255 * (1 - a) + pc_rgb[2] * 0.3 * a)
        draw.line([(0, y), (W, y)], fill=(int(r), int(g), int(b)))

    buf = io.BytesIO()
    story.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _draw_centered(draw, text, y, width, font, fill):
    """Draw text centered horizontally."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, y), text, fill=fill, font=font)


# ============================================================
# COACH IA (Gemini)
# ============================================================

def _format_scan_context():
    """Include last scan results in coach context if available."""
    scan = st.session_state.get("last_scan") if st else None
    if not scan:
        return ""
    lines = ["\nDERNIER SCAN VETEMENT :"]
    for hex_c, pct, score, sug in scan.get("colors", []):
        lines.append(f"- Couleur {hex_c} ({pct:.0f}% du vetement) : {score}% match — {sug}")
    lines.append(f"Score global : {scan.get('best_score', 0)}%")
    return "\n".join(lines)


def build_coach_system_prompt(season, advice, profile, diagnostic, hair_info, lip_undertone, quiz_data, light_type=None, gender="Femme"):
    """Build a comprehensive system prompt with all client context."""
    diag_text = "\n".join(
        f"- {d['feature']}: {d['title']} — {d['detail']}" for d in diagnostic
    )

    is_man = gender == "Homme"
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

    _persona_genre = (
        "Tu t'adresses a un homme. Adapte TOUS tes conseils en consequence : "
        "vetements masculins (costumes, chemises, vestes, pulls, manteaux), "
        "soins du visage a la place du maquillage, coiffure masculine. "
        "Pas de robe, pas de jupe, pas de maquillage. Dis 'mon ami' ou utilise le prenom. "
        "Reste energique et directe, mais avec un registre adapte a un homme."
        if is_man else
        "Tu t'adresses a une femme. Tous les conseils sont adaptes : vetements, maquillage, bijoux, etc."
    )

    return f"""**Persona :**
Tu es Iris, coach en image et colorimetrie, avec la franchise et l'energie de Cristina Cordula.
Tu es passionnee, directe, tu dis les choses telles qu'elles sont — avec bienveillance mais
SANS filtre. Si une couleur est catastrophique sur le client, tu le dis clairement. Si c'est
magnifique, tu t'enthousiasmes. Tu tutoies le client. Tu utilises des expressions vivantes.
Tu parles TOUJOURS en francais.
{_persona_genre}

Tu es experte en colorimetrie saisonniere (systeme 16 saisons), en stylisme, en maquillage
et en conseil en image. Tu connais les dernieres tendances mais tu privilegies TOUJOURS
ce qui va au client plutot que ce qui est a la mode.

**Grounding strict :**
Tu es limitee UNIQUEMENT aux informations du profil client ci-dessous.
Tu ne fais JAMAIS reference a des connaissances sur d'autres saisons que celle du client.
Si une question sort de ton expertise (medical, psycho, etc.), dis-le honnetement.
Ne repete JAMAIS ce que le client vient de dire. Chaque reponse doit apporter
une information nouvelle et actionnable.

**Style de reponse :**
- Franche et directe : "Non, cette couleur te terne completement !" ou "LA ! Ca c'est ta couleur !"
- Concis : 2-5 phrases par defaut. Plus long uniquement si le client demande du detail.
- Progressif : commence par le verdict, developpe si on te demande.
- Actionnable : noms de couleurs precis, types de vetements, suggestions concretes.
- Si le client mentionne une couleur ou un vetement, dis CLAIREMENT si ca lui va ou pas
  et POURQUOI en te referant a son profil (sous-ton, contraste, saison).
- N'hesite pas a etre expressive : "Magnifiiique !", "Non non non !", "Tu vas voir, ca va etre sublime !"

**Guardrails :**
- Jamais mechante ou humiliante. Franche OUI, cruelle NON.
- Si le client est negatif sur son physique, le reorienter avec energie vers ses atouts.
- Ne jamais dire "tout te va" — c'est faux et ca ne l'aide pas.
- Tu as acces a Google Search. Quand le client demande des produits specifiques
  (rouge a levres, vetements, etc.), UTILISE la recherche pour trouver des articles
  reels avec prix et liens. Cite toujours la source.
- Ne jamais INVENTER un produit ou un prix. Soit tu cherches et tu trouves, soit tu
  decris ce qu'il faut chercher sans citer de marque.
- Le client peut t'envoyer des PHOTOS (vetement, maquillage, tenue, accessoire).
  Analyse la couleur dominante de l'article et dis si ca correspond a sa palette.
  Sois precise : "Ce bordeaux est parfait pour toi !" ou "Ce bleu est trop froid,
  cherche plutot un bleu canard ou un teal."

---

PROFIL DU CLIENT :
- Genre : {gender}
- Saison : {season}
- Description : {advice.get('description', '')}
- Sous-ton : {profile['undertone']} (score brut: {profile['raw_undertone']:.2f})
- Eclairage au moment de l'analyse : {light_type or 'non renseigne'}
- Valeur/profondeur : {profile['depth']}
- Chroma : {profile['chroma']}
- Contraste : {profile['contrast']}
- Cheveux detectes : {hair_info.get('color', 'inconnu')} (temperature: {hair_info.get('warmth', 'inconnu')})
- Sous-ton levres : {lip_undertone}

DIAGNOSTIC FEATURE PAR FEATURE :
{diag_text}
{quiz_text}
PALETTE — A PORTER :
{advice.get('best_summary', '')}

PALETTE — A EVITER :
{advice.get('avoid_summary', '')}

ALTERNATIVES :
- Au lieu du noir : {advice.get('black_alt', '')}
- Au lieu du blanc : {advice.get('white_alt', '')}
- Metaux : {advice.get('metals', '')}

{"SOINS VISAGE (homme) : Conseille des soins adaptés au teint et au sous-ton de la saison. Pas de maquillage. Concentre-toi sur l'harmonie des couleurs vestimentaires proches du visage (col de chemise, pull, veste)." if is_man else f"""MAQUILLAGE :
- Fond de teint : {makeup.get('foundation', '')}
- Levres : {', '.join(makeup.get('lips', []))}
- Yeux : {', '.join(makeup.get('eyes', []))}
- Blush : {', '.join(makeup.get('blush', []))}
- Look naturel : {makeup.get('look_naturel', '')}
- Look soiree : {makeup.get('look_soiree', '')}
- Look bureau : {makeup.get('look_pro', '')}"""}

VETEMENTS :
- Meilleures combinaisons : {', '.join(clothing.get('best_combinations', []))}
- Motifs recommandes : {clothing.get('patterns', '')} (echelle: {clothing.get('pattern_scale', '')})
- Tissus : {clothing.get('fabrics', '')}
- Conseil contraste : {clothing.get('contrast_tip', '')}
- Capsule garde-robe : {', '.join(clothing.get('capsule', []))}
- Astuce shopping : {clothing.get('shopping_tip', '')}

CHEVEUX :
- Couleurs ideales : {', '.join(hair.get('ideal', []))}
- A eviter : {', '.join(hair.get('avoid', []))}
- Conseil coloriste : {hair.get('tips', '')}

ACCESSOIRES :
- Lunettes : {acc.get('glasses', '')}
- Bijoux : {acc.get('jewelry', '')}
- Ongles : {acc.get('nails', '')}
- Foulards : {acc.get('scarves', '')}
- Sacs/chaussures : {acc.get('bags_shoes', '')}
{_format_scan_context()}
"""


GEMINI_MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-lite-latest"]


def stream_coach_response(api_key, system_prompt, history, user_message):
    """Stream a response from Gemini with Google Search grounding."""
    from google.genai import types

    client = genai.Client(api_key=api_key)

    contents = []
    for m in history:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    # Google Search grounding: Gemini can search the web for product recommendations
    search_tool = types.Tool(google_search=types.GoogleSearch())

    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config={
                    "system_instruction": system_prompt,
                    "tools": [search_tool],
                },
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return
        except Exception as exc:
            last_error = exc
            if "quota" not in str(exc).lower() and "429" not in str(exc):
                raise
            continue

    if last_error:
        yield f"\n\nQuota Gemini epuisee. Reessayez dans quelques minutes. ({last_error})"


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
# MULTI-PHOTO HELPERS
# ============================================================

def _extract_skin_stats_silent(image_rgb, params):
    """Run the skin-stats pipeline on one image without any Streamlit UI.

    Returns (skin_stats_dict, has_sheet) or None if face not detected.
    """
    landmarks = detect_face(image_rgb)
    if landmarks is None:
        return None
    face_ok, _ = validate_face(landmarks, image_rgb.shape)
    if not face_ok:
        return None

    skin_mask = create_skin_mask(image_rgb.shape, landmarks)
    neck_mask = create_neck_mask(image_rgb.shape, landmarks)

    wb_reference = detect_white_region(image_rgb)
    exposure_corrected = correct_exposure(image_rgb, skin_mask)

    if wb_reference is not None:
        _wb_a_cast, _wb_b_cast = compute_wb_lab_cast(wb_reference)
        if _wb_b_cast > 10:
            _wb_strength, _wb_warm_offset = 0.43, 2.5
        elif _wb_b_cast > 5:
            _wb_strength, _wb_warm_offset = 0.55, 0.0
        else:
            _wb_strength, _wb_warm_offset = 1.0, 0.0
    else:
        _wb_a_cast, _wb_b_cast, _wb_warm_offset, _wb_strength = 0.0, 0.0, 0.0, 1.0

    has_makeup = st.session_state.get("chk_has_makeup", False)
    skin_pixels_rgb = extract_pixels(exposure_corrected, skin_mask)
    neck_pixels_rgb = extract_pixels(exposure_corrected, neck_mask)

    face_lab = apply_lab_cast_correction(
        pixels_to_lab(skin_pixels_rgb), _wb_a_cast, _wb_b_cast - _wb_warm_offset, _wb_strength
    )
    face_stats = compute_skin_stats_robust(face_lab)

    _neck_lab_raw = pixels_to_lab(neck_pixels_rgb) if len(neck_pixels_rgb) >= 30 else None
    neck_lab_pixels = apply_lab_cast_correction(
        _neck_lab_raw, _wb_a_cast, _wb_b_cast - _wb_warm_offset, _wb_strength
    ) if _neck_lab_raw is not None else None
    neck_stats = compute_skin_stats_robust(neck_lab_pixels) if neck_lab_pixels is not None else None

    if neck_stats is not None:
        neck_w = 0.80 if has_makeup else 0.65
        face_w = 1.0 - neck_w
        skin_stats = {k: face_stats[k] * face_w + neck_stats[k] * neck_w
                      for k in ("L", "a", "b", "C")}
    else:
        skin_stats = {k: face_stats[k] for k in ("L", "a", "b", "C")}

    return skin_stats, wb_reference is not None


def _average_skin_stats(stats_list):
    """Average a list of skin_stats dicts (L, a, b, C)."""
    return {k: float(np.mean([s[k] for s in stats_list])) for k in ("L", "a", "b", "C")}


def _consistency_label(stats_list):
    """Return a human-readable consistency label based on b* spread."""
    b_values = [s["b"] for s in stats_list]
    spread = max(b_values) - min(b_values)
    if spread < 1.5:
        return "✅ Mesures très cohérentes", "success"
    elif spread < 3.0:
        return "✅ Mesures cohérentes", "success"
    else:
        return f"⚠️ Variation détectée entre les prises (Δb*={spread:.1f}) — résultat indicatif", "warning"


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


PWA_HEAD = """
<link rel="manifest" href="/app/static/manifest.json">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="PikoLab">
<meta name="theme-color" content="#1c1917">
<link rel="apple-touch-icon" href="/app/static/icon-192.png">
<script>
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
      navigator.serviceWorker.register('/app/static/sw.js', { scope: '/app/static/' })
        .catch(function() {});
    });
  }
</script>
"""

MOBILE_CSS = "<style>\n" \
    "/* Hide Streamlit auto-generated page navigation */\n" \
    "[data-testid='stSidebarNav'] { display: none !important; }\n" \
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
    "[data-testid='stCameraInput'] video { max-height: 50vh; object-fit: contain; transform: scaleX(-1); }\n" \
    "</style>"


# ── Rendu du bloc consensus multi-agents ──────────────────────────────────────

_AGREEMENT_LABELS = {
    "unanimite": ("Unanimite", "#2ecc71"),
    "majorite":  ("Majorite",  "#f39c12"),
    "desaccord": ("Desaccord", "#e74c3c"),
}

_BASE_ICONS = {
    "Spring": "🌸",
    "Summer": "☀️",
    "Autumn": "🍂",
    "Winter": "❄️",
}


def _render_consensus_block(consensus_data: dict) -> None:
    """Affiche le tableau de votes des agents et le résultat consensuel."""
    agreement = consensus_data.get("agreement_level", "desaccord")
    label, color = _AGREEMENT_LABELS.get(agreement, ("Inconnu", "#888"))
    consensus_season = consensus_data["consensus_season"]
    overridden = consensus_data["overridden"]
    base_votes = consensus_data.get("base_votes", {})

    # Résumé des votes saison de base
    votes_str = " · ".join(
        f"{_BASE_ICONS.get(b, '')} {b} ×{n}"
        for b, n in sorted(base_votes.items(), key=lambda x: -x[1])
    )

    override_note = " (corrige l'algorithme)" if overridden else " (confirme l'algorithme)"

    with st.expander(f"Analyse multi-agents — {label}{override_note}", expanded=overridden):
        st.markdown(
            f"<span style='color:{color}; font-weight:700;'>● {label}</span> "
            f"&nbsp;|&nbsp; Votes base : {votes_str}",
            unsafe_allow_html=True,
        )

        agents = consensus_data.get("agents", [])
        for agent in agents:
            base = agent.get("base_season", "")
            icon = _BASE_ICONS.get(base, "")
            conf_pct = int(agent.get("confidence", 0) * 100)
            is_consensus = agent["sub_season"] == consensus_season
            marker = " ✓" if is_consensus else ""
            temp = agent.get("temperature", "")
            temp_label = {"chaud": "chaud", "froid": "froid", "neutre": "neutre"}.get(temp, temp)

            st.markdown(
                f"**{agent['name']}** — `{agent['sub_season']}`{marker} "
                f"&nbsp;·&nbsp; {icon} {base} &nbsp;·&nbsp; {temp_label} &nbsp;·&nbsp; "
                f"confiance {conf_pct} %",
                unsafe_allow_html=False,
            )
            if agent.get("reasoning"):
                st.caption(agent["reasoning"])

        errors = consensus_data.get("errors", [])
        if errors:
            with st.expander("Erreurs agents", expanded=False):
                for e in errors:
                    st.warning(e)


def main():
    st.set_page_config(
        page_title="PikoLab — Analyse Colorimetrique",
        page_icon="🎨",
        layout="centered",
        initial_sidebar_state="auto",
    )
    st.markdown(MOBILE_CSS + PWA_HEAD, unsafe_allow_html=True)

    # ---- Sidebar (visible on desktop) ----
    st.sidebar.header("PikoLab")
    st.sidebar.page_link("app.py", label="Analyse", icon="🎨")
    st.sidebar.page_link("pages/scanner.py", label="Scanner", icon="📷")
    st.sidebar.page_link("pages/coach_ia.py", label="Coach Iris", icon="💬")
    st.sidebar.markdown("---")

    if st.sidebar.button("Nouvelle analyse", use_container_width=True):
        for key in ["analysis_done", "ctx", "coach_messages", "demo_image",
                     "last_scan", "has_white_sheet",
                     "multi_step", "multi_skin_stats", "multi_last_image"]:
            st.session_state.pop(key, None)
        st.rerun()

    view_mode = st.sidebar.radio(
        "Mode d'affichage",
        ["Client", "Professionnel", "Avance"],
        index=0,
    )

    st.sidebar.markdown("---")
    gender = st.sidebar.radio(
        "Genre",
        ["Femme", "Homme"],
        index=0,
        horizontal=True,
        key="chk_gender",
    )
    st.sidebar.markdown("---")

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

    # ---- Clé API Gemini ----
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    except Exception:
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

    # ---- Acquisition ----
    image_rgb = None

    # ---- Hero section (only before first analysis) ----
    hero_placeholder = st.empty()
    if "analysis_done" not in st.session_state:
        hero_placeholder.markdown("""
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

    has_sheet = False

    # ---- Questions pré-capture (tous modes sauf Demo) ----
    if mode != "Demo":
        st.markdown("**Avant la photo :**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.checkbox(
                "Maquillage",
                key="chk_has_makeup",
                value=False,
                help="Fond de teint, blush ou contouring — le cou sera utilisé comme référence principale de teint",
            )
        with col2:
            st.checkbox(
                "Sourcils naturels",
                key="chk_natural_eyebrows",
                value=True,
                help="Non teints, non tatoués, non redessinés",
            )
        with col3:
            st.checkbox(
                "Cheveux naturels",
                key="chk_natural_hair",
                value=True,
                help="Couleur non traitée (ni colorée, ni décolorée, ni balayage). Si vos cheveux sont colorés, l'analyse de leur teinte sera ignorée.",
            )
        col4, col5, col6 = st.columns(3)
        with col4:
            st.checkbox(
                "Cheveux attachés",
                key="chk_hair_tied",
                value=False,
                help="Dégager le visage et le cou améliore la précision de l'analyse",
            )
        with col5:
            st.checkbox(
                "Sans lentilles colorées",
                key="chk_no_colored_contacts",
                value=True,
                help="Les lentilles de couleur faussent l'analyse de l'iris. Retirez-les ou cochez cette case uniquement si vos lentilles sont transparentes.",
            )
        with col6:
            st.checkbox(
                "Feuille blanche",
                key="chk_use_sheet",
                value=True,
                help="Feuille A4 pliée en 2, tenue à plat à côté du visage — nécessaire pour corriger le filtre couleur de la caméra",
            )

        # ---- Avertissement plage horaire lumière naturelle ----
        _now = datetime.now()
        _h = _now.hour + _now.minute / 60
        _month = _now.month
        _is_winter = _month in (11, 12, 1, 2)
        if _month in (5, 6, 7, 8):
            _ok_start, _ok_end = 10, 16
            _season_label = "en été"
        elif _is_winter:
            _ok_start, _ok_end = 11, 14
            _season_label = "en hiver"
        else:
            _ok_start, _ok_end = 10, 15
            _season_label = "au printemps/automne"

        if not (_ok_start <= _h <= _ok_end):
            st.warning(
                f"⚠️ Il est {_now.strftime('%Hh%M')} — en dehors de la plage recommandée "
                f"({_ok_start}h–{_ok_end}h {_season_label}). "
                "La lumière est trop basse et trop chaude, ce qui peut fausser le sous-ton. "
                "Préférez une photo près d'une fenêtre exposée au nord, ou revenez dans la plage horaire."
            )
        elif _is_winter:
            st.info(
                "Lumière hivernale : le soleil est bas et la lumière plus froide que la norme D65. "
                "La **feuille blanche est indispensable** pour corriger le cast et obtenir un résultat fiable. "
                "Sans elle, le résultat sera indicatif uniquement."
            )
        light_type = "Lumière naturelle (jour)"
    else:
        light_type = "Lumière naturelle (jour)"

    use_sheet = st.session_state.get("chk_use_sheet", False)

    if mode == "Appareil photo":
        analyse_mode = st.radio(
            "Mode d'analyse",
            ["Rapide (1 photo)", "Précise (3 photos)"],
            horizontal=True,
            key="chk_analyse_mode",
            help="3 photos dans des endroits différents améliore significativement la précision",
        )

        # ---- Mode 3 photos ----
        if analyse_mode == "Précise (3 photos)":
            _STEP_CONFIGS = [
                ("Photo 1/3", "Extérieur ou meilleure lumière naturelle disponible (porte, terrasse, jardin)"),
                ("Photo 2/3", "Près d'une fenêtre — lumière naturelle indirecte"),
                ("Photo 3/3", "Autre fenêtre ou autre endroit de la pièce"),
            ]
            _step = st.session_state.get("multi_step", 1)
            _multi_stats = st.session_state.get("multi_skin_stats", [])

            if _step <= 3:
                _label, _hint = _STEP_CONFIGS[_step - 1]
                st.progress(_step / 3, text=f"**{_label}** — {_hint}")
                st.caption("Lumiere naturelle · Visage de face" + (" · Feuille A4 à côté" if use_sheet else ""))

                if _multi_stats:
                    _cols = st.columns(len(_multi_stats))
                    for _i, _s in enumerate(_multi_stats):
                        _cols[_i].metric(f"Photo {_i+1} — b*", f"{_s['b']:.1f}")

                _cam = st.camera_input(f"{_label} — {_hint}", key=f"cam_multi_{_step}")
                if _cam:
                    _img = load_image(_cam)
                    if _img is not None:
                        _img = np.fliplr(_img).copy()
                        with st.spinner(f"Analyse photo {_step}/3..."):
                            try:
                                _result = _extract_skin_stats_silent(_img, params)
                            except Exception as _e:
                                _result = None
                                st.error(f"Erreur analyse photo {_step} : {_e}")
                        if _result is None:
                            st.error("Aucun visage détecté. Reprenez la photo.")
                        else:
                            _stats, _sheet_ok = _result
                            _multi_stats.append(_stats)
                            st.session_state["multi_skin_stats"] = _multi_stats
                            st.session_state["multi_last_image"] = _img
                            st.session_state["multi_step"] = _step + 1
                            if _sheet_ok:
                                st.success(f"✅ Photo {_step} — feuille détectée (b*={_stats['b']:.1f})")
                            else:
                                st.info(f"Photo {_step} enregistrée (b*={_stats['b']:.1f})")
                            st.rerun()
                return

            # Toutes les photos capturées → utiliser la dernière pour le pipeline
            image_rgb = st.session_state.get("multi_last_image")
            if image_rgb is None:
                st.error("Erreur : images perdues. Relancez une nouvelle analyse.")
                return
            st.session_state["_multi_averaged_stats"] = _average_skin_stats(_multi_stats)
            _cons_label, _cons_type = _consistency_label(_multi_stats)
            if _cons_type == "success":
                st.success(_cons_label)
            else:
                st.warning(_cons_label)

        else:
            # ---- Mode rapide (1 photo) ----
            st.caption("Lumiere naturelle · Visage de face" + (" · Feuille A4 pliée en 2 à côté du visage" if use_sheet else ""))

        # ---- Script de capture automatique + overlay canvas (mode rapide uniquement) ----
        if analyse_mode == "Rapide (1 photo)":
            _sheet_js = "true" if use_sheet else "false"
            components.html("""
<style>
  body { margin:0; padding:0; background:transparent; overflow:hidden; }
  #bar {
    background:#1c1917; color:#fff; border-radius:10px;
    padding:10px 16px; font-size:14px; font-family:sans-serif;
    display:flex; gap:16px; align-items:center; min-height:44px;
    box-sizing:border-box; border:2px solid #555;
    transition: background .3s, border .3s;
  }
  #cd { margin-left:auto; font-weight:700; font-size:16px; }
</style>
<div id="bar">
  <span id="fs">⏳ Démarrage caméra…</span>
  <span id="ss"></span>
  <span id="cd"></span>
</div>
<script>
(function() {
  const pwin = window.parent;
  const pdoc = pwin.document;
  if (pwin._pikolabRunning) return;
  pwin._pikolabRunning = true;

  const STABLE_NEEDED  = 10;   // 3s de décompte à 300ms/frame
  const SKIN_THRESHOLD = 0.07;
  const SHEET_REQUIRED = SHEET_REQUIRED_PLACEHOLDER;
  let stableCount = 0, captured = false;
  let overlayCanvas = null, overlayCtx = null;

  // ---- Détection ----
  function isSkin(r, g, b) {
    return r>60&&r<250&&g>35&&g<220&&b>15&&b<195&&r>g&&r>b&&(r-Math.min(g,b))>10;
  }

  function detectSheet(px, w, h) {
    // Zones couvrant toute l'image (gauche, droite, haut, bas, coins)
    const zones = [
      [0,0,w/2,h],[w/2,0,w,h],         // moitiés gauche / droite
      [0,0,w/3,h],[2*w/3,0,w,h],       // tiers gauche / droit
      [0,0,w,h/3],[0,2*h/3,w,h],       // tiers haut / bas
      [0,0,w/2,h/2],[w/2,h/2,w,h],    // coins diagonaux
    ];
    for (const [x1,y1,x2,y2] of zones) {
      let white=0, total=0;
      for (let y=y1; y<y2; y+=4) for (let x=x1; x<x2; x+=4) {
        const i=((y|0)*w+(x|0))*4;
        const r=px[i], g=px[i+1], b=px[i+2];
        const avg=(r+g+b)/3;
        const spread=Math.max(r,g,b)-Math.min(r,g,b);
        total++;
        // Seuils élargis : tolérance cast jaune (spread<85) + luminosité >140
        if (avg>140 && spread<85) white++;
      }
      if (total>0 && white/total>0.10) return true;
    }
    return false;
  }

  function detectFace(px, w, h) {
    const x1=Math.floor(w*.22), x2=Math.floor(w*.78);
    const y1=Math.floor(h*.10), y2=Math.floor(h*.90);
    let skin=0, total=0;
    for (let y=y1; y<y2; y+=5) for (let x=x1; x<x2; x+=5) {
      const i=(y*w+x)*4; total++;
      if (isSkin(px[i],px[i+1],px[i+2])) skin++;
    }
    return total>0 && skin/total>=SKIN_THRESHOLD;
  }

  // ---- Overlay canvas (injecté dans le DOM parent) ----
  function ensureOverlay() {
    if (pdoc.getElementById('pikolab-overlay')) return;
    overlayCanvas = pdoc.createElement('canvas');
    overlayCanvas.id = 'pikolab-overlay';
    overlayCanvas.style.cssText = 'position:fixed;pointer-events:none;z-index:99999;';
    pdoc.body.appendChild(overlayCanvas);
    overlayCtx = overlayCanvas.getContext('2d');
  }

  function syncOverlay(video) {
    if (!overlayCanvas) return;
    const r = video.getBoundingClientRect();
    const W = r.width|0, H = r.height|0;
    overlayCanvas.style.top  = r.top  + 'px';
    overlayCanvas.style.left = r.left + 'px';
    overlayCanvas.style.width  = W + 'px';
    overlayCanvas.style.height = H + 'px';
    if (overlayCanvas.width !== W || overlayCanvas.height !== H) {
      overlayCanvas.width = W; overlayCanvas.height = H;
    }
  }

  function drawOverlay(faceOk, sheetOk) {
    if (!overlayCtx) return;
    const W = overlayCanvas.width, H = overlayCanvas.height;
    if (!W || !H) return;
    overlayCtx.clearRect(0, 0, W, H);

    const ready = faceOk && (!SHEET_REQUIRED || sheetOk);
    const secsLeft = ready ? Math.ceil((STABLE_NEEDED - stableCount) * 0.3) : 0;

    // --- Ovale visage ---
    const fc = faceOk ? '#4ade80' : '#f87171';
    overlayCtx.save();
    overlayCtx.strokeStyle = fc;
    overlayCtx.lineWidth = 3;
    overlayCtx.setLineDash(faceOk ? [] : [10, 5]);
    overlayCtx.beginPath();
    overlayCtx.ellipse(W*0.5, H*0.44, W*0.22, H*0.37, 0, 0, Math.PI*2);
    overlayCtx.stroke();
    overlayCtx.restore();
    // label visage
    overlayCtx.save();
    overlayCtx.fillStyle = fc;
    overlayCtx.font = 'bold ' + Math.max(12, (H*0.045)|0) + 'px sans-serif';
    overlayCtx.textAlign = 'center';
    overlayCtx.shadowColor = '#000'; overlayCtx.shadowBlur = 4;
    overlayCtx.fillText(faceOk ? '✓ Visage centré' : 'Centrez votre visage ici', W*0.5, H*0.87);
    overlayCtx.restore();

    // --- Indicateur feuille — toujours visible ---
    {
      // Vert si détectée, orange si manquante+requise, gris si optionnelle+absente
      const sc = sheetOk ? '#4ade80' : (SHEET_REQUIRED ? '#f97316' : '#9ca3af');
      const lw = SHEET_REQUIRED ? 3 : 2;
      overlayCtx.save();
      overlayCtx.strokeStyle = sc;
      overlayCtx.lineWidth = lw;
      overlayCtx.setLineDash(sheetOk ? [] : [10, 5]);
      overlayCtx.strokeRect(W*0.65, H*0.18, W*0.30, H*0.54);
      overlayCtx.restore();
      overlayCtx.save();
      overlayCtx.fillStyle = sc;
      overlayCtx.font = 'bold ' + Math.max(11, (H*0.040)|0) + 'px sans-serif';
      overlayCtx.textAlign = 'center';
      overlayCtx.shadowColor = '#000'; overlayCtx.shadowBlur = 4;
      const label = sheetOk ? '✓ Feuille' : (SHEET_REQUIRED ? 'Feuille A4\nrequise ici' : 'Feuille A4\n(optionnelle)');
      overlayCtx.fillText(label, W*0.80, H*0.79);
      overlayCtx.restore();
    }

    // --- Décompte ---
    if (ready && secsLeft > 0) {
      overlayCtx.save();
      overlayCtx.fillStyle = 'rgba(0,0,0,0.38)';
      overlayCtx.fillRect(0, 0, W, H);
      overlayCtx.font = 'bold ' + ((H*0.38)|0) + 'px sans-serif';
      overlayCtx.fillStyle = '#fff';
      overlayCtx.textAlign = 'center';
      overlayCtx.textBaseline = 'middle';
      overlayCtx.shadowColor = '#000'; overlayCtx.shadowBlur = 12;
      overlayCtx.fillText(secsLeft, W*0.5, H*0.5);
      overlayCtx.restore();
    }
  }

  function removeOverlay() {
    const el = pdoc.getElementById('pikolab-overlay');
    if (el) el.remove();
    overlayCanvas = null; overlayCtx = null;
  }

  // ---- Barre de statut ----
  function updateBar(faceOk, sheetOk) {
    const bar = document.getElementById('bar'); if (!bar) return;
    document.getElementById('fs').textContent = faceOk ? '✅ Visage' : '❌ Centrez votre visage';
    const ssEl = document.getElementById('ss');
    ssEl.textContent = sheetOk ? '✅ Feuille détectée' : (SHEET_REQUIRED ? '🟠 Feuille manquante' : '⬜ Feuille non détectée');
    const cd = document.getElementById('cd');
    const ready = faceOk && (!SHEET_REQUIRED || sheetOk);
    if (ready) {
      const s = Math.ceil((STABLE_NEEDED - stableCount) * 0.3);
      cd.textContent = s > 0 ? '📸 ' + s + 's' : '📸';
      bar.style.cssText += 'background:#14532d;border:2px solid #4ade80;';
    } else if (faceOk && SHEET_REQUIRED) {
      cd.textContent = ''; bar.style.cssText += 'background:#78350f;border:2px solid #fbbf24;';
    } else {
      cd.textContent = ''; bar.style.cssText += 'background:#1c1917;border:2px solid #f87171;';
    }
  }

  function tryCapture() {
    removeOverlay();
    const btn = pdoc.querySelector('[data-testid="stCameraInputTakePhoto"]')
             || pdoc.querySelector('[data-testid="stCameraInput"] button');
    if (btn) btn.click();
  }

  function startLoop(video) {
    const offscreen = document.createElement('canvas');
    const octx = offscreen.getContext('2d', {willReadFrequently:true});
    ensureOverlay();

    setInterval(() => {
      if (captured || !video || video.readyState < 2) return;
      offscreen.width  = video.videoWidth  || 640;
      offscreen.height = video.videoHeight || 480;
      octx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
      const {data} = octx.getImageData(0, 0, offscreen.width, offscreen.height);

      const faceOk  = detectFace(data, offscreen.width, offscreen.height);
      const sheetOk = detectSheet(data, offscreen.width, offscreen.height);
      const ready   = faceOk && (!SHEET_REQUIRED || sheetOk);

      if (ready) stableCount++; else stableCount = 0;

      syncOverlay(video);
      drawOverlay(faceOk, sheetOk);
      updateBar(faceOk, sheetOk);

      if (ready && stableCount >= STABLE_NEEDED) {
        captured = true; pwin._pikolabRunning = false; tryCapture();
      }
    }, 300);
  }

  function waitForVideo() {
    const v = pdoc.querySelector('[data-testid="stCameraInput"] video');
    if (v) { startLoop(v); return; }
    const obs = new pwin.MutationObserver(() => {
      const v2 = pdoc.querySelector('[data-testid="stCameraInput"] video');
      if (v2) { obs.disconnect(); setTimeout(() => startLoop(v2), 800); }
    });
    obs.observe(pdoc.body, {childList:true, subtree:true});
  }

  waitForVideo();
})();
</script>
""".replace("SHEET_REQUIRED_PLACEHOLDER", _sheet_js), height=56, scrolling=False)

            camera_photo = st.camera_input("Centrez votre visage · Tenez la feuille A4 à côté")
            if camera_photo:
                image_rgb = load_image(camera_photo)
                if image_rgb is not None:
                    image_rgb = np.fliplr(image_rgb).copy()
                    # Signal de détection feuille immédiat
                    _sheet_check = detect_white_region(image_rgb)
                    if _sheet_check is not None:
                        st.success("✅ Feuille blanche détectée — correction colorimétrique totale activée")
                    else:
                        if use_sheet:
                            st.warning("⚠️ Feuille non détectée sur cette photo. Repositionnez-la bien visible à côté du visage et reprenez.")
                            if st.button("🔄 Reprendre la photo", use_container_width=True):
                                st.rerun()
                        else:
                            st.info("ℹ️ Aucune feuille blanche détectée — correction par type d'éclairage appliquée.")

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

    # ---- Detection feuille blanche — feedback immediat ----
    # Reset override when image changes (rough fingerprint: shape + first pixel)
    _img_sig = f"{image_rgb.shape}_{image_rgb[0, 0].tolist()}"
    if st.session_state.get("_last_img_sig") != _img_sig:
        st.session_state.pop("no_sheet_override", None)
        st.session_state["_last_img_sig"] = _img_sig

    wb_reference = None
    if mode != "Demo":
        wb_reference = detect_white_region(image_rgb)
        if wb_reference is not None:
            has_sheet = True
            a_cast, b_cast = compute_wb_lab_cast(wb_reference)
            st.success(f"✅ Feuille blanche détectée — cast corrigé (a*{a_cast:+.1f}, b*{b_cast:+.1f})")
            # Preview Lab-space : même correction que l'analyse (pas RGB gains)
            _img_lab = rgb2lab(image_rgb / 255.0)
            _img_lab_corr = _img_lab.copy()
            _img_lab_corr[:, :, 1] -= a_cast
            _img_lab_corr[:, :, 2] -= b_cast
            corrected_preview = np.clip(lab2rgb(_img_lab_corr) * 255, 0, 255).astype(np.uint8)
            _pc1, _pc2 = st.columns(2)
            _pc1.image(image_rgb, caption="Photo originale", use_container_width=True)
            _pc2.image(corrected_preview, caption="Après correction couleur", use_container_width=True)
        else:
            has_sheet = False
            st.warning(
                "⚠️ Aucune feuille blanche détectée. "
                "Sans référence neutre, un filtre coloré de votre caméra peut fausser le sous-ton. "
                "Reprenez la photo avec une feuille A4 blanche tenue à côté du visage."
            )
            if not st.session_state.get("no_sheet_override"):
                if st.button("Analyser quand même (résultat moins fiable)", type="secondary"):
                    st.session_state["no_sheet_override"] = True
                    st.rerun()
                return

    # ---- Pipeline with progress ----
    hero_placeholder.empty()
    progress = st.progress(0, text="Detection du visage...")
    landmarks = detect_face(image_rgb)
    if landmarks is None:
        progress.empty()
        st.error("Aucun visage humain detecte. Assurez-vous que la photo montre clairement un visage.")
        return

    face_ok, face_error = validate_face(landmarks, image_rgb.shape)
    if not face_ok:
        progress.empty()
        st.error(face_error)
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

    progress.progress(25, text="Detection cheveux, levres, sourcils, cou...")
    neck_mask = create_neck_mask(image_rgb.shape, landmarks)
    hair_mask = create_hair_mask(image_rgb.shape, landmarks)
    lip_mask = create_lip_mask(image_rgb.shape, landmarks)
    eyebrow_mask = create_eyebrow_mask(image_rgb.shape, landmarks)
    neck_px = int(np.count_nonzero(neck_mask))
    hair_px = int(np.count_nonzero(hair_mask))
    lip_px = int(np.count_nonzero(lip_mask))

    progress.progress(40, text="Correction des couleurs...")

    if wb_reference is not None:
        # Feuille blanche détectée → correction calibrée selon le cast mesuré.
        # La force est adaptée automatiquement : la peau n'est pas un réflecteur neutre
        # et ne nécessite pas 100% de la correction mesurée sur le papier blanc.
        # Calibration : cast ≈ b*+16 (tungstène) → correction réelle ≈ 5.8 → strength ≈ 0.43
        exposure_corrected = correct_exposure(image_rgb, skin_mask)
        corrected = correct_wb_with_reference(image_rgb, wb_reference)
        _wb_a_cast, _wb_b_cast = compute_wb_lab_cast(wb_reference)
        if _wb_b_cast > 10:
            # Cast fort → artificiel jaune/tungstène
            _wb_strength = 0.43
            _wb_warm_offset = 2.5
            _cast_label = "artificiel chaud"
        elif _wb_b_cast > 5:
            # Cast moyen → LED/néon
            _wb_strength = 0.55
            _wb_warm_offset = 0.0
            _cast_label = "artificiel blanc"
        else:
            # Cast faible → lumière naturelle
            _wb_strength = 1.0
            _wb_warm_offset = 0.0
            _cast_label = "naturelle"
        correction_method = f"Feuille blanche ({int(_wb_strength*100)}% — lumière {_cast_label})"
    else:
        # Pas de feuille → correction désactivée, résultat indicatif.
        exposure_corrected = correct_exposure(image_rgb, skin_mask)
        corrected = exposure_corrected
        _wb_a_cast, _wb_b_cast = 0.0, 0.0
        _wb_warm_offset = 0.0
        _wb_strength = 1.0
        correction_method = "Correction exposition uniquement"

    progress.progress(55, text="Analyse colorimetrique peau, cou et iris...")
    # Skin and neck: extracted from exposure-only image, then Lab cast correction applied
    skin_pixels_rgb = extract_pixels(exposure_corrected, skin_mask)
    neck_pixels_rgb = extract_pixels(exposure_corrected, neck_mask)
    has_makeup = st.session_state.get("chk_has_makeup", False)

    # --- Stats robustes séparées visage / cou ---
    # Lab cast correction: linear subtraction on Lab values, then warm residual offset
    face_lab = apply_lab_cast_correction(
        pixels_to_lab(skin_pixels_rgb), _wb_a_cast, _wb_b_cast - _wb_warm_offset, _wb_strength
    )
    face_stats = compute_skin_stats_robust(face_lab)

    _neck_lab_raw = pixels_to_lab(neck_pixels_rgb) if len(neck_pixels_rgb) >= 30 else None
    neck_lab_pixels = apply_lab_cast_correction(
        _neck_lab_raw, _wb_a_cast, _wb_b_cast - _wb_warm_offset, _wb_strength
    ) if _neck_lab_raw is not None else None
    neck_stats_raw = compute_skin_stats_robust(neck_lab_pixels) if neck_lab_pixels is not None else None

    if neck_stats_raw is not None:
        # Overtone = écart b* visage − cou (positif → overtone chaud sur le visage)
        overtone_delta = round(face_stats["b"] - neck_stats_raw["b"], 2)
        # Le cou est la référence principale pour b* (sous-ton pur)
        # Maquillage → cou pèse encore plus (80 %) ; sans maquillage → 65 %
        neck_w = 0.80 if has_makeup else 0.65
        face_w = 1.0 - neck_w
        skin_stats = {
            "L": face_stats["L"] * face_w + neck_stats_raw["L"] * neck_w,
            "a": face_stats["a"] * face_w + neck_stats_raw["a"] * neck_w,
            "b": face_stats["b"] * face_w + neck_stats_raw["b"] * neck_w,
            "C": face_stats["C"] * face_w + neck_stats_raw["C"] * neck_w,
        }
    else:
        overtone_delta = 0.0
        skin_stats = face_stats

    # --- Signature des imperfections (pixels haute chroma du visage) ---
    # Les taches et rougeurs révèlent le type d'overtone → indice de sous-saison
    imperfection_note = None
    if len(face_lab) >= 40:
        C_face = np.sqrt(face_lab[:, 1] ** 2 + face_lab[:, 2] ** 2)
        blemish_px = face_lab[C_face > np.percentile(C_face, 80)]
        if len(blemish_px) >= 10:
            b_blem = float(np.mean(blemish_px[:, 2]))
            a_blem = float(np.mean(blemish_px[:, 1]))
            if a_blem > 8 and b_blem < 10:
                imperfection_note = "rougeurs dominantes → signal teint froid/neutre"
            elif b_blem > 12 and a_blem < 8:
                imperfection_note = "taches jaune-dorées → signal teint chaud"
            elif a_blem > 6 and b_blem > 10:
                imperfection_note = "taches chauds-rosées → signal neutre-chaud"

    no_colored_contacts = st.session_state.get("chk_no_colored_contacts", True)
    iris_pixels_rgb = extract_pixels(corrected, iris_mask)
    iris_stats = extract_iris_dominant(iris_pixels_rgb) if (len(iris_pixels_rgb) > 0 and no_colored_contacts) else None

    progress.progress(70, text="Analyse cheveux et levres...")
    natural_hair = st.session_state.get("chk_natural_hair", True)
    hair_pixels = extract_pixels(corrected, hair_mask)
    hair_lab = pixels_to_lab(hair_pixels)
    hair_stats = compute_skin_stats(hair_lab) if len(hair_lab) > 0 and natural_hair else None
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
    # Mode précise : remplacer skin_stats par la moyenne des 3 photos
    if "_multi_averaged_stats" in st.session_state:
        skin_stats = st.session_state.pop("_multi_averaged_stats")
    scores = compute_scores(skin_stats, iris_stats, params)
    contrast = compute_contrast(skin_stats, iris_stats)
    # Référence b* pour la jauge : cou en priorité (non pollué par overtone)
    ref_b_for_gauge = neck_stats_raw["b"] if neck_stats_raw is not None else face_stats["b"]
    skin_temp_norm = float(np.clip(
        (ref_b_for_gauge - params["temp_center"]) / params["temp_scale"], -1, 1
    ))
    profile = compute_professional_profile(scores, contrast, skin_temp=skin_temp_norm)
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
    progress.empty()

    # ---- Consensus multi-agents ----
    consensus_data = None
    if gemini_api_key and mode != "Demo":
        with st.spinner("Validation multi-agents en cours (3 agents IA)..."):
            try:
                consensus_data = run_consensus_analysis(
                    corrected, season, confidence, gemini_api_key
                )
                # Override si le consensus est majoritaire/unanime ET vote une base différente.
                # On ré-exécute classify_season dans la base majoritaire pour garder
                # les scores pixel intacts → les jauges restent cohérentes avec la saison.
                _c_base  = consensus_data.get("consensus_base", "")
                _c_agree = consensus_data.get("agreement_level", "desaccord")
                _algo_base = season.split()[-1] if season else ""
                if _c_base and _c_base != _algo_base and _c_agree in ("majorite", "unanimite"):
                    season = classify_season_in_base(scores, _c_base)
                    advice = SEASON_ADVICE.get(season, {})
                    profile = compute_professional_profile(scores, contrast, skin_temp=skin_temp_norm)
                    diagnostic = generate_personal_diagnostic(
                        skin_stats, iris_stats, hair_info, lip_undertone,
                        profile, season, advice, contrast,
                    )
                    consensus_data["overridden"] = True
            except Exception as exc:
                st.warning(f"Validation multi-agents non disponible : {exc}")

    st.session_state["analysis_done"] = True
    light_type = "Lumière naturelle (jour)"
    st.session_state["ctx"] = {
        "season": season, "advice": advice, "profile": profile,
        "diagnostic": diagnostic, "hair_info": hair_info,
        "lip_undertone": lip_undertone, "scores": scores, "contrast": contrast,
        "consensus": consensus_data,
        "overtone_delta": overtone_delta,
        "imperfection_note": imperfection_note,
        "face_b": face_stats["b"],
        "neck_b": neck_stats_raw["b"] if neck_stats_raw is not None else None,
        "light_type": light_type,
        "gender": st.session_state.get("chk_gender", "Femme"),
        "skin_stats": {k: round(float(v), 2) for k, v in skin_stats.items()},
        "iris_stats": {k: round(float(v), 2) for k, v in iris_stats.items() if k != "rgb"} if iris_stats else None,
    }

    # ---- Season result card ----
    season_colors = SEASON_PALETTES.get(season, ["#E07A5F", "#F2CC8F"])
    c1 = season_colors[0]
    c2 = season_colors[1] if len(season_colors) > 1 else c1

    has_makeup = st.session_state.get("chk_has_makeup", False)
    _has_sheet = st.session_state.get("has_white_sheet", False)
    _now_month = datetime.now().month
    _is_winter_result = _now_month in (11, 12, 1, 2)
    if mode == "Demo":
        conf_text = "Demonstration"
    elif has_makeup and not _has_sheet:
        conf_text = "Analyse avec maquillage — reprenez sans maquillage ou avec une feuille blanche"
    elif has_makeup:
        conf_text = "Analyse avec maquillage — resultats bases principalement sur le teint du cou"
    elif _has_sheet and confidence >= 0.7:
        conf_text = "Resultat fiable — feuille blanche detectee"
    elif _has_sheet:
        conf_text = "Resultat indicatif — feuille blanche detectee"
    elif _is_winter_result:
        conf_text = "Resultat peu fiable — lumiere hivernale sans feuille blanche. Reprenez avec une feuille A4"
    elif confidence >= 0.7:
        conf_text = "Resultat indicatif — ajoutez une feuille blanche pour plus de precision"
    else:
        conf_text = "Resultat a confirmer — reprenez avec une feuille blanche en lumiere naturelle"

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

    # ---- Disclaimer maquillage ----
    if has_makeup and mode != "Demo":
        st.info(
            "**Analyse effectuee avec maquillage.** "
            "Le fond de teint et le contouring peuvent masquer le sous-ton reel du visage. "
            "Pour compenser, l'analyse s'appuie davantage sur le teint du cou (non maquille). "
            "Pour un resultat optimal, refaites l'analyse sans maquillage."
        )

    # ---- Bloc consensus multi-agents ----
    if consensus_data:
        _render_consensus_block(consensus_data)

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
    if view_mode == "Client":
        tab_labels = ["Profil", "Essayage", "Conseils"]
        tab_labels.append("Photo")
    elif view_mode == "Professionnel":
        tab_labels = ["Profil", "Essayage", "Conseils", "Coaching"]
        tab_labels.append("Photo")
    else:  # Avance
        tab_labels = ["Profil", "Essayage", "Conseils", "Coaching", "Detection", "Debug"]

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # ---- TAB: Profil ----
    with tabs[tab_idx]:
        st.markdown("### Votre profil en 4 dimensions")
        for label, raw, vmin, vmax, left, right in [
            (f"Sous-ton peau : {profile['undertone']}", profile["raw_undertone"], -1, 1, "Froid", "Chaud"),
            (f"Valeur : {profile['depth']}", profile["raw_depth"], -1, 1, "Fonce", "Clair"),
            (f"Chroma : {profile['chroma']}", profile["raw_chroma"], -1, 1, "Doux", "Vif"),
            (f"Contraste peau/yeux : {profile['contrast']}", profile["raw_contrast"], 0, 1, "Bas", "Eleve"),
        ]:
            fig = render_gauge(raw, vmin, vmax, left, right, label)
            st.pyplot(fig)
            plt.close(fig)
        st.caption(
            "Le **sous-ton peau** est mesuré sur le cou (référence non polluée) "
            "quand il est visible. La **valeur** indique la clarté globale — 'Foncé' "
            "correspond aux saisons 'Deep', indépendamment du contraste peau/yeux."
        )

        # --- Analyse overtone visage vs cou ---
        _nb = neck_stats_raw["b"] if neck_stats_raw is not None else None
        if _nb is not None:
            if abs(overtone_delta) >= 2.5:
                if overtone_delta > 0:
                    st.info(
                        f"**Overtone chaud détecté** : le visage est plus jaune/doré "
                        f"que le cou (Δb* = +{overtone_delta:.1f}). Cause possible : exposition solaire, "
                        f"rougeurs, fond de teint chaud. Le sous-ton mesuré s'appuie sur le cou."
                    )
                else:
                    st.info(
                        f"**Overtone froid détecté** : le visage est plus rosé/bleuté "
                        f"que le cou (Δb* = {overtone_delta:.1f}). Cause possible : rougeurs, maquillage "
                        f"froid, lumière. Le sous-ton mesuré s'appuie sur le cou."
                    )
        if imperfection_note:
            st.caption(f"Signature des imperfections : {imperfection_note}")

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
        natural_brows = st.session_state.get("chk_natural_eyebrows", True)
        if natural_brows and eyebrow_info["color"] != "inconnu" and eyebrow_info["color"] != hair_info["color"]:
            st.caption(f"Sourcils : {eyebrow_info['color']} — si differents des cheveux, c'est souvent un indice de votre couleur naturelle.")
        elif not natural_brows:
            st.caption("Sourcils non naturels — non pris en compte dans l'analyse.")
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

        # Export profil pour Scan Couleur
        _ctx = st.session_state.get("ctx", {})
        _skin = _ctx.get("skin_stats")
        if _skin:
            import json as _json
            import base64 as _b64
            import qrcode as _qr
            from io import BytesIO as _BytesIO

            SCAN_COULEUR_URL = "https://scan-couleur.vercel.app"

            _piko_profile = {
                "version": 1,
                "season": _ctx["season"],
                "skin": {"L": _skin["L"], "a": _skin["a"], "b": _skin["b"]},
                "iris": (
                    {"L": _ctx["iris_stats"]["L"], "a": _ctx["iris_stats"]["a"], "b": _ctx["iris_stats"]["b"]}
                    if _ctx.get("iris_stats") else None
                ),
                "hair": _ctx.get("hair_info"),
                "lip": _ctx.get("lip_undertone"),
                "profile": {
                    "undertone": _ctx["profile"]["undertone"],
                    "depth": _ctx["profile"]["depth"],
                    "chroma": _ctx["profile"]["chroma"],
                    "contrast": _ctx["profile"]["contrast"],
                },
            }
            _profile_b64 = _b64.urlsafe_b64encode(
                _json.dumps(_piko_profile, ensure_ascii=False).encode()
            ).decode()
            _deep_link = f"{SCAN_COULEUR_URL}/profil?piko={_profile_b64}"

            st.markdown("#### Utiliser mon profil dans Scan Couleur")
            st.caption("Ouvrez ce lien sur votre téléphone pour importer automatiquement votre profil.")

            col_link, col_qr = st.columns([2, 1])
            with col_link:
                st.link_button(
                    "📲 Ouvrir Scan Couleur avec mon profil",
                    _deep_link,
                    use_container_width=True,
                )
                st.download_button(
                    "💾 Télécharger le profil (.json)",
                    data=_json.dumps(_piko_profile, ensure_ascii=False, indent=2),
                    file_name=f"pikolab_profil_{_ctx['season'].lower().replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with col_qr:
                _qr_img = _qr.make(_deep_link)
                _buf = _BytesIO()
                _qr_img.save(_buf, format="PNG")
                st.image(_buf.getvalue(), caption="Scanner avec votre téléphone", use_container_width=True)

        # CTA Coach IA
        if st.button("💬 Parler a Iris, votre coach styliste", type="primary", use_container_width=True):
            st.switch_page("pages/coach_ia.py")
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
        default_test = good_hex[0] if good_hex else "#E07A5F"
        test_color = st.color_picker("Choisissez une couleur a essayer", default_test, key="drape_picker")
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

        st.markdown("---")
        if st.button("📷 Scanner un vetement en photo", use_container_width=True):
            st.switch_page("pages/scanner.py")
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

    # Coach IA is on a separate page (pages/coach_ia.py)

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
                (skin_mask, [0, 200, 0]),       # green = joues
                (neck_mask, [0, 240, 160]),      # teal = cou
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
            st.caption("Vert=joues, Turquoise=cou, Bleu=iris, Orange=cheveux, Rose=levres, Brun=sourcils")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Joues", f"{skin_px:,} px")
            col2.metric("Cou", f"{neck_px:,} px")
            col3.metric("Iris", f"{iris_px:,} px")
            col4.metric("Cheveux", f"{hair_px:,} px")
            col5.metric("Levres", f"{lip_px:,} px")

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

    # Coach IA is on a separate page (pages/coach_ia.py)


if __name__ == "__main__":
    main()
