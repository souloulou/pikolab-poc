"""
PikoLab — Scanner Couleur
Prenez en photo un vetement ou accessoire pour savoir s'il vous va.
"""

import os
import sys

import cv2
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from skimage.color import rgb2lab

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import (
    SEASON_PALETTES,
    detect_white_region,
    correct_wb_with_reference,
    compute_color_compatibility,
)

st.set_page_config(
    page_title="PikoLab — Scanner",
    page_icon="📷",
    layout="centered",
)

st.markdown(
    "<style>[data-testid='stSidebarNav'] { display: none !important; }</style>",
    unsafe_allow_html=True,
)
st.sidebar.header("PikoLab")
st.sidebar.page_link("app.py", label="Analyse", icon="🎨")
st.sidebar.page_link("pages/scanner.py", label="Scanner", icon="📷")
st.sidebar.page_link("pages/coach_ia.py", label="Coach Iris", icon="💬")

# ---- Check context ----
ctx = st.session_state.get("ctx")
if not ctx:
    st.info("Analysez d'abord une photo de votre visage pour activer le scanner.")
    if st.button("Aller a l'analyse", type="primary", use_container_width=True):
        st.switch_page("app.py")
    st.stop()

season = ctx["season"]
advice = ctx["advice"]
palette = list(SEASON_PALETTES.get(season, []))
palette += advice.get("palette_neutrals", [])

st.markdown(f"### Scanner couleur — {season}")
st.caption("Prenez en photo un vetement, un accessoire ou un tissu pour savoir s'il vous va.")

# ---- Calibration option ----
use_sheet = st.checkbox("J'ai une feuille blanche a cote du vetement", value=False)

# ---- Camera / Upload ----
input_mode = st.radio("Source", ["Appareil photo", "Galerie"], horizontal=True, label_visibility="collapsed")

img_source = None
if input_mode == "Appareil photo":
    img_source = st.camera_input("Photographiez le vetement", key="scanner_cam")
else:
    img_source = st.file_uploader("Photo du vetement", type=["jpg", "jpeg", "png", "webp"], key="scanner_upload")

if img_source is None:
    st.caption("Prenez en photo le vetement dont vous voulez verifier la couleur.")
    st.stop()

# ---- Load and process ----
data = img_source.getvalue()
arr = np.frombuffer(data, np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize if too large
MAX_DIM = 800
h, w = img_rgb.shape[:2]
if max(h, w) > MAX_DIM:
    scale = MAX_DIM / max(h, w)
    img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

# White balance correction if sheet present
if use_sheet:
    wb_ref = detect_white_region(img_rgb)
    if wb_ref is not None:
        img_rgb = correct_wb_with_reference(img_rgb, wb_ref)
        st.success("Calibration feuille blanche appliquee.")
    else:
        st.warning("Feuille blanche non detectee dans cette photo.")

st.image(img_rgb, caption="Photo analysee", use_container_width=True)

# ---- Extract dominant colors ----
# Use center region (avoid background edges)
ch, cw = img_rgb.shape[:2]
margin_h, margin_w = ch // 6, cw // 6
center = img_rgb[margin_h:ch - margin_h, margin_w:cw - margin_w]
pixels = center.reshape(-1, 3)

# KMeans to find top 3 colors
n_colors = min(3, max(1, len(pixels) // 100))
kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
kmeans.fit(pixels)

# Sort by cluster size
labels, counts = np.unique(kmeans.labels_, return_counts=True)
sorted_idx = np.argsort(-counts)

st.markdown("### Couleurs detectees")

results = []
for i, idx in enumerate(sorted_idx):
    rgb = kmeans.cluster_centers_[idx].astype(int)
    hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    pct = counts[idx] / counts.sum() * 100

    score, nearest, suggestion = compute_color_compatibility(hex_color, {season: palette}, season)
    results.append((hex_color, pct, score, nearest, suggestion))

# Display results
for hex_color, pct, score, nearest, suggestion in results:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.color_picker("", hex_color, disabled=True, key=f"sc_{hex_color}", label_visibility="collapsed")
        st.caption(f"{pct:.0f}%")
    with col2:
        if score >= 75:
            st.success(f"**{score}% match** — {suggestion}")
        elif score >= 50:
            st.warning(f"**{score}% match** — {suggestion}")
        else:
            st.error(f"**{score}% match** — {suggestion}")

# Overall verdict
best_score = max(r[2] for r in results) if results else 0
dominant_hex = results[0][0] if results else "#000000"

st.markdown("---")
if best_score >= 75:
    st.markdown(f"""
    <div style="background: #d4edda; border-left: 4px solid #28a745; padding: 1rem; border-radius: 0.5rem;">
        <h3 style="margin:0; color: #28a745;">Ce vetement vous va !</h3>
        <p style="margin:0.5rem 0 0 0; color: #333;">
            La couleur dominante ({dominant_hex}) est compatible avec votre palette {season}.
        </p>
    </div>
    """, unsafe_allow_html=True)
elif best_score >= 50:
    nearest_hex = results[0][3]
    st.markdown(f"""
    <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 0.5rem;">
        <h3 style="margin:0; color: #856404;">Acceptable, mais pas ideal</h3>
        <p style="margin:0.5rem 0 0 0; color: #333;">
            Preferez une nuance plus proche de {nearest_hex} pour un meilleur resultat.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    nearest_hex = results[0][3]
    st.markdown(f"""
    <div style="background: #f8d7da; border-left: 4px solid #dc3545; padding: 1rem; border-radius: 0.5rem;">
        <h3 style="margin:0; color: #dc3545;">Ce vetement ne vous va pas</h3>
        <p style="margin:0.5rem 0 0 0; color: #333;">
            Cette couleur n'est pas dans votre palette. Cherchez plutot du {nearest_hex}.
        </p>
    </div>
    """, unsafe_allow_html=True)
