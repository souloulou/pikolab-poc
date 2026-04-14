"""
PikoLab — Fond de teint & Maquillage
Trouve le fond de teint le plus adapte a ton teint reel,
et decouvre les blush et rouges a levres les mieux accordes.
"""

import io
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from skimage.color import rgb2lab

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from foundations_db import FOUNDATIONS_DB, MAKEUP_RECO

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PikoLab — Fond de teint",
    page_icon="💄",
    layout="centered",
)

st.markdown(
    "<style>[data-testid='stSidebarNav'] { display: none !important; }</style>"
    "<link rel='manifest' href='/app/static/manifest.json'>"
    "<meta name='apple-mobile-web-app-capable' content='yes'>"
    "<meta name='apple-mobile-web-app-title' content='PikoLab'>"
    "<meta name='theme-color' content='#1c1917'>",
    unsafe_allow_html=True,
)

st.sidebar.header("PikoLab")
st.sidebar.page_link("app.py",                    label="Analyse",       icon="🎨")
st.sidebar.page_link("pages/scanner.py",          label="Scanner",       icon="📷")
st.sidebar.page_link("pages/fond_de_teint.py",    label="Fond de teint", icon="💄")
st.sidebar.page_link("pages/coach_ia.py",         label="Coach Iris",    icon="💬")

# ── Guard : analyse requise (ou session sauvegardée) ──────────────────────────
ctx = st.session_state.get("ctx")
if not ctx:
    _save_path = Path(__file__).resolve().parent.parent / "dev_session.json"
    if _save_path.exists():
        try:
            _saved = json.loads(_save_path.read_text(encoding="utf-8"))
            # Réinjecter advice depuis SEASON_ADVICE (non sauvegardé pour alléger le fichier)
            from season_advice import SEASON_ADVICE as _SA
            _saved["advice"] = _SA.get(_saved.get("season"), {})
            st.session_state["ctx"] = _saved
            ctx = _saved
            st.toast(f"Session restaurée — {ctx.get('season', '?')}", icon="✅")
        except Exception as e:
            st.warning(f"Impossible de charger la session sauvegardée : {e}")
    if not ctx:
        st.info("Analysez d'abord une photo de votre visage pour activer cette page.")
        if st.button("Aller a l'analyse", type="primary", use_container_width=True):
            st.switch_page("app.py")
        st.stop()

season     = ctx["season"]
skin_stats = ctx["skin_stats"]   # L, a, b, C, L_foundation
# L_foundation = 70e percentile de L* (tons moyens-clairs, hors ombres)
# Plus représentatif que la moyenne pour le matching fond de teint.
skin_L     = skin_stats.get("L_foundation", skin_stats["L"])
skin_a     = skin_stats["a"]
skin_b     = skin_stats["b"]
_skin_L_raw = skin_stats["L"]  # conservé pour info debug

# ── Helpers ────────────────────────────────────────────────────────────────────

def hex_to_lab(hex_color: str) -> np.ndarray:
    """Convertit un hex RGB en valeurs CIELab."""
    h = hex_color.lstrip("#")
    rgb = np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
    lab = rgb2lab(rgb.reshape(1, 1, 3))
    return lab[0, 0]  # [L, a, b]


def delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Distance colorimetrique euclidienne en Lab (Delta-E 76)."""
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def depth_category(L: float) -> str:
    if L > 72:
        return "very_fair"
    if L > 62:
        return "fair"
    if L > 50:
        return "medium"
    if L > 38:
        return "deep"
    return "very_deep"


def season_family(season_name: str) -> str:
    """Retourne 'warm' ou 'cool' selon la saison."""
    warm_seasons = {
        "True Spring", "Light Spring", "Bright Spring",
        "True Autumn", "Soft Autumn", "Deep Autumn",
    }
    return "warm" if season_name in warm_seasons else "cool"


def makeup_key(L: float, season_name: str) -> str:
    return f"{depth_category(L)}_{season_family(season_name)}"


def color_swatch(hex_color: str, size: int = 28) -> str:
    return (
        f"<span style='display:inline-block;width:{size}px;height:{size}px;"
        f"background:{hex_color};border-radius:50%;border:1px solid #55555540;"
        f"vertical-align:middle;margin-right:6px'></span>"
    )


def delta_e_label(de: float) -> tuple[str, str]:
    """Retourne (emoji, texte) selon le ΔE."""
    if de < 5:
        return "🟢", f"Correspondance excellente (ΔE {de:.1f})"
    if de < 10:
        return "🟡", f"Bonne correspondance (ΔE {de:.1f})"
    if de < 18:
        return "🟠", f"Correspondance acceptable (ΔE {de:.1f})"
    return "🔴", f"Teinte eloignee (ΔE {de:.1f})"


def extract_jaw_neck_L(image_rgb: np.ndarray) -> float | None:
    """
    Extrait le L* de référence fond de teint depuis une photo de profil/45°.

    Cible deux zones complémentaires :
    - Mâchoire (bord inférieur du visage) : souvent maquillée, mais référence de forme
    - Cou sous le menton : jamais maquillé → référence vraie même avec fond de teint

    Sur un profil/45°, le cou n'est plus dans l'ombre de la mâchoire → les deux
    zones sont également exploitables.

    Stratégie :
    1. MediaPipe → masque mâchoire (tiers bas du visage) + cou (jusqu'à 40 % en dessous).
    2. Fallback : filtrage Lab par plage peau sur la moitié basse de l'image.
    Retourne le 70e percentile de L* des pixels peau combinés.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from app import detect_face, JAWLINE_IDX, pixels_to_lab
        landmarks = detect_face(image_rgb)
        if landmarks and len(landmarks) > max(JAWLINE_IDX):
            h, w = image_rgb.shape[:2]
            jaw_pts  = [landmarks[i] for i in JAWLINE_IDX]
            chin_y   = max(p[1] for p in jaw_pts)
            face_top = min(p[1] for p in jaw_pts)
            face_h   = max(1, chin_y - face_top)
            left_x   = min(p[0] for p in jaw_pts) + int(w * 0.04)
            right_x  = max(p[0] for p in jaw_pts) - int(w * 0.04)

            # Zone 1 : mâchoire — tiers inférieur du visage jusqu'au menton
            jaw_top = int(chin_y - face_h * 0.35)
            # Zone 2 : cou — jusqu'à 40 % de la hauteur du visage sous le menton
            neck_bot = min(h - 5, int(chin_y + face_h * 0.40))

            mask = np.zeros((h, w), dtype=np.uint8)
            # Mâchoire
            cv2.rectangle(mask, (left_x, jaw_top), (right_x, chin_y), 255, -1)
            # Cou (légèrement rétréci sur les côtés pour éviter les cheveux/vêtements)
            neck_inset = int((right_x - left_x) * 0.08)
            cv2.rectangle(
                mask,
                (left_x + neck_inset, chin_y),
                (right_x - neck_inset, neck_bot),
                255, -1,
            )

            combined_pixels = image_rgb[mask > 0].reshape(-1, 3)
            if len(combined_pixels) >= 80:
                lab = pixels_to_lab(combined_pixels)
                C = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
                clean = lab[C <= np.percentile(C, 75)]
                if len(clean) >= 30:
                    return float(np.percentile(clean[:, 0], 70))
    except Exception:
        pass  # MediaPipe indisponible ou profil trop marqué → fallback

    # Fallback : filtrage Lab par plage peau sur la moitié basse de l'image
    # (mâchoire + cou sont dans la moitié inférieure sur un selfie/portrait)
    h, w = image_rgb.shape[:2]
    region = image_rgb[int(h * 0.40):int(h * 0.90), int(w * 0.10):int(w * 0.90)]
    lab_all = rgb2lab(region.astype(np.float64) / 255.0).reshape(-1, 3)
    skin_mask = (
        (lab_all[:, 0] >= 28) & (lab_all[:, 0] <= 92) &
        (lab_all[:, 1] >= 4)  & (lab_all[:, 1] <= 22) &
        (lab_all[:, 2] >= 4)  & (lab_all[:, 2] <= 32)
    )
    skin_px = lab_all[skin_mask]
    if len(skin_px) < 50:
        return None
    return float(np.percentile(skin_px[:, 0], 70))


# ── Photo mâchoire (optionnelle) ───────────────────────────────────────────────

def _jaw_photo_section() -> float | None:
    """
    Section UI pour la photo de profil/45°.
    Retourne le L* mesuré si une photo est fournie et traitée avec succès, sinon None.
    """
    with st.expander("📸 Photo mâchoire — référence fond de teint (optionnel)", expanded=False):
        st.markdown(
            "Pour un matching plus précis, prenez une photo **de profil ou à 45°** "
            "qui expose votre mâchoire et votre cou **sans ombre portée**.\n\n"
            "**Comment faire :** tournez la tête d'environ 45°, éclairage frontal, "
            "mâchoire bien visible. Pas besoin de feuille blanche."
        )
        jaw_img_file = st.camera_input(
            "Photo mâchoire / cou",
            key="jaw_photo",
            help="Profil ou 45° — mâchoire et cou visibles, bien éclairés",
        )
        if jaw_img_file is None:
            jaw_img_file = st.file_uploader(
                "Ou importer une photo",
                type=["jpg", "jpeg", "png"],
                key="jaw_photo_upload",
            )
        if jaw_img_file is not None:
            pil_img = Image.open(io.BytesIO(jaw_img_file.read())).convert("RGB")
            img_rgb = np.array(pil_img)
            with st.spinner("Analyse de la mâchoire..."):
                jaw_L = extract_jaw_neck_L(img_rgb)
            if jaw_L is not None:
                st.success(f"Clarté mâchoire + cou mesurée : **L* {jaw_L:.1f}** (utilisée pour le matching)")
                # Aperçu miniature
                st.image(pil_img, width=220, caption="Photo analysée")
                return jaw_L
            else:
                st.warning(
                    "Pas assez de pixels peau détectés. "
                    "Vérifiez que la mâchoire est bien visible et éclairée."
                )
    return None


# ── En-tête ────────────────────────────────────────────────────────────────────
st.markdown("## 💄 Fond de teint & Maquillage")
st.caption(f"Profil actif : **{season}**")

# Photo mâchoire optionnelle — surcharge skin_L si mesurée
_jaw_L = _jaw_photo_section()
if _jaw_L is not None:
    skin_L = _jaw_L
    _source_label = f"mâchoire + cou (photo) L* {skin_L:.1f}"
else:
    _source_label = f"visage frontal L* {skin_L:.1f}"

skin_hex_approx = "#{:02X}{:02X}{:02X}".format(
    min(255, max(0, int(skin_L * 2.0))),
    min(255, max(0, int(skin_L * 1.8 + skin_a * 0.5))),
    min(255, max(0, int(skin_L * 1.6 - skin_b * 0.8))),
)

col1, col2, col3 = st.columns(3)
col1.metric(
    "Clarté teint (L*)",
    f"{skin_L:.1f}",
    delta=_source_label if _jaw_L is not None else (
        f"moy. brute {_skin_L_raw:.1f}" if abs(skin_L - _skin_L_raw) > 1 else None
    ),
    delta_color="off",
    help="Source : photo mâchoire si fournie, sinon 70e percentile L* visage frontal.",
)
col2.metric("Sous-ton rouge/vert (a*)", f"{skin_a:.1f}")
col3.metric("Sous-ton chaud/froid (b*)", f"{skin_b:.1f}")

st.divider()

# ── Section 1 : Fond de teint ──────────────────────────────────────────────────
st.markdown("### 🔍 Trouver mon fond de teint")

skin_lab = np.array([skin_L, skin_a, skin_b])

# Filtrage dur par sous-ton : une erreur de sous-ton est pire qu'une erreur de profondeur.
# Saison chaude (Spring/Autumn) → warm + neutral uniquement.
# Saison froide (Summer/Winter) → cool + neutral uniquement.
_family = season_family(season)
_allowed_undertones = {"warm", "neutral"} if _family == "warm" else {"cool", "neutral"}

# Sélection marque
brands = list(FOUNDATIONS_DB.keys())
selected_brand = st.selectbox("Marque", brands)

# Sélection gamme
products = list(FOUNDATIONS_DB[selected_brand].keys())
selected_product = st.selectbox("Gamme", products)

shades = FOUNDATIONS_DB[selected_brand][selected_product]

# Filtrage + calcul ΔE
compatible = [s for s in shades if s.get("undertone", "neutral") in _allowed_undertones]
if not compatible:
    compatible = shades  # fallback si la gamme n'a aucune teinte compatible

scored = sorted(
    [{**s, "delta_e": delta_e(skin_lab, hex_to_lab(s["hex"]))} for s in compatible],
    key=lambda x: x["delta_e"],
)

# ── Recommandations (1 ou 2 selon cohérence) ──────────────────────────────────
st.markdown(f"**Recommandations — {selected_brand} {selected_product}**")

_undertone_fr = {"warm": "Chaud", "cool": "Froid", "neutral": "Neutre"}
_rank_labels   = {1: "Meilleure correspondance", 2: "Alternative proche"}
_bg_colors     = {1: "#1a2a1a", 2: "#1e1e1e"}
_border_colors = {1: "#5a8a5a50", 2: "#33333380"}

# La 2ème option n'est affichée que si elle reste dans un ΔE acceptable (≤ 10)
# et proche de la 1ère (écart ≤ 3) — sinon elle serait visible sur la peau.
_to_show = [scored[0]]
if (
    len(scored) >= 2
    and scored[1]["delta_e"] <= 10
    and scored[1]["delta_e"] - scored[0]["delta_e"] <= 3
):
    _to_show.append(scored[1])

for rank, candidate in enumerate(_to_show, start=1):
    emoji_c, label_c = delta_e_label(candidate["delta_e"])
    undertone_fr_c = _undertone_fr.get(candidate["undertone"], candidate["undertone"])
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:16px;padding:14px;"
        f"background:{_bg_colors[rank]};border-radius:10px;"
        f"border:1px solid {_border_colors[rank]};margin-bottom:10px'>"
        f"<div style='width:52px;height:52px;background:{candidate['hex']};"
        f"border-radius:10px;border:1px solid #55555540;flex-shrink:0'></div>"
        f"<div style='flex:1'>"
        f"<div style='font-size:0.75rem;color:#888;margin-bottom:2px'>{_rank_labels[rank]}</div>"
        f"<div style='font-size:1.05rem;font-weight:700'>{candidate['ref']} — {candidate['name']}</div>"
        f"<div style='margin-top:3px'>{emoji_c} {label_c}</div>"
        f"<div style='color:#aaa;font-size:0.85rem;margin-top:2px'>Sous-ton : {undertone_fr_c}</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

# Teintes exclues (mauvais sous-ton) dans un expander discret
excluded = [s for s in shades if s.get("undertone", "neutral") not in _allowed_undertones]
with st.expander(f"Teintes exclues — sous-ton incompatible ({len(excluded)} teintes)"):
    st.caption(
        f"Votre saison **{season}** est {'chaude' if _family == 'warm' else 'froide'} — "
        f"les teintes {'froides' if _family == 'warm' else 'chaudes'} ont été écartées."
    )
    for s in excluded:
        undertone_fr = {"warm": "Chaud", "cool": "Froid", "neutral": "Neutre"}.get(s["undertone"], s["undertone"])
        st.markdown(
            color_swatch(s["hex"]) + f"~~{s['ref']} {s['name']}~~ — Sous-ton {undertone_fr}",
            unsafe_allow_html=True,
        )

st.divider()

# ── Saisie manuelle ────────────────────────────────────────────────────────────
with st.expander("Ma référence n'est pas dans la liste — Saisie manuelle"):
    st.caption(
        "Trouvez le code couleur hex de votre fond de teint sur le site de la marque "
        "ou avec une appli pipette. Entrez-le ci-dessous."
    )
    col_a, col_b = st.columns(2)
    manual_brand = col_a.text_input("Marque", placeholder="ex: Fenty Beauty")
    manual_ref   = col_b.text_input("Référence", placeholder="ex: 240W")
    manual_name  = st.text_input("Nom de la teinte", placeholder="ex: Warm Vanilla")
    manual_hex   = st.color_picker("Couleur de la teinte", value="#DEB887")

    if st.button("Calculer la correspondance", type="primary"):
        manual_lab = hex_to_lab(manual_hex)
        de_manual  = delta_e(skin_lab, manual_lab)
        emoji_m, label_m = delta_e_label(de_manual)

        st.markdown(
            f"<div style='display:flex;align-items:center;gap:12px;padding:12px;"
            f"background:#1e1e1e;border-radius:8px;margin-top:8px'>"
            f"<div style='width:48px;height:48px;background:{manual_hex};"
            f"border-radius:8px;flex-shrink:0'></div>"
            f"<div><strong>{manual_brand} {manual_ref}</strong> — {manual_name}<br>"
            f"{emoji_m} {label_m}</div></div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Section 2 : Blush & Rouge à lèvres ────────────────────────────────────────
st.markdown("### 🌸 Blush & Rouge à lèvres recommandés")
st.caption(
    f"Sélections basées sur votre profondeur de teint (L*={skin_L:.0f}) "
    f"et votre saison **{season}**."
)

key = makeup_key(skin_L, season)

depth_labels = {
    "very_fair": "Très clair (L* > 72)",
    "fair":      "Clair (L* 62–72)",
    "medium":    "Moyen (L* 50–62)",
    "deep":      "Foncé (L* 38–50)",
    "very_deep": "Très foncé (L* < 38)",
}
family_labels = {"warm": "Chaud (Printemps / Automne)", "cool": "Froid (Été / Hiver)"}
depth_key  = depth_category(skin_L)
family_key = season_family(season)

st.info(
    f"Profil : **{depth_labels[depth_key]}** — **{family_labels[family_key]}**"
)

tab_blush, tab_lip = st.tabs(["Blush", "Rouge à lèvres"])

with tab_blush:
    blush_recos = MAKEUP_RECO["blush"].get(key, [])
    if blush_recos:
        for r in blush_recos:
            with st.container(border=True):
                c1, c2 = st.columns([1, 6])
                with c1:
                    st.markdown(
                        f"<div style='width:44px;height:44px;background:{r['hex']};"
                        f"border-radius:50%;border:1px solid #55555540;margin-top:4px'></div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"**{r['brand']}** — {r['product']}  \n"
                        f"Teinte : *{r['shade']}*  \n"
                        f"{r['desc']}"
                    )
    else:
        st.info("Aucune recommandation disponible pour ce profil pour le moment.")

with tab_lip:
    lip_recos = MAKEUP_RECO["lipstick"].get(key, [])
    if lip_recos:
        for r in lip_recos:
            with st.container(border=True):
                c1, c2 = st.columns([1, 6])
                with c1:
                    st.markdown(
                        f"<div style='width:44px;height:44px;background:{r['hex']};"
                        f"border-radius:8px;border:1px solid #55555540;margin-top:4px'></div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"**{r['brand']}** — {r['product']}  \n"
                        f"Teinte : *{r['shade']}*  \n"
                        f"{r['desc']}"
                    )
    else:
        st.info("Aucune recommandation disponible pour ce profil pour le moment.")

st.divider()
st.caption(
    "Les correspondances sont calculees par distance colorimetrique (Delta-E) "
    "entre votre teint reel et la couleur de chaque teinte. "
    "Un Delta-E < 5 est imperceptible a l'oeil nu, < 10 est tres bon."
)
