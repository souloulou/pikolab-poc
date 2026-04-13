"""
PikoLab — Fond de teint & Maquillage
Trouve le fond de teint le plus adapte a ton teint reel,
et decouvre les blush et rouges a levres les mieux accordes.
"""

import sys
from pathlib import Path
import numpy as np
import streamlit as st
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

# ── Guard : analyse requise ────────────────────────────────────────────────────
ctx = st.session_state.get("ctx")
if not ctx:
    st.info("Analysez d'abord une photo de votre visage pour activer cette page.")
    if st.button("Aller a l'analyse", type="primary", use_container_width=True):
        st.switch_page("app.py")
    st.stop()

season     = ctx["season"]
skin_stats = ctx["skin_stats"]   # L, a, b, C
skin_L     = skin_stats["L"]
skin_a     = skin_stats["a"]
skin_b     = skin_stats["b"]

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


# ── En-tête ────────────────────────────────────────────────────────────────────
st.markdown("## 💄 Fond de teint & Maquillage")
st.caption(f"Profil actif : **{season}**")

skin_hex_approx = "#{:02X}{:02X}{:02X}".format(
    min(255, max(0, int(skin_L * 2.0))),
    min(255, max(0, int(skin_L * 1.8 + skin_a * 0.5))),
    min(255, max(0, int(skin_L * 1.6 - skin_b * 0.8))),
)

col1, col2, col3 = st.columns(3)
col1.metric("Luminosité (L*)", f"{skin_L:.1f}")
col2.metric("Sous-ton rouge/vert (a*)", f"{skin_a:.1f}")
col3.metric("Sous-ton chaud/froid (b*)", f"{skin_b:.1f}")

st.divider()

# ── Section 1 : Fond de teint ──────────────────────────────────────────────────
st.markdown("### 🔍 Trouver mon fond de teint")

skin_lab = np.array([skin_L, skin_a, skin_b])

# Sélection marque
brands = list(FOUNDATIONS_DB.keys())
selected_brand = st.selectbox("Marque", brands)

# Sélection gamme
products = list(FOUNDATIONS_DB[selected_brand].keys())
selected_product = st.selectbox("Gamme", products)

shades = FOUNDATIONS_DB[selected_brand][selected_product]

# Calcul ΔE pour chaque teinte
scored = []
for shade in shades:
    shade_lab = hex_to_lab(shade["hex"])
    de = delta_e(skin_lab, shade_lab)
    scored.append({**shade, "delta_e": de, "shade_lab": shade_lab})

scored.sort(key=lambda x: x["delta_e"])

# Affichage top 5
st.markdown(f"**Top 5 teintes les plus proches — {selected_brand} {selected_product}**")

for i, s in enumerate(scored[:5]):
    emoji, label = delta_e_label(s["delta_e"])
    undertone_fr = {"warm": "Chaud", "cool": "Froid", "neutral": "Neutre"}.get(s["undertone"], s["undertone"])

    with st.container(border=True):
        c1, c2 = st.columns([1, 6])
        with c1:
            st.markdown(
                f"<div style='width:44px;height:44px;background:{s['hex']};"
                f"border-radius:8px;border:1px solid #55555540;margin-top:4px'></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"**{s['ref']}** — {s['name']}  \n"
                f"{emoji} {label}  \n"
                f"Sous-ton : {undertone_fr}"
            )

# Toutes les teintes (expandable)
with st.expander("Voir toutes les teintes classées"):
    for s in scored:
        emoji, label = delta_e_label(s["delta_e"])
        undertone_fr = {"warm": "Chaud", "cool": "Froid", "neutral": "Neutre"}.get(s["undertone"], s["undertone"])
        st.markdown(
            color_swatch(s["hex"]) +
            f"**{s['ref']}** {s['name']} — {emoji} {label} — Sous-ton {undertone_fr}",
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
