"""
PikoLab — Quiz colorimetrique
Questionnaire subjectif (7 questions) pour pre-classifier la saison
avant l'analyse photo.
"""
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PikoLab — Quiz",
    page_icon="🎨",
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
st.sidebar.page_link("pages/quiz.py",             label="Quiz",          icon="📋")
st.sidebar.page_link("pages/scanner.py",          label="Scanner",       icon="📷")
st.sidebar.page_link("pages/fond_de_teint.py",    label="Fond de teint", icon="💄")
st.sidebar.page_link("pages/coach_ia.py",         label="Coach Iris",    icon="💬")

# ── Données du quiz ────────────────────────────────────────────────────────────

QUESTIONS = [
    {
        "id": "skin",
        "step": 1,
        "title": "Votre teint naturel",
        "subtitle": "Sans maquillage, quelle description se rapproche le plus de la couleur de votre peau ?",
        "weight": {"warmth": 2.5, "depth": 3.0, "chroma": 0.5},
        "options": [
            {"id": "skin-fair-cool",   "label": "Très clair rosé",     "sublabel": "Teint porcelaine, rose ou bleuté",           "hex": "#F4DDD8", "scores": {"warmth": -2,   "depth": 1,   "chroma": 1.5}},
            {"id": "skin-fair-warm",   "label": "Très clair pêche",    "sublabel": "Teint ivoire, pêche ou légèrement doré",      "hex": "#F5D5B8", "scores": {"warmth": 2,    "depth": 1,   "chroma": 1.5}},
            {"id": "skin-light-cool",  "label": "Clair rosé",          "sublabel": "Teint beige rosé ou beige neutre-froid",      "hex": "#DEC0B5", "scores": {"warmth": -1,   "depth": 2,   "chroma": 1.0}},
            {"id": "skin-light-warm",  "label": "Clair doré",          "sublabel": "Teint beige doré, miel ou abricot",           "hex": "#D8BC8C", "scores": {"warmth": 1.5,  "depth": 2,   "chroma": 1.5}},
            {"id": "skin-medium-cool", "label": "Moyen rosé-brun",     "sublabel": "Teint beige foncé, brun rosé ou cendré",      "hex": "#B89080", "scores": {"warmth": -1,   "depth": 3,   "chroma": 1.0}},
            {"id": "skin-medium-warm", "label": "Moyen doré-olive",    "sublabel": "Teint olivâtre, caramel ou doré moyen",       "hex": "#B89058", "scores": {"warmth": 2,    "depth": 3,   "chroma": 1.5}},
            {"id": "skin-deep-warm",   "label": "Foncé chaud",         "sublabel": "Teint brun, caramel foncé ou ébène chaud",    "hex": "#7A5030", "scores": {"warmth": 2,    "depth": 4.5, "chroma": 1.5}},
            {"id": "skin-deep-cool",   "label": "Foncé froid",         "sublabel": "Teint brun foncé, ébène ou brun violacé",     "hex": "#604040", "scores": {"warmth": -1.5, "depth": 4.5, "chroma": 1.0}},
        ],
    },
    {
        "id": "eyes",
        "step": 2,
        "title": "Vos yeux naturels",
        "subtitle": "Quelle couleur se rapproche le plus de votre iris (sans lentilles) ?",
        "weight": {"warmth": 2.0, "depth": 1.5, "chroma": 1.5},
        "options": [
            {"id": "eyes-dark-brown",  "label": "Brun très foncé",    "sublabel": "Brun-noir, presque noir",                    "hex": "#2A1000", "scores": {"warmth": 1,    "depth": 5,   "chroma": 1.0}},
            {"id": "eyes-medium-brown","label": "Brun moyen",         "sublabel": "Brun noisette, brun chaud",                  "hex": "#6B3410", "scores": {"warmth": 2,    "depth": 3.5, "chroma": 2.0}},
            {"id": "eyes-hazel",       "label": "Noisette",           "sublabel": "Mélange brun-vert-or",                       "hex": "#7A6020", "scores": {"warmth": 1.5,  "depth": 3,   "chroma": 2.5}},
            {"id": "eyes-green",       "label": "Vert",               "sublabel": "Vert herbe, vert émeraude, vert foncé",      "hex": "#386020", "scores": {"warmth": 0.5,  "depth": 2.5, "chroma": 3.0}},
            {"id": "eyes-blue-green",  "label": "Bleu-vert",          "sublabel": "Turquoise, aqua, bleu-vert",                 "hex": "#207060", "scores": {"warmth": -0.5, "depth": 2,   "chroma": 3.5}},
            {"id": "eyes-blue",        "label": "Bleu",               "sublabel": "Bleu moyen, bleu ciel, bleu acier",          "hex": "#385888", "scores": {"warmth": -1.5, "depth": 2.5, "chroma": 2.5}},
            {"id": "eyes-gray-blue",   "label": "Gris-bleu",          "sublabel": "Gris ardoise, bleu-gris, gris acier",        "hex": "#586878", "scores": {"warmth": -2,   "depth": 2,   "chroma": 1.5}},
            {"id": "eyes-light-gray",  "label": "Gris clair",         "sublabel": "Gris perle, bleu glacé, gris argenté",       "hex": "#90A0B0", "scores": {"warmth": -2,   "depth": 1,   "chroma": 1.0}},
        ],
    },
    {
        "id": "veins",
        "step": 3,
        "title": "Vos veines au poignet",
        "subtitle": "Regardez l'intérieur de votre poignet en pleine lumière. Quelle couleur décrit le mieux vos veines ?",
        "weight": {"warmth": 3.0, "depth": 0, "chroma": 0},
        "options": [
            {"id": "veins-blue-purple","label": "Bleu ou violacé",    "sublabel": "Nettement bleues ou avec un reflet violet",   "hex": "#5A4A8A", "scores": {"warmth": -2.5, "depth": 0, "chroma": 0}},
            {"id": "veins-blue",       "label": "Bleu-vert",          "sublabel": "Ni vraiment bleues ni vraiment vertes",       "hex": "#3A6A5A", "scores": {"warmth": 0,    "depth": 0, "chroma": 0}},
            {"id": "veins-green",      "label": "Vert",               "sublabel": "Nettement verdâtres ou avec un reflet olive", "hex": "#3A6828", "scores": {"warmth": 2.5,  "depth": 0, "chroma": 0}},
            {"id": "veins-unclear",    "label": "Difficile à dire",   "sublabel": "Très peu visibles ou couleur indistincte",    "hex": "#A89888", "scores": {"warmth": 0,    "depth": 0, "chroma": 0}},
        ],
    },
    {
        "id": "hair",
        "step": 4,
        "title": "Vos cheveux naturels",
        "subtitle": "Quelle couleur se rapproche le plus de vos cheveux naturels (racines, sans coloration) ?",
        "weight": {"warmth": 1.5, "depth": 2.0, "chroma": 0.5},
        "options": [
            {"id": "hair-platinum",    "label": "Blanc / Platine",    "sublabel": "Blond très clair, platine, blanc naturel",    "hex": "#EDE5C8", "scores": {"warmth": -0.5, "depth": 1,   "chroma": 1.0}},
            {"id": "hair-golden-blonde","label": "Blond doré",        "sublabel": "Blond chaud, doré, miel ou blond doré foncé", "hex": "#C09030", "scores": {"warmth": 2,    "depth": 1.5, "chroma": 2.5}},
            {"id": "hair-ash-blonde",  "label": "Blond cendré",       "sublabel": "Blond sans reflets chauds, blond froid",      "hex": "#B8B098", "scores": {"warmth": -1,   "depth": 1.5, "chroma": 1.0}},
            {"id": "hair-red",         "label": "Roux / Auburn",      "sublabel": "Roux vif, cuivré, auburn, châtain roux",      "hex": "#A03010", "scores": {"warmth": 2.5,  "depth": 2.5, "chroma": 3.5}},
            {"id": "hair-light-brown", "label": "Châtain clair",      "sublabel": "Brun moyen-clair, châtain doré",              "hex": "#7A5030", "scores": {"warmth": 1,    "depth": 3,   "chroma": 1.5}},
            {"id": "hair-dark-brown",  "label": "Brun foncé",         "sublabel": "Châtain foncé, brun naturel, brun-noir",      "hex": "#382010", "scores": {"warmth": 0.5,  "depth": 4,   "chroma": 1.0}},
            {"id": "hair-black-warm",  "label": "Noir chaud",         "sublabel": "Noir avec reflets bruns ou rougeâtres",       "hex": "#181000", "scores": {"warmth": 1,    "depth": 5,   "chroma": 1.0}},
            {"id": "hair-black-cool",  "label": "Noir froid",         "sublabel": "Noir bleuté, noir de jais, sans reflet",      "hex": "#081018", "scores": {"warmth": -1,   "depth": 5,   "chroma": 1.0}},
        ],
    },
    {
        "id": "undereye",
        "step": 5,
        "title": "Vos cernes",
        "subtitle": "Quelle couleur décrit le mieux la teinte de vos cernes ? (sans maquillage, en pleine lumière)",
        "multi": True,
        "max_select": 2,
        "weight": {"warmth": 2.0, "depth": 0.5, "chroma": 0},
        "options": [
            {"id": "undereye-blue",    "label": "Bleutés",            "sublabel": "Cernes nettement bleus ou violacés",          "hex": "#7878A8", "scores": {"warmth": -2.5, "depth": 2, "chroma": 0}},
            {"id": "undereye-purple",  "label": "Violacés / Mauves",  "sublabel": "Cernes roses-violets ou lavande",             "hex": "#9070A8", "scores": {"warmth": -1.5, "depth": 2, "chroma": 0}},
            {"id": "undereye-brown",   "label": "Brun / Caramel",     "sublabel": "Cernes beige-brun, marron ou miel",           "hex": "#A08060", "scores": {"warmth": 2,    "depth": 3, "chroma": 0}},
            {"id": "undereye-dark",    "label": "Sombres / Grisés",   "sublabel": "Cernes foncés, gris ou presque noirs",        "hex": "#485060", "scores": {"warmth": -1,   "depth": 4, "chroma": 0}},
            {"id": "undereye-none",    "label": "Peu visibles",       "sublabel": "Quasiment pas de cernes ou indéterminés",     "hex": "#D8C8B8", "scores": {"warmth": 0,    "depth": 0, "chroma": 0}},
        ],
    },
    {
        "id": "blemishes",
        "step": 6,
        "title": "Les marques laissées par vos imperfections",
        "subtitle": "Après qu'un bouton ou une rougeur guérit, quelle trace reste-t-il sur votre peau ?",
        "multi": True,
        "max_select": 2,
        "weight": {"warmth": 1.5, "depth": 0, "chroma": 0.5},
        "options": [
            {"id": "blemish-purple",   "label": "Une tache violacée ou mauve", "sublabel": "La marque vire au violet, mauve ou bordeaux froid",      "hex": "#907090", "scores": {"warmth": -0.5, "depth": 3.5, "chroma": 1.5}},
            {"id": "blemish-pink-red", "label": "Une tache rose ou rosée",    "sublabel": "La marque reste rose ou rouge rosé",                       "hex": "#E07080", "scores": {"warmth": -1.5, "depth": 2.5, "chroma": 2.0}},
            {"id": "blemish-red",      "label": "Une tache rouge ou orangée", "sublabel": "La marque reste rouge vif ou rouge-orangé",               "hex": "#D04030", "scores": {"warmth": 1,    "depth": 2.5, "chroma": 2.5}},
            {"id": "blemish-brown",    "label": "Une tache brune ou miel",    "sublabel": "La marque vire au brun, caramel ou hyperpigmentation chaude","hex": "#A06040", "scores": {"warmth": 2,    "depth": 3.5, "chroma": 1.0}},
            {"id": "blemish-rare",     "label": "Aucune trace visible",       "sublabel": "La peau cicatrise sans laisser de marque colorée",         "hex": "#D0B8A8", "scores": {"warmth": 0,    "depth": 0,   "chroma": 0.0}},
        ],
    },
]

# 16 saisons exactes (noms anglais = noms de app.py) avec labels français et scores
# Scores dérivés des SEASON_CENTROIDS de app.py :
#   warmth = temp * 3  |  depth = 3 - val*2  |  chroma = 3 + sat*2
SEASONS = {
    "Light Spring":  {"label": "Printemps Clair",   "warmth":  1.05, "depth": 1.56, "chroma": 3.44, "emoji": "🌤"},
    "Warm Spring":   {"label": "Printemps Chaud",   "warmth":  2.16, "depth": 2.36, "chroma": 3.44, "emoji": "🌸"},
    "Bright Spring": {"label": "Printemps Lumineux","warmth":  0.96, "depth": 2.24, "chroma": 4.56, "emoji": "✨"},
    "True Spring":   {"label": "Printemps Véritable","warmth":  1.05, "depth": 1.96, "chroma": 3.76, "emoji": "🌼"},
    "Light Summer":  {"label": "Été Clair",         "warmth": -1.14, "depth": 1.50, "chroma": 2.44, "emoji": "☁️"},
    "Cool Summer":   {"label": "Été Froid",         "warmth": -2.25, "depth": 2.56, "chroma": 2.50, "emoji": "🌊"},
    "Soft Summer":   {"label": "Été Doux",          "warmth": -0.90, "depth": 2.60, "chroma": 1.64, "emoji": "🌸"},
    "True Summer":   {"label": "Été Véritable",      "warmth": -1.50, "depth": 2.10, "chroma": 2.36, "emoji": "🏖"},
    "Soft Autumn":   {"label": "Automne Doux",      "warmth":  0.60, "depth": 3.44, "chroma": 1.64, "emoji": "🌾"},
    "Warm Autumn":   {"label": "Automne Chaud",     "warmth":  2.16, "depth": 3.50, "chroma": 3.36, "emoji": "🍂"},
    "Deep Autumn":   {"label": "Automne Profond",   "warmth":  1.65, "depth": 4.44, "chroma": 3.10, "emoji": "🍁"},
    "True Autumn":   {"label": "Automne Véritable",  "warmth":  1.05, "depth": 3.90, "chroma": 2.76, "emoji": "🌰"},
    "Deep Winter":   {"label": "Hiver Profond",     "warmth": -0.96, "depth": 4.50, "chroma": 3.44, "emoji": "🌑"},
    "Cool Winter":   {"label": "Hiver Froid",       "warmth": -2.25, "depth": 3.44, "chroma": 3.36, "emoji": "❄️"},
    "Bright Winter": {"label": "Hiver Brillant",    "warmth": -0.84, "depth": 3.50, "chroma": 4.56, "emoji": "💎"},
    "True Winter":   {"label": "Hiver Véritable",   "warmth": -1.56, "depth": 4.00, "chroma": 3.50, "emoji": "🌨"},
}

# ── Logique de scoring ─────────────────────────────────────────────────────────

def compute_scores(answers: dict) -> dict:
    total = {"warmth": 0, "depth": 0, "chroma": 0}
    weights = {"warmth": 0, "depth": 0, "chroma": 0}

    for q in QUESTIONS:
        qid = q["id"]
        answer_id = answers.get(qid)
        if not answer_id:
            continue
        w = q["weight"]
        # Multi-select : moyenne des scores des options choisies
        if isinstance(answer_id, list):
            if not answer_id:
                continue
            selected_opts = [o for o in q.get("options", []) if o["id"] in answer_id]
            if not selected_opts:
                continue
            s = {dim: sum(o["scores"][dim] for o in selected_opts) / len(selected_opts)
                 for dim in ("warmth", "depth", "chroma")}
        else:
            opt = next((o for o in q.get("options", []) if o["id"] == answer_id), None)
            if not opt:
                continue
            s = opt["scores"]
        for dim in ("warmth", "depth", "chroma"):
            total[dim] += s[dim] * w[dim]
            weights[dim] += w[dim]

    return {
        dim: (total[dim] / weights[dim]) if weights[dim] > 0 else 0
        for dim in ("warmth", "depth", "chroma")
    }


def season_distance(scores: dict, season_data: dict) -> float:
    return (
        ((scores["warmth"] - season_data["warmth"]) * 1.5) ** 2
        + (scores["depth"]  - season_data["depth"])  ** 2
        + (scores["chroma"] - season_data["chroma"]) ** 2
    )


def determine_season(answers: dict) -> dict:
    scores = compute_scores(answers)
    ranked = sorted(
        SEASONS.items(),
        key=lambda item: season_distance(scores, item[1])
    )
    best_id,   best_s   = ranked[0]
    second_id, second_s = ranked[1]

    gap        = season_distance(scores, second_s) - season_distance(scores, best_s)
    confidence = min(100, round(50 + gap * 15))

    return {
        "season":        best_id,                          # nom anglais = clé app.py
        "season_label":  best_s["label"],                  # nom français pour affichage
        "emoji":         best_s["emoji"],
        "confidence":    confidence,
        "runner_up":     second_id        if confidence < 75 else None,
        "runner_up_label": second_s["label"] if confidence < 75 else None,
        "scores":        scores,
    }

# ── UI helpers ─────────────────────────────────────────────────────────────────

def swatch_button(option: dict, selected: bool, key: str) -> bool:
    border = "3px solid #E8C4A0" if selected else "2px solid #33333360"
    bg     = "#2a2a2a" if selected else "#1a1a1a"
    check  = "✓ " if selected else ""

    clicked = st.button(
        f"{check}{option['label']}",
        key=key,
        use_container_width=True,
        type="primary" if selected else "secondary",
    )
    st.markdown(
        f"<div style='width:100%;height:10px;background:{option['hex']};"
        f"border-radius:4px;margin-top:-8px;margin-bottom:4px'></div>"
        f"<p style='font-size:11px;color:#aaa;margin:-4px 0 8px 0'>{option['sublabel']}</p>",
        unsafe_allow_html=True,
    )
    return clicked

# ── État du quiz ───────────────────────────────────────────────────────────────

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_step" not in st.session_state:
    st.session_state.quiz_step = 0
if "quiz_done" not in st.session_state:
    st.session_state.quiz_done = False

# ── Résultat final ─────────────────────────────────────────────────────────────

if st.session_state.quiz_done:
    result = determine_season(st.session_state.quiz_answers)
    st.session_state["quiz_result"] = result

    emoji        = result["emoji"]
    season_label = result["season_label"]
    conf         = result["confidence"]

    st.markdown(f"## {emoji} Résultat du quiz")
    st.markdown(f"### {season_label}")

    conf_color = "#4CAF50" if conf >= 75 else "#FF9800" if conf >= 55 else "#F44336"
    st.markdown(
        f"<div style='background:#1e1e1e;padding:16px;border-radius:10px;margin-bottom:16px'>"
        f"<span style='font-size:18px'>Confiance : </span>"
        f"<span style='font-size:22px;font-weight:bold;color:{conf_color}'>{conf}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if result.get("runner_up_label"):
        st.info(f"Résultat proche : **{result['runner_up_label']}** — l'analyse photo affinera le résultat.")

    scores = result["scores"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Chaleur",    f"{scores['warmth']:+.1f}", help="-3 (froid) → +3 (chaud)")
    col2.metric("Profondeur", f"{scores['depth']:.1f}",  help="1 (très clair) → 5 (très foncé)")
    col3.metric("Chroma",     f"{scores['chroma']:.1f}", help="1 (estompé) → 5 (vif)")

    st.success("Quiz terminé ! Passez maintenant à l'analyse photo pour croiser les résultats.")

    col_a, col_b = st.columns(2)
    if col_a.button("Aller a l'analyse photo →", type="primary", use_container_width=True):
        st.switch_page("app.py")
    if col_b.button("Refaire le quiz", use_container_width=True):
        st.session_state.quiz_answers = {}
        st.session_state.quiz_step    = 0
        st.session_state.quiz_done    = False
        st.rerun()
    st.stop()

# ── Quiz en cours ──────────────────────────────────────────────────────────────

step       = st.session_state.quiz_step
n_steps    = len(QUESTIONS)
question   = QUESTIONS[step]
answers    = st.session_state.quiz_answers

# Barre de progression
st.markdown(f"## 📋 Quiz colorimétrique")
st.progress((step) / n_steps, text=f"Question {step + 1} / {n_steps}")
st.markdown(f"### {question['title']}")
st.caption(question["subtitle"])
st.markdown("")

# Grille de swatches (2 colonnes)
options    = question.get("options", [])
is_multi   = question.get("multi", False)
max_select = question.get("max_select", 1)
selected   = answers.get(question["id"])

if is_multi:
    if not isinstance(selected, list):
        selected = []
    if max_select > 1:
        st.caption(f"Sélectionnez jusqu'à {max_select} réponses.")

if options:
    cols_per_row = 2
    rows = [options[i:i+cols_per_row] for i in range(0, len(options), cols_per_row)]
    for row in rows:
        cols = st.columns(cols_per_row)
        for i, opt in enumerate(row):
            with cols[i]:
                if is_multi:
                    is_selected = opt["id"] in selected
                    if swatch_button(opt, is_selected, key=f"{question['id']}_{opt['id']}"):
                        current = list(selected)
                        if opt["id"] in current:
                            current.remove(opt["id"])
                        elif len(current) < max_select:
                            current.append(opt["id"])
                        st.session_state.quiz_answers[question["id"]] = current
                        st.rerun()
                else:
                    is_selected = (selected == opt["id"])
                    if swatch_button(opt, is_selected, key=f"{question['id']}_{opt['id']}"):
                        st.session_state.quiz_answers[question["id"]] = opt["id"]
                        st.rerun()

st.markdown("---")

# Navigation
col_prev, col_next = st.columns(2)

with col_prev:
    if step > 0:
        if st.button("← Précédent", use_container_width=True):
            st.session_state.quiz_step -= 1
            st.rerun()

with col_next:
    answer_val = answers.get(question["id"])
    if question.get("multi"):
        can_next = isinstance(answer_val, list) and len(answer_val) > 0
    else:
        can_next = question["id"] in answers
    if step < n_steps - 1:
        if st.button("Suivant →", type="primary", use_container_width=True, disabled=not can_next):
            st.session_state.quiz_step += 1
            st.rerun()
    else:
        if st.button("Voir mon résultat ✓", type="primary", use_container_width=True, disabled=not can_next):
            st.session_state.quiz_done = True
            st.rerun()
