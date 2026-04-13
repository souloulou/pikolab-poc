"""
Pipeline de validation colorimétrique hybride.
3 sources : Quiz (25%) + CIELab algorithmique (35%) + 3 agents Vision Gemini (40%).
Le contexte quiz est injecté dans chaque prompt Vision.
"""

import base64
import json
import time

import cv2
import numpy as np
from google import genai

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-flash-lite-latest",
]

VALID_SEASONS = {
    "Light Spring", "Warm Spring", "Bright Spring", "True Spring",
    "Light Summer", "Cool Summer", "Soft Summer", "True Summer",
    "Soft Autumn", "Warm Autumn", "Deep Autumn", "True Autumn",
    "Deep Winter", "Cool Winter", "Bright Winter", "True Winter",
}

BASE_TO_SEASONS = {
    "Spring": {"Light Spring", "Warm Spring", "Bright Spring", "True Spring"},
    "Summer": {"Light Summer", "Cool Summer", "Soft Summer", "True Summer"},
    "Autumn": {"Soft Autumn", "Warm Autumn", "Deep Autumn", "True Autumn"},
    "Winter": {"Deep Winter", "Cool Winter", "Bright Winter", "True Winter"},
}

SEASON_TO_BASE = {s: base for base, seasons in BASE_TO_SEASONS.items() for s in seasons}

# Poids des sources
_W_QUIZ   = 0.25
_W_ALGO   = 0.20
_W_VISION = 0.55   # partagé entre les 3 agents Vision

# ── Construction du contexte quiz ─────────────────────────────────────────────

def _quiz_context_str(quiz_result: dict | None) -> str:
    """Retourne une ligne de contexte quiz à injecter dans les prompts Vision."""
    if not quiz_result or not quiz_result.get("season"):
        return ""
    label  = quiz_result.get("season_label") or quiz_result.get("season")
    conf   = quiz_result.get("confidence", 0)
    scores = quiz_result.get("scores", {})
    w_str  = f"{scores.get('warmth', 0):+.1f}" if scores else "?"
    d_str  = f"{scores.get('depth',  0):.1f}"  if scores else "?"
    return (
        f"\n[CONTEXTE QUIZ — données subjectives] Saison probable : {label} "
        f"({conf}% confiance) | chaleur {w_str}/3 | profondeur {d_str}/5. "
        f"Confirme ou corrige si ton analyse visuelle diverge."
    )


# ── Définition des 3 agents Vision ───────────────────────────────────────────

def _build_agents(quiz_context: str) -> list[dict]:
    seasons_list = (
        "Light Spring, Warm Spring, Bright Spring, True Spring, "
        "Light Summer, Cool Summer, Soft Summer, True Summer, "
        "Soft Autumn, Warm Autumn, Deep Autumn, True Autumn, "
        "Deep Winter, Cool Winter, Bright Winter, True Winter"
    )
    json_schema = (
        '{"sub_season": "...", "base_season": "Spring|Summer|Autumn|Winter", '
        '"temperature": "chaud|neutre|froid", "confidence": 0.0-1.0, "reasoning": "max 80 mots"}'
    )

    return [
        {
            "name": "Vision — Sous-ton peau",
            "system": (
                "Tu es un expert en analyse colorimétrique de teint. "
                "Tu te concentres UNIQUEMENT sur la peau du visage (joues, front, menton). "
                "Ignore la couleur des yeux. Repère les reflets chauds (dorés, pêche, abricot, jaune) "
                "ou froids (rosés, lilas, bleutés, grisés) dans le teint. "
                "Un teint neutre-chaud doit être classé Autumn ou Spring. "
                "Réponds UNIQUEMENT en JSON valide, sans markdown ni backticks."
            ),
            "user": (
                f"Analyse la peau de ce visage (joues, front, menton — ignore les yeux).{quiz_context}\n"
                f"Classe dans une des 16 saisons : {seasons_list}.\n"
                f"Retourne ce JSON exact : {json_schema}"
            ),
        },
        {
            "name": "Vision — Expert saisonnier",
            "system": (
                "Tu es un expert colorimétrie saisonnier avec 20 ans d'expérience. "
                "Tu analyses le visage complet : peau, yeux, sourcils, cheveux visibles. "
                "Tu maîtrises le système des 16 saisons (4 bases × 4 sous-saisons). "
                "Tu es particulièrement précis sur la distinction chaud/froid/neutre. "
                "Réponds UNIQUEMENT en JSON valide, sans markdown ni backticks."
            ),
            "user": (
                f"Classe cette personne dans une des 16 saisons colorimétriques.{quiz_context}\n"
                f"Saisons : {seasons_list}.\n"
                "Analyse : peau (reflets dorés/rosés), yeux (chauds/froids/clairs/foncés), "
                "sourcils et cheveux visibles.\n"
                f"Retourne ce JSON exact : {json_schema}"
            ),
        },
        {
            "name": "Vision — Détecteur chaud/froid",
            "system": (
                "Tu es spécialisé dans la détection du caractère chaud vs froid d'un teint. "
                "Signaux CHAUDS : reflets dorés/pêche/abricot dans la peau, "
                "sourcils/cheveux tirant vers le roux ou le doré, tons terre dans le regard. "
                "Signaux FROIDS : reflets roses/lilas/grisés dans la peau, yeux bleus/gris/verts froids, "
                "cheveux cendré ou platine. "
                "ATTENTION : un teint 'neutre' qui penche vers le chaud = Autumn ou Spring. "
                "Réponds UNIQUEMENT en JSON valide, sans markdown ni backticks."
            ),
            "user": (
                f"Cherche tous les signaux chauds ET froids dans ce visage.{quiz_context}\n"
                f"Tranche et classe dans une des 16 saisons : {seasons_list}.\n"
                f"Retourne ce JSON exact : {json_schema}"
            ),
        },
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _image_to_b64(image_rgb: np.ndarray) -> str:
    _, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    return base64.b64encode(buf.tobytes()).decode()


def _call_gemini_vision(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    image_b64: str,
    max_retries: int = 2,
) -> str:
    """Appel Gemini Vision avec fallback modèles et retry sur quota."""
    client = genai.Client(api_key=api_key)
    contents = [{
        "role": "user",
        "parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
            {"text": user_prompt},
        ],
    }]
    last_error = None
    for model_name in GEMINI_MODELS:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config={
                        "system_instruction": system_prompt,
                        "response_mime_type": "application/json",
                    },
                )
                return response.text or ""
            except Exception as exc:
                last_error = exc
                is_quota = "quota" in str(exc).lower() or "429" in str(exc)
                if is_quota and attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                if not is_quota:
                    raise
    raise RuntimeError(f"Vision non disponible : {last_error}")


def _parse_response(raw: str, agent_name: str) -> dict | None:
    try:
        data = json.loads(raw)
        season = data.get("sub_season", data.get("season", ""))
        if season not in VALID_SEASONS:
            for valid in sorted(VALID_SEASONS):
                if valid.lower() in raw.lower():
                    season = valid
                    break
        if season not in VALID_SEASONS:
            return None
        return {
            "name": agent_name,
            "sub_season": season,
            "base_season": SEASON_TO_BASE.get(season, data.get("base_season", "")),
            "temperature": data.get("temperature", "neutre"),
            "confidence": float(data.get("confidence", 0.7)),
            "reasoning": str(data.get("reasoning", ""))[:300],
        }
    except Exception:
        for season in sorted(VALID_SEASONS):
            if season.lower() in raw.lower():
                return {
                    "name": agent_name,
                    "sub_season": season,
                    "base_season": SEASON_TO_BASE.get(season, ""),
                    "temperature": "neutre",
                    "confidence": 0.5,
                    "reasoning": "(réponse partiellement parsée)",
                }
        return None


# ── Synthèse finale ───────────────────────────────────────────────────────────

def _synthesize(
    algo_season: str,
    algo_confidence: float,
    quiz_season: str | None,
    quiz_confidence: float,
    vision_results: list[dict],
) -> tuple[str, str, str]:
    """
    Combine Quiz + CIELab + agents Vision → (saison finale, base, accord).
    """
    algo_base = SEASON_TO_BASE.get(algo_season, "")
    quiz_base = SEASON_TO_BASE.get(quiz_season, "") if quiz_season else None

    # Vote pondéré sur la base
    base_weights: dict[str, float] = {}
    base_weights[algo_base] = base_weights.get(algo_base, 0) + _W_ALGO
    if quiz_base:
        base_weights[quiz_base] = base_weights.get(quiz_base, 0) + _W_QUIZ * (quiz_confidence / 100)
    vision_weight_each = _W_VISION / max(len(vision_results), 1)
    for vr in vision_results:
        vb = vr.get("base_season", "")
        if vb:
            base_weights[vb] = base_weights.get(vb, 0) + vision_weight_each * vr.get("confidence", 0.7)

    consensus_base = max(base_weights, key=base_weights.get) if base_weights else algo_base

    # Niveau d'accord (sur toutes les sources)
    all_bases = (
        [algo_base]
        + ([quiz_base] if quiz_base else [])
        + [vr.get("base_season", "") for vr in vision_results]
    )
    agreeing     = sum(1 for b in all_bases if b == consensus_base)
    total_sources = len(all_bases)
    if agreeing == total_sources:
        agreement = "unanimite"
    elif agreeing >= max(2, (total_sources + 1) // 2):
        agreement = "majorite"
    else:
        agreement = "desaccord"

    # Sous-saison : vote pondéré parmi les sources qui s'accordent sur la base
    sub_votes: dict[str, float] = {}
    if algo_base == consensus_base:
        sub_votes[algo_season] = sub_votes.get(algo_season, 0) + _W_ALGO
    if quiz_season and quiz_base == consensus_base:
        sub_votes[quiz_season] = sub_votes.get(quiz_season, 0) + _W_QUIZ * (quiz_confidence / 100)
    for vr in vision_results:
        if vr.get("base_season") == consensus_base:
            s = vr.get("sub_season", "")
            if s in VALID_SEASONS:
                sub_votes[s] = sub_votes.get(s, 0) + vision_weight_each * vr.get("confidence", 0.7)

    consensus_season = max(sub_votes, key=sub_votes.get) if sub_votes else algo_season
    if consensus_season not in VALID_SEASONS:
        consensus_season = algo_season

    return consensus_season, consensus_base, agreement


# ── Point d'entrée principal ──────────────────────────────────────────────────

def run_consensus_analysis(
    image_rgb: np.ndarray,
    algo_season: str,
    algo_confidence: float,
    api_key: str,
    skin_stats: dict | None = None,
    quiz_result: dict | None = None,
) -> dict:
    """
    Validation hybride : Quiz + CIELab + 3 agents Vision Gemini.
    Le contexte quiz est injecté dans chaque prompt Vision.
    """
    image_b64   = _image_to_b64(image_rgb)
    algo_base   = SEASON_TO_BASE.get(algo_season, "")
    quiz_season = quiz_result.get("season") if quiz_result else None
    quiz_conf   = quiz_result.get("confidence", 0) if quiz_result else 0
    quiz_ctx    = _quiz_context_str(quiz_result)

    # Source 1 : CIELab algorithmique
    agents_results = [{
        "name": "Algo (CIELab)",
        "sub_season": algo_season,
        "base_season": algo_base,
        "temperature": "chaud" if algo_base in ("Spring", "Autumn") else "froid",
        "confidence": algo_confidence,
        "reasoning": "Analyse CIELab peau (70%) + iris (30%)",
    }]

    # Source 2 : Quiz (si disponible)
    if quiz_season and quiz_season in VALID_SEASONS:
        agents_results.append({
            "name": "Quiz colorimétrique",
            "sub_season": quiz_season,
            "base_season": SEASON_TO_BASE.get(quiz_season, ""),
            "temperature": "chaud" if SEASON_TO_BASE.get(quiz_season, "") in ("Spring", "Autumn") else "froid",
            "confidence": quiz_conf / 100,
            "reasoning": f"Questionnaire : teint, yeux, veines, cheveux, cernes ({quiz_conf}% confiance)",
        })

    # Source 3 : 3 agents Vision Gemini avec contexte quiz injecté
    errors = []
    vision_results = []
    for cfg in _build_agents(quiz_ctx):
        try:
            raw    = _call_gemini_vision(api_key, cfg["system"], cfg["user"], image_b64)
            parsed = _parse_response(raw, cfg["name"])
            if parsed:
                agents_results.append(parsed)
                vision_results.append(parsed)
            else:
                errors.append(f"{cfg['name']} : réponse non parsable ({raw[:60]}...)")
        except Exception as exc:
            errors.append(f"{cfg['name']} : {str(exc)[:100]}")

    # Synthèse
    consensus_season, consensus_base, agreement = _synthesize(
        algo_season, algo_confidence,
        quiz_season, quiz_conf,
        vision_results,
    )

    # Votes bruts pour affichage
    base_votes: dict[str, int] = {}
    for r in agents_results:
        b = r.get("base_season", "")
        if b:
            base_votes[b] = base_votes.get(b, 0) + 1

    return {
        "consensus_season": consensus_season,
        "consensus_base":   consensus_base,
        "agents":           agents_results,
        "base_votes":       base_votes,
        "agreement_level":  agreement,
        "overridden":       consensus_season != algo_season,
        "errors":           errors,
        "quiz_used":        quiz_season is not None,
    }
