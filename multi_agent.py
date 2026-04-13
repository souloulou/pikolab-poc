"""
Pipeline de validation colorimétrique hybride.
3 sources : Quiz (25%) + CIELab algorithmique (35%) + Vision Gemini (40%).
1 seul appel Vision (au lieu de 3) pour éviter les erreurs de quota.
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

# Poids des 3 sources
_W_QUIZ  = 0.25
_W_ALGO  = 0.35
_W_VISION = 0.40

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
            "reasoning": str(data.get("reasoning", ""))[:400],
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


def _build_vision_prompt(
    algo_season: str,
    algo_confidence: float,
    skin_stats: dict | None,
    quiz_result: dict | None,
) -> tuple[str, str]:
    """Construit le prompt Vision enrichi avec le contexte Quiz et CIELab."""

    # Contexte quiz
    quiz_ctx = ""
    if quiz_result and quiz_result.get("season"):
        quiz_label = quiz_result.get("season_label") or quiz_result.get("season")
        quiz_conf  = quiz_result.get("confidence", 0)
        quiz_scores = quiz_result.get("scores", {})
        warmth_str = f"{quiz_scores.get('warmth', 0):+.1f}" if quiz_scores else "?"
        depth_str  = f"{quiz_scores.get('depth', 0):.1f}"  if quiz_scores else "?"
        quiz_ctx = (
            f"\n\nCONTEXTE QUIZ (données subjectives de l'utilisatrice) :\n"
            f"- Saison probable : {quiz_label} (confiance {quiz_conf}%)\n"
            f"- Score chaleur : {warmth_str} (-3=froid, +3=chaud)\n"
            f"- Score profondeur : {depth_str} (1=très clair, 5=très foncé)\n"
            f"- Ce contexte est basé sur : teint déclaré, couleur des yeux, "
            f"veines, cheveux, cernes et imperfections."
        )

    # Contexte CIELab
    algo_ctx = (
        f"\n\nCONTEXTE ANALYSE ALGORITHMIQUE (CIELab) :\n"
        f"- Saison calculée : {algo_season} (confiance {round(algo_confidence*100)}%)\n"
    )
    if skin_stats:
        algo_ctx += (
            f"- L* peau : {skin_stats.get('L', '?'):.1f} "
            f"(luminosité, 0=noir, 100=blanc)\n"
            f"- a* peau : {skin_stats.get('a', '?'):.1f} "
            f"(rouge/vert)\n"
            f"- b* peau : {skin_stats.get('b', '?'):.1f} "
            f"(chaud/froid — positif=jaune, négatif=bleu)\n"
        )

    system = (
        "Tu es une experte en colorimétrie saisonnière avec 20 ans d'expérience. "
        "Tu analyses le visage entier : peau, yeux, sourcils, cheveux visibles. "
        "Tu as accès à des données contextuelles (quiz subjectif et mesures CIELab). "
        "Ton rôle est de VALIDER ou CORRIGER ces données par ton analyse visuelle. "
        "Sois particulièrement attentive à la distinction chaud/froid/neutre. "
        "Signaux CHAUDS : reflets dorés, pêche, abricot, roux dans la peau ou les cheveux. "
        "Signaux FROIDS : reflets rosés, lilas, grisés, bleutés dans la peau. "
        "Réponds UNIQUEMENT en JSON valide, sans markdown ni backticks."
    )

    user = (
        f"Analyse ce visage et classe-le dans l'une des 16 saisons colorimétriques :{quiz_ctx}{algo_ctx}\n"
        "Saisons possibles : "
        "Light Spring, Warm Spring, Bright Spring, True Spring, "
        "Light Summer, Cool Summer, Soft Summer, True Summer, "
        "Soft Autumn, Warm Autumn, Deep Autumn, True Autumn, "
        "Deep Winter, Cool Winter, Bright Winter, True Winter.\n\n"
        "Analyse : peau (reflets), yeux (couleur et intensité), "
        "sourcils et cheveux visibles, profondeur générale du teint.\n"
        "Si le quiz et l'algo s'accordent, confirme sauf raison visuelle forte. "
        "Si tu corriges, explique pourquoi.\n\n"
        "Retourne ce JSON exact :\n"
        '{"sub_season": "...", "base_season": "Spring|Summer|Autumn|Winter", '
        '"temperature": "chaud|neutre|froid", "confidence": 0.0-1.0, '
        '"quiz_confirmed": true|false, "algo_confirmed": true|false, '
        '"reasoning": "max 120 mots"}'
    )

    return system, user


# ── Synthèse finale ───────────────────────────────────────────────────────────

def _synthesize(
    algo_season: str,
    algo_confidence: float,
    quiz_season: str | None,
    quiz_confidence: float,
    vision_result: dict | None,
) -> tuple[str, str, str]:
    """
    Combine Quiz + CIELab + Vision → (saison finale, base, niveau d'accord).
    Retourne (consensus_season, consensus_base, agreement_level).
    """
    # Bases de chaque source
    algo_base  = SEASON_TO_BASE.get(algo_season, "")
    quiz_base  = SEASON_TO_BASE.get(quiz_season, "") if quiz_season else None
    vision_base = SEASON_TO_BASE.get(vision_result["sub_season"], "") if vision_result else None
    vision_season = vision_result["sub_season"] if vision_result else None
    vision_conf   = vision_result.get("confidence", 0.7) if vision_result else 0.0

    # Vote pondéré sur la base
    base_weights: dict[str, float] = {}
    if algo_base:
        base_weights[algo_base] = base_weights.get(algo_base, 0) + _W_ALGO
    if quiz_base:
        base_weights[quiz_base] = base_weights.get(quiz_base, 0) + _W_QUIZ * (quiz_confidence / 100)
    if vision_base:
        base_weights[vision_base] = base_weights.get(vision_base, 0) + _W_VISION * vision_conf

    consensus_base = max(base_weights, key=base_weights.get) if base_weights else algo_base
    total_weight = base_weights.get(consensus_base, 0)

    # Niveau d'accord
    sources_agreeing = sum([
        algo_base  == consensus_base,
        quiz_base  == consensus_base if quiz_base  else False,
        vision_base == consensus_base if vision_base else False,
    ])
    total_sources = 1 + (1 if quiz_base else 0) + (1 if vision_base else 0)

    if sources_agreeing == total_sources:
        agreement = "unanimite"
    elif sources_agreeing >= max(2, (total_sources + 1) // 2):
        agreement = "majorite"
    else:
        agreement = "desaccord"

    # Sous-saison finale (priorité Vision si dispo, sinon algo)
    if vision_season and vision_base == consensus_base:
        consensus_season = vision_season
    elif quiz_season and quiz_base == consensus_base:
        consensus_season = algo_season  # garde l'algo pour la précision numerique
    else:
        consensus_season = algo_season

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
    Validation hybride : Quiz + CIELab + 1 agent Vision Gemini.

    Retour :
    {
        "consensus_season": str,
        "consensus_base": str,
        "agents": list[dict],
        "base_votes": dict[str, float],
        "agreement_level": str,
        "overridden": bool,
        "errors": list[str],
        "quiz_used": bool,
    }
    """
    image_b64  = _image_to_b64(image_rgb)
    algo_base  = SEASON_TO_BASE.get(algo_season, "")
    quiz_season = quiz_result.get("season") if quiz_result else None
    quiz_conf   = quiz_result.get("confidence", 0) if quiz_result else 0

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
            "reasoning": f"Questionnaire subjectif : teint, yeux, veines, cheveux, cernes (confiance {quiz_conf}%)",
        })

    # Source 3 : Vision Gemini (1 agent enrichi)
    errors = []
    vision_result = None
    system_prompt, user_prompt = _build_vision_prompt(
        algo_season, algo_confidence, skin_stats, quiz_result
    )
    try:
        raw = _call_gemini_vision(api_key, system_prompt, user_prompt, image_b64)
        vision_result = _parse_response(raw, "Vision Gemini")
        if vision_result:
            agents_results.append(vision_result)
        else:
            errors.append(f"Vision : réponse non parsable ({raw[:80]}...)")
    except Exception as exc:
        errors.append(f"Vision : {str(exc)[:120]}")

    # Synthèse
    consensus_season, consensus_base, agreement = _synthesize(
        algo_season, algo_confidence,
        quiz_season, quiz_conf,
        vision_result,
    )

    # Votes (pour affichage)
    base_votes: dict[str, int] = {}
    for r in agents_results:
        b = r.get("base_season", "")
        if b:
            base_votes[b] = base_votes.get(b, 0) + 1

    return {
        "consensus_season":  consensus_season,
        "consensus_base":    consensus_base,
        "agents":            agents_results,
        "base_votes":        base_votes,
        "agreement_level":   agreement,
        "overridden":        consensus_season != algo_season,
        "errors":            errors,
        "quiz_used":         quiz_season is not None,
    }
