"""
Système multi-agents pour l'analyse colorimétrique saisonnière.
3 agents Gemini Vision indépendants + 1 agent algorithmique → vote majoritaire.
"""

import base64
import json

import cv2
import numpy as np
from google import genai

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash", "gemini-flash-lite-latest"]

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

# ── Prompts des 3 agents Vision ─────────────────────────────────────────────

_AGENTS = [
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
            "Analyse la peau de ce visage (joues, front, menton uniquement — ignore les yeux). "
            "Classe dans une des 16 saisons colorimétriques : "
            "Light Spring, Warm Spring, Bright Spring, True Spring, "
            "Light Summer, Cool Summer, Soft Summer, True Summer, "
            "Soft Autumn, Warm Autumn, Deep Autumn, True Autumn, "
            "Deep Winter, Cool Winter, Bright Winter, True Winter. "
            "Retourne ce JSON exact : "
            '{"sub_season": "...", "base_season": "Spring|Summer|Autumn|Winter", '
            '"temperature": "chaud|neutre|froid", "confidence": 0.0-1.0, "reasoning": "max 80 mots"}'
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
            "Classe cette personne dans une des 16 saisons colorimétriques. "
            "Saisons : Light Spring, Warm Spring, Bright Spring, True Spring, "
            "Light Summer, Cool Summer, Soft Summer, True Summer, "
            "Soft Autumn, Warm Autumn, Deep Autumn, True Autumn, "
            "Deep Winter, Cool Winter, Bright Winter, True Winter. "
            "Analyse : peau (reflets dorés/rosés), yeux (chauds/froids/clairs/foncés), "
            "sourcils et cheveux visibles. "
            "Retourne ce JSON exact : "
            '{"sub_season": "...", "base_season": "Spring|Summer|Autumn|Winter", '
            '"temperature": "chaud|neutre|froid", "confidence": 0.0-1.0, "reasoning": "max 80 mots"}'
        ),
    },
    {
        "name": "Vision — Détecteur chaud/froid",
        "system": (
            "Tu es spécialisé dans la détection du caractère chaud vs froid d'un teint. "
            "Signaux CHAUDS à chercher : reflets dorés/pêche/abricot dans la peau, "
            "sourcils/cheveux tirant vers le roux ou le doré, tons terre dans le regard. "
            "Signaux FROIDS : reflets roses/lilas/grisés dans la peau, yeux bleus/gris/verts froids, "
            "cheveux cendré ou platine. "
            "ATTENTION : un teint 'neutre' qui penche vers le chaud = Autumn ou Spring, pas Summer/Winter. "
            "Réponds UNIQUEMENT en JSON valide, sans markdown ni backticks."
        ),
        "user": (
            "Cherche tous les signaux chauds ET froids dans ce visage. "
            "Tranche : chaud, neutre-chaud, neutre, neutre-froid ou froid ? "
            "Classe dans une des 16 saisons : "
            "Light Spring, Warm Spring, Bright Spring, True Spring, "
            "Light Summer, Cool Summer, Soft Summer, True Summer, "
            "Soft Autumn, Warm Autumn, Deep Autumn, True Autumn, "
            "Deep Winter, Cool Winter, Bright Winter, True Winter. "
            "Retourne ce JSON exact : "
            '{"sub_season": "...", "base_season": "Spring|Summer|Autumn|Winter", '
            '"temperature": "chaud|neutre|froid", "confidence": 0.0-1.0, "reasoning": "max 80 mots"}'
        ),
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _image_to_b64(image_rgb: np.ndarray) -> str:
    """Convertit un tableau numpy RGB en JPEG base64."""
    _, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    return base64.b64encode(buf.tobytes()).decode()


def _call_gemini_vision(api_key: str, system_prompt: str, user_prompt: str, image_b64: str) -> str:
    """Appel Gemini Vision non-streamé avec fallback sur les modèles."""
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
            if "quota" not in str(exc).lower() and "429" not in str(exc):
                raise
            continue
    raise RuntimeError(f"Quota Gemini épuisé : {last_error}")


def _parse_response(raw: str, agent_name: str) -> dict | None:
    """Parse la réponse JSON d'un agent avec fallback sur recherche de nom de saison."""
    try:
        data = json.loads(raw)
        season = data.get("sub_season", data.get("season", ""))
        # Correction si la saison renvoyée n'est pas exactement dans la liste
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


# ── Point d'entrée principal ─────────────────────────────────────────────────

def run_consensus_analysis(
    image_rgb: np.ndarray,
    algo_season: str,
    algo_confidence: float,
    api_key: str,
) -> dict:
    """
    Lance 3 agents Gemini Vision + intègre le résultat algorithmique.
    Retourne le résultat consensuel par vote majoritaire.

    Retour :
    {
        "consensus_season": str,
        "consensus_base": str,
        "agents": list[dict],          # 4 entrées : algo + 3 vision
        "base_votes": dict[str, int],
        "agreement_level": str,        # "unanimite" | "majorite" | "desaccord"
        "overridden": bool,
        "errors": list[str],
    }
    """
    image_b64 = _image_to_b64(image_rgb)
    algo_base = SEASON_TO_BASE.get(algo_season, "")

    # Agent 0 : résultat algorithmique
    agents_results = [{
        "name": "Algo (CIELab)",
        "sub_season": algo_season,
        "base_season": algo_base,
        "temperature": "chaud" if algo_base in ("Spring", "Autumn") else "froid",
        "confidence": algo_confidence,
        "reasoning": "Analyse algorithmique CIELab sur peau (70 %) + iris (30 %)",
    }]

    errors = []
    for cfg in _AGENTS:
        try:
            raw = _call_gemini_vision(api_key, cfg["system"], cfg["user"], image_b64)
            parsed = _parse_response(raw, cfg["name"])
            if parsed:
                agents_results.append(parsed)
            else:
                errors.append(f"{cfg['name']} : réponse non parsable ({raw[:60]}...)")
        except Exception as exc:
            errors.append(f"{cfg['name']} : {str(exc)[:100]}")

    # ── Vote majoritaire sur la saison de base ────────────────────────────
    base_votes: dict[str, int] = {}
    for r in agents_results:
        b = r.get("base_season", "")
        if b:
            base_votes[b] = base_votes.get(b, 0) + 1

    consensus_base = max(base_votes, key=base_votes.get) if base_votes else algo_base
    base_vote_count = base_votes.get(consensus_base, 0)

    # ── Vote pondéré sur la sous-saison (parmi ceux qui s'accordent sur la base) ──
    sub_votes: dict[str, float] = {}
    for r in agents_results:
        if r.get("base_season") == consensus_base:
            s = r.get("sub_season", "")
            if s in VALID_SEASONS:
                sub_votes[s] = sub_votes.get(s, 0) + r.get("confidence", 0.7)

    consensus_season = max(sub_votes, key=sub_votes.get) if sub_votes else algo_season

    # ── Niveau d'accord ──────────────────────────────────────────────────
    total = len(agents_results)
    if base_vote_count == total:
        agreement = "unanimite"
    elif base_vote_count >= max(2, (total + 1) // 2):
        agreement = "majorite"
    else:
        agreement = "desaccord"

    return {
        "consensus_season": consensus_season,
        "consensus_base": consensus_base,
        "agents": agents_results,
        "base_votes": base_votes,
        "agreement_level": agreement,
        "overridden": consensus_season != algo_season,
        "errors": errors,
    }
