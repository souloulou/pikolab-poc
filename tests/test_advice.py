"""Tests for season advice data integrity and new pro features."""

import re
import pytest
from season_advice import (
    SEASON_ADVICE,
    REQUIRED_ADVICE_KEYS,
    REQUIRED_MAKEUP_KEYS,
    REQUIRED_CLOTHING_KEYS,
    REQUIRED_HAIR_KEYS,
    REQUIRED_ACCESSORIES_KEYS,
    REQUIRED_EXPERT_KEYS,
)
from app import (
    SEASON_PALETTES,
    SEASON_CENTROIDS,
    compute_contrast,
    compute_professional_profile,
    classify_top3,
)


class TestSeasonAdviceIntegrity:
    def test_all_16_seasons_present(self):
        assert len(SEASON_ADVICE) == 16

    def test_advice_matches_palettes(self):
        for name in SEASON_PALETTES:
            assert name in SEASON_ADVICE, f"Season '{name}' missing from SEASON_ADVICE"

    def test_all_required_keys_present(self):
        for name, advice in SEASON_ADVICE.items():
            for key in REQUIRED_ADVICE_KEYS:
                assert key in advice, f"'{name}' missing key '{key}'"

    def test_all_makeup_keys(self):
        for name, advice in SEASON_ADVICE.items():
            makeup = advice["makeup"]
            for key in REQUIRED_MAKEUP_KEYS:
                assert key in makeup, f"'{name}' makeup missing '{key}'"

    def test_all_clothing_keys(self):
        for name, advice in SEASON_ADVICE.items():
            clothing = advice["clothing"]
            for key in REQUIRED_CLOTHING_KEYS:
                assert key in clothing, f"'{name}' clothing missing '{key}'"

    def test_all_hair_keys(self):
        for name, advice in SEASON_ADVICE.items():
            hair = advice["hair"]
            for key in REQUIRED_HAIR_KEYS:
                assert key in hair, f"'{name}' hair missing '{key}'"

    def test_all_accessories_keys(self):
        for name, advice in SEASON_ADVICE.items():
            acc = advice["accessories"]
            for key in REQUIRED_ACCESSORIES_KEYS:
                assert key in acc, f"'{name}' accessories missing '{key}'"

    def test_palettes_are_valid_hex(self):
        hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for name, advice in SEASON_ADVICE.items():
            for key in ["palette_neutrals", "palette_accents", "palette_avoid"]:
                for color in advice[key]:
                    assert hex_re.match(color), f"Invalid hex '{color}' in {name}.{key}"

    def test_palette_avoid_not_empty(self):
        for name, advice in SEASON_ADVICE.items():
            assert len(advice["palette_avoid"]) >= 3, f"'{name}' should have at least 3 colors to avoid"

    def test_lips_and_eyes_not_empty(self):
        for name, advice in SEASON_ADVICE.items():
            assert len(advice["makeup"]["lips"]) >= 3, f"'{name}' should have >= 3 lip colors"
            assert len(advice["makeup"]["eyes"]) >= 3, f"'{name}' should have >= 3 eye colors"

    def test_hair_ideal_and_avoid(self):
        for name, advice in SEASON_ADVICE.items():
            assert len(advice["hair"]["ideal"]) >= 2, f"'{name}' needs >= 2 ideal hair colors"
            assert len(advice["hair"]["avoid"]) >= 2, f"'{name}' needs >= 2 hair colors to avoid"

    def test_description_is_nonempty_string(self):
        for name, advice in SEASON_ADVICE.items():
            desc = advice["description"]
            assert isinstance(desc, str) and len(desc) > 20, f"'{name}' description too short"

    def test_all_expert_keys(self):
        for name, advice in SEASON_ADVICE.items():
            expert = advice.get("expert", {})
            assert expert, f"'{name}' missing expert section"
            for key in REQUIRED_EXPERT_KEYS:
                assert key in expert, f"'{name}' expert missing '{key}'"

    def test_tagline_present(self):
        for name, advice in SEASON_ADVICE.items():
            assert "tagline" in advice, f"'{name}' missing tagline"
            assert len(advice["tagline"]) > 5, f"'{name}' tagline too short"

    def test_icons_present(self):
        for name, advice in SEASON_ADVICE.items():
            assert "icons" in advice, f"'{name}' missing icons"
            assert len(advice["icons"]) >= 2, f"'{name}' needs >= 2 icons"

    def test_black_white_alternatives(self):
        for name, advice in SEASON_ADVICE.items():
            assert "black_alt" in advice, f"'{name}' missing black_alt"
            assert "white_alt" in advice, f"'{name}' missing white_alt"

    def test_makeup_looks_present(self):
        for name, advice in SEASON_ADVICE.items():
            makeup = advice["makeup"]
            for look in ["look_naturel", "look_soiree", "look_pro"]:
                assert look in makeup, f"'{name}' makeup missing '{look}'"

    def test_capsule_wardrobe_present(self):
        for name, advice in SEASON_ADVICE.items():
            capsule = advice["clothing"].get("capsule", [])
            assert len(capsule) >= 5, f"'{name}' capsule needs >= 5 items"

    def test_shopping_tip_present(self):
        for name, advice in SEASON_ADVICE.items():
            assert advice["clothing"].get("shopping_tip"), f"'{name}' missing shopping_tip"

    def test_hair_tips_present(self):
        for name, advice in SEASON_ADVICE.items():
            assert advice["hair"].get("tips"), f"'{name}' missing hair tips"

    def test_nails_and_scarves_present(self):
        for name, advice in SEASON_ADVICE.items():
            assert advice["accessories"].get("nails"), f"'{name}' missing nails"
            assert advice["accessories"].get("scarves"), f"'{name}' missing scarves"


class TestSeasonCentroids:
    def test_all_16_centroids(self):
        assert len(SEASON_CENTROIDS) == 16

    def test_centroids_match_palettes(self):
        for name in SEASON_PALETTES:
            assert name in SEASON_CENTROIDS, f"'{name}' missing from centroids"

    def test_centroid_values_in_range(self):
        for name, (t, v, s) in SEASON_CENTROIDS.items():
            assert -1 <= t <= 1, f"'{name}' temp {t} out of range"
            assert -1 <= v <= 1, f"'{name}' value {v} out of range"
            assert -1 <= s <= 1, f"'{name}' sat {s} out of range"

    def test_spring_centroids_warm_light(self):
        for name, (t, v, s) in SEASON_CENTROIDS.items():
            if "Spring" in name:
                assert t > 0, f"Spring '{name}' should be warm (t>0)"
                assert v > 0, f"Spring '{name}' should be light (v>0)"

    def test_winter_centroids_cool_dark(self):
        for name, (t, v, s) in SEASON_CENTROIDS.items():
            if "Winter" in name:
                assert t < 0, f"Winter '{name}' should be cool (t<0)"
                assert v < 0, f"Winter '{name}' should be dark (v<0)"


class TestComputeContrast:
    def test_high_contrast(self):
        skin = {"L": 70.0, "a": 12.0, "b": 18.0, "C": 20.0}
        iris = {"L": 25.0, "a": 5.0, "b": 10.0, "C": 12.0, "rgb": [60, 50, 40]}
        c = compute_contrast(skin, iris)
        assert c > 0.4, f"High L* difference should give high contrast, got {c}"

    def test_low_contrast(self):
        skin = {"L": 55.0, "a": 12.0, "b": 18.0, "C": 20.0}
        iris = {"L": 50.0, "a": 10.0, "b": 15.0, "C": 18.0, "rgb": [100, 90, 80]}
        c = compute_contrast(skin, iris)
        assert c < 0.3, f"Similar L* should give low contrast, got {c}"

    def test_no_iris_returns_medium(self):
        skin = {"L": 55.0, "a": 12.0, "b": 18.0, "C": 20.0}
        assert compute_contrast(skin, None) == 0.5

    def test_contrast_clamped(self):
        skin = {"L": 95.0, "a": 5.0, "b": 5.0, "C": 7.0}
        iris = {"L": 5.0, "a": 30.0, "b": 30.0, "C": 42.0, "rgb": [0, 0, 0]}
        c = compute_contrast(skin, iris)
        assert 0 <= c <= 1


class TestComputeProfessionalProfile:
    def test_warm_profile(self):
        scores = {"temperature": 0.7, "value": 0.3, "saturation": 0.5}
        profile = compute_professional_profile(scores, 0.5)
        assert profile["undertone"] == "Chaud"
        assert profile["chroma"] == "Vif"

    def test_cool_profile(self):
        scores = {"temperature": -0.7, "value": -0.3, "saturation": -0.5}
        profile = compute_professional_profile(scores, 0.3)
        assert profile["undertone"] == "Froid"
        assert profile["chroma"] == "Doux"

    def test_neutral_profile(self):
        scores = {"temperature": 0.0, "value": 0.0, "saturation": 0.0}
        profile = compute_professional_profile(scores, 0.3)
        assert profile["undertone"] == "Neutre"
        assert profile["depth"] == "Medium"
        assert profile["chroma"] == "Modere"
        assert profile["contrast"] == "Moyen"

    def test_all_5_undertone_levels(self):
        levels = set()
        for t in [-0.8, -0.3, 0.0, 0.3, 0.8]:
            scores = {"temperature": t, "value": 0, "saturation": 0}
            profile = compute_professional_profile(scores, 0.5)
            levels.add(profile["undertone"])
        assert len(levels) == 5, f"Expected 5 undertone levels, got {levels}"

    def test_all_5_depth_levels(self):
        levels = set()
        for v in [-0.8, -0.3, 0.0, 0.3, 0.8]:
            scores = {"temperature": 0, "value": v, "saturation": 0}
            profile = compute_professional_profile(scores, 0.5)
            levels.add(profile["depth"])
        assert len(levels) == 5, f"Expected 5 depth levels, got {levels}"

    def test_raw_values_preserved(self):
        scores = {"temperature": 0.42, "value": -0.33, "saturation": 0.15}
        profile = compute_professional_profile(scores, 0.6)
        assert profile["raw_undertone"] == 0.42
        assert profile["raw_depth"] == -0.33
        assert profile["raw_chroma"] == 0.15
        assert profile["raw_contrast"] == 0.6


class TestClassifyTop3:
    def test_returns_3_entries(self):
        scores = {"temperature": 0.5, "value": 0.5, "saturation": 0.3}
        top3 = classify_top3(scores)
        assert len(top3) == 3

    def test_top1_is_closest(self):
        scores = {"temperature": 0.8, "value": 0.3, "saturation": 0.3}
        top3 = classify_top3(scores)
        assert top3[0]["season"] == "Warm Spring"

    def test_percentages_sum_to_100(self):
        scores = {"temperature": -0.3, "value": -0.5, "saturation": 0.2}
        top3 = classify_top3(scores)
        total = sum(e["match_pct"] for e in top3)
        assert abs(total - 100.0) < 1.0, f"Percentages sum to {total}, expected ~100"

    def test_first_has_highest_pct(self):
        scores = {"temperature": 0.5, "value": -0.8, "saturation": 0.0}
        top3 = classify_top3(scores)
        assert top3[0]["match_pct"] >= top3[1]["match_pct"]
        assert top3[1]["match_pct"] >= top3[2]["match_pct"]

    def test_entries_have_required_fields(self):
        scores = {"temperature": 0.0, "value": 0.0, "saturation": 0.0}
        top3 = classify_top3(scores)
        for entry in top3:
            assert "season" in entry
            assert "match_pct" in entry
            assert "distance" in entry

    def test_all_seasons_are_valid(self):
        scores = {"temperature": 0.3, "value": 0.3, "saturation": 0.3}
        top3 = classify_top3(scores)
        for entry in top3:
            assert entry["season"] in SEASON_PALETTES
