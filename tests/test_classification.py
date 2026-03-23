"""Tests for scoring and season classification."""

import pytest
from app import (
    DEFAULTS,
    SEASON_PALETTES,
    SUBSEASON_RULES,
    classify_season,
    compute_confidence,
    compute_scores,
)


DEFAULT_PARAMS = {
    "temp_center": DEFAULTS["temp_center"],
    "temp_scale": DEFAULTS["temp_scale"],
    "value_center": DEFAULTS["value_center"],
    "sat_center": DEFAULTS["sat_center"],
    "sat_scale": DEFAULTS["sat_scale"],
}


class TestComputeScores:
    def test_warm_skin(self):
        skin = {"L": 65.0, "a": 15.0, "b": 28.0, "C": 25.0}
        scores = compute_scores(skin, None, DEFAULT_PARAMS)
        assert scores["temperature"] > 0, "High b* should give warm temperature"

    def test_cool_skin(self):
        skin = {"L": 65.0, "a": 8.0, "b": 8.0, "C": 12.0}
        scores = compute_scores(skin, None, DEFAULT_PARAMS)
        assert scores["temperature"] < 0, "Low b* should give cool temperature"

    def test_light_skin(self):
        skin = {"L": 75.0, "a": 12.0, "b": 18.0, "C": 20.0}
        scores = compute_scores(skin, None, DEFAULT_PARAMS)
        assert scores["value"] > 0, "High L* should give light value"

    def test_dark_skin(self):
        skin = {"L": 35.0, "a": 12.0, "b": 18.0, "C": 20.0}
        scores = compute_scores(skin, None, DEFAULT_PARAMS)
        assert scores["value"] < 0, "Low L* should give dark value"

    def test_scores_clamped(self):
        # Extreme values
        skin = {"L": 100.0, "a": 50.0, "b": 60.0, "C": 70.0}
        scores = compute_scores(skin, None, DEFAULT_PARAMS)
        assert -1 <= scores["temperature"] <= 1
        assert -1 <= scores["value"] <= 1
        assert -1 <= scores["saturation"] <= 1

    def test_iris_contribution(self):
        skin = {"L": 55.0, "a": 12.0, "b": 17.0, "C": 20.0}
        # Warm iris (brown)
        warm_iris = {"L": 40.0, "a": 15.0, "b": 25.0, "C": 25.0, "rgb": [120, 90, 60]}
        # Cool iris (blue)
        cool_iris = {"L": 50.0, "a": -5.0, "b": -20.0, "C": 22.0, "rgb": [80, 100, 150]}

        scores_warm = compute_scores(skin, warm_iris, DEFAULT_PARAMS)
        scores_cool = compute_scores(skin, cool_iris, DEFAULT_PARAMS)
        assert scores_warm["temperature"] > scores_cool["temperature"]

    def test_no_iris_still_works(self):
        skin = {"L": 55.0, "a": 12.0, "b": 17.0, "C": 20.0}
        scores = compute_scores(skin, None, DEFAULT_PARAMS)
        assert "temperature" in scores
        assert "value" in scores
        assert "saturation" in scores


class TestClassifySeason:
    # ---- Base season tests ----

    def test_warm_light_is_spring(self):
        scores = {"temperature": 0.5, "value": 0.5, "saturation": 0.1}
        season = classify_season(scores, 0.3)
        assert "Spring" in season

    def test_cool_light_is_summer(self):
        scores = {"temperature": -0.5, "value": 0.5, "saturation": 0.1}
        season = classify_season(scores, 0.3)
        assert "Summer" in season

    def test_warm_dark_is_autumn(self):
        scores = {"temperature": 0.5, "value": -0.5, "saturation": 0.1}
        season = classify_season(scores, 0.3)
        assert "Autumn" in season

    def test_cool_dark_is_winter(self):
        scores = {"temperature": -0.5, "value": -0.5, "saturation": 0.1}
        season = classify_season(scores, 0.3)
        assert "Winter" in season

    # ---- Sub-season tests ----

    def test_bright_spring(self):
        scores = {"temperature": 0.4, "value": 0.4, "saturation": 0.8}
        assert classify_season(scores, 0.3) == "Bright Spring"

    def test_light_spring(self):
        scores = {"temperature": 0.3, "value": 0.9, "saturation": 0.2}
        assert classify_season(scores, 0.3) == "Light Spring"

    def test_warm_spring(self):
        scores = {"temperature": 0.9, "value": 0.3, "saturation": 0.2}
        assert classify_season(scores, 0.3) == "Warm Spring"

    def test_cool_summer(self):
        scores = {"temperature": -0.8, "value": 0.4, "saturation": 0.2}
        assert classify_season(scores, 0.3) == "Cool Summer"

    def test_soft_summer(self):
        scores = {"temperature": -0.3, "value": 0.3, "saturation": -0.6}
        assert classify_season(scores, 0.3) == "Soft Summer"

    def test_deep_autumn(self):
        scores = {"temperature": 0.3, "value": -0.9, "saturation": 0.1}
        assert classify_season(scores, 0.3) == "Deep Autumn"

    def test_warm_autumn(self):
        scores = {"temperature": 0.9, "value": -0.3, "saturation": 0.1}
        assert classify_season(scores, 0.3) == "Warm Autumn"

    def test_deep_winter(self):
        scores = {"temperature": -0.3, "value": -0.9, "saturation": 0.1}
        assert classify_season(scores, 0.3) == "Deep Winter"

    def test_bright_winter(self):
        scores = {"temperature": -0.3, "value": -0.3, "saturation": 0.9}
        assert classify_season(scores, 0.3) == "Bright Winter"

    # ---- True season tests (no dominant axis) ----

    def test_true_spring(self):
        scores = {"temperature": 0.2, "value": 0.2, "saturation": 0.2}
        assert classify_season(scores, 0.3) == "True Spring"

    def test_true_summer(self):
        scores = {"temperature": -0.2, "value": 0.2, "saturation": -0.1}
        assert classify_season(scores, 0.3) == "True Summer"

    def test_true_autumn(self):
        scores = {"temperature": 0.2, "value": -0.2, "saturation": -0.1}
        assert classify_season(scores, 0.3) == "True Autumn"

    def test_true_winter(self):
        scores = {"temperature": -0.2, "value": -0.2, "saturation": 0.1}
        assert classify_season(scores, 0.3) == "True Winter"

    # ---- Edge cases ----

    def test_all_zeros_is_true_winter(self):
        scores = {"temperature": 0.0, "value": 0.0, "saturation": 0.0}
        season = classify_season(scores, 0.3)
        # temp<=0 and val<=0 → Winter, no dominance → True Winter
        assert season == "True Winter"

    def test_boundary_values(self):
        # Exactly on boundaries
        scores = {"temperature": 0.0, "value": 0.0, "saturation": 0.0}
        season = classify_season(scores, 0.0)
        assert season is not None


class TestComputeConfidence:
    def test_high_confidence_extreme_scores(self):
        scores = {"temperature": 0.9, "value": 0.9, "saturation": 0.5}
        conf = compute_confidence(scores)
        assert conf > 0.7

    def test_low_confidence_near_boundary(self):
        scores = {"temperature": 0.05, "value": 0.05, "saturation": 0.0}
        conf = compute_confidence(scores)
        assert conf < 0.7

    def test_confidence_between_0_and_1(self):
        for t in [-1, -0.5, 0, 0.5, 1]:
            for v in [-1, -0.5, 0, 0.5, 1]:
                scores = {"temperature": t, "value": v, "saturation": 0}
                conf = compute_confidence(scores)
                assert 0 <= conf <= 1


class TestSeasonDataIntegrity:
    def test_all_16_seasons_have_palettes(self):
        assert len(SEASON_PALETTES) == 16

    def test_all_palettes_have_8_colors(self):
        for name, colors in SEASON_PALETTES.items():
            assert len(colors) == 8, f"{name} has {len(colors)} colors, expected 8"

    def test_all_palette_colors_are_valid_hex(self):
        import re
        for name, colors in SEASON_PALETTES.items():
            for c in colors:
                assert re.match(r"^#[0-9A-Fa-f]{6}$", c), f"Invalid hex {c} in {name}"

    def test_all_base_seasons_have_4_subsections(self):
        for base, rules in SUBSEASON_RULES.items():
            assert len(rules) == 4, f"{base} has {len(rules)} sub-seasons"

    def test_each_base_has_one_true_season(self):
        for base, rules in SUBSEASON_RULES.items():
            true_count = sum(1 for _, axis, _ in rules if axis is None)
            assert true_count == 1, f"{base} has {true_count} True seasons"

    def test_all_subseason_names_in_palette(self):
        all_sub_names = set()
        for rules in SUBSEASON_RULES.values():
            for name, _, _ in rules:
                all_sub_names.add(name)
        for name in all_sub_names:
            assert name in SEASON_PALETTES, f"Sub-season '{name}' missing from palettes"
