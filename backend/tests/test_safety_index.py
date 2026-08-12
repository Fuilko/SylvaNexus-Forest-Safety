"""
Tests for the GIS-AHP Forest Work Safety Index.

Covers AHP weight derivation and consistency checking, per-criterion scoring,
composite index behaviour, and that the region profiles behave differently in
the ways their pairwise judgements claim.
"""

import pytest

from app.weather.ahp import (
    InconsistentMatrixError,
    MAX_CONSISTENCY_RATIO,
    derive_weights,
    weights_for,
)
from app.weather.safety_criteria import (
    CRITERIA_GROUPS,
    GROUP_ORDER,
    REGION_PROFILES,
    get_profile,
)
from app.weather.safety_index import (
    assess_safety_index,
    classify_index,
    derive_region_weights,
    score_aspect,
    score_geomorphon,
    score_rainfall,
    score_slope,
    score_vegetation_density,
)


# ---------------------------------------------------------------------------
# AHP mechanics
# ---------------------------------------------------------------------------

def test_identity_matrix_gives_equal_weights():
    weights, cr = derive_weights([[1, 1], [1, 1]], "equal")
    assert weights == pytest.approx([0.5, 0.5])
    assert cr == 0.0


def test_perfectly_consistent_matrix_has_zero_cr():
    # a[i][j] = w_i / w_j for w = (0.6, 0.3, 0.1) is perfectly consistent
    matrix = [
        [1, 2, 6],
        [1 / 2, 1, 3],
        [1 / 6, 1 / 3, 1],
    ]
    weights, cr = derive_weights(matrix, "consistent")
    assert cr < 1e-9
    assert weights == pytest.approx([0.6, 0.3, 0.1], abs=1e-9)


def test_weights_always_sum_to_one():
    weights, _ = derive_weights([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]], "m")
    assert sum(weights) == pytest.approx(1.0)


def test_contradictory_matrix_is_rejected():
    """A says A>>B, B>>C, but C>>A — circular, so it must be refused."""
    matrix = [
        [1, 9, 1 / 9],
        [1 / 9, 1, 9],
        [9, 1 / 9, 1],
    ]
    _, cr = derive_weights(matrix, "circular")
    assert cr > MAX_CONSISTENCY_RATIO
    with pytest.raises(InconsistentMatrixError):
        weights_for(["a", "b", "c"], matrix, "circular")


def test_non_reciprocal_matrix_is_rejected():
    with pytest.raises(ValueError, match="not reciprocal"):
        derive_weights([[1, 2], [2, 1]], "bad")


def test_non_square_matrix_is_rejected():
    with pytest.raises(ValueError, match="not square"):
        derive_weights([[1, 2], [1 / 2]], "bad")


def test_non_positive_entry_is_rejected():
    with pytest.raises(ValueError, match="non-positive"):
        derive_weights([[1, 0], [0, 1]], "bad")


def test_label_count_must_match_matrix():
    with pytest.raises(ValueError, match="labels"):
        weights_for(["only_one"], [[1, 2], [1 / 2, 1]], "m")


# ---------------------------------------------------------------------------
# Region profiles — every shipped matrix must be consistent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("region", sorted(REGION_PROFILES))
def test_all_shipped_matrices_are_consistent(region):
    """If this fails, the expert judgements contradict themselves."""
    weights = derive_region_weights(get_profile(region))
    for name, cr in weights["consistency_ratios"].items():
        assert cr <= MAX_CONSISTENCY_RATIO, f"{region}.{name} CR={cr}"


@pytest.mark.parametrize("region", sorted(REGION_PROFILES))
def test_global_weights_sum_to_one(region):
    weights = derive_region_weights(get_profile(region))
    assert sum(weights["global_weights"].values()) == pytest.approx(1.0)
    assert sum(weights["group_weights"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("region", sorted(REGION_PROFILES))
def test_every_criterion_has_a_weight(region):
    weights = derive_region_weights(get_profile(region))
    expected = {c for members in CRITERIA_GROUPS.values() for c in members}
    assert set(weights["global_weights"]) == expected


def test_taiwan_prioritises_climate_and_hydro():
    """Taiwan's dominant fatal hazard is typhoon rainfall on steep terrain."""
    weights = derive_region_weights(get_profile("taiwan"))["group_weights"]
    assert weights["climate"] > weights["hydro"] > weights["biophysical"] > weights["threat"]


def test_indonesia_prioritises_threat():
    """Matches the source study's AHP result for Indonesian production forest."""
    weights = derive_region_weights(get_profile("indonesia"))["group_weights"]
    assert weights["threat"] == max(weights.values())
    assert weights["threat"] > weights["climate"]


def test_rainfall_is_the_single_heaviest_criterion_in_taiwan():
    weights = derive_region_weights(get_profile("taiwan"))["global_weights"]
    assert max(weights, key=weights.get) == "rainfall"


def test_unknown_region_is_rejected():
    with pytest.raises(ValueError, match="Unknown region"):
        get_profile("antarctica")


def test_default_region_is_taiwan():
    assert get_profile(None).code == "taiwan"


def test_group_order_matches_criteria_groups():
    assert set(GROUP_ORDER) == set(CRITERIA_GROUPS)


# ---------------------------------------------------------------------------
# Criterion scoring
# ---------------------------------------------------------------------------

def test_slope_score_increases_with_steepness():
    assert score_slope(5) == 1
    assert score_slope(38) == 4
    assert score_slope(60) == 5


def test_dense_vegetation_scores_low_risk():
    """Root cohesion reinforces soil, so high NDVI must LOWER the score."""
    assert score_vegetation_density(0.8) == 1
    assert score_vegetation_density(0.3) == 4
    assert score_vegetation_density(0.1) == 5


def test_missing_ndvi_is_not_treated_as_safe():
    assert score_vegetation_density(0) == 3


def test_convergent_landforms_score_highest():
    hollow, valley, ridge, flat = 7, 9, 3, 1
    assert score_geomorphon(hollow) == 5
    assert score_geomorphon(valley) == 5
    assert score_geomorphon(ridge) == 2
    assert score_geomorphon(flat) == 1


def test_windward_aspect_scores_above_leeward():
    east, northwest = 90, 315
    assert score_aspect(east) > score_aspect(northwest)


def test_zero_aspect_is_due_north_not_missing_data():
    """GDAL reports flat ground as negative; 0 is a real north-facing slope."""
    due_north, flat, unknown = 0, -9999, None
    assert score_aspect(due_north) == 1
    assert score_aspect(flat) == 2
    assert score_aspect(unknown) == 2


def test_rainfall_scored_against_regional_threshold():
    """The same rainfall is more severe where the official threshold is lower."""
    taiwan = get_profile("taiwan")        # 350mm
    indonesia = get_profile("indonesia")  # 200mm
    assert score_rainfall(210, taiwan) < score_rainfall(210, indonesia)


def test_reaching_the_threshold_is_extreme():
    taiwan = get_profile("taiwan")
    assert score_rainfall(taiwan.rainfall_reference_mm, taiwan) == 5


def test_index_classification_boundaries():
    assert classify_index(1.0)[0] == "very_low"
    assert classify_index(3.0)[0] == "moderate"
    assert classify_index(4.9)[0] == "very_high"


# ---------------------------------------------------------------------------
# Composite index
# ---------------------------------------------------------------------------

BAXIANSHAN = dict(
    slope_deg=38, elevation_m=1800, soil_type="slate", ndvi=0.72,
    accessibility_minutes=35, twi=5.2, flow_accumulation=8000,
    geomorphon=5, aspect_deg=180, wildlife="hornets",
    abnormal_trees="dead_standing",
)


def test_composite_stays_within_scale():
    result = assess_safety_index("taiwan", rt_mm=120, wind_speed_ms=6, wbgt_c=24, **BAXIANSHAN)
    assert 1.0 <= result["composite_score"] <= 5.0


def test_best_case_scores_minimum():
    result = assess_safety_index(
        "taiwan", slope_deg=2, elevation_m=100, soil_type="granite", ndvi=0.9,
        accessibility_minutes=5, twi=1, flow_accumulation=10, geomorphon=1,
        aspect_deg=0, rt_mm=0, wind_speed_ms=0, wbgt_c=0,
        wildlife="none", abnormal_trees="none",
    )
    assert result["composite_score"] == pytest.approx(1.0)
    assert result["class"] == "very_low"


def test_worst_case_scores_maximum():
    result = assess_safety_index(
        "taiwan", slope_deg=60, elevation_m=3000, soil_type="colluvium", ndvi=0.05,
        accessibility_minutes=120, twi=9, flow_accumulation=20000, geomorphon=9,
        aspect_deg=90, rt_mm=400, wind_speed_ms=25, wbgt_c=35,
        wildlife="black_bear", abnormal_trees="hanging_limb",
    )
    assert result["composite_score"] == pytest.approx(5.0)
    assert result["class"] == "very_high"


def test_typhoon_conditions_raise_the_index():
    calm = assess_safety_index("taiwan", rt_mm=50, wind_speed_ms=3, wbgt_c=22, **BAXIANSHAN)
    storm = assess_safety_index("taiwan", rt_mm=400, wind_speed_ms=20, wbgt_c=22, **BAXIANSHAN)
    assert storm["composite_score"] > calm["composite_score"]
    assert storm["dynamic_share"] > calm["dynamic_share"]


def test_dynamic_share_reveals_how_much_is_live_weather():
    """Guards against a static index that never responds to conditions."""
    storm = assess_safety_index("taiwan", rt_mm=400, wind_speed_ms=20, wbgt_c=33, **BAXIANSHAN)
    assert storm["dynamic_share"] > 0.4


def test_static_score_ignores_weather():
    """The zoning layer must be stable so a work-zone map does not flicker."""
    calm = assess_safety_index("taiwan", rt_mm=0, wind_speed_ms=0, wbgt_c=0, **BAXIANSHAN)
    storm = assess_safety_index("taiwan", rt_mm=400, wind_speed_ms=25, wbgt_c=35, **BAXIANSHAN)
    assert calm["static_score"] == storm["static_score"]
    assert calm["composite_score"] < storm["composite_score"]


def test_static_score_reaches_the_top_of_the_scale_on_bad_terrain():
    """The composite cannot do this on a calm day, which is why static exists."""
    result = assess_safety_index(
        "taiwan", slope_deg=60, elevation_m=3000, soil_type="colluvium", ndvi=0.05,
        accessibility_minutes=120, twi=9, flow_accumulation=20000, geomorphon=9,
        aspect_deg=90, rt_mm=0, wind_speed_ms=0, wbgt_c=0,
        wildlife="black_bear", abnormal_trees="hanging_limb",
    )
    assert result["static_score"] == pytest.approx(5.0)
    assert result["static_class"] == "very_high"
    assert result["class"] == "moderate"


def test_top_drivers_identify_the_worst_contributors():
    result = assess_safety_index("taiwan", rt_mm=400, wind_speed_ms=20, wbgt_c=22, **BAXIANSHAN)
    assert result["top_drivers"][0]["criterion"] == "rainfall"
    assert len(result["top_drivers"]) == 3


def test_contributions_sum_to_composite():
    result = assess_safety_index("taiwan", rt_mm=200, wind_speed_ms=9, wbgt_c=27, **BAXIANSHAN)
    total = sum(c["contribution"] for c in result["criteria"].values())
    assert total == pytest.approx(result["composite_score"], abs=0.01)


def test_group_subtotals_sum_to_composite():
    result = assess_safety_index("taiwan", rt_mm=200, wind_speed_ms=9, wbgt_c=27, **BAXIANSHAN)
    total = sum(g["contribution"] for g in result["groups"].values())
    assert total == pytest.approx(result["composite_score"], abs=0.01)


def test_same_site_scores_differently_by_region():
    """Region profiles must actually change the answer, not just the label."""
    scores = {
        region: assess_safety_index(region, rt_mm=150, wind_speed_ms=6, wbgt_c=30,
                                    **BAXIANSHAN)["composite_score"]
        for region in ("taiwan", "japan", "indonesia")
    }
    assert len(set(scores.values())) == 3


def test_wildlife_scoring_is_region_specific():
    """Elephants are an Indonesian hazard; the term is meaningless in Taiwan."""
    indonesia = get_profile("indonesia")
    taiwan = get_profile("taiwan")
    assert indonesia.wildlife_hazards["elephant"] == 5
    assert "elephant" not in taiwan.wildlife_hazards


def test_unknown_soil_type_is_not_optimistic():
    """Missing survey data must not make a site look safe."""
    unknown = assess_safety_index("taiwan", rt_mm=100, wind_speed_ms=5, wbgt_c=24,
                                  **{**BAXIANSHAN, "soil_type": "unmapped_xyz"})
    granite = assess_safety_index("taiwan", rt_mm=100, wind_speed_ms=5, wbgt_c=24,
                                  **{**BAXIANSHAN, "soil_type": "granite"})
    assert unknown["composite_score"] > granite["composite_score"]


def test_result_reports_consistency_ratios():
    result = assess_safety_index("taiwan", rt_mm=100, wind_speed_ms=5, wbgt_c=24, **BAXIANSHAN)
    assert set(result["consistency_ratios"]) == {"groups", *GROUP_ORDER}
    assert all(cr <= MAX_CONSISTENCY_RATIO for cr in result["consistency_ratios"].values())
