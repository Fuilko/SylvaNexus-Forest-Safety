"""
SylvaNexus — Forest Work Safety Index (GIS-AHP)
=================================================
Scores every safety criterion onto a common 1-5 scale and combines them using
AHP-derived, region-specific weights into a composite Forest Work Safety Index.

Relationship to the trigger-based hazard assessment
---------------------------------------------------
These two mechanisms answer different questions and are deliberately kept apart:

  * `FloodRiskEngine.assess_landslide_risk` answers "must work stop right now?"
    It compares live measurements against official thresholds (SWCB rainfall,
    職安署 wind and WBGT limits) and drives notifications. A threshold breach is
    absolute — it cannot be averaged away by favourable conditions elsewhere.

  * This module answers "how hazardous is this work zone overall?" It is a
    weighted composite for planning, zoning and comparison between sites.

The composite index must NOT gate alerts. A permanently steep, remote site
would otherwise sit at a high index forever and desensitise its crew, while a
gentle site could average away a lethal rainfall reading. `dynamic_share` is
reported so users can see how much of the score is live weather versus fixed
terrain.

Scoring scale (applies to every criterion):
  1 = negligible, 2 = low, 3 = moderate, 4 = high, 5 = extreme

References:
  Rahmawati, Yovi & Setiawan (2025) Advancing occupational safety in forest
    management through a new GIS-AHP integrated framework. Eur. J. Forest Eng.
    11(2). — framework, criteria groups, 5-class output
  SWCB 水土保持技術規範 坡度分級 — slope classes
  Beven & Kirkby (1979) — TWI
  Jasiewicz & Stepinski (2013) Geomorphons — landform classification
  Sidle & Ochiai (2006) — root reinforcement / vegetation density
  勞動部職安署 高氣溫戶外作業熱危害預防指引 — WBGT classes
"""

from typing import Dict, Optional

from app.weather.ahp import weights_for
from app.weather.safety_criteria import (
    CRITERIA_GROUPS,
    CRITERION_LABELS_ZH,
    GROUP_LABELS_ZH,
    GROUP_ORDER,
    RegionProfile,
    get_profile,
)


# ---------------------------------------------------------------------------
# Composite index classification (5 classes, per the source framework)
# ---------------------------------------------------------------------------

INDEX_CLASSES = [
    (1.8, "very_low", "極低", "#2e7d32"),
    (2.6, "low", "低", "#7cb342"),
    (3.4, "moderate", "中", "#fbc02d"),
    (4.2, "high", "高", "#f57c00"),
    (float("inf"), "very_high", "極高", "#c62828"),
]

# Criteria whose value changes hour to hour. Everything else is fixed terrain
# or a slowly-changing site survey.
DYNAMIC_CRITERIA = {"rainfall", "wind", "heat"}


# ---------------------------------------------------------------------------
# Per-criterion scoring
# ---------------------------------------------------------------------------

def _banded_score(value: float, bands: list) -> int:
    """Score a value against ascending (upper_bound, score) bands."""
    for upper, score in bands:
        if value < upper:
            return score
    return bands[-1][1]


def score_slope(slope_deg: float) -> int:
    """SWCB slope classes: gentle <15°, to >45° which fails readily when wet."""
    return _banded_score(slope_deg, [(15, 1), (25, 2), (35, 3), (45, 4), (float("inf"), 5)])


def score_elevation(elevation_m: float) -> int:
    """Higher ground means colder, more exposed and further from rescue."""
    return _banded_score(elevation_m, [(500, 1), (1000, 2), (1500, 3), (2500, 4), (float("inf"), 5)])


def score_soil_type(soil_type: str, profile: RegionProfile) -> int:
    """Look up lithology susceptibility for the region; unknown is mid-scale."""
    key = (soil_type or "unknown").strip().lower()
    return profile.geology_scores.get(key, profile.geology_scores.get("unknown", 3))


def score_vegetation_density(ndvi: float) -> int:
    """
    Dense vegetation reinforces soil through root cohesion, so high NDVI is
    LOW risk. Ref: Sidle & Ochiai (2006).
    """
    if ndvi <= 0:
        return 3  # no reading — do not assume the site is well vegetated
    return _banded_score(-ndvi, [(-0.7, 1), (-0.6, 2), (-0.4, 3), (-0.2, 4), (float("inf"), 5)])


def score_accessibility(minutes_to_access: float) -> int:
    """Minutes to the nearest vehicle-accessible point — governs rescue time."""
    if minutes_to_access <= 0:
        return 3  # unsurveyed
    return _banded_score(minutes_to_access, [(10, 1), (20, 2), (40, 3), (60, 4), (float("inf"), 5)])


def score_twi(twi: float) -> int:
    """Topographic Wetness Index — above ~4.5 the zone concentrates runoff."""
    return _banded_score(twi, [(3, 1), (4, 2), (5, 3), (7, 4), (float("inf"), 5)])


def score_flow_accumulation(flow_accum: float) -> int:
    """Upslope contributing cells — proxy for stream surge potential."""
    return _banded_score(flow_accum, [(500, 1), (2000, 2), (5000, 3), (10000, 4), (float("inf"), 5)])


# GRASS r.geomorphon classes → landslide/debris-flow exposure
# 1 flat, 2 summit, 3 ridge, 4 shoulder, 5 spur, 6 slope,
# 7 hollow, 8 footslope, 9 valley, 10 depression
GEOMORPHON_SCORES = {
    1: 1,   # flat — stable
    2: 2,   # summit — exposed to wind, little convergence
    3: 2,   # ridge — sheds water
    4: 4,   # shoulder — convex break, shallow failure initiation
    5: 4,   # spur — shallow failure initiation
    6: 3,   # slope — general hillslope
    7: 5,   # hollow — convergent, saturates first
    8: 3,   # footslope — deposition
    9: 5,   # valley — debris flow path
    10: 4,  # depression — ponding
}


def score_geomorphon(geomorphon: int) -> int:
    """Landform position; convergent forms (hollow, valley) score highest."""
    return GEOMORPHON_SCORES.get(int(geomorphon or 0), 3)


def score_aspect(aspect_deg: float) -> int:
    """
    Aspect as a rainfall-exposure modifier.

    Taiwan and Japan typhoons approach from the east and southeast, so east and
    southeast facing slopes receive orographically enhanced rainfall. Aspect is
    the lowest-weighted hydro criterion, so this is a gentle modifier only.

    0 degrees is due north, which is a valid reading. Flat ground has no aspect
    and is reported as negative by GDAL, which is treated as unknown.
    """
    if aspect_deg is None or aspect_deg < 0:
        return 2
    octant = round(aspect_deg / 45) % 8  # 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
    return {0: 1, 1: 3, 2: 5, 3: 5, 4: 3, 5: 2, 6: 2, 7: 1}.get(octant, 2)


def score_rainfall(rt_mm: float, profile: RegionProfile) -> int:
    """
    Effective cumulative rainfall as a fraction of the region's official
    warning threshold. Reaching the threshold is by definition extreme.
    """
    reference = profile.rainfall_reference_mm
    if reference <= 0:
        return 1
    ratio = rt_mm / reference
    return _banded_score(ratio, [(0.3, 1), (0.5, 2), (0.75, 3), (1.0, 4), (float("inf"), 5)])


def score_wind(wind_ms: float) -> int:
    """Beaufort-aligned bands; 10.8 m/s suspends felling, 17.2 stops work."""
    return _banded_score(wind_ms, [(5.0, 1), (8.0, 2), (10.8, 3), (17.2, 4), (float("inf"), 5)])


def score_heat(wbgt_c: float) -> int:
    """職安署 WBGT classes for heavy outdoor work."""
    if wbgt_c <= 0:
        return 1
    return _banded_score(wbgt_c, [(25, 1), (28, 2), (30, 3), (32, 4), (float("inf"), 5)])


def score_wildlife(wildlife: str, profile: RegionProfile) -> int:
    """Region-specific wildlife encounter severity."""
    key = (wildlife or "none").strip().lower()
    return profile.wildlife_hazards.get(key, profile.wildlife_hazards.get("unknown", 2))


# Standing-tree defects, as surveyed on site or from UAV imagery
ABNORMAL_TREE_SCORES = {
    "none": 1,
    "minor_lean": 2,       # 輕微傾斜
    "severe_lean": 4,      # 明顯傾斜
    "dead_standing": 4,    # 枯立木
    "broken_top": 4,       # 斷梢
    "hanging_limb": 5,     # 懸垂枝（掛枝）— 最易致死
    "uprooted": 5,         # 已倒木／根盤翹起
    "unknown": 2,
}


def score_abnormal_trees(condition: str) -> int:
    """Standing-tree defect severity. Ref: FAO forest harvesting safety guides."""
    key = (condition or "unknown").strip().lower()
    return ABNORMAL_TREE_SCORES.get(key, 2)


# ---------------------------------------------------------------------------
# Weight derivation (cached per region — matrices are static)
# ---------------------------------------------------------------------------

_weight_cache: Dict[str, dict] = {}


def derive_region_weights(profile: RegionProfile) -> dict:
    """
    Derive group and global criterion weights for a region, with the
    consistency ratio of every matrix so the judgements stay auditable.

    Raises InconsistentMatrixError if any matrix exceeds CR 0.10.
    """
    if profile.code in _weight_cache:
        return _weight_cache[profile.code]

    group_weights, group_cr = weights_for(
        GROUP_ORDER, profile.group_matrix, f"{profile.code}.groups"
    )

    sub_weights: Dict[str, Dict[str, float]] = {}
    consistency: Dict[str, float] = {"groups": round(group_cr, 4)}
    global_weights: Dict[str, float] = {}

    for group in GROUP_ORDER:
        matrix = profile.sub_matrices[group]
        weights, cr = weights_for(
            CRITERIA_GROUPS[group], matrix, f"{profile.code}.{group}"
        )
        sub_weights[group] = weights
        consistency[group] = round(cr, 4)
        for criterion, weight in weights.items():
            global_weights[criterion] = group_weights[group] * weight

    result = {
        "group_weights": group_weights,
        "sub_weights": sub_weights,
        "global_weights": global_weights,
        "consistency_ratios": consistency,
    }
    _weight_cache[profile.code] = result
    return result


# ---------------------------------------------------------------------------
# Composite index
# ---------------------------------------------------------------------------

def classify_index(score: float) -> tuple:
    """Map a 1-5 composite score onto (class_code, label_zh, colour)."""
    for upper, code, label, colour in INDEX_CLASSES:
        if score < upper:
            return code, label, colour
    return INDEX_CLASSES[-1][1], INDEX_CLASSES[-1][2], INDEX_CLASSES[-1][3]


def assess_safety_index(
    region: Optional[str] = None,
    *,
    slope_deg: float = 0,
    elevation_m: float = 0,
    soil_type: str = "unknown",
    ndvi: float = 0,
    accessibility_minutes: float = 0,
    twi: float = 0,
    flow_accumulation: float = 0,
    geomorphon: int = 0,
    aspect_deg: float = 0,
    rt_mm: float = 0,
    wind_speed_ms: float = 0,
    wbgt_c: float = 0,
    wildlife: str = "none",
    abnormal_trees: str = "unknown",
) -> dict:
    """
    Compute the composite Forest Work Safety Index for a work zone.

    Returns the 1-5 composite score, its 5-class rating, a per-criterion
    breakdown (raw score, weight, weighted contribution), the group subtotals,
    the AHP consistency ratios, and `dynamic_share` — the fraction of the score
    contributed by live weather rather than fixed terrain.
    """
    profile = get_profile(region)
    weights = derive_region_weights(profile)

    scores = {
        "slope": score_slope(slope_deg),
        "elevation": score_elevation(elevation_m),
        "soil_type": score_soil_type(soil_type, profile),
        "vegetation_density": score_vegetation_density(ndvi),
        "accessibility": score_accessibility(accessibility_minutes),
        "twi": score_twi(twi),
        "flow_accumulation": score_flow_accumulation(flow_accumulation),
        "geomorphon": score_geomorphon(geomorphon),
        "aspect": score_aspect(aspect_deg),
        "rainfall": score_rainfall(rt_mm, profile),
        "wind": score_wind(wind_speed_ms),
        "heat": score_heat(wbgt_c),
        "wildlife": score_wildlife(wildlife, profile),
        "abnormal_trees": score_abnormal_trees(abnormal_trees),
    }

    global_weights = weights["global_weights"]
    composite = sum(scores[c] * global_weights[c] for c in scores)

    breakdown = {}
    for criterion, score in scores.items():
        weight = global_weights[criterion]
        breakdown[criterion] = {
            "label": CRITERION_LABELS_ZH.get(criterion, criterion),
            "score": score,
            "weight": round(weight, 4),
            "contribution": round(score * weight, 4),
            "dynamic": criterion in DYNAMIC_CRITERIA,
        }

    group_subtotals = {}
    for group in GROUP_ORDER:
        members = CRITERIA_GROUPS[group]
        contribution = sum(breakdown[c]["contribution"] for c in members)
        # Group score on the same 1-5 scale, independent of its group weight
        group_score = sum(
            scores[c] * weights["sub_weights"][group][c] for c in members
        )
        group_subtotals[group] = {
            "label": GROUP_LABELS_ZH[group],
            "weight": round(weights["group_weights"][group], 4),
            "score": round(group_score, 2),
            "contribution": round(contribution, 4),
        }

    dynamic_contribution = sum(
        breakdown[c]["contribution"] for c in DYNAMIC_CRITERIA
    )
    class_code, class_label, class_colour = classify_index(composite)

    # Static zoning score — terrain and site survey only, with the weights
    # renormalised over the non-weather criteria.
    #
    # This exists because the composite alone cannot serve as a zoning map.
    # Where climate carries most of the weight (Taiwan: 0.48), a site with the
    # worst possible terrain still scores only "moderate" on a calm day, so
    # comparing sites by composite on a dry day understates terrain danger.
    # The static score answers "how dangerous is this ground, whatever the
    # weather?" — which is what a work-zone map needs.
    static_criteria = [c for c in scores if c not in DYNAMIC_CRITERIA]
    static_weight_total = sum(global_weights[c] for c in static_criteria)
    static_score = (
        sum(scores[c] * global_weights[c] for c in static_criteria) / static_weight_total
        if static_weight_total else 0.0
    )
    static_class, static_label, static_colour = classify_index(static_score)

    # The highest-contributing criteria are what a supervisor should act on
    drivers = sorted(
        breakdown.items(), key=lambda kv: kv[1]["contribution"], reverse=True
    )[:3]

    return {
        "region": profile.code,
        "region_name": profile.name,
        "composite_score": round(composite, 2),
        "class": class_code,
        "class_label": class_label,
        "class_colour": class_colour,
        "static_score": round(static_score, 2),
        "static_class": static_class,
        "static_class_label": static_label,
        "static_class_colour": static_colour,
        "criteria": breakdown,
        "groups": group_subtotals,
        "dynamic_share": round(dynamic_contribution / composite, 3) if composite else 0.0,
        "top_drivers": [
            {"criterion": k, "label": v["label"], "score": v["score"],
             "contribution": v["contribution"]}
            for k, v in drivers
        ],
        "consistency_ratios": weights["consistency_ratios"],
        "rainfall_reference_mm": profile.rainfall_reference_mm,
        "rainfall_authority": profile.rainfall_authority,
        "method": (
            "GIS-AHP 複合風險指數（1-5 分，5 級分類）。權重由區域專家成對比較矩陣經 "
            "AHP 特徵向量法推導，一致性比率 CR<=0.10。"
            "此指數用於區位比較與作業規劃，不用於觸發警報。"
        ),
        "rationale": profile.rationale,
    }
