"""
SylvaNexus — Forest Work Safety Criteria & Region Profiles
============================================================
Defines the criteria hierarchy for occupational safety in forest operations,
the region-specific AHP pairwise judgements, and how each raw measurement is
scored onto a common 1-5 risk scale.

Framework adapted from:
  Rahmawati, Yovi & Setiawan (2025) Advancing occupational safety in forest
  management through a new GIS-AHP integrated framework.
  European Journal of Forest Engineering, 11(2).

That study assessed 12 parameters in three groups (biophysical, climate,
threat) for Indonesian production forest. This module keeps that structure but:

  1. Adds a HYDRO group (TWI, flow accumulation, geomorphon, aspect). Taiwan's
     dominant fatal hazard is rainfall-triggered shallow landslide and debris
     flow, which is governed by hydrological convergence — not captured by the
     original biophysical group.
  2. Re-weights the groups per region. In Indonesian peat forest the THREAT
     group (wildlife, abnormal trees) carried the highest AHP weight. In Taiwan
     and Japan, typhoon rainfall dominates, so CLIMATE and HYDRO lead.
  3. Makes scoring thresholds region-specific (rainfall reference threshold,
     wildlife species, geology classes).

Criteria hierarchy:

  BIOPHYSICAL   slope, elevation, soil_type, vegetation_density, accessibility
  HYDRO         twi, flow_accumulation, geomorphon, aspect
  CLIMATE       rainfall, wind, heat
  THREAT        wildlife, abnormal_trees

Every parameter is scored 1 (negligible) to 5 (extreme) so that groups with
different units can be combined by weight.
"""

from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Criteria hierarchy — labels must match the pairwise matrix row order
# ---------------------------------------------------------------------------

CRITERIA_GROUPS: Dict[str, list] = {
    "biophysical": ["slope", "elevation", "soil_type", "vegetation_density", "accessibility"],
    "hydro": ["twi", "flow_accumulation", "geomorphon", "aspect"],
    "climate": ["rainfall", "wind", "heat"],
    "threat": ["wildlife", "abnormal_trees"],
}

GROUP_ORDER = ["biophysical", "hydro", "climate", "threat"]

GROUP_LABELS_ZH = {
    "biophysical": "生物物理環境",
    "hydro": "水文",
    "climate": "氣候",
    "threat": "現場威脅",
}

CRITERION_LABELS_ZH = {
    "slope": "坡度",
    "elevation": "海拔",
    "soil_type": "土壤／地質",
    "vegetation_density": "植被密度",
    "accessibility": "可及性",
    "twi": "地形濕潤指數",
    "flow_accumulation": "匯流面積",
    "geomorphon": "地形位",
    "aspect": "坡向",
    "rainfall": "降雨",
    "wind": "風速",
    "heat": "熱危害",
    "wildlife": "野生動物",
    "abnormal_trees": "異常樹木",
}


# ---------------------------------------------------------------------------
# Region profiles
# ---------------------------------------------------------------------------

class RegionProfile:
    """
    Region-specific AHP judgements and scoring references.

    `group_matrix` compares the four criteria groups against each other.
    `sub_matrices` compares the criteria within each group.
    Both use Saaty's 1-9 scale and must be reciprocal.
    """

    def __init__(self, code: str, name: str,
                 group_matrix: list,
                 sub_matrices: Dict[str, list],
                 rainfall_reference_mm: float,
                 rainfall_authority: str,
                 wildlife_hazards: Dict[str, int],
                 geology_scores: Dict[str, int],
                 rationale: str):
        self.code = code
        self.name = name
        self.group_matrix = group_matrix
        self.sub_matrices = sub_matrices
        self.rainfall_reference_mm = rainfall_reference_mm
        self.rainfall_authority = rainfall_authority
        self.wildlife_hazards = wildlife_hazards
        self.geology_scores = geology_scores
        self.rationale = rationale


# ── Taiwan ──────────────────────────────────────────────────────────────────
# Group priority: climate > hydro > biophysical > threat.
# Justification: Taiwan mountain forestry fatalities are dominated by typhoon
# rainfall triggering shallow landslide, debris flow and stream surge. Wildlife
# and standing-tree defects matter but cause far fewer fatal events than
# rainfall-driven ground failure.
# Row/column order: biophysical, hydro, climate, threat
_TAIWAN_GROUP_MATRIX = [
    [1,      1/2,   1/3,   3],      # biophysical
    [2,      1,     1/2,   3],      # hydro
    [3,      2,     1,     5],      # climate
    [1/3,    1/3,   1/5,   1],      # threat
]

_TAIWAN_SUB_MATRICES = {
    # slope, elevation, soil_type, vegetation_density, accessibility
    # Slope dominates shallow-landslide susceptibility; geology next (mudstone
    # and colluvium fail readily when wet); accessibility governs rescue time.
    "biophysical": [
        [1,     3,     2,     3,     2],     # slope
        [1/3,   1,     1/2,   1,     1/2],   # elevation
        [1/2,   2,     1,     2,     1],     # soil_type
        [1/3,   1,     1/2,   1,     1/2],   # vegetation_density
        [1/2,   2,     1,     2,     1],     # accessibility
    ],
    # twi, flow_accumulation, geomorphon, aspect
    # TWI is the primary convergence indicator; aspect is a weak modifier.
    "hydro": [
        [1,     2,     3,     5],     # twi
        [1/2,   1,     2,     4],     # flow_accumulation
        [1/3,   1/2,   1,     3],     # geomorphon
        [1/5,   1/4,   1/3,   1],     # aspect
    ],
    # rainfall, wind, heat
    # Rainfall is the trigger for the fatal hazards; wind causes treefall;
    # heat stress is debilitating but rarely fatal under closed canopy.
    "climate": [
        [1,     3,     5],     # rainfall
        [1/3,   1,     3],     # wind
        [1/5,   1/3,   1],     # heat
    ],
    # wildlife, abnormal_trees
    # Standing dead/leaning trees are the more common injury source in Taiwan
    # thinning operations; hornets are seasonal but locally severe.
    "threat": [
        [1,     1/2],   # wildlife
        [2,     1],     # abnormal_trees
    ],
}

# Taiwan geology → landslide susceptibility score (1-5)
# Ref: 經濟部中央地質調查所 1:50,000 地質圖; 土石流潛勢溪流地質分類
_TAIWAN_GEOLOGY = {
    "colluvium": 5,        # 崩積層 — 既有崩積物，最易再次滑動
    "alluvium": 4,         # 沖積層 — 疏鬆未固結
    "mudstone": 5,         # 泥岩 — 遇水軟化、泥岩惡地
    "shale": 4,            # 頁岩 — 易風化剝離
    "slate": 4,            # 板岩 — 葉理面順向坡滑動
    "schist": 3,           # 片岩 — 片理面弱面
    "sandstone": 3,        # 砂岩 — 中等
    "sandstone_shale": 4,  # 砂頁岩互層 — 互層弱面
    "limestone": 3,        # 石灰岩 — 溶蝕孔隙
    "gneiss": 2,           # 片麻岩 — 較堅硬
    "andesite": 2,         # 安山岩 — 火成岩，較穩定
    "granite": 1,          # 花崗岩 — 最穩定
    "unknown": 3,          # 無資料時取中間值，避免低估
}

# Taiwan wildlife hazards → severity score (1-5)
# Ref: 林業試驗所 林業作業安全; 疾管署毒蛇咬傷統計
_TAIWAN_WILDLIFE = {
    "none": 1,
    "leeches": 1,          # 螞蟥
    "wild_boar": 2,        # 野豬
    "macaque": 2,          # 台灣獼猴
    "venomous_snake": 4,   # 毒蛇（龜殼花、赤尾青竹絲）
    "hornets": 5,          # 蜂群（秋季虎頭蜂，致死率最高）
    "black_bear": 5,       # 台灣黑熊
}

TAIWAN = RegionProfile(
    code="taiwan",
    name="台灣",
    group_matrix=_TAIWAN_GROUP_MATRIX,
    sub_matrices=_TAIWAN_SUB_MATRICES,
    rainfall_reference_mm=350,
    rainfall_authority="SWCB 土石流警戒基準值",
    wildlife_hazards=_TAIWAN_WILDLIFE,
    geology_scores=_TAIWAN_GEOLOGY,
    rationale=(
        "颱風降雨為主導致災因子，故 climate 與 hydro 權重最高；"
        "threat（野生動物、異常樹木）致死事件相對少，權重最低。"
    ),
)


# ── Japan ───────────────────────────────────────────────────────────────────
# Similar hazard profile to Taiwan (typhoon + steep terrain), but hydro is
# raised because JMA operates a soil-water-index warning system at 1km mesh,
# and heavy snow adds a treefall/loading component to wind.
_JAPAN_GROUP_MATRIX = [
    [1,      1/2,   1/2,   3],      # biophysical
    [2,      1,     1,     4],      # hydro
    [2,      1,     1,     4],      # climate
    [1/3,    1/4,   1/4,   1],      # threat
]

_JAPAN_SUB_MATRICES = {
    "biophysical": _TAIWAN_SUB_MATRICES["biophysical"],
    "hydro": _TAIWAN_SUB_MATRICES["hydro"],
    # Wind weighted higher than Taiwan: snow loading plus winter storms are a
    # major cause of windthrow in Japanese plantation forest.
    "climate": [
        [1,     2,     5],     # rainfall
        [1/2,   1,     3],     # wind
        [1/5,   1/3,   1],     # heat
    ],
    # Bear encounters are a leading cause of serious forest injury in Japan.
    "threat": [
        [1,     1],     # wildlife
        [1,     1],     # abnormal_trees
    ],
}

_JAPAN_WILDLIFE = {
    "none": 1,
    "leeches": 1,
    "wild_boar": 3,        # イノシシ
    "macaque": 2,          # ニホンザル
    "venomous_snake": 3,   # マムシ
    "hornets": 5,          # スズメバチ — 年間死亡例最多
    "black_bear": 5,       # ツキノワグマ
    "brown_bear": 5,       # ヒグマ（北海道）
}

# Ref: 産業技術総合研究所 地質図Navi; 国土交通省 土砂災害警戒区域
_JAPAN_GEOLOGY = {
    "colluvium": 5,
    "alluvium": 4,
    "mudstone": 5,
    "shale": 4,
    "slate": 4,
    "schist": 3,
    "sandstone": 3,
    "sandstone_shale": 4,
    "limestone": 3,
    "gneiss": 2,
    "andesite": 2,
    "granite": 1,
    "volcanic_ash": 5,     # シラス台地・火山灰土 — 崩壊しやすい
    "unknown": 3,
}

JAPAN = RegionProfile(
    code="japan",
    name="日本",
    group_matrix=_JAPAN_GROUP_MATRIX,
    sub_matrices=_JAPAN_SUB_MATRICES,
    rainfall_reference_mm=300,
    rainfall_authority="JMA 土砂災害警戒情報 (土壌雨量指数)",
    wildlife_hazards=_JAPAN_WILDLIFE,
    geology_scores=_JAPAN_GEOLOGY,
    rationale=(
        "颱風與豪雨為主要致災因子，hydro 因 JMA 土壌雨量指数 1km 網格警戒制度"
        "而權重提高；冬季雪載與暴風使 wind 權重高於台灣；熊類遭遇為重大職災來源。"
    ),
)


# ── Indonesia ───────────────────────────────────────────────────────────────
# Follows the source paper's AHP result: the THREAT group carried the highest
# weight for Indonesian production forest, where wildlife encounters and
# abnormal tree conditions drive most recorded accidents, and terrain is far
# flatter than Taiwan or Japan.
_INDONESIA_GROUP_MATRIX = [
    [1,      2,     1,     1/3],    # biophysical
    [1/2,    1,     1/2,   1/5],    # hydro
    [1,      2,     1,     1/3],    # climate
    [3,      5,     3,     1],      # threat
]

_INDONESIA_SUB_MATRICES = {
    # Accessibility ranks higher: remote peat forest means very long evacuation
    # times, which the source study identified as a key vulnerability.
    "biophysical": [
        [1,     2,     2,     2,     1],     # slope
        [1/2,   1,     1,     1,     1/2],   # elevation
        [1/2,   1,     1,     1,     1/2],   # soil_type
        [1/2,   1,     1,     1,     1/2],   # vegetation_density
        [1,     2,     2,     2,     1],     # accessibility
    ],
    "hydro": _TAIWAN_SUB_MATRICES["hydro"],
    # Heat weighted far higher: equatorial lowland work without the temperate
    # canopy relief, and the source study flagged solar exposure explicitly.
    "climate": [
        [1,     2,     1],     # rainfall
        [1/2,   1,     1/2],   # wind
        [1,     2,     1],     # heat
    ],
    "threat": [
        [1,     2],     # wildlife
        [1/2,   1],     # abnormal_trees
    ],
}

_INDONESIA_WILDLIFE = {
    "none": 1,
    "leeches": 1,
    "wild_boar": 3,
    "macaque": 2,
    "venomous_snake": 5,   # king cobra, pit vipers
    "hornets": 4,
    "elephant": 5,         # Sumatran elephant
    "tiger": 5,            # Sumatran tiger
    "orangutan": 2,
    "crocodile": 5,
    "unknown": 3,
}

_INDONESIA_GEOLOGY = {
    "peat": 5,             # 泥炭地 — subsidence, fire, unstable footing
    "colluvium": 5,
    "alluvium": 4,
    "mudstone": 4,
    "shale": 4,
    "sandstone": 3,
    "limestone": 3,
    "volcanic_ash": 4,
    "andesite": 2,
    "granite": 1,
    "unknown": 3,
}

INDONESIA = RegionProfile(
    code="indonesia",
    name="Indonesia",
    group_matrix=_INDONESIA_GROUP_MATRIX,
    sub_matrices=_INDONESIA_SUB_MATRICES,
    rainfall_reference_mm=200,
    rainfall_authority="BMKG heavy rainfall advisory",
    wildlife_hazards=_INDONESIA_WILDLIFE,
    geology_scores=_INDONESIA_GEOLOGY,
    rationale=(
        "依原論文 AHP 結果，threat（野生動物、異常樹木）權重最高；"
        "地形較平緩故 hydro 權重最低；赤道低地熱危害顯著故 heat 權重高。"
    ),
)


REGION_PROFILES: Dict[str, RegionProfile] = {
    "taiwan": TAIWAN,
    "japan": JAPAN,
    "indonesia": INDONESIA,
}

DEFAULT_REGION = "taiwan"


def get_profile(region: Optional[str] = None) -> RegionProfile:
    """Look up a region profile, falling back to the default region."""
    if not region:
        return REGION_PROFILES[DEFAULT_REGION]
    profile = REGION_PROFILES.get(region.lower())
    if profile is None:
        raise ValueError(
            f"Unknown region '{region}'. "
            f"Available: {', '.join(sorted(REGION_PROFILES))}."
        )
    return profile
