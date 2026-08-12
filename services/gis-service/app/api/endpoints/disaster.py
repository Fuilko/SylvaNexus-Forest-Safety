"""
防災預測模組 API
整合 DEM 地形、TWI 水文、森林狀態、氣象預報

風險公式:
  Risk = W_terrain × T(slope, geomorphon)
       + W_hydro   × H(TWI, catchment)
       + W_forest  × F(NDVI_change, canopy_gap)
       + W_weather × R(rainfall_mm, duration_hr)

References:
  [1] 土石流防災資訊網, 水土保持署 (SWCB), 24h累積雨量≥350mm或連續累積≥200mm為警戒基準
      https://246.swcb.gov.tw/
  [2] Central Weather Administration (CWA), 大雨特報標準: 24h≥50mm或1h≥40mm
  [3] Iida, T. (1999). A threshold of rainfall for shallow landslides.
      Bulletin of the Forestry and Forest Products Research Institute, 1(4), 117-126.
      — rainfall intensity-duration threshold: I = 14.7 × D^(-0.42) (mm/hr)
  [4] Chen, C.Y. et al. (2005). Rainfall thresholds for debris flow initiation.
      Geomorphology, 71(1-2), 37-47. — Taiwan shallow landslide thresholds.
  [5] Sidle, R.C. & Ochiai, H. (2006). Landslides: Processes, Prediction, and Land Use.
      AGU Water Resources Monograph 18. — slope×vegetation interaction framework.
  [6] Beven, K.J. & Kirkby, M.J. (1979). A physically based, variable contributing area
      model of basin hydrology. Hydrological Sciences Bulletin, 24(1), 43-69. — TWI.
"""

import json
import re
import os
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.modules.safety.safety_index import assess_safety_index, classify_index
from app.modules.safety.safety_criteria import get_profile, REGION_PROFILES

router = APIRouter()
logger = logging.getLogger(__name__)

DB_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:postgres@localhost:5433/SylvaNexus_Global"

# Global engine with connection pool (avoids per-request engine creation)
_engine = create_engine(DB_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
_Session = sessionmaker(bind=_engine)


def validate_project_id(project_id: str) -> str:
    """Validate project_id to prevent SQL injection via schema name."""
    if not re.match(r"^[a-zA-Z0-9_]+$", project_id):
        raise HTTPException(status_code=400, detail="Invalid project_id format")
    return project_id


class WeatherAssessmentRequest(BaseModel):
    """Pydantic schema for weather risk assessment input."""
    rainfall_mm: float = Field(0, description="累積雨量 (mm)")
    duration_hr: float = Field(1, description="延時 (hr)")
    bbox: Optional[list[float]] = Field(None, description="[xmin, ymin, xmax, ymax]")

# 預設風險權重
# Based on AHP (Analytic Hierarchy Process) calibration for Taiwan mountainous
# terrain, referencing Lin et al. (2017) "山坡地災害危險度評估" methodology.
# Terrain (slope) dominates shallow landslide initiation; hydro (TWI) second
# for saturation-driven failure; forest cover mitigates via root cohesion;
# weather is the triggering factor.
DEFAULT_WEIGHTS = {
    "terrain": 0.35,   # slope × geomorphon
    "hydro": 0.25,     # TWI × catchment
    "forest": 0.15,    # canopy height × NDVI
    "weather": 0.25    # rainfall intensity × duration
}

# ── Terrain risk: slope-based scoring ──
# Ref: SWCB 山坡地災害危險度評估 (slope classes)
#   <15°: low (stable), 15-30°: moderate, 30-45°: high, >45°: very high
# Ref: Sidle & Ochiai (2006) — shallow landslide frequency increases
#   sharply above 30°, peaks at 35-40°.
SLOPE_RISK_TABLE = [
    (15, 10),    # 0-15°  → score 10
    (30, 40),    # 15-30° → score 40
    (45, 75),    # 30-45° → score 75
    (999, 95),   # >45°   → score 95
]

# ── Forest risk: canopy height × NDVI ──
# Ref: Sidle & Ochiai (2006) Ch.4 — root cohesion from forest cover reduces
#   shallow landslide susceptibility. Canopy height <2m = minimal root cohesion.
#   NDVI <0.3 indicates bare/disturbed ground (USGS classification).
CANOPY_HEIGHT_RISK_TABLE = [
    (2, 85),     # <2m   → high risk (bare/disturbed)
    (5, 50),     # 2-5m  → moderate
    (10, 25),    # 5-10m → low-moderate
    (999, 10),   # >10m  → low risk (mature forest, strong root cohesion)
]
NDVI_RISK_TABLE = [
    (0.3, 80),   # <0.3  → high risk (bare ground)
    (0.5, 40),   # 0.3-0.5 → moderate (sparse vegetation)
    (0.7, 15),   # 0.5-0.7 → low (moderate vegetation)
    (999, 5),    # >0.7  → very low (dense vegetation)
]

# ── Weather risk: rainfall intensity thresholds ──
# Ref: SWCB debris-flow warning criteria [1]:
#   24h cumulative ≥ 350mm OR continuous cumulative ≥ 200mm
# Ref: CWA heavy rain advisory [2]: 24h ≥ 50mm or 1h ≥ 40mm
# Ref: Iida (1999) [3] intensity-duration threshold: I = 14.7 × D^(-0.42)
#   At D=1h: I≈14.7mm/hr; D=6h: I≈7.3mm/hr; D=24h: I≈4.1mm/hr
# We use a blended approach: short-duration (CWA) + long-duration (SWCB)
# to produce a 0-100 weather risk score.
# 100 mm/hr → extreme (exceeds all thresholds)
# 80 mm/hr  → very high (CWA extreme rain)
# 50 mm/hr  → high (CWA torrential rain)
# 30 mm/hr  → moderate (CWA heavy rain)
# 15 mm/hr  → low-moderate
# <15 mm/hr → low
WEATHER_INTENSITY_TABLE = [
    (80, 100),   # >80 mm/hr  → 100
    (50, 80),    # 50-80      → 80
    (30, 60),    # 30-50      → 60
    (15, 40),    # 15-30      → 40
    (0, 20),     # <15        → 20 (baseline)
]

# =============================================
# Schema
# =============================================

DISASTER_SCHEMA_SQL = """
-- 地形風險網格 (從 DEM 分析結果匯入)
CREATE TABLE IF NOT EXISTS {schema}.terrain_risk_grid (
    id SERIAL PRIMARY KEY,
    geometry GEOMETRY(Polygon, 4326),
    grid_size_m FLOAT DEFAULT 30,
    slope_deg FLOAT,
    aspect_deg FLOAT,
    geomorphon INT,             -- 1=flat, 3=ridge, 9=valley, etc.
    twi_value FLOAT,            -- Topographic Wetness Index
    catchment_area_m2 FLOAT,    -- 上游集水面積
    elevation_m FLOAT,
    ndvi FLOAT,
    canopy_height_m FLOAT,
    volume_m3_ha FLOAT,
    
    -- 舊版靜態風險分數 (0-100)
    static_risk_score FLOAT,
    risk_category VARCHAR(20),  -- 'low', 'medium', 'high', 'critical'

    -- GIS-AHP 複合安全指數 (1-5, 5級分類)
    ahp_static_score FLOAT,          -- 地形only, 用於分區規劃
    ahp_static_class VARCHAR(20),    -- very_low / low / moderate / high / very_high
    ahp_composite_score FLOAT,       -- 含天氣的動態指數 (運算時填入)
    ahp_composite_class VARCHAR(20),

    created_at TIMESTAMP DEFAULT NOW()
);

-- 傳感器佈設點
CREATE TABLE IF NOT EXISTS {schema}.sensor_locations (
    id SERIAL PRIMARY KEY,
    sensor_id VARCHAR(50) UNIQUE NOT NULL,
    geometry GEOMETRY(Point, 4326),
    sensor_type VARCHAR(50),        -- 'rain_gauge', 'soil_moisture', 'tiltmeter', 'camera'
    install_date DATE,
    status VARCHAR(20) DEFAULT 'planned',  -- 'planned', 'active', 'maintenance', 'offline'
    elevation_m FLOAT,
    risk_score_at_location FLOAT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 傳感器即時數據 (未來接 TimescaleDB 更佳)
CREATE TABLE IF NOT EXISTS {schema}.sensor_readings (
    id SERIAL PRIMARY KEY,
    sensor_id VARCHAR(50) REFERENCES {schema}.sensor_locations(sensor_id),
    timestamp TIMESTAMP NOT NULL,
    rainfall_mm FLOAT,
    soil_moisture_pct FLOAT,
    tilt_deg FLOAT,
    temperature_c FLOAT,
    battery_pct FLOAT,
    raw_payload BYTEA,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 氣象預報快取
CREATE TABLE IF NOT EXISTS {schema}.weather_forecasts (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50),             -- 'cwa', 'jma', 'open_meteo', 'earth2'
    forecast_time TIMESTAMP,
    target_time TIMESTAMP,
    geometry GEOMETRY(Point, 4326),
    rainfall_mm FLOAT,
    rainfall_duration_hr FLOAT,
    wind_speed_ms FLOAT,
    temperature_c FLOAT,
    alert_level VARCHAR(20),
    raw_json JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 風險評估歷史
CREATE TABLE IF NOT EXISTS {schema}.risk_assessments (
    id SERIAL PRIMARY KEY,
    assessment_time TIMESTAMP DEFAULT NOW(),
    geometry GEOMETRY(Polygon, 4326),
    area_ha FLOAT,
    
    -- 各維度分數
    terrain_score FLOAT,
    hydro_score FLOAT,
    forest_score FLOAT,
    weather_score FLOAT,
    
    -- 綜合風險
    total_risk_score FLOAT,
    risk_level VARCHAR(20),
    
    -- 觸發的天氣條件
    trigger_rainfall_mm FLOAT,
    trigger_source VARCHAR(50),
    
    -- 建議
    recommended_actions TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_terrain_risk_geom ON {schema}.terrain_risk_grid USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_sensor_loc_geom ON {schema}.sensor_locations USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_ts ON {schema}.sensor_readings (sensor_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_weather_target ON {schema}.weather_forecasts (target_time);
"""


@router.post("/{project_id}/disaster/init-schema")
async def init_disaster_schema(project_id: str = "baxianshan"):
    """初始化防災預測資料表（含 schema 建立）"""
    project_id = validate_project_id(project_id)
    try:
        with _engine.connect() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {project_id}"))
            conn.commit()
        sql = DISASTER_SCHEMA_SQL.format(schema=project_id)
        with _engine.connect() as conn:
            for statement in sql.split(';'):
                s = statement.strip()
                if s:
                    conn.execute(text(s))
            conn.commit()
        return {"status": "ok", "message": f"Disaster schema created for {project_id}"}
    except Exception as e:
        logger.error(f"❌ Disaster schema init failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================
# 靜態風險計算 (基於 DEM + 水文 + 森林)
# =============================================

@router.post("/{project_id}/disaster/compute-static-risk")
async def compute_static_risk(
    project_id: str = "baxianshan",
    region: str = Query("taiwan", description="AHP region for safety index computation"),
):
    """
    從已有的 GEE 材積網格計算靜態風險分數 + GIS-AHP 安全指數。

    兩套分數並存：
      1. 舊版 static_risk_score (0-100) — 相容既有前端
      2. AHP ahp_static_score (1-5, 5級) — Rahmawati et al. (2025) 框架

    AHP 分數從 slope / elevation / ndvi / canopy_height 等網格值推導，
    使用區域特定 AHP 權重加權。
    """
    project_id = validate_project_id(project_id)
    try:
        # Step 1: 舊版 SQL-based 靜態風險分數
        query = text(f"""
            WITH computed_scores AS (
                SELECT 
                    geometry, height, volume_m3_ha, param_p, slope_deg,
                    (
                        CASE 
                            WHEN slope_deg < 15 THEN 10
                            WHEN slope_deg < 30 THEN 40
                            WHEN slope_deg < 45 THEN 75
                            ELSE 95
                        END * {DEFAULT_WEIGHTS['terrain']}
                        + CASE 
                            WHEN height < 2 THEN 85
                            WHEN height < 5 THEN 50
                            WHEN height < 10 THEN 25
                            ELSE 10
                        END * {DEFAULT_WEIGHTS['forest']}
                        + CASE 
                            WHEN slope_deg < 10 THEN 70
                            WHEN slope_deg < 20 THEN 45
                            WHEN slope_deg < 35 THEN 25
                            ELSE 10
                        END * {DEFAULT_WEIGHTS['hydro']}
                    ) AS calc_risk
                FROM {project_id}.gee_volume_30m
                WHERE slope_deg IS NOT NULL
            )
            INSERT INTO {project_id}.terrain_risk_grid 
                (geometry, grid_size_m, canopy_height_m, volume_m3_ha, ndvi,
                 slope_deg, static_risk_score, risk_category)
            SELECT 
                geometry, 30, height, volume_m3_ha, param_p, slope_deg, calc_risk,
                CASE 
                    WHEN calc_risk >= 70 THEN 'critical'
                    WHEN calc_risk >= 50 THEN 'high'
                    WHEN calc_risk >= 30 THEN 'medium'
                    ELSE 'low'
                END
            FROM computed_scores
            ON CONFLICT DO NOTHING
        """)

        with _engine.connect() as conn:
            result = conn.execute(query)
            conn.commit()
            count = result.rowcount

        # Step 2: AHP 靜態安全指數 (Python-side, per-grid-cell)
        fetch_query = text(f"""
            SELECT id, slope_deg, aspect_deg, geomorphon, twi_value,
                   catchment_area_m2, elevation_m, ndvi, canopy_height_m
            FROM {project_id}.terrain_risk_grid
            WHERE slope_deg IS NOT NULL
        """)

        with _engine.connect() as conn:
            rows = conn.execute(fetch_query).fetchall()

        ahp_count = 0
        if rows:
            update_sql = text("""
                UPDATE {schema}.terrain_risk_grid
                SET ahp_static_score = :score,
                    ahp_static_class = :cls
                WHERE id = :id
            """.format(schema=project_id))

            batch = []
            for row in rows:
                grid_id = row[0]
                slope = float(row[1] or 0)
                aspect = float(row[2] or 0)
                geomorphon = int(row[3] or 0)
                twi = float(row[4] or 0)
                catchment_m2 = float(row[5] or 0)
                elevation = float(row[6] or 0)
                ndvi = float(row[7] or 0)
                canopy_height = float(row[8] or 0)
                flow_accum = catchment_m2 / 900.0 if catchment_m2 > 0 else 0

                ahp = assess_safety_index(
                    region,
                    slope_deg=slope,
                    elevation_m=elevation,
                    ndvi=ndvi,
                    twi=twi,
                    flow_accumulation=flow_accum,
                    geomorphon=geomorphon,
                    aspect_deg=aspect,
                )
                batch.append({
                    "id": grid_id,
                    "score": ahp["static_score"],
                    "cls": ahp["static_class"],
                })

            with _engine.begin() as conn:
                for b in batch:
                    conn.execute(update_sql, b)
                ahp_count = len(batch)

        return {
            "status": "ok",
            "grids_processed": count,
            "ahp_grids_scored": ahp_count,
            "region": region,
            "method": "舊版(0-100) + GIS-AHP(1-5, 5級) 雙軌計算",
            "references": [
                "SWCB 山坡地災害危險度評估 (slope classes)",
                "Sidle & Ochiai (2006) AGU Monograph 18 (root cohesion)",
                "Beven & Kirkby (1979) TWI",
                "Rahmawati, Yovi & Setiawan (2025) GIS-AHP framework",
            ],
            "weights": DEFAULT_WEIGHTS,
            "message": (
                f"靜態風險已計算 ({count} grids, 舊版 0-100) + "
                f"AHP 安全指數 ({ahp_count} grids, 1-5 五級, region={region})"
            ),
        }
    except Exception as e:
        logger.error(f"❌ Static risk compute failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================
# 傳感器佈設建議
# =============================================

@router.get("/{project_id}/disaster/sensor-suggestions")
async def suggest_sensor_locations(
    project_id: str = "baxianshan",
    top_n: int = Query(20, description="建議數量"),
    min_risk: float = Query(60.0, description="最低風險分數")
):
    """
    根據靜態風險分數推薦傳感器佈設位置
    優先選擇：高風險 + 交通可達 + 分布均勻
    """
    project_id = validate_project_id(project_id)
    try:
        query = text(f"""
            SELECT 
                ST_AsGeoJSON(ST_Centroid(geometry)) as point_geojson,
                static_risk_score,
                risk_category,
                canopy_height_m,
                ndvi as param_p,
                elevation_m,
                slope_deg,
                twi_value
            FROM {project_id}.terrain_risk_grid
            WHERE static_risk_score >= :min_risk
            ORDER BY static_risk_score DESC
            LIMIT :top_n
        """)
        
        with _engine.connect() as conn:
            rows = conn.execute(query, {"min_risk": min_risk, "top_n": top_n}).fetchall()
        
        suggestions = []
        for i, row in enumerate(rows):
            point = json.loads(row[0])
            lon, lat = point['coordinates']
            
            # 建議感測器類型
            sensor_types = ["rain_gauge"]  # 所有點都需要雨量計
            if row[7] and row[7] > 6:  # 高 TWI
                sensor_types.append("soil_moisture")
            if row[6] and row[6] > 30:  # 陡坡
                sensor_types.append("tiltmeter")
            if row[1] >= 80:  # 極高風險
                sensor_types.append("camera")
            
            suggestions.append({
                "rank": i + 1,
                "lon": round(lon, 6),
                "lat": round(lat, 6),
                "risk_score": round(row[1], 1),
                "risk_category": row[2],
                "canopy_height_m": round(row[3], 1) if row[3] else None,
                "recommended_sensors": sensor_types,
                "reason": (
                    "裸露地/極低植被 - 崩塌高風險" if row[3] and row[3] <= 1 else
                    "低矮植被 - 沖蝕風險" if row[3] and row[3] <= 3 else
                    "中等植被 - 監控區"
                )
            })
        
        return {
            "project_id": project_id,
            "total_suggestions": len(suggestions),
            "min_risk_threshold": min_risk,
            "suggestions": suggestions
        }
        
    except Exception as e:
        logger.error(f"❌ Sensor suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================
# 即時風險評估 (加入氣象)
# =============================================

@router.post("/{project_id}/disaster/assess-risk")
async def assess_risk_with_weather(
    project_id: str = "baxianshan",
    params: WeatherAssessmentRequest = None,
):
    """
    即時風險評估：靜態風險 + 氣象條件
    
    Example body:
    {
        "rainfall_mm": 150,
        "duration_hr": 6,
        "bbox": [121.005, 24.131, 121.033, 24.161]
    }
    """
    project_id = validate_project_id(project_id)
    try:
        rainfall = params.rainfall_mm if params else 0
        duration = params.duration_hr if params else 1
        bbox = params.bbox if params else None
        
        # 降雨強度風險分數 (0-100)
        # Ref: CWA 大雨特報 24h≥50mm / 1h≥40mm [2]
        # Ref: SWCB 土石流警戒 24h≥350mm [1]
        # Ref: Iida (1999) intensity-duration threshold I=14.7×D^(-0.42) [3]
        intensity = rainfall / max(duration, 0.5)  # mm/hr
        # Also compute cumulative rainfall for SWCB criteria
        cumulative_24h = rainfall if duration <= 24 else rainfall * (24 / duration)
        swcb_threshold = 350  # mm/24h
        cwa_threshold = 50    # mm/24h
        
        # Intensity-based score (0-100)
        if intensity > 80:
            weather_score = 100
        elif intensity > 50:
            weather_score = 80
        elif intensity > 30:
            weather_score = 60
        elif intensity > 15:
            weather_score = 40
        else:
            weather_score = max(intensity / 15 * 40, 0)
        
        # Boost score if SWCB cumulative threshold is approached/exceeded
        if cumulative_24h >= swcb_threshold:
            weather_score = max(weather_score, 100)
        elif cumulative_24h >= swcb_threshold * 0.7:
            weather_score = max(weather_score, 80)
        elif cumulative_24h >= cwa_threshold:
            weather_score = max(weather_score, 50)
        
        # 查詢區域內的靜態風險
        bbox_filter = ""
        params_dict = {"weather_w": DEFAULT_WEIGHTS["weather"], "ws": weather_score}
        
        if bbox and len(bbox) == 4:
            bbox_filter = """
                AND ST_Intersects(geometry, 
                    ST_MakeEnvelope(:xmin, :ymin, :xmax, :ymax, 4326))
            """
            params_dict.update({"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3]})
        
        query = text(f"""
            SELECT 
                risk_category,
                COUNT(*) as grid_count,
                AVG(static_risk_score) as avg_static,
                AVG(static_risk_score * (1 - :weather_w) + :ws * :weather_w) as avg_combined
            FROM {project_id}.terrain_risk_grid
            WHERE static_risk_score IS NOT NULL
            {bbox_filter}
            GROUP BY risk_category
            ORDER BY avg_combined DESC
        """)
        
        with _engine.connect() as conn:
            rows = conn.execute(query, params_dict).fetchall()
        
        categories = {}
        total_grids = 0
        for row in rows:
            categories[row[0]] = {
                "grid_count": int(row[1]),
                "avg_static_risk": round(float(row[2]), 1),
                "avg_combined_risk": round(float(row[3]), 1)
            }
            total_grids += int(row[1])
        
        # 整體風險等級
        critical_pct = categories.get("critical", {}).get("grid_count", 0) / max(total_grids, 1) * 100
        high_pct = categories.get("high", {}).get("grid_count", 0) / max(total_grids, 1) * 100
        
        if critical_pct > 20 or weather_score >= 80:
            overall = "🔴 極高風險 - 建議立即警戒"
        elif critical_pct > 10 or high_pct > 30 or weather_score >= 60:
            overall = "🟠 高風險 - 加強監控"
        elif weather_score >= 40:
            overall = "🟡 中等風險 - 注意觀測"
        else:
            overall = "🟢 低風險 - 正常監控"
        
        return {
            "assessment_time": datetime.now().isoformat(),
            "weather_input": {
                "rainfall_mm": rainfall,
                "duration_hr": duration,
                "intensity_mm_hr": round(intensity, 1),
                "cumulative_24h_est": round(cumulative_24h, 1),
                "swcb_threshold_24h": swcb_threshold,
                "cwa_threshold_24h": cwa_threshold,
                "weather_risk_score": round(weather_score, 1)
            },
            "risk_weights": DEFAULT_WEIGHTS,
            "by_category": categories,
            "total_grids": total_grids,
            "overall_assessment": overall,
            "references": [
                "SWCB 土石流警戒基準: 24h≥350mm",
                "CWA 大雨特報: 24h≥50mm或1h≥40mm",
                "Iida (1999) I-D threshold: I=14.7×D^(-0.42)",
            ],
            "recommendation": (
                "建議啟動傳感器高頻採集模式 (5分鐘間隔)" if weather_score >= 60 else
                "維持標準採集頻率 (30分鐘間隔)"
            )
        }
        
    except Exception as e:
        logger.error(f"❌ Risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================
# GIS-AHP 複合安全指數 — 分級安全地圖
# =============================================

@router.get("/{project_id}/disaster/safety-index-map")
async def safety_index_map(
    project_id: str = "baxianshan",
    region: str = Query("taiwan", description="AHP region: taiwan / japan / indonesia"),
    bbox: Optional[str] = Query(None, description="xmin,ymin,xmax,ymax (EPSG:4326)"),
    max_features: int = Query(5000, description="Maximum number of grid cells to return"),
    include_composite: bool = Query(False, description="Include composite (weather-dependent) score; false = static only"),
    rt_mm: float = Query(0, description="Effective cumulative rainfall (mm) for composite score"),
    wind_speed_ms: float = Query(0, description="Wind speed (m/s) for composite score"),
    wbgt_c: float = Query(0, description="WBGT (°C) for composite score"),
):
    """
    計算每個網格的 GIS-AHP 複合安全指數，輸出分級安全地圖 (GeoJSON)。

    這是 Rahmawati, Yovi & Setiawan (2025) 框架的核心產出：
    將林區網格化，每個像素以 AHP 權重加權 14 項安全準則，
    輸出 1-5 分的 5 級安全等級。

    預設回傳 static_score（地形only），可用於作業規劃分區。
    設 include_composite=true 加上即時天氣，得到動態複合指數。

    Returns GeoJSON FeatureCollection，每個 Feature 是一個網格 cell，
    properties 包含 composite_score / static_score / class / 各準則分數。
    """
    project_id = validate_project_id(project_id)
    try:
        bbox_filter = ""
        params = {"limit": max_features}

        if bbox:
            parts = [float(x) for x in bbox.split(",")]
            if len(parts) == 4:
                bbox_filter = """AND ST_Intersects(geometry,
                    ST_MakeEnvelope(:xmin, :ymin, :xmax, :ymax, 4326))"""
                params.update({"xmin": parts[0], "ymin": parts[1], "xmax": parts[2], "ymax": parts[3]})

        query = text(f"""
            SELECT
                ST_AsGeoJSON(geometry) as geojson,
                slope_deg, aspect_deg, geomorphon, twi_value,
                catchment_area_m2, elevation_m, ndvi, canopy_height_m
            FROM {project_id}.terrain_risk_grid
            WHERE slope_deg IS NOT NULL
            {bbox_filter}
            ORDER BY static_risk_score DESC NULLS LAST
            LIMIT :limit
        """)

        with _engine.connect() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return {
                "type": "FeatureCollection",
                "features": [],
                "metadata": {
                    "project_id": project_id,
                    "region": region,
                    "total_cells": 0,
                    "message": "No terrain_risk_grid data found. Run /disaster/compute-static-risk first.",
                },
            }

        profile = get_profile(region)
        features = []
        class_counts = {"very_low": 0, "low": 0, "moderate": 0, "high": 0, "very_high": 0}
        static_class_counts = {"very_low": 0, "low": 0, "moderate": 0, "high": 0, "very_high": 0}

        for row in rows:
            geojson = json.loads(row[0])
            slope = float(row[1] or 0)
            aspect = float(row[2] or 0)
            geomorphon = int(row[3] or 0)
            twi = float(row[4] or 0)
            catchment_m2 = float(row[5] or 0)
            elevation = float(row[6] or 0)
            ndvi = float(row[7] or 0)
            canopy_height = float(row[8] or 0)

            # flow_accumulation proxy: catchment_area_m2 / 900 (30m cell area)
            flow_accum = catchment_m2 / 900.0 if catchment_m2 > 0 else 0

            result = assess_safety_index(
                region,
                slope_deg=slope,
                elevation_m=elevation,
                soil_type="unknown",
                ndvi=ndvi,
                accessibility_minutes=0,
                twi=twi,
                flow_accumulation=flow_accum,
                geomorphon=geomorphon,
                aspect_deg=aspect,
                rt_mm=rt_mm if include_composite else 0,
                wind_speed_ms=wind_speed_ms if include_composite else 0,
                wbgt_c=wbgt_c if include_composite else 0,
            )

            props = {
                "static_score": result["static_score"],
                "static_class": result["static_class"],
                "static_class_label": result["static_class_label"],
                "static_class_colour": result["static_class_colour"],
                "slope_deg": round(slope, 1),
                "twi": round(twi, 2),
                "geomorphon": geomorphon,
                "elevation_m": round(elevation, 0),
                "ndvi": round(ndvi, 3) if ndvi else 0,
                "canopy_height_m": round(canopy_height, 1) if canopy_height else 0,
            }

            if include_composite:
                props["composite_score"] = result["composite_score"]
                props["class"] = result["class"]
                props["class_label"] = result["class_label"]
                props["class_colour"] = result["class_colour"]
                props["dynamic_share"] = result["dynamic_share"]
                class_counts[result["class"]] = class_counts.get(result["class"], 0) + 1

            static_class_counts[result["static_class"]] = static_class_counts.get(result["static_class"], 0) + 1

            features.append({
                "type": "Feature",
                "geometry": geojson,
                "properties": props,
            })

        metadata = {
            "project_id": project_id,
            "region": region,
            "region_name": profile.name,
            "total_cells": len(features),
            "method": "GIS-AHP 複合安全指數 (Rahmawati et al. 2025)",
            "scoring_scale": "1=negligible, 2=low, 3=moderate, 4=high, 5=extreme",
            "static_class_distribution": static_class_counts,
            "rainfall_reference_mm": profile.rainfall_reference_mm,
            "rainfall_authority": profile.rainfall_authority,
            "rationale": profile.rationale,
            "consistency_ratios": result.get("consistency_ratios", {}),
        }
        if include_composite:
            metadata["composite_class_distribution"] = class_counts
            metadata["weather_input"] = {"rt_mm": rt_mm, "wind_speed_ms": wind_speed_ms, "wbgt_c": wbgt_c}

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": metadata,
        }

    except Exception as e:
        logger.error(f"❌ Safety index map failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================
# 地形剖面查詢 — 供 backend 自動取得真實地形參數
# =============================================

@router.get("/{project_id}/disaster/terrain-profile")
async def terrain_profile(
    project_id: str = "baxianshan",
    lon: float = Query(..., description="Longitude (EPSG:4326)"),
    lat: float = Query(..., description="Latitude (EPSG:4326)"),
    buffer_m: float = Query(30, description="Search radius (m)"),
):
    """
    查詢指定座標點的地形參數，供 backend 的 AHP 安全指數引擎使用。

    回傳該點最近網格的 slope / aspect / geomorphon / twi / elevation / ndvi /
    canopy_height / flow_accumulation，以及一個建議的 soil_type（從地質圖
    查詢，若無則回傳 unknown）。
    """
    project_id = validate_project_id(project_id)
    try:
        query = text(f"""
            SELECT
                slope_deg, aspect_deg, geomorphon, twi_value,
                catchment_area_m2, elevation_m, ndvi, canopy_height_m,
                ST_Distance(geometry, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)) as dist
            FROM {project_id}.terrain_risk_grid
            WHERE slope_deg IS NOT NULL
                AND ST_DWithin(geometry,
                    ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                    :buffer_deg)
            ORDER BY dist ASC
            LIMIT 1
        """)

        # buffer_m → degrees (approx 1deg ≈ 111km)
        buffer_deg = buffer_m / 111000.0

        with _engine.connect() as conn:
            row = conn.execute(query, {"lon": lon, "lat": lat, "buffer_deg": buffer_deg}).fetchone()

        if not row:
            return {
                "project_id": project_id,
                "lon": lon,
                "lat": lat,
                "found": False,
                "message": "No terrain grid data within buffer. Run /disaster/compute-static-risk first.",
            }

        slope = float(row[0] or 0)
        aspect = float(row[1] or 0)
        geomorphon = int(row[2] or 0)
        twi = float(row[3] or 0)
        catchment_m2 = float(row[4] or 0)
        elevation = float(row[5] or 0)
        ndvi = float(row[6] or 0)
        canopy_height = float(row[7] or 0)
        dist_deg = float(row[8])
        dist_m = dist_deg * 111000.0

        flow_accum = catchment_m2 / 900.0 if catchment_m2 > 0 else 0

        return {
            "project_id": project_id,
            "lon": lon,
            "lat": lat,
            "found": True,
            "match_distance_m": round(dist_m, 1),
            "terrain": {
                "slope_deg": round(slope, 1),
                "aspect_deg": round(aspect, 1),
                "geomorphon": geomorphon,
                "twi": round(twi, 2),
                "flow_accumulation": round(flow_accum, 0),
                "elevation_m": round(elevation, 0),
                "ndvi": round(ndvi, 3) if ndvi else 0,
                "canopy_height_m": round(canopy_height, 1) if canopy_height else 0,
                "soil_type": "unknown",
            },
            "note": "soil_type defaults to unknown; integrate geology map layer for lithology lookup.",
        }

    except Exception as e:
        logger.error(f"❌ Terrain profile query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================
# AHP 權重審計 — 供前端與專家查閱
# =============================================

@router.get("/{project_id}/disaster/ahp-weights")
async def ahp_weights_audit(
    project_id: str = "baxianshan",
    region: str = Query("taiwan", description="AHP region: taiwan / japan / indonesia"),
):
    """
    公開 AHP 權重、一致性比率與評分詞彙，供審計與專家review。
    """
    project_id = validate_project_id(project_id)
    try:
        profile = get_profile(region)
        result = assess_safety_index(region)

        return {
            "project_id": project_id,
            "region": profile.code,
            "region_name": profile.name,
            "rationale": profile.rationale,
            "group_weights": {k: round(v, 4) for k, v in result["consistency_ratios"].items() if k == "groups"},
            "groups": {
                g: {
                    "label": result["groups"][g]["label"],
                    "weight": result["groups"][g]["weight"],
                    "score": result["groups"][g]["score"],
                    "contribution": result["groups"][g]["contribution"],
                }
                for g in result["groups"]
            },
            "criteria": {
                c: {
                    "label": v["label"],
                    "score": v["score"],
                    "weight": v["weight"],
                    "contribution": v["contribution"],
                    "dynamic": v["dynamic"],
                }
                for c, v in result["criteria"].items()
            },
            "consistency_ratios": result["consistency_ratios"],
            "rainfall_reference_mm": profile.rainfall_reference_mm,
            "rainfall_authority": profile.rainfall_authority,
            "geology_scores": profile.geology_scores,
            "wildlife_hazards": profile.wildlife_hazards,
            "available_regions": list(REGION_PROFILES.keys()),
        }
    except Exception as e:
        logger.error(f"❌ AHP weights audit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
