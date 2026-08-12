"""
SylvaNexus — Weather & Flood Risk Router
==========================================
Endpoints for weather data and flood risk assessment.
Taiwan → CWA, Japan → JMA, auto-detected by project region.
"""

from datetime import datetime, timezone
from typing import Optional
import os
import httpx
import logging

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, EmailStr
from app.weather.providers import (
    JMAProvider, CWAProvider, FloodRiskEngine,
    WeatherForecast, FloodRiskAssessment,
    CWARainFilter, CWAWeatherFilter, BAXIANSHAN_STATIONS
)
from app.weather.ahp import MAX_CONSISTENCY_RATIO
from app.weather.safety_criteria import (
    CRITERIA_GROUPS, CRITERION_LABELS_ZH, REGION_PROFILES, get_profile
)
from app.weather.safety_index import (
    ABNORMAL_TREE_SCORES, DYNAMIC_CRITERIA, INDEX_CLASSES, derive_region_weights
)

logger = logging.getLogger(__name__)
_GIS_SERVICE_URL = os.getenv("GIS_SERVICE_URL", "http://gis-service:8000")

router = APIRouter(prefix="/weather", tags=["weather"])


# ---------------------------------------------------------------------------
# Region detection
# ---------------------------------------------------------------------------

REGION_COUNTRY = {
    "baxianshan": "tw",
    "taiwan": "tw",
    "kochi": "jp",
    "shimanto": "jp",
    "japan": "jp",
}


def detect_country(project_id: str) -> str:
    """Detect country from project ID."""
    for key, country in REGION_COUNTRY.items():
        if key in project_id.lower():
            return country
    return "jp"  # default


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

@router.get("/forecast", summary="Get weather forecast for project region")
async def get_forecast(
    project_id: str = Query(..., description="Project ID (e.g. baxianshan, kochi)"),
    location: Optional[str] = Query(None, description="Specific location name"),
):
    """
    Get weather forecast. Auto-detects provider based on project region.

    Sources:
    - Japan: 気象庁 (JMA) https://www.jma.go.jp/
    - Taiwan: 中央氣象署 (CWA) https://opendata.cwa.gov.tw/
    """
    country = detect_country(project_id)

    if country == "jp":
        region_code = JMAProvider.REGION_CODES.get(project_id, "390000")
        data = await JMAProvider.get_forecast(region_code)
        return {
            "provider": "JMA",
            "source": JMAProvider.SOURCE,
            "source_url": JMAProvider.SOURCE_URL,
            "country": "jp",
            "data": data,
        }
    else:
        loc = location or "臺中市"  # Baxianshan is in Taichung
        data = await CWAProvider.get_forecast_36h(location=loc)
        return {
            "provider": "CWA",
            "source": CWAProvider.SOURCE,
            "source_url": CWAProvider.SOURCE_URL,
            "country": "tw",
            "data": data,
        }


# ---------------------------------------------------------------------------
# Rain observation
# ---------------------------------------------------------------------------

@router.get("/rain", summary="Get current rainfall observations")
async def get_rain(
    project_id: str = Query(..., description="Project ID"),
):
    """
    Get current rain gauge observations near the project area.

    Sources:
    - Japan: 気象庁 AMeDAS
    - Taiwan: 中央氣象署 自動雨量站
    """
    country = detect_country(project_id)

    if country == "jp":
        data = await JMAProvider.get_amedas_latest()
        return {
            "provider": "JMA_AMeDAS",
            "source": "気象庁 AMeDAS (Automated Meteorological Data Acquisition System)",
            "source_url": "https://www.jma.go.jp/bosai/amedas/",
            "data": data,
        }
    else:
        data = await CWAProvider.get_rain_observation()
        return {
            "provider": "CWA_RainStation",
            "source": "中央氣象署 自動雨量站",
            "source_url": "https://opendata.cwa.gov.tw/",
            "data": data,
        }


# ---------------------------------------------------------------------------
# Flood risk assessment
# ---------------------------------------------------------------------------

@router.get("/flood-risk", response_model=FloodRiskAssessment,
            summary="Assess flood risk for project area")
async def assess_flood_risk(
    project_id: str = Query(...),
    rainfall_forecast_mm: float = Query(0, description="Forecast rainfall (mm/24h)"),
    twi_max: float = Query(0, description="Maximum TWI value in area"),
    lat: float = Query(0),
    lng: float = Query(0),
):
    """
    Assess flood risk by combining weather forecast with terrain hydrology.

    Method:
    - Rainfall forecast from JMA/CWA
    - TWI (Topographic Wetness Index) from DEM analysis
    - Flow accumulation from D8 flow routing
    - Rainfall-runoff threshold comparison

    References:
    - TWI: Beven & Kirkby (1979)
    - Flow accumulation: O'Callaghan & Mark (1984)
    """
    country = detect_country(project_id)
    region = "jp_" + project_id if country == "jp" else "tw_" + project_id

    # Check for active warnings
    has_warning = False
    if country == "jp":
        region_code = JMAProvider.REGION_CODES.get(project_id, "390000")
        try:
            warnings = await JMAProvider.get_rain_warning(region_code)
            # JMA returns a dict even when no warnings — check for actual warning entries
            if warnings and isinstance(warnings, dict):
                area_warnings = warnings.get("areaTypes", [])
                for area in area_warnings:
                    for area_entry in area.get("areas", []):
                        for w in area_entry.get("warnings", []):
                            if w.get("status") == "発表" or w.get("status") == "継続":
                                has_warning = True
                                break
        except Exception:
            has_warning = False

    # Auto-fetch forecast if not provided
    if rainfall_forecast_mm <= 0:
        if country == "jp":
            region_code = JMAProvider.REGION_CODES.get(project_id, "390000")
            forecast = await JMAProvider.get_forecast(region_code)
            # Parse rainfall from forecast (simplified)
            rainfall_forecast_mm = 0  # Would parse from forecast data
        else:
            forecast = await CWAProvider.get_forecast_36h()
            rainfall_forecast_mm = 0

    risk_level = FloodRiskEngine.assess_risk(
        rainfall_forecast_mm, twi_max, has_warning
    )
    warning_msg = FloodRiskEngine.generate_warning(
        risk_level, rainfall_forecast_mm, region
    )

    threshold = FloodRiskEngine.THRESHOLDS.get(
        "medium" if twi_max > 4.5 else "high", 100
    )

    return FloodRiskAssessment(
        location_name=project_id,
        lat=lat,
        lng=lng,
        risk_level=risk_level,
        rainfall_forecast_mm=rainfall_forecast_mm,
        rainfall_threshold_mm=threshold,
        twi_max=twi_max,
        warning_message=warning_msg,
        source=f"{'JMA' if country == 'jp' else 'CWA'} forecast + DEM TWI analysis",
        assessment_time=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

@router.get("/warnings", summary="Get active weather warnings")
async def get_warnings(
    project_id: str = Query(...),
):
    """
    Get active weather warnings (heavy rain, flood, landslide).

    Sources:
    - Japan: 気象庁 警報・注意報
    - Taiwan: 中央氣象署 警特報
    """
    country = detect_country(project_id)

    if country == "jp":
        region_code = JMAProvider.REGION_CODES.get(project_id, "390000")
        data = await JMAProvider.get_rain_warning(region_code)
        return {
            "provider": "JMA",
            "source": "気象庁 警報・注意報",
            "source_url": "https://www.jma.go.jp/bosai/warning/",
            "data": data,
        }
    else:
        return {
            "provider": "CWA",
            "source": "中央氣象署 警特報",
            "source_url": "https://opendata.cwa.gov.tw/",
            "data": None,
            "note": "Set CWA_API_KEY to enable Taiwan weather warnings",
        }


# ---------------------------------------------------------------------------
# Baxianshan area rain stations (filtered)
# ---------------------------------------------------------------------------

def _classify_station_risk(past_24h_mm: float, past_1h_mm: float) -> dict:
    """
    Per-station risk classification using SWCB (水土保持署) debris-flow
    rainfall criteria as the upper anchor and a softer 'watch' band below.

    Returns: {level, color, label, label_en}
    """
    # 1-hour intensity short-circuits 24h accumulation
    if past_1h_mm >= 70 or past_24h_mm >= 350:
        return {"level": "red_alert", "color": "#d32f2f",
                "label": "🔴 紅色警戒", "label_en": "Red Alert"}
    if past_1h_mm >= 40 or past_24h_mm >= 200:
        return {"level": "yellow_alert", "color": "#f57c00",
                "label": "🟠 黃色警戒", "label_en": "Yellow Alert"}
    if past_24h_mm >= 100:
        return {"level": "watch", "color": "#fbc02d",
                "label": "🟡 觀察", "label_en": "Watch"}
    if past_24h_mm >= 50:
        return {"level": "advisory", "color": "#7cb342",
                "label": "🟢 提示", "label_en": "Advisory"}
    return {"level": "safe", "color": "#26a69a",
            "label": "🔵 安全", "label_en": "Safe"}


@router.get("/rain/baxianshan/geojson", summary="Rainfall stations as GeoJSON with per-station risk")
async def get_baxianshan_rain_geojson():
    """
    GPS-tagged rainfall observations for Baxianshan area as a GeoJSON
    FeatureCollection. Each feature is a CWA rain gauge station with
    coordinates + per-station risk classification (SWCB thresholds).

    UX rationale: replaces the previous 'whole Baxianshan, one banner'
    aggregate alert with point-level information that can be plotted
    directly on the map and sized by past_1h_mm or past_24h_mm.
    """
    raw_data = await CWAProvider.get_rain_observation()
    features = []
    error = None
    summary = {"max_1h_mm": 0, "max_24h_mm": 0, "by_level": {}}

    if not raw_data:
        error = "CWA_API_KEY not set or upstream unavailable"
    else:
        stations = CWARainFilter.filter_stations(
            raw_data,
            station_names=BAXIANSHAN_STATIONS["names"],
            bbox=BAXIANSHAN_STATIONS["bbox"],
        )
        for s in stations:
            risk = _classify_station_risk(s["past_24h_mm"], s["past_1h_mm"])
            summary["by_level"][risk["level"]] = summary["by_level"].get(risk["level"], 0) + 1
            summary["max_1h_mm"] = max(summary["max_1h_mm"], s["past_1h_mm"])
            summary["max_24h_mm"] = max(summary["max_24h_mm"], s["past_24h_mm"])
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [s["lng"], s["lat"]]},
                "properties": {
                    "station_id": s.get("station_id"),
                    "station_name": s.get("station_name"),
                    "elevation_m": s.get("elevation_m"),
                    "obs_time": s.get("obs_time"),
                    "past_1h_mm": s["past_1h_mm"],
                    "past_3h_mm": s.get("past_3h_mm", 0),
                    "past_6h_mm": s.get("past_6h_mm", 0),
                    "past_12h_mm": s.get("past_12h_mm", 0),
                    "past_24h_mm": s["past_24h_mm"],
                    "daily_mm": s.get("daily_mm", 0),
                    "risk_level": risk["level"],
                    "risk_color": risk["color"],
                    "risk_label": risk["label"],
                    "risk_label_en": risk["label_en"],
                },
            })

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "provider": "CWA",
            "source": "中央氣象署 自動雨量站",
            "area": "八仙山及周邊山區",
            "station_count": len(features),
            "summary": summary,
            "thresholds": {
                "advisory_24h_mm": 50,
                "watch_24h_mm": 100,
                "yellow_alert_24h_mm": 200,
                "yellow_alert_1h_mm": 40,
                "red_alert_24h_mm": 350,
                "red_alert_1h_mm": 70,
            },
            "error": error,
        },
    }


@router.get("/rain/baxianshan", summary="Get rainfall near Baxianshan")
async def get_baxianshan_rain():
    """
    Get current rainfall from stations near Baxianshan (八仙山).
    Filters CWA rain observations to only nearby mountain stations.

    Returns station list sorted by 24h rainfall (descending).
    """
    raw_data = await CWAProvider.get_rain_observation()
    if not raw_data:
        return {
            "provider": "CWA",
            "source": "中央氣象署 自動雨量站",
            "stations": [],
            "error": "CWA_API_KEY not set or API unavailable",
        }

    stations = CWARainFilter.filter_stations(
        raw_data,
        station_names=BAXIANSHAN_STATIONS["names"],
        bbox=BAXIANSHAN_STATIONS["bbox"]
    )

    # Summary stats
    max_1h = max((s["past_1h_mm"] for s in stations), default=0)
    max_24h = max((s["past_24h_mm"] for s in stations), default=0)

    return {
        "provider": "CWA",
        "source": "中央氣象署 自動雨量站",
        "area": "八仙山及周邊山區",
        "station_count": len(stations),
        "summary": {
            "max_1h_mm": max_1h,
            "max_24h_mm": max_24h,
        },
        "stations": stations,
    }


# ---------------------------------------------------------------------------
# Landslide Risk Assessment (rainfall + hydrology)
# ---------------------------------------------------------------------------

@router.get("/landslide-risk", summary="Assess forest work safety risk for Baxianshan")
async def assess_landslide_risk(
    project_id: str = Query("baxianshan"),
    rain_1h_mm: Optional[float] = Query(None, description="Override: 1-hour rainfall (mm)"),
    rain_3h_mm: Optional[float] = Query(None, description="Override: 3-hour rainfall (mm)"),
    rain_24h_mm: Optional[float] = Query(None, description="Override: 24-hour rainfall (mm)"),
    twi_max: float = Query(5.2, description="Max TWI value in area (from DEM analysis)"),
    slope_deg: float = Query(38, description="Representative steep slope (degrees)"),
    flow_accum_max: float = Query(8000, description="Max flow accumulation value"),
    canopy_height_m: float = Query(12, description="Canopy height (m, from GEE)"),
    ndvi: float = Query(0.72, description="NDVI vegetation index"),
    aspect_deg: float = Query(180, description="Slope aspect (degrees from north)"),
    geomorphon: int = Query(5, description="Geomorphon landform class (1-10)"),
    elevation_m: float = Query(1800, description="Representative elevation (m)"),
    wind_speed_ms: Optional[float] = Query(None, description="Override: wind speed (m/s)"),
    temperature_c: Optional[float] = Query(None, description="Override: air temperature (°C)"),
    humidity_pct: Optional[float] = Query(None, description="Override: relative humidity (%)"),
    region: str = Query("taiwan", description="AHP weighting region: taiwan / japan / indonesia"),
    soil_type: str = Query("slate", description="Lithology class, e.g. slate, mudstone, colluvium"),
    accessibility_minutes: float = Query(35, description="Minutes to nearest vehicle access"),
    wildlife: str = Query("hornets", description="Dominant wildlife hazard on site"),
    abnormal_trees: str = Query("unknown", description="Standing-tree defect surveyed on site"),
    use_gis_terrain: bool = Query(False, description="Auto-fetch terrain from gis-service PostGIS"),
    lon: Optional[float] = Query(None, description="Site longitude (for gis-service terrain lookup)"),
    lat: Optional[float] = Query(None, description="Site latitude (for gis-service terrain lookup)"),
):
    """
    Assess forest work safety risk across multiple hazards.

    The purpose is protecting worker life and safety, so landslide is one hazard
    among several. Combines:
    1. Real-time rainfall and weather from CWA stations near Baxianshan
    2. Terrain factors (slope, TWI, flow accumulation, aspect, geomorphon,
       canopy height, NDVI, elevation) from DEM/satellite analysis
    3. Taiwan SWCB (水土保持署) debris flow warning criteria
    4. 職安署 wind and heat-stress guidance for outdoor forest work

    Hazards assessed: 崩塌/土石流, 溪流暴漲, 倒木/落枝, 熱危害.

    If rain or weather parameters are not provided, live CWA data is fetched.
    Terrain defaults are pre-calculated values for Baxianshan.
    Set use_gis_terrain=true with lon/lat to auto-fetch real terrain from gis-service.

    Returns both:
    - `risk_level`: the official SWCB/JMA landslide alert level
    - `safety_level`: the worst level across all hazards (drives notifications)

    Levels: safe / watch (注意) / yellow_alert (預警) / red_alert (警戒)
    """
    # Auto-fetch terrain from gis-service if requested
    terrain_source = "default"
    if use_gis_terrain and lon is not None and lat is not None:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{_GIS_SERVICE_URL}/{project_id}/disaster/terrain-profile",
                    params={"lon": lon, "lat": lat, "buffer_m": 60},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("found"):
                        t = data["terrain"]
                        slope_deg = t["slope_deg"] or slope_deg
                        aspect_deg = t["aspect_deg"] or aspect_deg
                        geomorphon = t["geomorphon"] or geomorphon
                        twi_max = t["twi"] or twi_max
                        flow_accum_max = t["flow_accumulation"] or flow_accum_max
                        elevation_m = t["elevation_m"] or elevation_m
                        ndvi = t["ndvi"] or ndvi
                        canopy_height_m = t["canopy_height_m"] or canopy_height_m
                        soil_type = t.get("soil_type", soil_type)
                        terrain_source = f"gis-service (match {data.get('match_distance_m', '?')}m)"
        except Exception as e:
            logger.warning(f"gis-service terrain lookup failed: {e}")
            terrain_source = f"gis-service error: {e}"

    # Auto-fetch rainfall if not provided
    if rain_1h_mm is None or rain_24h_mm is None:
        raw_data = await CWAProvider.get_rain_observation()
        if raw_data:
            stations = CWARainFilter.filter_stations(
                raw_data,
                station_names=BAXIANSHAN_STATIONS["names"],
                bbox=BAXIANSHAN_STATIONS["bbox"]
            )
            if stations:
                # Use the maximum values from nearby stations (worst case)
                rain_1h_mm = rain_1h_mm or max(s["past_1h_mm"] for s in stations)
                rain_3h_mm = rain_3h_mm or max(s["past_3h_mm"] for s in stations)
                rain_24h_mm = rain_24h_mm or max(s["past_24h_mm"] for s in stations)

    # Fallback to 0 if still None (API unavailable)
    rain_1h_mm = rain_1h_mm or 0
    rain_3h_mm = rain_3h_mm or 0
    rain_24h_mm = rain_24h_mm or 0

    # Fetch forecast for yellow alert (預測超過)
    forecast_36h = await CWAProvider.get_forecast_36h(location="臺中市")
    forecast_24h_mm = CWAProvider.parse_forecast_rainfall(forecast_36h)

    # Auto-fetch wind / temperature / humidity if not provided.
    # Needed for the non-landslide hazards (treefall, heat stress).
    wind_gust_ms = 0.0
    if wind_speed_ms is None or temperature_c is None or humidity_pct is None:
        weather_raw = await CWAProvider.get_weather_observation()
        weather_stations = CWAWeatherFilter.filter_stations(
            weather_raw,
            station_names=BAXIANSHAN_STATIONS["names"],
            bbox=BAXIANSHAN_STATIONS["bbox"],
        )
        weather = CWAWeatherFilter.summarize(weather_stations)
        wind_speed_ms = wind_speed_ms if wind_speed_ms is not None else weather["wind_speed_ms"]
        temperature_c = temperature_c if temperature_c is not None else weather["temperature_c"]
        humidity_pct = humidity_pct if humidity_pct is not None else weather["humidity_pct"]
        wind_gust_ms = weather["wind_gust_ms"]

    # Fetch earthquake data for threshold auto-reduction
    earthquake_data = await CWAProvider.get_earthquake_report(days_back=30)

    # Run multi-hazard assessment
    assessment = FloodRiskEngine.assess_landslide_risk(
        rain_1h_mm=rain_1h_mm,
        rain_3h_mm=rain_3h_mm,
        rain_24h_mm=rain_24h_mm,
        forecast_24h_mm=forecast_24h_mm,
        twi_max=twi_max,
        slope_deg=slope_deg,
        flow_accum_max=flow_accum_max,
        canopy_height_m=canopy_height_m,
        ndvi=ndvi,
        aspect_deg=aspect_deg,
        geomorphon=geomorphon,
        elevation_m=elevation_m,
        wind_speed_ms=wind_speed_ms,
        wind_gust_ms=wind_gust_ms,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
        soil_type=soil_type,
        accessibility_minutes=accessibility_minutes,
        wildlife=wildlife,
        abnormal_trees=abnormal_trees,
        region=region,
        project_id=project_id,
        earthquake_data=earthquake_data,
    )

    # Warning message reflects the overall safety level across all hazards
    safety_level = assessment["safety_level"]
    hazards = assessment.get("hazards", [])
    warning_messages = {
        "safe": "✅ 目前各項條件未達警戒標準，可正常作業。",
        "watch": "🟡 注意 — 現場條件變化中，請提高警覺並確認撤離路線。",
        "yellow_alert": "⚠️ 預警 — 預期作業環境將達警戒標準，建議延期上山作業。",
        "red_alert": "⛔ 警戒 — 作業環境已達警戒標準，建議停止作業並移至安全處所。",
    }
    message = warning_messages.get(safety_level, warning_messages["safe"])
    if hazards:
        message += "（主要危害：" + "、".join(h["name"] for h in hazards) + "）"
    assessment["warning_message"] = message

    assessment["project_id"] = project_id
    assessment["assessment_time"] = datetime.now(timezone.utc).isoformat()
    assessment["terrain_source"] = terrain_source
    assessment["source"] = (
        "CWA 即時雨量 + 自動氣象站 + 鄉鎮預報 + 地震報告 + DEM 水文分析 "
        "+ SWCB 土石流警戒基準 + 職安署風速/熱危害指引"
    )
    assessment["hydrology_layers"] = [
        {"layer": "hydrology_flowlines", "description": "水文流向線（溢洪時危險區）"},
        {"layer": "hydrology_risk", "description": "水文風險區（優先撤離）"},
        {"layer": "twi_runoff", "description": "TWI 逕流潛勢圖"},
        {"layer": "landslide_avoidance_points", "description": "崩塌避讓點"},
    ]

    return assessment


# ---------------------------------------------------------------------------
# AHP weighting transparency
# ---------------------------------------------------------------------------

@router.get("/safety-criteria", summary="Inspect AHP weights and scoring vocabulary")
async def get_safety_criteria(
    region: Optional[str] = Query(None, description="taiwan / japan / indonesia; omit for all"),
):
    """
    Expose the AHP-derived criteria weights for review.

    The weights come from expert pairwise judgements, so they must be auditable
    rather than buried in code. This returns each region's group and global
    weights, the consistency ratio of every matrix (must be <= 0.10), the
    reasoning behind the regional priority ordering, and the accepted values for
    the categorical criteria.
    """
    codes = [region.lower()] if region else sorted(REGION_PROFILES)

    regions = {}
    for code in codes:
        profile = get_profile(code)
        weights = derive_region_weights(profile)
        regions[code] = {
            "name": profile.name,
            "group_weights": {
                g: round(w, 4) for g, w in weights["group_weights"].items()
            },
            "global_weights": {
                c: round(w, 4)
                for c, w in sorted(
                    weights["global_weights"].items(),
                    key=lambda kv: kv[1], reverse=True
                )
            },
            "consistency_ratios": weights["consistency_ratios"],
            "max_consistency_ratio": MAX_CONSISTENCY_RATIO,
            "consistent": all(
                cr <= MAX_CONSISTENCY_RATIO
                for cr in weights["consistency_ratios"].values()
            ),
            "rainfall_reference_mm": profile.rainfall_reference_mm,
            "rainfall_authority": profile.rainfall_authority,
            "rationale": profile.rationale,
            "accepted_values": {
                "soil_type": sorted(profile.geology_scores),
                "wildlife": sorted(profile.wildlife_hazards),
                "abnormal_trees": sorted(ABNORMAL_TREE_SCORES),
            },
        }

    return {
        "criteria_groups": CRITERIA_GROUPS,
        "criterion_labels": CRITERION_LABELS_ZH,
        "dynamic_criteria": sorted(DYNAMIC_CRITERIA),
        "score_scale": {
            1: "極低 negligible", 2: "低 low", 3: "中 moderate",
            4: "高 high", 5: "極高 extreme",
        },
        "index_classes": [
            {"upper_bound": None if u == float("inf") else u,
             "class": c, "label": label, "colour": colour}
            for u, c, label, colour in INDEX_CLASSES
        ],
        "regions": regions,
        "method": (
            "階層式 AHP（Saaty 1980）：先比較四大類，再比較類內準則，"
            "全域權重 = 類權重 × 類內權重。每個矩陣皆檢核一致性比率 CR<=0.10。"
        ),
        "reference": (
            "Rahmawati, Yovi & Setiawan (2025) Advancing occupational safety in "
            "forest management through a new GIS-AHP integrated framework. "
            "European Journal of Forest Engineering, 11(2)."
        ),
    }


# ---------------------------------------------------------------------------
# Scheduler status & manual trigger
# ---------------------------------------------------------------------------

@router.get("/alert-status", summary="Get last scheduled alert check result")
async def get_alert_status():
    """
    Returns the result of the most recent hourly landslide check.
    Includes risk level, rainfall, notification status.
    """
    from app.weather.scheduler import get_last_check
    last = get_last_check()
    if not last:
        return {"status": "no_check_yet", "message": "Scheduler has not completed a check yet."}
    return last


@router.post("/alert-check-now", summary="Manually trigger landslide check + notification")
async def manual_alert_check():
    """
    Manually trigger a landslide risk check and send notifications
    if risk level warrants it.
    """
    from app.weather.scheduler import run_landslide_check
    result = await run_landslide_check()
    return result


@router.post("/test-notify", summary="Test notification channels")
async def test_notify():
    """Send a test notification to LINE and Email."""
    from app.weather.notifier import LINENotifier, EmailAlertNotifier

    test_assessment = {
        "risk_level": "watch",
        "factors": ["測試通知 — 此為系統測試，非真實警報"],
        "evacuation_needed": False,
        "affected_zones": ["測試"],
        "rain_summary": {"1h_mm": 0, "3h_mm": 0, "24h_mm": 0},
        "warning_message": "🔔 這是 HiiForest 預警通知測試。如收到此訊息，表示通知管道設定成功。",
        "assessment_time": datetime.now(timezone.utc).isoformat(),
    }

    line_ok = await LINENotifier.send(
        "\n🔔 HiiForest 通知測試\n"
        "====================\n"
        "此為系統測試，非真實警報。\n"
        "如您收到此訊息，LINE 通知設定成功！\n"
        "\n🔗 https://hiiforest.com"
    )
    email_ok = EmailAlertNotifier.send(test_assessment)

    return {
        "line_notify": "✅ 成功" if line_ok else "❌ 失敗（LINE_NOTIFY_TOKEN 未設定？）",
        "email": "✅ 成功" if email_ok else "❌ 失敗（SMTP 設定問題？）",
    }


# ---------------------------------------------------------------------------
# Alert subscription management (unsubscribe / resubscribe)
# ---------------------------------------------------------------------------

class UnsubscribeBody(BaseModel):
    email: EmailStr
    project_id: str = "baxianshan"


@router.post("/unsubscribe", summary="Unsubscribe from weather alert emails")
async def unsubscribe_alerts(body: UnsubscribeBody):
    """Unsubscribe a user's email from weather alerts for a specific project."""
    from app.auth.database import get_db_config
    import psycopg2
    from psycopg2.extras import RealDictCursor

    config = get_db_config()
    conn = psycopg2.connect(**config)
    conn.autocommit = True
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                UPDATE auth.project_permissions
                SET alert_subscribed = false
                FROM auth.users u
                WHERE project_permissions.user_id = u.user_id
                  AND u.email = %s
                  AND project_permissions.project_id = %s
                RETURNING project_permissions.permission_id
            """, (body.email, body.project_id))
            row = cur.fetchone()
            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"No permission found for {body.email} on project '{body.project_id}'"
                )
            print(f"[Unsubscribe] {body.email} unsubscribed from {body.project_id}")
            return {
                "status": "unsubscribed",
                "email": body.email,
                "project_id": body.project_id,
                "message": "您已取消訂閱八仙山天氣預警通知。"
            }
    finally:
        conn.close()


@router.post("/resubscribe", summary="Re-subscribe to weather alert emails")
async def resubscribe_alerts(body: UnsubscribeBody):
    """Re-subscribe a user's email to weather alerts for a specific project."""
    from app.auth.database import get_db_config
    import psycopg2
    from psycopg2.extras import RealDictCursor

    config = get_db_config()
    conn = psycopg2.connect(**config)
    conn.autocommit = True
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                UPDATE auth.project_permissions
                SET alert_subscribed = true
                FROM auth.users u
                WHERE project_permissions.user_id = u.user_id
                  AND u.email = %s
                  AND project_permissions.project_id = %s
                RETURNING project_permissions.permission_id
            """, (body.email, body.project_id))
            row = cur.fetchone()
            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"No permission found for {body.email} on project '{body.project_id}'"
                )
            print(f"[Resubscribe] {body.email} resubscribed to {body.project_id}")
            return {
                "status": "subscribed",
                "email": body.email,
                "project_id": body.project_id,
                "message": "您已重新訂閱八仙山天氣預警通知。"
            }
    finally:
        conn.close()
