"""
Tests for the multi-hazard occupational safety assessment.

Covers the non-landslide hazards added on top of the SWCB landslide logic:
soil saturation estimation, treefall risk from wind, heat stress from WBGT,
and aggregation of all hazards into one overall safety level.
"""

import pytest

from app.weather.providers import CWAWeatherFilter, FloodRiskEngine


# ---------------------------------------------------------------------------
# Soil saturation
# ---------------------------------------------------------------------------

def test_soil_saturation_low_when_dry():
    soil = FloodRiskEngine.soil_saturation_index(rain_24h_mm=10, twi_max=4.0)
    assert soil["level"] == "low"
    assert soil["saturation_pct"] < 40


def test_soil_saturation_saturated_after_heavy_rain():
    soil = FloodRiskEngine.soil_saturation_index(
        rain_24h_mm=200, antecedent_7d_mm=[50, 40, 30, 0, 0, 0, 0], twi_max=5.2
    )
    assert soil["level"] == "saturated"
    assert soil["saturation_pct"] == 100.0


def test_soil_saturation_twi_amplifies():
    dry_terrain = FloodRiskEngine.soil_saturation_index(rain_24h_mm=100, twi_max=2.0)
    wet_terrain = FloodRiskEngine.soil_saturation_index(rain_24h_mm=100, twi_max=8.0)
    assert wet_terrain["saturation_pct"] > dry_terrain["saturation_pct"]


# ---------------------------------------------------------------------------
# Wind / treefall hazard
# ---------------------------------------------------------------------------

def test_wind_safe_when_calm():
    wind = FloodRiskEngine.assess_wind_hazard(wind_speed_ms=3.0)
    assert wind["level"] == "safe"
    assert wind["action"] == ""


def test_wind_warning_stops_felling():
    wind = FloodRiskEngine.assess_wind_hazard(wind_speed_ms=12.0)
    assert wind["level"] == "warning"
    assert "停止伐木" in wind["action"]


def test_wind_gust_drives_level_when_higher_than_mean():
    wind = FloodRiskEngine.assess_wind_hazard(wind_speed_ms=4.0, wind_gust_ms=18.0)
    assert wind["level"] == "danger"
    assert wind["effective_wind_ms"] == 18.0


def test_saturated_soil_lowers_wind_threshold():
    dry = FloodRiskEngine.assess_wind_hazard(wind_speed_ms=9.5, soil_saturation_pct=10)
    wet = FloodRiskEngine.assess_wind_hazard(wind_speed_ms=9.5, soil_saturation_pct=95)
    assert dry["level"] == "caution"
    assert wet["level"] == "warning"


# ---------------------------------------------------------------------------
# Heat stress
# ---------------------------------------------------------------------------

def test_heat_safe_in_mild_weather():
    heat = FloodRiskEngine.assess_heat_hazard(temperature_c=18, humidity_pct=60)
    assert heat["level"] == "safe"


def test_heat_danger_when_hot_and_humid():
    heat = FloodRiskEngine.assess_heat_hazard(temperature_c=36, humidity_pct=85)
    assert heat["level"] == "danger"
    assert "停止" in heat["action"]


def test_wbgt_rises_with_humidity():
    dry = FloodRiskEngine.estimate_wbgt(30, 40)
    humid = FloodRiskEngine.estimate_wbgt(30, 90)
    assert humid > dry


def test_wbgt_zero_without_temperature_reading():
    assert FloodRiskEngine.estimate_wbgt(0, 0) == 0.0


# ---------------------------------------------------------------------------
# Overall assessment
# ---------------------------------------------------------------------------

def test_calm_dry_day_is_safe():
    result = FloodRiskEngine.assess_landslide_risk(
        rain_1h_mm=0, rain_3h_mm=0, rain_24h_mm=0,
        slope_deg=38, twi_max=5.2,
        wind_speed_ms=2.0, temperature_c=15, humidity_pct=60,
    )
    assert result["risk_level"] == "safe"
    assert result["safety_level"] == "safe"
    assert result["hazards"] == []


def test_wind_alone_raises_safety_level_without_landslide_alert():
    """A dangerous wind must alert workers even when no rain has fallen."""
    result = FloodRiskEngine.assess_landslide_risk(
        rain_1h_mm=0, rain_3h_mm=0, rain_24h_mm=0,
        slope_deg=38, canopy_height_m=12,
        wind_speed_ms=20.0, temperature_c=15, humidity_pct=60,
    )
    assert result["risk_level"] == "safe"
    assert result["safety_level"] == "red_alert"
    assert [h["type"] for h in result["hazards"]] == ["treefall"]


def test_heat_alone_raises_safety_level():
    result = FloodRiskEngine.assess_landslide_risk(
        rain_1h_mm=0, rain_3h_mm=0, rain_24h_mm=0,
        wind_speed_ms=1.0, temperature_c=36, humidity_pct=85,
    )
    assert result["risk_level"] == "safe"
    assert result["safety_level"] == "red_alert"
    assert [h["type"] for h in result["hazards"]] == ["heat_stress"]


def test_stream_surge_reported_on_intense_rain():
    result = FloodRiskEngine.assess_landslide_risk(
        rain_1h_mm=35, rain_3h_mm=60, rain_24h_mm=120,
        flow_accum_max=8000, twi_max=5.2, slope_deg=38,
    )
    hazard_types = [h["type"] for h in result["hazards"]]
    assert "stream_surge" in hazard_types
    assert result["safety_level"] == "red_alert"


def test_landslide_red_alert_when_rt_exceeds_threshold():
    result = FloodRiskEngine.assess_landslide_risk(
        rain_1h_mm=20, rain_3h_mm=60, rain_24h_mm=400,
        slope_deg=38, twi_max=5.2, project_id="baxianshan",
    )
    assert result["risk_level"] == "red_alert"
    assert result["safety_level"] == "red_alert"
    assert result["evacuation_needed"] is True
    landslide = next(h for h in result["hazards"] if h["type"] == "landslide")
    assert landslide["level"] == "danger"


def test_safety_level_takes_worst_hazard():
    """Landslide only at watch, but wind at danger → overall must be red_alert."""
    result = FloodRiskEngine.assess_landslide_risk(
        rain_1h_mm=25, rain_3h_mm=45, rain_24h_mm=90,
        slope_deg=38, twi_max=5.2,
        wind_speed_ms=20.0, canopy_height_m=12,
    )
    assert result["risk_level"] == "watch"
    assert result["safety_level"] == "red_alert"


def test_assessment_includes_composite_index():
    result = FloodRiskEngine.assess_landslide_risk(
        rain_1h_mm=0, rain_3h_mm=0, rain_24h_mm=0,
        slope_deg=38, twi_max=5.2, soil_type="slate",
        accessibility_minutes=35, wildlife="hornets",
        abnormal_trees="dead_standing",
    )
    index = result["safety_index"]
    assert 1.0 <= index["composite_score"] <= 5.0
    assert index["region"] == "taiwan"
    assert index["criteria"]["soil_type"]["score"] == 4  # slate


def test_composite_index_does_not_soften_a_threshold_breach():
    """A rainfall breach must stay red even though most criteria are benign."""
    result = FloodRiskEngine.assess_landslide_risk(
        rain_1h_mm=20, rain_3h_mm=60, rain_24h_mm=400,
        slope_deg=5, twi_max=1, flow_accum_max=10,
        ndvi=0.9, soil_type="granite", accessibility_minutes=5,
        wildlife="none", abnormal_trees="none",
        project_id="baxianshan",
    )
    assert result["risk_level"] == "red_alert"
    assert result["safety_level"] == "red_alert"
    # The weighted average is dragged down by the benign terrain...
    assert result["safety_index"]["composite_score"] < 4.2
    # ...but that must not soften the alert.
    assert result["evacuation_needed"] is True


def test_high_composite_index_alone_does_not_raise_an_alert():
    """Permanently hazardous terrain must not hold the crew at alert forever."""
    result = FloodRiskEngine.assess_landslide_risk(
        rain_1h_mm=0, rain_3h_mm=0, rain_24h_mm=0,
        slope_deg=60, elevation_m=3000, twi_max=9, flow_accum_max=20000,
        geomorphon=9, ndvi=0.1, soil_type="colluvium",
        accessibility_minutes=120, wildlife="black_bear",
        abnormal_trees="hanging_limb",
        wind_speed_ms=1.0, temperature_c=15, humidity_pct=50,
    )
    index = result["safety_index"]
    # Terrain-only score sees the danger...
    assert index["static_class"] == "very_high"
    # ...but with no weather driving it, nothing is triggered.
    assert result["safety_level"] == "safe"
    assert result["hazards"] == []


def test_composite_understates_terrain_on_a_calm_day():
    """
    Documents a real limit: Taiwan weights climate at ~0.48, so the worst
    possible terrain cannot exceed 'moderate' on the composite when the weather
    is benign. This is why `static_score` exists for zoning.
    """
    result = FloodRiskEngine.assess_landslide_risk(
        rain_1h_mm=0, rain_3h_mm=0, rain_24h_mm=0,
        slope_deg=60, elevation_m=3000, twi_max=9, flow_accum_max=20000,
        geomorphon=9, ndvi=0.1, soil_type="colluvium",
        accessibility_minutes=120, wildlife="black_bear",
        abnormal_trees="hanging_limb",
        wind_speed_ms=1.0, temperature_c=15, humidity_pct=50,
    )
    index = result["safety_index"]
    assert index["class"] == "moderate"
    assert index["static_score"] > index["composite_score"]


def test_region_selects_different_weights():
    params = dict(
        rain_1h_mm=0, rain_3h_mm=0, rain_24h_mm=150,
        slope_deg=38, twi_max=5.2, soil_type="unknown",
    )
    taiwan = FloodRiskEngine.assess_landslide_risk(region="taiwan", **params)
    indonesia = FloodRiskEngine.assess_landslide_risk(region="indonesia", **params)
    assert taiwan["safety_index"]["composite_score"] != \
        indonesia["safety_index"]["composite_score"]


def test_assessment_exposes_weather_conditions():
    result = FloodRiskEngine.assess_landslide_risk(
        rain_1h_mm=0, rain_3h_mm=0, rain_24h_mm=0,
        wind_speed_ms=6.0, wind_gust_ms=9.0,
        temperature_c=28, humidity_pct=70,
    )
    conditions = result["weather_conditions"]
    assert conditions["wind_speed_ms"] == 6.0
    assert conditions["wind_gust_ms"] == 9.0
    assert conditions["temperature_c"] == 28.0
    assert conditions["wbgt_c"] > 0


# ---------------------------------------------------------------------------
# Weather station parsing
# ---------------------------------------------------------------------------

def test_weather_filter_ignores_sentinel_values():
    """CWA uses -99 / -990 for missing readings; these must not become data."""
    raw = {
        "records": {
            "Station": [{
                "StationName": "谷關",
                "StationId": "C0F9A0",
                "GeoInfo": {
                    "Coordinates": [{
                        "StationLatitude": "24.20",
                        "StationLongitude": "120.95",
                    }],
                    "StationAltitude": "800",
                },
                "ObsTime": {"DateTime": "2026-08-11T18:00:00+08:00"},
                "WeatherElement": {
                    "WindSpeed": "-99",
                    "AirTemperature": "24.5",
                    "RelativeHumidity": "-990",
                    "GustInfo": {"PeakGustSpeed": "8.2"},
                },
            }]
        }
    }
    stations = CWAWeatherFilter.filter_stations(
        raw, bbox={"min_lat": 24.0, "max_lat": 24.5, "min_lng": 120.7, "max_lng": 121.3}
    )
    assert len(stations) == 1
    assert stations[0]["wind_speed_ms"] == 0.0
    assert stations[0]["humidity_pct"] == 0.0
    assert stations[0]["temperature_c"] == 24.5
    assert stations[0]["wind_gust_ms"] == 8.2


def test_weather_summary_takes_worst_case():
    stations = [
        {"wind_speed_ms": 5.0, "wind_gust_ms": 8.0, "temperature_c": 20.0, "humidity_pct": 60.0},
        {"wind_speed_ms": 12.0, "wind_gust_ms": 15.0, "temperature_c": 28.0, "humidity_pct": 80.0},
    ]
    summary = CWAWeatherFilter.summarize(stations)
    assert summary["wind_speed_ms"] == 12.0
    assert summary["wind_gust_ms"] == 15.0
    assert summary["temperature_c"] == 28.0
    assert summary["station_count"] == 2


def test_weather_summary_empty_is_zeroed():
    summary = CWAWeatherFilter.summarize([])
    assert summary["station_count"] == 0
    assert summary["wind_speed_ms"] == 0.0
