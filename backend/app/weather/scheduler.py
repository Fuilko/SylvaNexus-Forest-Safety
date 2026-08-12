"""
SylvaNexus — Weather Alert Scheduler
=======================================
Hourly background task that:
1. Fetches CWA rainfall data for Baxianshan
2. Runs landslide risk assessment
3. Sends notifications if risk escalates

Uses asyncio background task (no extra dependencies).
Interval: every 3 hours (configurable via _check_interval_seconds).
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Optional


# State
_scheduler_task: Optional[asyncio.Task] = None
_check_interval_seconds = 10800  # 3 hours
_last_check_result: Optional[dict] = None


async def run_landslide_check() -> dict:
    """Run one landslide risk check cycle."""
    global _last_check_result

    from app.weather.providers import (
        CWAProvider, CWARainFilter, CWAWeatherFilter, FloodRiskEngine,
        BAXIANSHAN_STATIONS
    )
    from app.weather.notifier import AlertDispatcher

    print(f"[Scheduler] Running landslide check at {datetime.now(timezone.utc).isoformat()}")

    # ── Fetch observed rain data ──
    raw_data = await CWAProvider.get_rain_observation()
    rain_1h = 0.0
    rain_3h = 0.0
    rain_24h = 0.0
    station_count = 0

    if raw_data:
        stations = CWARainFilter.filter_stations(
            raw_data,
            station_names=BAXIANSHAN_STATIONS["names"],
            bbox=BAXIANSHAN_STATIONS["bbox"]
        )
        station_count = len(stations)
        if stations:
            rain_1h = max(s["past_1h_mm"] for s in stations)
            rain_3h = max(s["past_3h_mm"] for s in stations)
            rain_24h = max(s["past_24h_mm"] for s in stations)

    # ── Fetch CWA forecast (for yellow alert: 預測超過) ──
    # F-D0047-089 township 7-day forecast does not provide quantitative rainfall.
    # Use F-C0032-001 36h forecast with Wx weather codes to estimate rainfall.
    forecast_36h = await CWAProvider.get_forecast_36h(location="臺中市")
    forecast_24h = CWAProvider.parse_forecast_rainfall(forecast_36h)

    # ── Fetch weather observations (wind / temperature / humidity) ──
    # Needed for non-landslide occupational hazards: treefall and heat stress.
    weather_raw = await CWAProvider.get_weather_observation()
    weather_stations = CWAWeatherFilter.filter_stations(
        weather_raw,
        station_names=BAXIANSHAN_STATIONS["names"],
        bbox=BAXIANSHAN_STATIONS["bbox"],
    )
    weather = CWAWeatherFilter.summarize(weather_stations)

    # ── Fetch earthquake data (for threshold auto-reduction) ──
    earthquake_data = await CWAProvider.get_earthquake_report(days_back=30)

    # Baxianshan terrain defaults (from DEM/GIS analysis)
    twi_max = float(os.getenv("BAXIANSHAN_TWI_MAX", "5.2"))
    slope_deg = float(os.getenv("BAXIANSHAN_SLOPE_DEG", "38"))
    flow_accum_max = float(os.getenv("BAXIANSHAN_FLOW_ACCUM_MAX", "8000"))
    canopy_height_m = float(os.getenv("BAXIANSHAN_CANOPY_HEIGHT", "12"))  # mature forest
    ndvi = float(os.getenv("BAXIANSHAN_NDVI", "0.72"))  # dense vegetation
    aspect_deg = float(os.getenv("BAXIANSHAN_ASPECT", "180"))  # south-facing

    # Site survey inputs for the GIS-AHP composite index.
    # Baxianshan sits on the 廬山層 slate/phyllite formation; access is via the
    # 八仙山林道 so a typical work face is ~35 min from vehicle access.
    region = os.getenv("SAFETY_REGION", "taiwan")
    soil_type = os.getenv("BAXIANSHAN_SOIL_TYPE", "slate")
    accessibility_minutes = float(os.getenv("BAXIANSHAN_ACCESS_MIN", "35"))
    wildlife = os.getenv("BAXIANSHAN_WILDLIFE", "hornets")  # 秋季虎頭蜂為主要威脅
    abnormal_trees = os.getenv("BAXIANSHAN_TREE_CONDITION", "unknown")
    geomorphon = int(os.getenv("BAXIANSHAN_GEOMORPHON", "5"))  # steep slope
    elevation_m = float(os.getenv("BAXIANSHAN_ELEVATION", "1800"))

    # Assess risk with two-layer method
    assessment = FloodRiskEngine.assess_landslide_risk(
        rain_1h_mm=rain_1h,
        rain_3h_mm=rain_3h,
        rain_24h_mm=rain_24h,
        forecast_24h_mm=forecast_24h,
        antecedent_7d_mm=None,  # TODO: populate from DB once rain history persistence is added
        twi_max=twi_max,
        slope_deg=slope_deg,
        flow_accum_max=flow_accum_max,
        canopy_height_m=canopy_height_m,
        ndvi=ndvi,
        aspect_deg=aspect_deg,
        geomorphon=geomorphon,
        elevation_m=elevation_m,
        wind_speed_ms=weather["wind_speed_ms"],
        wind_gust_ms=weather["wind_gust_ms"],
        temperature_c=weather["temperature_c"],
        humidity_pct=weather["humidity_pct"],
        soil_type=soil_type,
        accessibility_minutes=accessibility_minutes,
        wildlife=wildlife,
        abnormal_trees=abnormal_trees,
        region=region,
        project_id="baxianshan",
        earthquake_data=earthquake_data,
    )

    # Add metadata
    risk_level = assessment["risk_level"]
    safety_level = assessment["safety_level"]
    hazards = assessment.get("hazards", [])

    # Warning message reflects the overall safety level across all hazards,
    # not the landslide alert level alone.
    base_messages = {
        "safe": "✅ 目前各項條件未達警戒標準，可正常作業。",
        "watch": "🟡 注意 — 現場條件變化中，請提高警覺並確認撤離路線。",
        "yellow_alert": "⚠️ 預警 — 預期作業環境將達警戒標準，建議延期上山作業。",
        "red_alert": "⛔ 警戒 — 作業環境已達警戒標準，建議停止作業並移至安全處所。",
    }
    message = base_messages.get(safety_level, base_messages["safe"])
    if hazards:
        hazard_names = "、".join(h["name"] for h in hazards)
        message += f"（主要危害：{hazard_names}）"
    assessment["warning_message"] = message

    assessment["project_id"] = "baxianshan"
    assessment["location_name"] = "八仙山"
    assessment["lat"] = 24.2633
    assessment["lng"] = 120.9500
    assessment["assessment_time"] = datetime.now(timezone.utc).isoformat()
    assessment["source"] = (
        "CWA 即時雨量 + 自動氣象站 + 鄉鎮預報 + 地震報告 + DEM 水文分析 "
        "+ SWCB 土石流警戒基準 + 職安署風速/熱危害指引"
    )
    assessment["station_count"] = station_count
    assessment["weather_station_count"] = weather["station_count"]

    _last_check_result = assessment

    print(f"[Scheduler] Safety={safety_level} (landslide={risk_level}), "
          f"1h={rain_1h}mm, 24h={rain_24h}mm, forecast={forecast_24h}mm, "
          f"wind={weather['wind_speed_ms']}m/s, temp={weather['temperature_c']}°C, "
          f"hazards={[h['type'] for h in hazards]}, stations={station_count}")

    # Dispatch notifications if needed
    notify_result = await AlertDispatcher.dispatch(assessment)
    assessment["notification"] = notify_result

    return assessment


async def _scheduler_loop():
    """Background loop: check every hour."""
    # Wait 30s after startup before first check
    await asyncio.sleep(30)
    print(f"[Scheduler] Started. Interval: {_check_interval_seconds}s")

    consecutive_failures = 0
    _SENTRY_DSN = os.getenv("SENTRY_DSN_BACKEND")

    while True:
        try:
            await run_landslide_check()
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            print(f"[Scheduler] ❌ Error in check cycle ({consecutive_failures} consecutive): {e}")
            # Send to Sentry every 3 consecutive failures to avoid spam
            if consecutive_failures % 3 == 0 and _SENTRY_DSN:
                try:
                    import sentry_sdk
                    sentry_sdk.capture_message(
                        f"Weather scheduler failed {consecutive_failures} consecutive times: {e}",
                        level="error",
                    )
                except Exception:
                    pass
        await asyncio.sleep(_check_interval_seconds)


def start_scheduler():
    """Start the background scheduler. Call from FastAPI startup event."""
    global _scheduler_task

    cwa_key = os.getenv("CWA_API_KEY", "")
    if not cwa_key:
        print("[Scheduler] CWA_API_KEY not set, scheduler disabled")
        return

    if _scheduler_task and not _scheduler_task.done():
        print("[Scheduler] Already running")
        return

    loop = asyncio.get_event_loop()
    _scheduler_task = loop.create_task(_scheduler_loop())
    print("[Scheduler] Background landslide check scheduled (every 3 hours)")


def stop_scheduler():
    """Stop the background scheduler."""
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        _scheduler_task.cancel()
        print("[Scheduler] Stopped")


def get_last_check() -> Optional[dict]:
    """Get the result of the last scheduled check."""
    return _last_check_result
