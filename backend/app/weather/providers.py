"""
SylvaNexus — Weather Data Providers
======================================
Adapters for Japan (JMA) and Taiwan (CWA) weather APIs.
Each provider returns a unified WeatherData format.

Japan:  気象庁 (JMA) — https://www.jma.go.jp/bosai/forecast/
        AMeDAS 10-min observation data
        土砂災害警戒情報 (Level 4 equivalent)
Taiwan: 中央氣象署 (CWA) — https://opendata.cwa.gov.tw/
        Requires API key (free registration)
        鄉鎮預報 (F-D0047-089) for yellow alert (forecast-driven)
        雨量站觀測 (O-A0002-001) for red alert (actual rainfall)
        地震報告 (E-A0014-001) for threshold auto-reduction

Source attribution:
  - JMA: 気象庁ホームページ (https://www.jma.go.jp/)
  - CWA: 交通部中央氣象署 開放資料平台 (https://opendata.cwa.gov.tw/)
  - SWCB: 農業部農村發展及水土保持署 土石流防災資訊網 (https://246.ardswc.gov.tw/)

References:
  [1] SWCB 土石流警戒基準值訂定方法 — RTI = Rt × I, Rt = R₀ + 0.7×ΣRᵢ (前7日)
      https://246.ardswc.gov.tw/DebrisFlow/AlertSet
  [2] Chen et al. (2015) Rainfall intensity–duration conditions for mass movements
      in Taiwan. Progress in Earth and Planetary Science, 2:14.
      I = 18.10 × D^(-0.17) (Taiwan-specific, 263 events)
  [3] Jan & Lee (2004) RTI model — basis for SWCB official warning system.
  [4] Guzzetti et al. (2008) The rainfall intensity–duration control of shallow
      landslides and debris flows: an update. Landslides, 5:3-17.
      Global I = 0.82 × D^(-0.20) (Bayesian, 2626 events)
  [5] Iida (1999) A threshold of rainfall for shallow landslides.
      Bull. FFPR 1(4):117-126. I = 14.7 × D^(-0.42)
  [6] Sidle & Ochiai (2006) Landslides: Processes, Prediction, and Land Use.
      AGU Water Resources Monograph 18.
  [7] Beven & Kirkby (1979) A physically based, variable contributing area
      model of basin hydrology. Hydro. Sci. Bull. 24:43-69.
  [8] O'Callaghan & Mark (1984) The extraction of drainage networks from DEMs.
  [9] JMA 土砂災害警戒情報 — CL (Critical Line) = SWI × 60min rainfall,
      per 1km mesh, RBFN-calibrated.
      https://www.mlit.go.jp/river/shishin_guideline/sabo/dsk_kizyun_kensho_r0503.pdf
  [10] Chen et al. (2024) Combining rainfall parameter and landslide susceptibility
       to forecast shallow landslide in Taiwan. SEAGS & AGSSEA J. 47(2):72-82.
       R24 + I3 hazard matrix, 2-9h lead time.
"""

import math
import os
import httpx
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from pydantic import BaseModel, Field

from app.weather.safety_index import assess_safety_index


# ---------------------------------------------------------------------------
# Unified Weather Data Model
# ---------------------------------------------------------------------------

class WeatherObservation(BaseModel):
    """Single weather observation point."""
    station_id: str
    station_name: str
    lat: float
    lng: float
    elevation_m: Optional[float] = None
    timestamp: datetime
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    precipitation_mm: Optional[float] = None
    precipitation_1h_mm: Optional[float] = None
    precipitation_24h_mm: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    wind_direction: Optional[str] = None
    pressure_hpa: Optional[float] = None
    source: str
    source_url: str


class WeatherForecast(BaseModel):
    """Weather forecast for a region."""
    region: str
    forecast_date: str
    weather: str
    max_temp_c: Optional[float] = None
    min_temp_c: Optional[float] = None
    precip_probability_pct: Optional[int] = None
    precip_amount_mm: Optional[float] = None
    wind_warning: Optional[str] = None
    source: str
    source_url: str


class FloodRiskAssessment(BaseModel):
    """Flood/surge risk for a forest area based on weather + terrain."""
    location_name: str
    lat: float
    lng: float
    risk_level: str  # low / medium / high / critical
    rainfall_forecast_mm: float
    rainfall_threshold_mm: float
    twi_max: Optional[float] = None
    flow_accumulation_max: Optional[float] = None
    warning_message: str
    source: str
    assessment_time: datetime


# ---------------------------------------------------------------------------
# Japan — JMA (気象庁)
# ---------------------------------------------------------------------------

class JMAProvider:
    """
    Japan Meteorological Agency data provider.
    Uses public JSON APIs (no API key required).

    Source: 気象庁ホームページ
    URL: https://www.jma.go.jp/bosai/
    License: 気象庁が公開するデータは、利用規約に基づき利用可能
    """

    BASE_URL = "https://www.jma.go.jp/bosai"
    SOURCE = "気象庁 (Japan Meteorological Agency)"
    SOURCE_URL = "https://www.jma.go.jp/"

    # Region codes for forest areas
    # Reference: https://www.jma.go.jp/bosai/forecast/data/overview_forecast/{code}.json
    REGION_CODES = {
        "kochi": "390000",      # 高知県
        "shimanto": "390000",   # 四万十（高知県）
        "shizuoka": "220000",   # 静岡県
        "nagano": "200000",     # 長野県
        "hokkaido": "016000",   # 北海道
    }

    @classmethod
    async def get_forecast(cls, region_code: str) -> Optional[dict]:
        """
        Fetch JMA forecast overview for a region.
        Returns raw JSON from JMA API.
        """
        url = f"{cls.BASE_URL}/forecast/data/overview_forecast/{region_code}.json"
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f"[Weather/JMA] Failed to fetch forecast: {e}")
                return None

    @classmethod
    async def get_amedas_latest(cls) -> Optional[dict]:
        """
        Fetch latest AMeDAS observation data.
        Returns station-keyed dict of observations.

        Source: 気象庁 AMeDAS (Automated Meteorological Data Acquisition System)
        """
        url = f"{cls.BASE_URL}/amedas/data/latest_time.txt"
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                # Get latest timestamp
                time_resp = await client.get(url)
                latest_time = time_resp.text.strip().replace('"', '')

                # Fetch observation data
                data_url = f"{cls.BASE_URL}/amedas/data/point/{latest_time}.json"
                data_resp = await client.get(data_url)
                data_resp.raise_for_status()
                return data_resp.json()
            except Exception as e:
                print(f"[Weather/JMA] Failed to fetch AMeDAS: {e}")
                return None

    @classmethod
    async def get_rain_warning(cls, region_code: str) -> Optional[dict]:
        """
        Fetch weather warnings for a region.
        Includes heavy rain, flood, and landslide warnings.

        Source: 気象庁 警報・注意報
        """
        url = f"{cls.BASE_URL}/warning/data/warning/{region_code}.json"
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f"[Weather/JMA] Failed to fetch warnings: {e}")
                return None


# ---------------------------------------------------------------------------
# Taiwan — CWA (中央氣象署)
# ---------------------------------------------------------------------------

class CWAProvider:
    """
    Taiwan Central Weather Administration data provider.
    Requires API key from https://opendata.cwa.gov.tw/

    Source: 交通部中央氣象署 開放資料平台
    URL: https://opendata.cwa.gov.tw/
    License: 政府資料開放授權條款 (Open Government Data License)
    """

    BASE_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
    SOURCE = "中央氣象署 (Central Weather Administration, Taiwan)"
    SOURCE_URL = "https://opendata.cwa.gov.tw/"

    # Dataset IDs
    FORECAST_36H = "F-C0032-001"        # 一般天氣預報-今明36小時天氣預報
    FORECAST_TOWN_7D = "F-D0047-089"    # 臺灣各鄉鎮預報資料-未來1週(12小時)
    OBSERVATION = "O-A0001-001"          # 自動氣象站-氣象觀測資料
    RAIN_OBSERVATION = "O-A0002-001"     # 自動雨量站-雨量觀測資料
    RAINFALL_FORECAST = "F-B0046-001"    # 定量降水預報
    EARTHQUAKE_REPORT = "E-A0014-001"    # 地震報告-地震活動

    @classmethod
    def _get_api_key(cls) -> str:
        key = os.getenv("CWA_API_KEY", "")
        if not key:
            print("[Weather/CWA] WARNING: CWA_API_KEY not set")
        return key

    @classmethod
    async def get_forecast_36h(cls, location: str = "") -> Optional[dict]:
        """
        Fetch 36-hour weather forecast.
        Source: 中央氣象署 一般天氣預報
        """
        api_key = cls._get_api_key()
        if not api_key:
            return None

        params = {"Authorization": api_key, "format": "JSON"}
        if location:
            params["locationName"] = location

        url = f"{cls.BASE_URL}/{cls.FORECAST_36H}"
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f"[Weather/CWA] Failed to fetch forecast: {e}")
                return None

    @classmethod
    async def get_rain_observation(cls) -> Optional[dict]:
        """
        Fetch current rain gauge observations.
        Source: 中央氣象署 自動雨量站
        """
        api_key = cls._get_api_key()
        if not api_key:
            return None

        url = f"{cls.BASE_URL}/{cls.RAIN_OBSERVATION}"
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.get(url, params={
                    "Authorization": api_key, "format": "JSON"
                })
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f"[Weather/CWA] Failed to fetch rain data: {e}")
                return None

    @classmethod
    async def get_weather_observation(cls) -> Optional[dict]:
        """
        Fetch current automatic weather station observations.
        Source: 中央氣象署 自動氣象站-氣象觀測資料 (O-A0001-001)

        Provides wind speed (WDSD), gust (WSGust), air temperature (AirTemperature)
        and relative humidity (RelativeHumidity) — required for non-landslide
        occupational hazards (treefall risk, heat stress).
        """
        api_key = cls._get_api_key()
        if not api_key:
            return None

        url = f"{cls.BASE_URL}/{cls.OBSERVATION}"
        async with httpx.AsyncClient(timeout=15) as client:
            try:
                resp = await client.get(url, params={
                    "Authorization": api_key, "format": "JSON"
                })
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f"[Weather/CWA] Failed to fetch weather observation: {e}")
                return None

    @classmethod
    async def get_town_forecast_7d(cls, county: str = "臺中市", town: str = "和平區") -> Optional[dict]:
        """
        Fetch 7-day township forecast (3-hour intervals).
        Source: 中央氣象署 臺灣各鄉鎮預報資料-未來1週
        Used for yellow alert: forecast cumulative rainfall > SWCB threshold.
        """
        api_key = cls._get_api_key()
        if not api_key:
            return None

        url = f"{cls.BASE_URL}/{cls.FORECAST_TOWN_7D}"
        async with httpx.AsyncClient(timeout=15) as client:
            try:
                resp = await client.get(url, params={
                    "Authorization": api_key,
                    "format": "JSON",
                    "CountyName": county,
                    "TownName": town,
                })
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f"[Weather/CWA] Failed to fetch town forecast: {e}")
                return None

    @classmethod
    async def get_earthquake_report(cls, days_back: int = 30) -> Optional[dict]:
        """
        Fetch recent earthquake reports for threshold auto-reduction.
        Source: 中央氣象署 地震報告

        SWCB rule:
          - Intensity >= 5強 (5Upper): reduce threshold by 50-100mm (1-2 級距)
          - Intensity >= 6弱 (6Lower): reduce threshold by 100-200mm (2-4 級距)
        """
        api_key = cls._get_api_key()
        if not api_key:
            return None

        time_to = datetime.now(timezone.utc)
        time_from = time_to - timedelta(days=days_back)

        url = f"{cls.BASE_URL}/{cls.EARTHQUAKE_REPORT}"
        async with httpx.AsyncClient(timeout=15) as client:
            try:
                resp = await client.get(url, params={
                    "Authorization": api_key,
                    "format": "JSON",
                    "timeFrom": time_from.strftime("%Y-%m-%dT%H:%M:%S"),
                    "timeTo": time_to.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f"[Weather/CWA] Failed to fetch earthquake report: {e}")
                return None

    @classmethod
    def parse_forecast_rainfall(cls, forecast_data: dict) -> float:
        """
        Parse CWA 36h forecast data to estimate 24h cumulative rainfall.

        CWA F-C0032-001 Wx element uses weather codes:
          23 = 豪雨 (heavy rain)   → estimate ~100mm/24h
          24 = 大豪雨 (very heavy) → estimate ~200mm/24h
          25 = 超大豪雨 (extreme)  → estimate ~350mm/24h

        These are conservative estimates based on CWA advisory thresholds:
          豪雨特報: 24h≥100mm
          大豪雨特報: 24h≥200mm
          超大豪雨特報: 24h≥350mm
        Ref: CWA 大雨特報標準 [2]

        Returns estimated 24h rainfall in mm (0 if no significant rain forecast).
        """
        if not forecast_data:
            return 0.0
        try:
            elements = forecast_data.get("records", {}).get("location", [{}])[0]
            wx_elements = elements.get("weatherElement", [])
            max_rain = 0.0
            for elem in wx_elements:
                if elem.get("elementName") == "Wx":
                    time_periods = elem.get("time", [])
                    for period in time_periods:
                        wx_value = period.get("elementValue", [{}])
                        if wx_value:
                            weather_code = wx_value[0].get("weatherCode", "")
                            code_rain = {"23": 100, "24": 200, "25": 350}
                            max_rain = max(max_rain, code_rain.get(weather_code, 0))
            return max_rain
        except (KeyError, IndexError, TypeError) as e:
            print(f"[Weather/CWA] Failed to parse forecast rainfall: {e}")
            return 0.0


# ---------------------------------------------------------------------------
# Baxianshan Area Rain Stations
# ---------------------------------------------------------------------------

# Nearby CWA rain gauge stations for Baxianshan (八仙山)
# Coordinates: ~24.20°N, 120.95°E, elevation ~1000-2400m
BAXIANSHAN_STATIONS = {
    "names": ["德基", "梨山", "谷關", "稍來", "大甲溪", "白冷", "天輪", "麟趾山",
              "思源", "武陵", "大禹嶺", "合歡山", "鞍部", "雪山"],
    "bbox": {"min_lat": 24.05, "max_lat": 24.45, "min_lng": 120.75, "max_lng": 121.30},
}


class CWARainFilter:
    """Filter CWA rain observation data for specific area."""

    @classmethod
    def filter_stations(cls, raw_data: dict, station_names: list = None,
                        bbox: dict = None) -> List[dict]:
        """Extract relevant stations from raw CWA rain observation response."""
        if not raw_data:
            return []

        stations = []
        try:
            records = raw_data.get("records", {}).get("Station", [])
        except (AttributeError, KeyError):
            return []

        for stn in records:
            stn_name = stn.get("StationName", "")
            lat = float(stn.get("GeoInfo", {}).get("Coordinates", [{}])[0].get("StationLatitude", 0) or
                        stn.get("StationLatitude", 0))
            lng = float(stn.get("GeoInfo", {}).get("Coordinates", [{}])[0].get("StationLongitude", 0) or
                        stn.get("StationLongitude", 0))

            # Filter by name or bbox
            name_match = station_names and any(n in stn_name for n in station_names)
            bbox_match = bbox and (bbox["min_lat"] <= lat <= bbox["max_lat"] and
                                   bbox["min_lng"] <= lng <= bbox["max_lng"])

            if not (name_match or bbox_match):
                continue

            # Parse rainfall data
            rain_info = stn.get("RainfallElement", {})
            obs_time = stn.get("ObsTime", {}).get("DateTime", "")

            stations.append({
                "station_name": stn_name,
                "station_id": stn.get("StationId", ""),
                "lat": lat,
                "lng": lng,
                "elevation_m": float(stn.get("GeoInfo", {}).get("StationAltitude", 0) or 0),
                "obs_time": obs_time,
                "now_mm": float(rain_info.get("Now", {}).get("Precipitation", 0) or 0),
                "past_10min_mm": float(rain_info.get("Past10Min", {}).get("Precipitation", 0) or 0),
                "past_1h_mm": float(rain_info.get("Past1hr", {}).get("Precipitation", 0) or 0),
                "past_3h_mm": float(rain_info.get("Past3hr", {}).get("Precipitation", 0) or 0),
                "past_6h_mm": float(rain_info.get("Past6hr", {}).get("Precipitation", 0) or 0),
                "past_12h_mm": float(rain_info.get("Past12hr", {}).get("Precipitation", 0) or 0),
                "past_24h_mm": float(rain_info.get("Past24hr", {}).get("Precipitation", 0) or 0),
                "daily_mm": float(rain_info.get("Daily", {}).get("Precipitation", 0) or 0),
            })

        return sorted(stations, key=lambda x: x["past_24h_mm"], reverse=True)


class CWAWeatherFilter:
    """
    Filter CWA automatic weather station data (O-A0001-001) for a specific area.

    Extracts the meteorological elements needed for non-landslide occupational
    hazards: wind speed / gust (treefall, felling safety) and temperature /
    humidity (heat stress).
    """

    # CWA uses -99 / -990 / -99.0 as "no data" sentinels
    _INVALID = (-99, -990, -9996, -9997, -9998, -9999)

    @classmethod
    def _num(cls, value, default: float = 0.0) -> float:
        """Parse a CWA numeric field, mapping sentinel values to default."""
        try:
            num = float(value)
        except (TypeError, ValueError):
            return default
        if num <= -90:
            return default
        return num

    @classmethod
    def filter_stations(cls, raw_data: dict, station_names: list = None,
                        bbox: dict = None) -> List[dict]:
        """Extract relevant weather stations from raw O-A0001-001 response."""
        if not raw_data:
            return []

        stations = []
        try:
            records = raw_data.get("records", {}).get("Station", [])
        except (AttributeError, KeyError):
            return []

        for stn in records:
            stn_name = stn.get("StationName", "")
            coords = stn.get("GeoInfo", {}).get("Coordinates", [{}])
            lat = cls._num(coords[0].get("StationLatitude") if coords else 0)
            lng = cls._num(coords[0].get("StationLongitude") if coords else 0)

            name_match = station_names and any(n in stn_name for n in station_names)
            bbox_match = bbox and (bbox["min_lat"] <= lat <= bbox["max_lat"] and
                                   bbox["min_lng"] <= lng <= bbox["max_lng"])
            if not (name_match or bbox_match):
                continue

            el = stn.get("WeatherElement", {})
            stations.append({
                "station_name": stn_name,
                "station_id": stn.get("StationId", ""),
                "lat": lat,
                "lng": lng,
                "elevation_m": cls._num(stn.get("GeoInfo", {}).get("StationAltitude")),
                "obs_time": stn.get("ObsTime", {}).get("DateTime", ""),
                "weather": el.get("Weather", ""),
                "wind_speed_ms": cls._num(el.get("WindSpeed")),
                "wind_gust_ms": cls._num(el.get("GustInfo", {}).get("PeakGustSpeed")),
                "wind_direction_deg": cls._num(el.get("WindDirection")),
                "temperature_c": cls._num(el.get("AirTemperature")),
                "humidity_pct": cls._num(el.get("RelativeHumidity")),
                "visibility": el.get("VisibilityDescription", ""),
            })

        return sorted(stations, key=lambda x: x["wind_gust_ms"], reverse=True)

    @classmethod
    def summarize(cls, stations: List[dict]) -> dict:
        """
        Aggregate stations into a single worst-case weather summary.

        Wind uses the maximum (worst case for treefall), temperature uses the
        maximum (worst case for heat stress), humidity uses the maximum
        (worst case for WBGT / evaporative cooling loss).
        """
        if not stations:
            return {
                "wind_speed_ms": 0.0,
                "wind_gust_ms": 0.0,
                "temperature_c": 0.0,
                "humidity_pct": 0.0,
                "station_count": 0,
            }
        return {
            "wind_speed_ms": max(s["wind_speed_ms"] for s in stations),
            "wind_gust_ms": max(s["wind_gust_ms"] for s in stations),
            "temperature_c": max(s["temperature_c"] for s in stations),
            "humidity_pct": max(s["humidity_pct"] for s in stations),
            "station_count": len(stations),
        }


# ---------------------------------------------------------------------------
# Flood Risk Assessment Engine
# ---------------------------------------------------------------------------

class FloodRiskEngine:
    """
    Two-layer debris flow / landslide risk assessment engine.

    Layer 1 — Official SWCB threshold (Taiwan) / JMA warning (Japan):
      Taiwan: Single threshold per SWCB警戒基準值.
        - Yellow alert: CWA forecast rainfall >= threshold (預測超過)
        - Red alert:   CWA actual rainfall >= threshold (實際超過)
        Ref: SWCB 土石流警戒基準值訂定方法 [1]
             Rt = R₀ + 0.7 × Σᵢ₌₁⁷ Rᵢ  (effective cumulative rainfall, α=0.7)
             RTI = Rt × I  (rainfall-triggering index)
             Threshold = R₇₀ at I=10mm/hr, per 50mm class (200-650mm range)

      Japan: Directly use JMA 土砂災害警戒情報 (Level 4).
        Ref: JMA CL = SWI × 60min rainfall [9]

    Layer 2 — Work-zone amplification (our value-add):
      Within the alert area, use DEM-derived terrain to identify
      the most dangerous sub-zones for forest workers:
        - Slope > 35° + rainfall > 80mm → slope instability [6]
        - TWI > 4.5 → high runoff potential, upgrade one level [7]
        - Flow accumulation > 5000 + 1h > 20mm → stream surge [8]

    Earthquake auto-reduction (SWCB rule):
      - Intensity >= 5強: reduce threshold by 50mm (1 級距)
      - Intensity >= 6弱: reduce threshold by 100mm (2 級距)
      Ref: SWCB 警戒值調降規則 [1]

    Site-specific thresholds (SWCB 115年明細表):
      八仙山 (和平區): 350mm (達觀/自由/南勢/天輪/平等/梨山里)
      和平區一般:      400mm
    """

    # SWCB 警戒基準值 (mm) — per-site, from 246.ardswc.gov.tw official table
    # 八仙山工作區位在和平區，鄰近村里警戒值為 350mm
    SITE_THRESHOLDS = {
        "baxianshan": 350,
        "heping": 350,
        "default_taiwan": 400,
    }

    # Watch level thresholds (below yellow alert, for worker awareness)
    # Ref: SWCB 民眾防災指南最低級 — 24h>=80mm or 3h>=40mm
    WATCH_THRESHOLDS = {
        "rain_24h_mm": 80,
        "rain_3h_mm": 40,
        "rain_1h_mm": 20,
    }

    # Wind thresholds for treefall / felling hazards (m/s)
    # Ref: 勞動部職安署 林業伐木作業安全指引 — 強風時應停止伐木作業
    #      CWA 蒲福風級: 6級(10.8-13.8m/s)樹枝搖動, 8級(17.2-20.7m/s)小枝折斷
    #      ISO/FAO forest harvesting guidance: suspend felling above ~11 m/s
    WIND_THRESHOLDS = {
        "caution_ms": 8.0,    # 蒲福5級 — 樹木搖動，注意落枝
        "warning_ms": 10.8,   # 蒲福6級 — 建議停止伐木作業
        "danger_ms": 17.2,    # 蒲福8級 — 全面停止山區作業
    }

    # Heat stress thresholds (WBGT-approximated, °C)
    # Ref: 勞動部職安署 高氣溫戶外作業熱危害預防指引 (WBGT 分級)
    #      ISO 7243 — moderate metabolic rate (forestry ≈ 300W, heavy work)
    HEAT_THRESHOLDS = {
        "caution_wbgt": 28.0,  # 注意 — 補充水分、增加休息
        "warning_wbgt": 30.0,  # 警告 — 縮短作業時間
        "danger_wbgt": 32.0,   # 危險 — 建議停止戶外重度作業
    }

    # Earthquake intensity → threshold reduction (SWCB rule)
    # Ref: SWCB 警戒值調降 — 5強降1-2級距(50-100mm), 6弱降2-4級距(100-200mm)
    EARTHQUAKE_REDUCTION = {
        "5強": 75,    # midpoint of 50-100mm
        "5Upper": 75,
        "6弱": 150,   # midpoint of 100-200mm
        "6Lower": 150,
        "6Upper": 200,
        "6強": 200,
    }

    # α weighting for antecedent rainfall (SWCB standard)
    ALPHA = 0.7

    @classmethod
    def get_site_threshold(cls, project_id: str = "baxianshan",
                           earthquake_reduction_mm: float = 0) -> float:
        """Get SWCB threshold for site, adjusted for recent earthquake."""
        base = cls.SITE_THRESHOLDS.get(project_id, cls.SITE_THRESHOLDS["default_taiwan"])
        adjusted = max(base - earthquake_reduction_mm, 200)  # 200mm is SWCB minimum
        return adjusted

    @classmethod
    def calculate_rt(cls, rain_24h_mm: float,
                     antecedent_7d_mm: list = None) -> float:
        """
        Calculate effective cumulative rainfall (Rt).
        Ref: SWCB RTI method [1], Jan & Lee (2004) [3]

        Rt = R₀ + α × Σᵢ₌₁⁷ Rᵢ
          R₀ = current 24h rainfall
          Rᵢ = daily rainfall of i-th previous day
          α  = 0.7 (SWCB standard weighting)
        """
        if not antecedent_7d_mm:
            return rain_24h_mm
        weighted_antecedent = cls.ALPHA * sum(antecedent_7d_mm[:7])
        return rain_24h_mm + weighted_antecedent

    @classmethod
    def soil_saturation_index(cls, rain_24h_mm: float,
                              antecedent_7d_mm: list = None,
                              twi_max: float = 0) -> dict:
        """
        Estimate soil saturation without in-situ soil moisture sensors.

        Proxy for JMA 土壌雨量指数 (Soil Water Index): antecedent rainfall drives
        the tank-model storage, TWI scales how much of that water converges on
        the work zone.

        saturation_pct = min(100, (Rt / reference_storage) × twi_factor × 100)
          reference_storage = 200mm (approximate field capacity of a forest
                              soil column on Taiwan mountain slopes)
          twi_factor        = 1 + (TWI - 4.0) × 0.1, clamped to [0.8, 1.5]

        Ref: JMA 土壌雨量指数 (tank model) [9]; Beven & Kirkby (1979) TWI [7]
        Note: this is an estimate. Install soil moisture sensors for direct
              measurement — schema field `soil_moisture_pct` is reserved.
        """
        rt = cls.calculate_rt(rain_24h_mm, antecedent_7d_mm)
        reference_storage_mm = 200.0

        twi_factor = 1.0
        if twi_max > 0:
            twi_factor = max(0.8, min(1.5, 1.0 + (twi_max - 4.0) * 0.1))

        saturation = min(100.0, (rt / reference_storage_mm) * twi_factor * 100)

        if saturation >= 90:
            level = "saturated"
            note = "土壤接近飽和 — 邊坡失穩與根系拔起風險顯著升高"
        elif saturation >= 70:
            level = "high"
            note = "土壤含水偏高 — 注意邊坡與樹木根系穩定性"
        elif saturation >= 40:
            level = "moderate"
            note = "土壤含水中等"
        else:
            level = "low"
            note = "土壤含水偏低"

        return {
            "saturation_pct": round(saturation, 1),
            "level": level,
            "note": note,
            "rt_mm": round(rt, 1),
            "twi_factor": round(twi_factor, 2),
            "method": "estimated (JMA 土壌雨量指数 proxy: Rt × TWI factor)",
        }

    @classmethod
    def estimate_wbgt(cls, temperature_c: float, humidity_pct: float,
                      in_shade: bool = True) -> float:
        """
        Estimate WBGT (Wet Bulb Globe Temperature) from air temperature and
        relative humidity.

        Uses the Australian Bureau of Meteorology simplified approximation:
          WBGT ≈ 0.567×Ta + 0.393×e + 3.94
          e = (RH/100) × 6.105 × exp(17.27×Ta / (237.7+Ta))   [vapour pressure, hPa]

        Forest work under canopy is largely shaded, so no solar radiation
        correction is applied by default. For open/clearcut work add ~2-3°C.

        Ref: ABM WBGT approximation; ISO 7243
        """
        if temperature_c <= 0:
            return 0.0
        vapour_pressure = (humidity_pct / 100.0) * 6.105 * math.exp(
            17.27 * temperature_c / (237.7 + temperature_c)
        )
        wbgt = 0.567 * temperature_c + 0.393 * vapour_pressure + 3.94
        if not in_shade:
            wbgt += 2.5  # open-area solar radiation correction
        return round(wbgt, 1)

    @classmethod
    def assess_wind_hazard(cls, wind_speed_ms: float, wind_gust_ms: float = 0,
                           soil_saturation_pct: float = 0,
                           canopy_height_m: float = 0) -> dict:
        """
        Assess treefall / falling-branch hazard from wind.

        Saturated soil markedly reduces root anchorage, so the wind speed
        needed to uproot a tree drops. Tall stands catch more wind load.

        Ref: 職安署 林業伐木作業安全指引; Peltola et al. (1999) wind damage model;
             Sidle & Ochiai (2006) root reinforcement
        """
        effective_wind = max(wind_speed_ms, wind_gust_ms)

        # Saturated soil lowers the effective threshold (root anchorage loss)
        threshold_adjust = 0.0
        adjust_note = ""
        if soil_saturation_pct >= 90:
            threshold_adjust = 3.0
            adjust_note = "（土壤飽和，根系固定力下降，門檻調降 3m/s）"
        elif soil_saturation_pct >= 70:
            threshold_adjust = 1.5
            adjust_note = "（土壤含水高，門檻調降 1.5m/s）"

        # Tall stands are more exposed to wind loading
        if canopy_height_m >= 25:
            threshold_adjust += 1.0
            adjust_note += "（高林分受風面大，門檻再降 1m/s）"

        danger = cls.WIND_THRESHOLDS["danger_ms"] - threshold_adjust
        warning = cls.WIND_THRESHOLDS["warning_ms"] - threshold_adjust
        caution = cls.WIND_THRESHOLDS["caution_ms"] - threshold_adjust

        if effective_wind >= danger:
            level, action = "danger", "建議全面停止山區作業，撤至安全處所"
        elif effective_wind >= warning:
            level, action = "warning", "建議停止伐木與高空作業，注意倒木與落枝"
        elif effective_wind >= caution:
            level, action = "caution", "樹木搖動，注意落枝，避免於大樹下停留"
        else:
            level, action = "safe", ""

        return {
            "level": level,
            "wind_speed_ms": round(wind_speed_ms, 1),
            "wind_gust_ms": round(wind_gust_ms, 1),
            "effective_wind_ms": round(effective_wind, 1),
            "beaufort": cls._beaufort_scale(effective_wind),
            "action": action,
            "threshold_ms": round(warning, 1),
            "threshold_note": adjust_note,
        }

    @classmethod
    def _beaufort_scale(cls, wind_ms: float) -> str:
        """Convert wind speed (m/s) to Beaufort scale description."""
        scale = [
            (0.3, "0級 無風"), (1.6, "1級 軟風"), (3.4, "2級 輕風"),
            (5.5, "3級 微風"), (8.0, "4級 和風"), (10.8, "5級 清風"),
            (13.9, "6級 強風"), (17.2, "7級 疾風"), (20.8, "8級 大風"),
            (24.5, "9級 烈風"), (28.5, "10級 狂風"), (32.7, "11級 暴風"),
        ]
        for limit, label in scale:
            if wind_ms < limit:
                return label
        return "12級 颶風"

    @classmethod
    def assess_heat_hazard(cls, temperature_c: float, humidity_pct: float,
                           in_shade: bool = True) -> dict:
        """
        Assess heat stress hazard for outdoor forest work.

        Ref: 勞動部職安署 高氣溫戶外作業熱危害預防指引; ISO 7243 (WBGT)
        """
        wbgt = cls.estimate_wbgt(temperature_c, humidity_pct, in_shade)

        if wbgt >= cls.HEAT_THRESHOLDS["danger_wbgt"]:
            level = "danger"
            action = "建議停止重度戶外作業，每小時休息 ≥30 分鐘並補充含電解質飲水"
        elif wbgt >= cls.HEAT_THRESHOLDS["warning_wbgt"]:
            level = "warning"
            action = "縮短連續作業時間，每小時休息 ≥15 分鐘，設置遮蔭與飲水點"
        elif wbgt >= cls.HEAT_THRESHOLDS["caution_wbgt"]:
            level = "caution"
            action = "增加飲水頻率，注意同伴是否出現頭暈、噁心等熱危害徵兆"
        else:
            level = "safe"
            action = ""

        return {
            "level": level,
            "wbgt_c": wbgt,
            "temperature_c": round(temperature_c, 1),
            "humidity_pct": round(humidity_pct, 1),
            "action": action,
            "method": "WBGT 推估 (ABM 近似式，樹冠遮蔭)" if in_shade else "WBGT 推估 (含日射修正)",
        }

    # Shared hazard severity scale, ordered from least to most severe.
    # All hazards (landslide, stream surge, treefall, heat) map onto this scale
    # so that one overall safety level can be derived.
    HAZARD_LEVELS = ["safe", "caution", "warning", "danger"]

    # Mapping between the official SWCB/JMA alert level and the hazard scale
    _RISK_TO_HAZARD = {
        "safe": "safe",
        "watch": "caution",
        "yellow_alert": "warning",
        "red_alert": "danger",
    }
    _HAZARD_TO_RISK = {
        "safe": "safe",
        "caution": "watch",
        "warning": "yellow_alert",
        "danger": "red_alert",
    }

    @classmethod
    def _to_hazard_level(cls, risk_level: str) -> str:
        """Map an official alert level onto the shared hazard severity scale."""
        return cls._RISK_TO_HAZARD.get(risk_level, "safe")

    @classmethod
    def _aggregate_safety_level(cls, hazards: list) -> str:
        """
        Derive the overall safety level from all assessed hazards.

        Takes the worst hazard level and expresses it on the alert-level scale
        (safe / watch / yellow_alert / red_alert) so downstream consumers keep
        a single vocabulary.
        """
        if not hazards:
            return "safe"
        worst_idx = max(
            (
                cls.HAZARD_LEVELS.index(h["level"])
                for h in hazards
                if h.get("level") in cls.HAZARD_LEVELS
            ),
            default=0,
        )
        return cls._HAZARD_TO_RISK[cls.HAZARD_LEVELS[worst_idx]]

    @classmethod
    def check_earthquake_reduction(cls, earthquake_data: dict = None) -> float:
        """
        Check recent earthquake data and return threshold reduction (mm).
        Returns 0 if no significant earthquake in the period.

        Ref: SWCB rule —震度5強降50-100mm, 6弱降100-200mm
        """
        if not earthquake_data:
            return 0
        try:
            eq_records = earthquake_data.get("records", {}).get("Earthquake", [])
        except (AttributeError, KeyError):
            return 0

        max_reduction = 0
        for eq in eq_records:
            intensities = eq.get("Intensity", [])
            for station_intensity in intensities:
                area_intensity = station_intensity.get("AreaIntensity", "")
                for level, reduction in cls.EARTHQUAKE_REDUCTION.items():
                    if level in str(area_intensity):
                        max_reduction = max(max_reduction, reduction)
        return max_reduction

    @classmethod
    def assess_landslide_risk(cls,
                              rain_1h_mm: float, rain_3h_mm: float,
                              rain_24h_mm: float,
                              forecast_24h_mm: float = 0,
                              antecedent_7d_mm: list = None,
                              twi_max: float = 0,
                              slope_deg: float = 0,
                              flow_accum_max: float = 0,
                              canopy_height_m: float = 0,
                              ndvi: float = 0,
                              aspect_deg: float = 0,
                              geomorphon: int = 0,
                              elevation_m: float = 0,
                              wind_speed_ms: float = 0,
                              wind_gust_ms: float = 0,
                              temperature_c: float = 0,
                              humidity_pct: float = 0,
                              soil_type: str = "unknown",
                              accessibility_minutes: float = 0,
                              wildlife: str = "none",
                              abnormal_trees: str = "unknown",
                              region: str = "taiwan",
                              project_id: str = "baxianshan",
                              earthquake_data: dict = None,
                              jma_warning: dict = None) -> dict:
        """
        Multi-hazard occupational safety assessment for forest operations.

        The primary purpose is protecting worker life and property, so landslide
        is one hazard among several. Landslide keeps the official SWCB/JMA alert
        level (`risk_level`), while `safety_level` is the worst level across all
        assessed hazards and is what drives notifications.

        Hazards assessed:
          - landslide / debris flow  (rainfall Rt vs SWCB threshold + terrain)
          - stream surge             (short-duration rainfall + flow accumulation)
          - treefall / falling limb  (wind, reduced by soil saturation)
          - heat stress              (WBGT from temperature + humidity)

        Layer 1 — Official alert level (SWCB for TW, JMA for JP):
          safe < watch < yellow_alert < red_alert

          Taiwan (SWCB method):
            Yellow: CWA forecast 24h >= threshold (預測超過 → 勸告疏散)
            Red:    actual Rt >= threshold (實際超過 → 強制疏散)
            Single threshold per site (350mm for 八仙山), earthquake-adjusted.

          Japan (JMA):
            Directly use JMA 土砂災害警戒情報 if provided.

        Layer 2 — Work-zone amplification:
          Terrain factors that identify high-risk sub-zones within the alert area.
          These do NOT change the official alert level, but provide actionable
          detail for forest workers (which slope, which road segment, which zone).

        Returns dict with:
          - risk_level: safe/watch/yellow_alert/red_alert
          - alert_source: 'swcb_forecast' / 'swcb_actual' / 'jma_official' / 'terrain_watch'
          - factors: triggered criteria (Layer 1)
          - work_zone_risks: terrain-specific risks (Layer 2)
          - evacuation_needed: bool (True only for red_alert)
          - affected_zones: description of at-risk work areas
          - rain_summary: observed + forecast rainfall
          - thresholds: values used for this assessment
          - methodology: citation list
        """
        factors = []
        work_zone_risks = []
        risk_level = "safe"
        alert_source = ""
        alert_type = ""  # "forecast" or "realtime"
        timing_info = ""  # human-readable timing context

        # ── Earthquake threshold reduction ──
        eq_reduction = cls.check_earthquake_reduction(earthquake_data)
        threshold = cls.get_site_threshold(project_id, eq_reduction)

        # ── Calculate Rt (effective cumulative rainfall) ──
        rt = cls.calculate_rt(rain_24h_mm, antecedent_7d_mm)

        # ── Japan: use JMA official warning if provided ──
        if jma_warning:
            jma_level = jma_warning.get("level", "")
            if "level5" in jma_level.lower() or "特別警報" in jma_warning.get("headline", ""):
                risk_level = "red_alert"
                alert_source = "jma_official"
                factors.append(f"JMA土砂災害特別警報 (Level 5): {jma_warning.get('headline', '')}")
            elif "level4" in jma_level.lower() or "警戒情報" in jma_warning.get("headline", ""):
                risk_level = "red_alert"
                alert_source = "jma_official"
                factors.append(f"JMA土砂災害警戒情報 (Level 4): {jma_warning.get('headline', '')}")
            elif "level3" in jma_level.lower():
                risk_level = "yellow_alert"
                alert_source = "jma_official"
                factors.append(f"JMA土砂災害注意情報 (Level 3): {jma_warning.get('headline', '')}")

        # ── Taiwan: SWCB single-threshold method ──
        if risk_level == "safe" and forecast_24h_mm > 0:
            # Yellow alert: forecast exceeds threshold (預測超過)
            if forecast_24h_mm >= threshold:
                risk_level = "yellow_alert"
                alert_source = "swcb_forecast"
                alert_type = "forecast"
                timing_info = f"基於 CWA 36小時天氣預報，預估未來 24 小時累積雨量將達 {forecast_24h_mm:.0f}mm，超過警戒基準值 {threshold:.0f}mm"
                factors.append(
                    f"預報24h雨量 {forecast_24h_mm:.0f}mm >= 警戒基準值 {threshold:.0f}mm"
                    f"{' (地震調降' + f'{eq_reduction:.0f}mm)' if eq_reduction > 0 else ''}"
                )

        if risk_level in ("safe", "yellow_alert"):
            # Red alert: actual Rt exceeds threshold (實際超過)
            if rt >= threshold:
                risk_level = "red_alert"
                alert_source = "swcb_actual"
                alert_type = "realtime"
                remaining = threshold - rt
                timing_info = f"實際有效累積雨量 Rt={rt:.0f}mm 已超過警戒基準值 {threshold:.0f}mm，目前雨量持續累積中"
                factors.append(
                    f"有效累積雨量 Rt={rt:.0f}mm (R₀={rain_24h_mm:.0f} + 0.7×API="
                    f"{rt - rain_24h_mm:.0f}) >= 警戒基準值 {threshold:.0f}mm"
                )

        # ── Watch level (below yellow, for worker awareness) ──
        if risk_level == "safe":
            if (rain_24h_mm >= cls.WATCH_THRESHOLDS["rain_24h_mm"] or
                rain_3h_mm >= cls.WATCH_THRESHOLDS["rain_3h_mm"] or
                rain_1h_mm >= cls.WATCH_THRESHOLDS["rain_1h_mm"]):
                risk_level = "watch"
                alert_source = "terrain_watch"
                alert_type = "realtime"
                pct = (rt / threshold * 100) if threshold > 0 else 0
                timing_info = f"目前 Rt={rt:.0f}mm / 警戒值={threshold:.0f}mm ({pct:.0f}%)，降雨持續中但尚未達警戒標準"
                factors.append(
                    f"降雨持續中：1h={rain_1h_mm:.0f}mm, 3h={rain_3h_mm:.0f}mm, "
                    f"24h={rain_24h_mm:.0f}mm (低於警戒值，需注意)"
                )

        # ── Layer 2: Work-zone terrain amplification ──
        # These do NOT change official alert level, but identify specific danger zones
        # All factors are from DEM/GIS analysis (static), combined with realtime rainfall
        if risk_level in ("watch", "yellow_alert", "red_alert"):
            # Hydro factors
            if twi_max > 4.5:
                work_zone_risks.append(
                    f"TWI={twi_max:.1f} (高逕流潛勢區) — 水文敏感區，溪流暴漲風險高"
                )
            if flow_accum_max > 5000 and rain_1h_mm > 20:
                work_zone_risks.append(
                    f"匯流面積大 (accum={flow_accum_max:.0f}) — 溪流暴漲風險高"
                )

            # Terrain factors
            if slope_deg > 35 and rain_24h_mm > 80:
                work_zone_risks.append(
                    f"坡度 {slope_deg:.0f}° > 35° 搭配降雨 — 邊坡不穩定，林道崩塌風險高"
                )
            elif slope_deg > 30 and rain_24h_mm > 50:
                work_zone_risks.append(
                    f"坡度 {slope_deg:.0f}° > 30° 搭配降雨 — 邊坡穩定性下降，注意林道路況"
                )
            if aspect_deg > 0:
                aspect_name = {0: "北", 45: "東北", 90: "東", 135: "東南",
                               180: "南", 225: "西南", 270: "西", 315: "西北"}.get(
                    round(aspect_deg / 45) * 45 % 360, "")
                if aspect_name in ("南", "西南", "東南") and slope_deg > 25:
                    work_zone_risks.append(
                        f"坡向 {aspect_name}向 ({aspect_deg:.0f}°) — 陽坡面，土壤含水較低但日雨後易快速飽和"
                    )
            if geomorphon in (3, 4, 5) and slope_deg > 25:
                geomorphon_name = {3: "山脊", 4: "肩部", 5: "陡坡"}.get(geomorphon, "")
                work_zone_risks.append(
                    f"地形位 {geomorphon_name} (geomorphon={geomorphon}) — 陵線/陡坡區，淺層崩塌風險高"
                )
            if geomorphon == 9 and twi_max > 3:
                work_zone_risks.append(
                    f"地形位 谷地 (geomorphon=9) — 匯流區，降雨後地下水位快速上升"
                )

            # Forest/vegetation factors
            if canopy_height_m > 0 and canopy_height_m < 5:
                risk_note = "裸露地/新植栽" if canopy_height_m < 2 else "低矮植被"
                work_zone_risks.append(
                    f"冠層高 {canopy_height_m:.1f}m ({risk_note}) — 根系固土能力弱，淺層崩塌風險高"
                )
            if ndvi > 0 and ndvi < 0.4:
                work_zone_risks.append(
                    f"NDVI={ndvi:.2f} (<0.4) — 植被覆蓋差，地表裸露，沖蝕/崩塌風險高"
                )

            # Elevation factor
            if elevation_m > 1500 and rain_1h_mm > 15:
                work_zone_risks.append(
                    f"海拔 {elevation_m:.0f}m — 高海拔區降雨強度可能放大，注意強風倒木"
                )

        # ── Soil saturation (JMA 土壌雨量指数 proxy) ──
        soil = cls.soil_saturation_index(rain_24h_mm, antecedent_7d_mm, twi_max)
        if soil["level"] in ("saturated", "high") and risk_level != "safe":
            work_zone_risks.append(
                f"土壤飽和度 {soil['saturation_pct']:.0f}% ({soil['level']}) — {soil['note']}"
            )

        # ── Non-landslide occupational hazards ──
        hazards = []

        # Hazard 1: landslide / debris flow (the official alert level)
        if risk_level != "safe":
            hazards.append({
                "type": "landslide",
                "name": "崩塌／土石流",
                "level": cls._to_hazard_level(risk_level),
                "detail": f"Rt={rt:.0f}mm / 警戒值={threshold:.0f}mm",
                "action": {
                    "watch": "注意邊坡與溪流狀態，確認撤離路線",
                    "yellow_alert": "建議延期上山作業，避開溪流匯流處與崩塌地",
                    "red_alert": "建議立即停止作業，避開水文敏感區與高崩塌風險路段",
                }.get(risk_level, ""),
            })

        # Hazard 2: stream surge (short-duration rainfall on a large catchment)
        if flow_accum_max > 3000 and (rain_1h_mm >= 15 or rain_3h_mm >= 30):
            surge_level = "danger" if rain_1h_mm >= 30 else "warning"
            hazards.append({
                "type": "stream_surge",
                "name": "溪流暴漲",
                "level": surge_level,
                "detail": f"1h={rain_1h_mm:.0f}mm, 3h={rain_3h_mm:.0f}mm, "
                          f"匯流面積={flow_accum_max:.0f}",
                "action": "立即遠離溪床與河道，勿試圖涉水或渡溪",
            })

        # Hazard 3: treefall / falling limbs (wind, amplified by wet soil)
        wind = cls.assess_wind_hazard(
            wind_speed_ms, wind_gust_ms,
            soil_saturation_pct=soil["saturation_pct"],
            canopy_height_m=canopy_height_m,
        )
        if wind["level"] != "safe":
            hazards.append({
                "type": "treefall",
                "name": "倒木／落枝",
                "level": wind["level"],
                "detail": f"風速 {wind['effective_wind_ms']:.1f}m/s ({wind['beaufort']})，"
                          f"門檻 {wind['threshold_ms']:.1f}m/s{wind['threshold_note']}",
                "action": wind["action"],
            })

        # Hazard 4: heat stress (WBGT)
        heat = cls.assess_heat_hazard(temperature_c, humidity_pct, in_shade=True)
        if heat["level"] != "safe":
            hazards.append({
                "type": "heat_stress",
                "name": "熱危害",
                "level": heat["level"],
                "detail": f"WBGT≈{heat['wbgt_c']:.1f}°C "
                          f"(氣溫 {heat['temperature_c']:.1f}°C, 濕度 {heat['humidity_pct']:.0f}%)",
                "action": heat["action"],
            })

        # ── Overall safety level (worst across all hazards) ──
        safety_level = cls._aggregate_safety_level(hazards)

        # ── Composite GIS-AHP safety index ──
        # Reported for planning and site comparison. Deliberately NOT fed into
        # safety_level: a weighted average would let good conditions elsewhere
        # dilute a threshold breach, and fixed terrain would pin a permanent
        # elevated score that desensitises the crew.
        safety_index = assess_safety_index(
            region,
            slope_deg=slope_deg,
            elevation_m=elevation_m,
            soil_type=soil_type,
            ndvi=ndvi,
            accessibility_minutes=accessibility_minutes,
            twi=twi_max,
            flow_accumulation=flow_accum_max,
            geomorphon=geomorphon,
            aspect_deg=aspect_deg,
            rt_mm=rt,
            wind_speed_ms=wind["effective_wind_ms"],
            wbgt_c=heat["wbgt_c"],
            wildlife=wildlife,
            abnormal_trees=abnormal_trees,
        )

        # ── Evacuation decision (only red_alert) ──
        evacuation_needed = risk_level == "red_alert"

        # ── Affected work zones ──
        affected_zones = []
        if risk_level in ("yellow_alert", "red_alert"):
            affected_zones.append("水文敏感區（TWI高值區）")
            affected_zones.append("溪流匯流處下游")
            affected_zones.append("坡度>35°且崩塌比例高之路段")
            if flow_accum_max > 3000:
                affected_zones.append("主要排水線兩側50m範圍")

        return {
            "risk_level": risk_level,
            "safety_level": safety_level,
            "alert_source": alert_source,
            "alert_type": alert_type,
            "timing_info": timing_info,
            "hazards": hazards,
            "safety_index": safety_index,
            "region": region,
            "soil_saturation": soil,
            "weather_conditions": {
                "wind_speed_ms": round(wind_speed_ms, 1),
                "wind_gust_ms": round(wind_gust_ms, 1),
                "beaufort": wind["beaufort"],
                "temperature_c": round(temperature_c, 1),
                "humidity_pct": round(humidity_pct, 1),
                "wbgt_c": heat["wbgt_c"],
            },
            "factors": factors,
            "work_zone_risks": work_zone_risks,
            "evacuation_needed": evacuation_needed,
            "affected_zones": affected_zones,
            "terrain_factors": {
                "slope_deg": slope_deg,
                "twi_max": twi_max,
                "flow_accum_max": flow_accum_max,
                "canopy_height_m": canopy_height_m,
                "ndvi": ndvi,
                "aspect_deg": aspect_deg,
                "geomorphon": geomorphon,
                "elevation_m": elevation_m,
            },
            "rain_summary": {
                "1h_mm": rain_1h_mm,
                "3h_mm": rain_3h_mm,
                "24h_mm": rain_24h_mm,
                "forecast_24h_mm": forecast_24h_mm,
                "rt_effective_mm": round(rt, 1),
                "antecedent_7d_mm": sum(antecedent_7d_mm[:7]) if antecedent_7d_mm else 0,
            },
            "thresholds": {
                "site_base_mm": cls.SITE_THRESHOLDS.get(project_id, 400),
                "earthquake_reduction_mm": eq_reduction,
                "effective_threshold_mm": threshold,
                "watch_24h_mm": cls.WATCH_THRESHOLDS["rain_24h_mm"],
                "watch_3h_mm": cls.WATCH_THRESHOLDS["rain_3h_mm"],
            },
            "methodology": [
                "SWCB 土石流警戒基準值 (單一閾值，黃=預測/紅=實際) [1]",
                f"預報型警報：CWA 36h 預報 Wx 代碼估算 24h 雨量 >= 警戒值 → 預警（提前 24h）",
                f"即時型警報：實際有效累積雨量 Rt >= 警戒值 → 警戒（已發生）",
                f"Rt = R₀ + 0.7×API (有效累積雨量) [3], 本次 Rt={rt:.0f}mm",
                f"警戒基準值={threshold:.0f}mm (和平區350mm{f'-地震調降{eq_reduction:.0f}mm' if eq_reduction > 0 else ''})",
                "工作區位加值: slope×TWI×flow_accum [6][7][8]",
                "TWI (地形濕潤指數) = ln(上游集水面積 / tan(局部坡度))，由 DEM 水文分析計算。值越高代表越容易積水匯流，>4.5 為水文敏感區",
                "JMA 日本做法：土壌雨量指数 + 60分雨量，Level 3(3h前) → Level 4(2h前) → Level 5(實況)，各級有明確 lead time",
                f"土壤飽和度推估 {soil['saturation_pct']:.0f}% — {soil['method']}（無土壤水分感測器時之替代指標）",
                "倒木／落枝：風速門檻 8/10.8/17.2 m/s（蒲福5/6/8級），土壤飽和時門檻調降（根系固定力下降）[6]",
                "熱危害：WBGT 由氣溫與相對濕度推估（ABM 近似式），分級依職安署高氣溫戶外作業指引 (28/30/32°C)",
                "多危害綜合：safety_level = 各危害等級之最大值（崩塌、溪流暴漲、倒木、熱危害）",
                "多危害架構參考 Rahmawati, Yovi & Setiawan (2025) GIS-AHP 森林作業職安評估框架, Eur. J. Forest Eng. 11(2)",
                f"GIS-AHP 複合風險指數 = {safety_index['composite_score']:.2f}/5 "
                f"（{safety_index['class_label']}級，區域={safety_index['region_name']}，"
                f"動態氣象占比 {safety_index['dynamic_share']*100:.0f}%）",
                "AHP 權重由階層式成對比較矩陣推導（Saaty 1980 特徵向量法），"
                f"一致性比率 CR={max(safety_index['consistency_ratios'].values()):.3f} <= 0.10",
                "複合指數用於區位比較與作業規劃；警報觸發仍以官方門檻為準，不受加權平均稀釋",
            ],
        }

    @classmethod
    def generate_warning(cls, risk_level: str, rainfall_mm: float,
                         region: str) -> str:
        """Generate warning message in appropriate language."""
        if region.startswith("jp") or region in ["kochi", "shimanto"]:
            messages = {
                "red_alert": f"⛔ 警戒 — {rainfall_mm:.0f}mm 降水。土砂災害・河川増水に厳重警戒。林道通行禁止。作業中止を推奨。",
                "yellow_alert": f"⚠️ 預警 — {rainfall_mm:.0f}mm 降水予報。土砂災害に警戒。上山を控えることを推奨。",
                "watch": f"🟡 注意 — {rainfall_mm:.0f}mm 降水。渓流の急増水に注意。排水施設の事前点検を推奨。",
                "safe": f"✅ 降水量 {rainfall_mm:.0f}mm — 通常作業可能。",
            }
        else:
            messages = {
                "red_alert": f"⛔ 警戒 — 實際雨量已達警戒值，建議立即停止作業並避開水文敏感區、溪流兩側與高崩塌風險路段。",
                "yellow_alert": f"⚠️ 預警 — 預報降雨將達警戒值，建議避免上山，避開溪流匯流處與崩塌地。",
                "watch": f"🟡 注意 — 降雨持續中，注意溪流變化。建議事先檢查排水設施。",
                "safe": f"✅ 降水量 {rainfall_mm:.0f}mm — 可正常作業。",
            }
        return messages.get(risk_level, messages["safe"])
