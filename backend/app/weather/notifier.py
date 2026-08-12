"""
SylvaNexus — Weather Alert Notification Service
=================================================
Sends landslide/flood warnings via LINE Notify and Email.

LINE Notify: https://notify-bot.line.me/
Email: Uses existing SMTP configuration (Gmail app password)
"""

import os
import httpx
from datetime import datetime, timezone
from typing import Optional, List

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

# Lazy singleton connection pool for recipient lookups
_db_pool = None

def _get_db_pool():
    """Get or create a psycopg2 connection pool."""
    global _db_pool
    if _db_pool is None or _db_pool.closed:
        try:
            from app.auth.database import get_db_config
            config = get_db_config()
            _db_pool = pool.ThreadedConnectionPool(
                minconn=1, maxconn=5, **config
            )
        except Exception as e:
            print(f"[Notify] Failed to create DB pool: {e}")
            return None
    return _db_pool


# ---------------------------------------------------------------------------
# LINE Notify
# ---------------------------------------------------------------------------

class LINENotifier:
    """
    LINE Notify API wrapper.
    Token obtained from https://notify-bot.line.me/my/

    Setup:
    1. Go to https://notify-bot.line.me/my/
    2. Click "Generate token"
    3. Select a chat room (personal or group)
    4. Copy token → set as LINE_NOTIFY_TOKEN env var
    """

    API_URL = "https://notify-api.line.me/api/notify"

    @classmethod
    def _get_token(cls) -> str:
        return os.getenv("LINE_NOTIFY_TOKEN", "")

    @classmethod
    async def send(cls, message: str) -> bool:
        """Send a LINE Notify message. Returns True on success."""
        token = cls._get_token()
        if not token:
            print("[Notify/LINE] LINE_NOTIFY_TOKEN not set, skipping")
            return False

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.post(
                    cls.API_URL,
                    headers={"Authorization": f"Bearer {token}"},
                    data={"message": message},
                )
                if resp.status_code == 200:
                    print(f"[Notify/LINE] ✅ Sent: {message[:60]}...")
                    return True
                else:
                    print(f"[Notify/LINE] ❌ {resp.status_code}: {resp.text}")
                    return False
            except Exception as e:
                print(f"[Notify/LINE] ❌ Error: {e}")
                return False


# ---------------------------------------------------------------------------
# Email Alert
# ---------------------------------------------------------------------------

def get_project_recipients(project_id: str) -> List[str]:
    """Fetch email addresses of all users with access to the given project."""
    pg_pool = _get_db_pool()
    if not pg_pool:
        return []
    try:
        conn = pg_pool.getconn()
        conn.autocommit = True
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT u.email FROM auth.users u
                    JOIN auth.project_permissions pp ON u.user_id = pp.user_id
                    WHERE pp.project_id = %s
                      AND u.is_active = true
                      AND COALESCE(pp.alert_subscribed, true) = true
                    ORDER BY u.user_id
                """, (project_id,))
                rows = cur.fetchall()
                emails = [r["email"] for r in rows if r.get("email")]
                print(f"[Notify] Found {len(emails)} recipients for project '{project_id}'")
                return emails
        finally:
            pg_pool.putconn(conn)
    except Exception as e:
        print(f"[Notify] Failed to fetch project recipients: {e}")
        return []


class EmailAlertNotifier:
    """Send weather alert emails using existing EmailService."""

    @classmethod
    def send(cls, assessment: dict, recipients: list = None) -> bool:
        """Send email alert for landslide risk."""
        try:
            from app.email_service import EmailService
            svc = EmailService()

            if not recipients:
                # Fetch all users with access to this project
                project_id = assessment.get("project_id", "baxianshan")
                recipients = get_project_recipients(project_id)
                if not recipients:
                    # Fallback to admin email if no project users found
                    recipients = [svc.admin_email]
                    print(f"[Notify/Email] No project recipients found, falling back to admin: {svc.admin_email}")

            risk_level = assessment.get("risk_level", "safe")
            safety_level = assessment.get("safety_level", risk_level)
            hazards = assessment.get("hazards", [])
            soil = assessment.get("soil_saturation", {})
            conditions = assessment.get("weather_conditions", {})
            rain = assessment.get("rain_summary", {})
            factors = assessment.get("factors", [])
            work_zone_risks = assessment.get("work_zone_risks", [])
            warning = assessment.get("warning_message", "")
            evac = assessment.get("evacuation_needed", False)
            zones = assessment.get("affected_zones", [])
            thresholds = assessment.get("thresholds", {})
            methodology = assessment.get("methodology", [])
            ts = assessment.get("assessment_time", datetime.now(timezone.utc).isoformat())
            alert_type = assessment.get("alert_type", "")
            timing_info = assessment.get("timing_info", "")
            terrain = assessment.get("terrain_factors", {})
            type_label = "預報型" if alert_type == "forecast" else "即時型" if alert_type == "realtime" else ""

            level_colors = {
                "safe": "#4caf50",
                "watch": "#ff9800",
                "yellow_alert": "#ffc107",
                "red_alert": "#f44336",
            }
            level_labels = {
                "safe": "✅ 安全",
                "watch": "🟡 注意",
                "yellow_alert": "⚠️ 預警",
                "red_alert": "⛔ 警戒",
            }

            # Header reflects the overall safety level across all hazards
            color = level_colors.get(safety_level, "#999")
            label = level_labels.get(safety_level, safety_level)

            # Location info
            lat = assessment.get("lat", 24.2633)
            lng = assessment.get("lng", 120.9500)
            location_name = assessment.get("location_name", "八仙山")
            hiiforest_url = f"https://hiiforest.com/app/index_modular.html#lat={lat}&lng={lng}&z=14"
            gmaps_url = f"https://www.google.com/maps?q={lat},{lng}"

            hazard_names = "、".join(h["name"] for h in hazards)
            subject = (
                f"⚠️ HiiForest {location_name}作業安全預警 — {label}"
                f"{'：' + hazard_names if hazard_names else ''}"
            )

            # Hazard breakdown — one card per hazard with its suggested action
            hazard_level_style = {
                "caution": ("#ff9800", "注意"),
                "warning": ("#ffc107", "預警"),
                "danger": ("#f44336", "警戒"),
            }
            hazard_html = ""
            for h in hazards:
                h_color, h_label = hazard_level_style.get(h["level"], ("#999", h["level"]))
                action_html = (
                    f'<div style="margin-top:4px; color:#333;">建議：{h["action"]}</div>'
                    if h.get("action") else ""
                )
                hazard_html += (
                    f'<div class="info-row" style="border-color:{h_color};">'
                    f'<span class="label">{h["name"]}</span> '
                    f'<span style="background:{h_color}; color:white; padding:1px 8px; '
                    f'border-radius:10px; font-size:12px;">{h_label}</span>'
                    f'<div style="font-size:13px; color:#666; margin-top:4px;">{h["detail"]}</div>'
                    f'{action_html}</div>'
                )

            # Current field conditions (wind / temperature / soil saturation)
            condition_items = []
            if conditions.get("wind_speed_ms", 0) > 0:
                condition_items.append(
                    f"風速: {conditions['wind_speed_ms']:.1f}m/s ({conditions.get('beaufort', '')})"
                )
            if conditions.get("wind_gust_ms", 0) > 0:
                condition_items.append(f"最大陣風: {conditions['wind_gust_ms']:.1f}m/s")
            if conditions.get("temperature_c", 0) > 0:
                condition_items.append(f"氣溫: {conditions['temperature_c']:.1f}°C")
            if conditions.get("humidity_pct", 0) > 0:
                condition_items.append(f"濕度: {conditions['humidity_pct']:.0f}%")
            if conditions.get("wbgt_c", 0) > 0:
                condition_items.append(f"WBGT: {conditions['wbgt_c']:.1f}°C")
            if soil.get("saturation_pct", 0) > 0:
                condition_items.append(f"土壤飽和度: {soil['saturation_pct']:.0f}% (推估)")
            conditions_html = " ｜ ".join(condition_items) if condition_items else "無資料"

            # Composite index — labelled as a planning reference so nobody reads
            # it as the reason work was stopped.
            index = assessment.get("safety_index", {})
            index_html = ""
            if index:
                drivers = "、".join(
                    f"{d['label']}({d['score']})" for d in index.get("top_drivers", [])
                )
                index_html = (
                    f'<div class="section-title">作業環境風險指數（規劃參考）</div>'
                    f'<div class="info-row" style="border-color:{index["class_colour"]};">'
                    f'<span class="label">GIS-AHP 複合指數：</span> '
                    f'<span style="font-size:20px; font-weight:bold; '
                    f'color:{index["class_colour"]};">{index["composite_score"]:.2f}</span> / 5.00 '
                    f'（{index["class_label"]}級）'
                    f'<div style="font-size:12px; color:#666; margin-top:4px;">'
                    f'地形固有風險 {index["static_score"]:.2f}（{index["static_class_label"]}級）'
                    f' ｜ 動態氣象占比 {index["dynamic_share"]*100:.0f}%</div>'
                    f'<div style="font-size:12px; color:#666;">主要貢獻：{drivers}</div>'
                    f'<div style="font-size:11px; color:#999; margin-top:4px;">'
                    f'※ 此指數供區位比較與作業規劃，警報判定仍以官方門檻為準</div>'
                    f'</div>'
                )

            factors_html = "".join(f"<li>{f}</li>" for f in factors) if factors else "<li>無觸發因子</li>"
            zones_html = "".join(f"<li>{z}</li>" for z in zones) if zones else "<li>無</li>"
            work_zone_html = "".join(f"<li>{w}</li>" for w in work_zone_risks) if work_zone_risks else ""
            methodology_html = "".join(f"<li>{m}</li>" for m in methodology) if methodology else ""

            # Terrain factor summary table
            terrain_items = []
            if terrain.get("slope_deg", 0) > 0:
                terrain_items.append(f"坡度: {terrain['slope_deg']:.0f}°")
            if terrain.get("twi_max", 0) > 0:
                terrain_items.append(f"TWI: {terrain['twi_max']:.1f}")
            if terrain.get("flow_accum_max", 0) > 0:
                terrain_items.append(f"匯流面積: {terrain['flow_accum_max']:.0f}")
            if terrain.get("canopy_height_m", 0) > 0:
                terrain_items.append(f"冠層高: {terrain['canopy_height_m']:.1f}m")
            if terrain.get("ndvi", 0) > 0:
                terrain_items.append(f"NDVI: {terrain['ndvi']:.2f}")
            if terrain.get("elevation_m", 0) > 0:
                terrain_items.append(f"海拔: {terrain['elevation_m']:.0f}m")
            if terrain.get("aspect_deg", 0) > 0:
                _aspect = round(terrain['aspect_deg'] / 45) * 45 % 360
                _aspect_name = {0:"北",45:"東北",90:"東",135:"東南",180:"南",225:"西南",270:"西",315:"西北"}.get(_aspect,"")
                terrain_items.append(f"坡向: {_aspect_name}({terrain['aspect_deg']:.0f}°)")
            if terrain.get("geomorphon", 0) > 0:
                _geo_name = {1:"平坦",2:"凹地",3:"山脊",4:"肩部",5:"陡坡",6:"緩坡",7:"山麓",8:"開闊坡",9:"谷地"}.get(terrain['geomorphon'], str(terrain['geomorphon']))
                terrain_items.append(f"地形位: {_geo_name}")
            terrain_html = " ｜ ".join(terrain_items) if terrain_items else "無資料"

            # Rain grid: now includes forecast and Rt
            forecast_val = rain.get('forecast_24h_mm', 0)
            rt_val = rain.get('rt_effective_mm', 0)
            # Display forecast as '--' when 0 (CWA Wx code has no heavy rain advisory → not a quantitative 0)
            forecast_display = f'{forecast_val:.0f}' if forecast_val > 0 else '--'

            html = f"""
            <!DOCTYPE html>
            <html>
            <head><style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, {color} 0%, #333 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .risk-badge {{ display: inline-block; padding: 8px 20px; background: {color}; color: white; border-radius: 20px; font-size: 18px; font-weight: bold; }}
                .info-row {{ margin: 12px 0; padding: 10px; background: white; border-left: 4px solid {color}; }}
                .label {{ font-weight: bold; color: #333; }}
                .rain-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr 1fr; gap: 6px; margin: 10px 0; }}
                .rain-cell {{ background: white; padding: 8px; text-align: center; border-radius: 6px; }}
                .rain-val {{ font-size: 18px; font-weight: bold; color: {color}; }}
                .rain-label {{ font-size: 11px; color: #666; }}
                .disclaimer {{ background: #fff3e0; border: 1px solid #ffcc02; border-radius: 6px; padding: 12px; margin: 15px 0; font-size: 12px; color: #666; }}
                .footer {{ text-align: center; margin-top: 20px; color: #999; font-size: 12px; }}
                .section-title {{ font-size: 14px; font-weight: bold; color: #333; margin: 15px 0 5px; padding-bottom: 3px; border-bottom: 1px solid #ddd; }}
            </style></head>
            <body>
            <div class="container">
                <div class="header">
                    <h1>🌲 HiiForest 作業安全預警</h1>
                    <div class="risk-badge">{label}</div>
                </div>
                <div class="content">
                    <p style="font-size:16px; font-weight:bold;">{warning}</p>

                    {f'<div class="info-row" style="background:#e3f2fd; border-color:#2196f3;"><span class="label">⏰ {type_label}警報：</span> {timing_info}</div>' if timing_info else ''}

                    {f'<div class="section-title">危害項目與建議作為</div>{hazard_html}' if hazard_html else ''}

                    <div class="section-title">現場條件</div>
                    <div class="info-row" style="font-size:13px;">{conditions_html}</div>

                    {index_html}

                    <div class="section-title">降雨觀測</div>
                    <div class="rain-grid">
                        <div class="rain-cell"><div class="rain-val">{rain.get('1h_mm', 0):.1f}</div><div class="rain-label">mm/1h</div></div>
                        <div class="rain-cell"><div class="rain-val">{rain.get('3h_mm', 0):.1f}</div><div class="rain-label">mm/3h</div></div>
                        <div class="rain-cell"><div class="rain-val">{rain.get('24h_mm', 0):.1f}</div><div class="rain-label">mm/24h</div></div>
                        <div class="rain-cell"><div class="rain-val">{forecast_display}</div><div class="rain-label">預報24h</div></div>
                        <div class="rain-cell"><div class="rain-val">{rt_val:.0f}</div><div class="rain-label">Rt有效</div></div>
                    </div>

                    {"<div class='info-row' style='background:#fff3cd; border-color:#f44336;'><span class='label'>⛔ 建議停止作業並避開危險區域</span></div>" if evac else ""}

                    <div class="section-title">Layer 1 — 官方警戒判定</div>
                    <div class="info-row">
                        <span class="label">觸發因子：</span>
                        <ul style="margin:5px 0;">{factors_html}</ul>
                    </div>

                    {f'<div class="section-title">Layer 2 — 工作區位風險加值</div><div class="info-row"><span class="label">工作區高風險點：</span><ul style="margin:5px 0;">{work_zone_html}</ul></div>' if work_zone_html else ''}

                    <div class="info-row" style="font-size:12px;">
                        <span class="label">地形因子（DEM 水文分析）：</span> {terrain_html}
                    </div>

                    <div class="info-row">
                        <span class="label">建議避開區域：</span>
                        <ul style="margin:5px 0;">{zones_html}</ul>
                    </div>

                    <div class="info-row">
                        <span class="label">警戒基準值：</span> {thresholds.get('effective_threshold_mm', 350):.0f}mm
                        {f" (原{thresholds.get('site_base_mm', 350):.0f}mm - 地震調降{thresholds.get('earthquake_reduction_mm', 0):.0f}mm)" if thresholds.get('earthquake_reduction_mm', 0) > 0 else ""}
                    </div>

                    <div class="info-row">
                        <span class="label">📍 位置：</span> {location_name} ({lat:.4f}, {lng:.4f})
                    </div>

                    <div class="info-row">
                        <span class="label">評估時間：</span> {ts}
                    </div>

                    <div style="text-align:center; margin-top:20px;">
                        <a href="{hiiforest_url}" style="display:inline-block; padding:12px 30px; background:#2d7d46; color:white; text-decoration:none; border-radius:5px; margin:5px;">
                            🗺️ HiiForest 地圖
                        </a>
                        <a href="{gmaps_url}" style="display:inline-block; padding:12px 30px; background:#4285f4; color:white; text-decoration:none; border-radius:5px; margin:5px;">
                            📍 Google Maps
                        </a>
                    </div>

                    <div class="disclaimer">
                        <strong>免責聲明：</strong>本預警基於客觀雨量數據與地形分析，提供森林工作者安全規劃參考。
                        正式防災警戒請依農業部農村發展及水土保持署（SWCB）及地方政府公告為準。
                    </div>

                    {f'<div class="info-row" style="font-size:11px;"><span class="label">方法學：</span><ul style="margin:5px 0;">{methodology_html}</ul></div>' if methodology_html else ''}

                    <div class="footer">
                        <p>資料來源：CWA 即時雨量 + 鄉鎮預報 + 地震報告 + DEM 水文分析 + SWCB 土石流警戒基準</p>
                        <p>此郵件由 HiiForest 系統自動發送</p>
                        <p><a href="https://hiiforest.com/landing/unsubscribe.html?project=baxianshan&email=__EMAIL__" style="color:#999;">取消訂閱預警通知</a></p>
                    </div>
                </div>
            </div>
            </body></html>
            """

            work_zone_text = chr(10).join('- ' + w for w in work_zone_risks) if work_zone_risks else '無'
            methodology_text = chr(10).join('- ' + m for m in methodology) if methodology else ''

            hazard_lines = []
            for h in hazards:
                _, h_label = hazard_level_style.get(h["level"], ("", h["level"]))
                hazard_lines.append(f"- [{h_label}] {h['name']}：{h['detail']}")
                if h.get("action"):
                    hazard_lines.append(f"  建議：{h['action']}")
            hazard_text = chr(10).join(hazard_lines) if hazard_lines else '無'

            if index:
                index_text = (
                    f"GIS-AHP 複合指數 {index['composite_score']:.2f}/5.00"
                    f"（{index['class_label']}級）"
                    f"  地形固有風險 {index['static_score']:.2f}"
                    f"（{index['static_class_label']}級）"
                    f"  動態氣象占比 {index['dynamic_share']*100:.0f}%"
                    f"{chr(10)}※ 供區位比較與作業規劃，警報判定仍以官方門檻為準"
                )
            else:
                index_text = "無資料"

            text = f"""
HiiForest {location_name}作業安全預警 — {label}

{warning}

{'【' + type_label + '警報】' + timing_info if timing_info else ''}

【危害項目與建議作為】
{hazard_text}

【現場條件】
{conditions_html}

【作業環境風險指數（規劃參考）】
{index_text}

雨量：1h={rain.get('1h_mm', 0):.1f}mm  3h={rain.get('3h_mm', 0):.1f}mm  24h={rain.get('24h_mm', 0):.1f}mm
預報24h={forecast_display}mm  Rt有效累積={rt_val:.0f}mm
警戒基準值={thresholds.get('effective_threshold_mm', 350):.0f}mm
建議停止作業：{'是' if evac else '否'}

【Layer 1 — 官方警戒判定】
觸發因子：
{chr(10).join('- ' + f for f in factors)}

【Layer 2 — 工作區位風險】
{work_zone_text}

地形因子：{terrain_html}

建議避開區域：
{chr(10).join('- ' + z for z in zones)}

位置：{location_name} ({lat:.4f}, {lng:.4f})
評估時間：{ts}

🗺️ HiiForest 地圖：{hiiforest_url}
📍 Google Maps：{gmaps_url}

免責聲明：本預警基於客觀雨量數據與地形分析，提供森林工作者安全規劃參考。
正式防災警戒請依SWCB及地方政府公告為準。

來源：CWA 即時雨量 + 鄉鎮預報 + 地震報告 + DEM 水文分析 + SWCB 土石流警戒基準
{methodology_text}

取消訂閱預警通知：https://hiiforest.com/landing/unsubscribe.html?project=baxianshan&email=__EMAIL__
"""

            success = True
            for email in recipients:
                # Replace per-recipient email in unsubscribe links
                html_personalized = html.replace("__EMAIL__", email)
                text_personalized = text.replace("__EMAIL__", email)
                if not svc.send_email(email, subject, html_personalized, text_personalized):
                    success = False
            return success

        except Exception as e:
            print(f"[Notify/Email] ❌ Error: {e}")
            return False


# ---------------------------------------------------------------------------
# Unified Alert Dispatcher
# ---------------------------------------------------------------------------

class AlertDispatcher:
    """Dispatch alerts to all configured channels."""

    # Only notify when risk reaches the most dangerous level
    # watch/yellow_alert = dashboard reference only, red_alert = notify
    NOTIFY_THRESHOLD = "red_alert"  # safe < watch < yellow_alert < red_alert
    RISK_ORDER = ["safe", "watch", "yellow_alert", "red_alert"]

    # Track last notified level — only send on ESCALATION (never re-send same level)
    _last_notified_level = "safe"
    _last_notified_time: Optional[datetime] = None
    # No cooldown needed — we only notify on level UPgrade, not repeats

    @classmethod
    def should_notify(cls, risk_level: str) -> bool:
        """Check if notification should fire.

        Rule: Only send on ESCALATION (level upgrade).
        safe → watch → yellow_alert → red_alert
        Each upgrade sends exactly one notification. No repeats.
        De-escalation (downgrade) does NOT trigger notification.
        """
        current_idx = cls.RISK_ORDER.index(risk_level) if risk_level in cls.RISK_ORDER else 0
        threshold_idx = cls.RISK_ORDER.index(cls.NOTIFY_THRESHOLD)
        last_idx = cls.RISK_ORDER.index(cls._last_notified_level) if cls._last_notified_level in cls.RISK_ORDER else 0

        # Only notify if: above threshold AND strictly escalated
        if current_idx >= threshold_idx and current_idx > last_idx:
            return True
        return False

    @classmethod
    async def dispatch(cls, assessment: dict) -> dict:
        """Send notifications via all channels. Returns status dict."""
        risk_level = assessment.get("risk_level", "safe")
        # Notifications are gated on the overall safety level, so a severe
        # non-landslide hazard (e.g. destructive wind) can also trigger an alert.
        safety_level = assessment.get("safety_level", risk_level)
        results = {"line": False, "email": False, "skipped": False}

        if not cls.should_notify(safety_level):
            results["skipped"] = True
            return results

        # Build LINE message
        rain = assessment.get("rain_summary", {})
        warning = assessment.get("warning_message", "")
        factors = assessment.get("factors", [])
        work_zone_risks = assessment.get("work_zone_risks", [])
        thresholds = assessment.get("thresholds", {})
        timing_info = assessment.get("timing_info", "")
        alert_type = assessment.get("alert_type", "")
        terrain = assessment.get("terrain_factors", {})
        hazards = assessment.get("hazards", [])
        soil = assessment.get("soil_saturation", {})
        conditions = assessment.get("weather_conditions", {})
        index = assessment.get("composite_index")
        type_label = "預報型" if alert_type == "forecast" else "即時型" if alert_type == "realtime" else ""

        # Build terrain summary for LINE
        _terrain_parts = []
        if terrain.get("slope_deg", 0) > 0:
            _terrain_parts.append(f"坡度{terrain['slope_deg']:.0f}°")
        if terrain.get("twi_max", 0) > 0:
            _terrain_parts.append(f"TWI={terrain['twi_max']:.1f}")
        if terrain.get("canopy_height_m", 0) > 0:
            _terrain_parts.append(f"冠層{terrain['canopy_height_m']:.0f}m")
        if terrain.get("ndvi", 0) > 0:
            _terrain_parts.append(f"NDVI={terrain['ndvi']:.2f}")
        if terrain.get("elevation_m", 0) > 0:
            _terrain_parts.append(f"海拔{terrain['elevation_m']:.0f}m")
        _terrain_summary = " / ".join(_terrain_parts) if _terrain_parts else "無"

        # Build hazard breakdown for LINE — the most actionable part of the message
        _hazard_labels = {"caution": "注意", "warning": "預警", "danger": "警戒"}
        _hazard_lines = []
        for h in hazards:
            _hazard_lines.append(f"• [{_hazard_labels.get(h['level'], h['level'])}] {h['name']}")
            if h.get("action"):
                _hazard_lines.append(f"  → {h['action']}")

        # Build current field conditions for LINE
        _cond_parts = []
        if conditions.get("wind_speed_ms", 0) > 0:
            _cond_parts.append(f"風速{conditions['wind_speed_ms']:.1f}m/s")
        if conditions.get("temperature_c", 0) > 0:
            _cond_parts.append(f"氣溫{conditions['temperature_c']:.0f}°C")
        if conditions.get("wbgt_c", 0) > 0:
            _cond_parts.append(f"WBGT{conditions['wbgt_c']:.0f}°C")
        if soil.get("saturation_pct", 0) > 0:
            _cond_parts.append(f"土壤飽和{soil['saturation_pct']:.0f}%")
        _cond_summary = " / ".join(_cond_parts) if _cond_parts else "無"

        line_msg = f"\n🌲 HiiForest {assessment.get('location_name', '八仙山')}作業安全預警\n"
        line_msg += f"{'='*30}\n"
        line_msg += f"{warning}\n\n"
        if timing_info:
            line_msg += f"⏰ {type_label}警報：{timing_info}\n\n"
        if _hazard_lines:
            line_msg += "⚠️ 危害項目與建議：\n" + "\n".join(_hazard_lines) + "\n\n"
        line_msg += f"🌡 現場條件：{_cond_summary}\n"
        if index:
            line_msg += (
                f"📊 風險指數：{index['composite_score']:.1f}/5"
                f"（{index['class_label']}，地形{index['static_score']:.1f}）\n"
            )
        line_msg += f"🌧 雨量：1h={rain.get('1h_mm',0):.0f}mm / 3h={rain.get('3h_mm',0):.0f}mm / 24h={rain.get('24h_mm',0):.0f}mm\n"
        _fc = rain.get('forecast_24h_mm', 0)
        _fc_disp = f'{_fc:.0f}' if _fc > 0 else '--'
        line_msg += f"預報24h={_fc_disp}mm / Rt有效={rain.get('rt_effective_mm',0):.0f}mm\n"
        line_msg += f"警戒值={thresholds.get('effective_threshold_mm',350):.0f}mm\n"
        line_msg += f"🏔️ 地形：{_terrain_summary}\n"
        if factors:
            line_msg += f"\n📋 警戒因子：\n" + "\n".join(f"• {f}" for f in factors[:4])
        if work_zone_risks:
            line_msg += f"\n⚠️ 工作區風險：\n" + "\n".join(f"• {w}" for w in work_zone_risks[:5])
        if assessment.get("evacuation_needed"):
            line_msg += f"\n\n⛔ 建議停止作業並避開危險區域"
        lat = assessment.get("lat", 24.2633)
        lng = assessment.get("lng", 120.9500)
        line_msg += f"\n\n📍 位置：{lat:.4f}, {lng:.4f}"
        line_msg += f"\n🗺️ https://hiiforest.com/app/index_modular.html#lat={lat}&lng={lng}&z=14"
        line_msg += f"\n📍 https://www.google.com/maps?q={lat},{lng}"
        line_msg += f"\n\n免責：本預警供森林工作者參考，正式警戒請依SWCB及地方政府公告。"

        # Send LINE
        results["line"] = await LINENotifier.send(line_msg)

        # Send Email
        results["email"] = EmailAlertNotifier.send(assessment)

        # Update last notified level and time
        cls._last_notified_level = safety_level
        cls._last_notified_time = datetime.now(timezone.utc)

        print(f"[AlertDispatcher] Dispatched safety={safety_level} "
              f"(landslide={risk_level}): LINE={results['line']}, Email={results['email']}")
        return results
