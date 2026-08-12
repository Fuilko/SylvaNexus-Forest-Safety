"""
Tests for multi-hazard notification dispatch.

Verifies that notifications are gated on the overall safety level (so a severe
non-landslide hazard still reaches workers) and that the LINE and email
templates render the hazard breakdown without errors.
"""

import asyncio

import pytest

from app.weather.notifier import AlertDispatcher, EmailAlertNotifier
from app.weather.providers import FloodRiskEngine


@pytest.fixture(autouse=True)
def reset_dispatcher_state():
    """Dispatcher tracks the last notified level in class state."""
    AlertDispatcher._last_notified_level = "safe"
    AlertDispatcher._last_notified_time = None
    yield
    AlertDispatcher._last_notified_level = "safe"
    AlertDispatcher._last_notified_time = None


@pytest.fixture
def captured(monkeypatch):
    """Capture outgoing messages instead of sending them."""
    sent = {"line": None, "email": None}

    async def fake_line_send(message):
        sent["line"] = message
        return True

    def fake_email_send(assessment, recipients=None):
        sent["email"] = assessment
        return True

    monkeypatch.setattr("app.weather.notifier.LINENotifier.send", fake_line_send)
    monkeypatch.setattr(EmailAlertNotifier, "send", staticmethod(fake_email_send))
    return sent


def build_assessment(**overrides):
    """Build a realistic assessment via the engine, then attach scheduler metadata."""
    params = {
        "rain_1h_mm": 0, "rain_3h_mm": 0, "rain_24h_mm": 0,
        "slope_deg": 38, "twi_max": 5.2, "canopy_height_m": 12,
    }
    params.update(overrides)
    assessment = FloodRiskEngine.assess_landslide_risk(**params)
    assessment["warning_message"] = "測試訊息"
    assessment["location_name"] = "八仙山"
    assessment["lat"] = 24.2633
    assessment["lng"] = 120.9500
    return assessment


def test_safe_assessment_is_skipped(captured):
    assessment = build_assessment(wind_speed_ms=2.0, temperature_c=15, humidity_pct=60)
    result = asyncio.run(AlertDispatcher.dispatch(assessment))
    assert result["skipped"] is True
    assert captured["line"] is None


def test_dangerous_wind_notifies_without_landslide_alert(captured):
    """Wind alone must reach workers — this is the point of the safety framing."""
    assessment = build_assessment(wind_speed_ms=20.0, temperature_c=15, humidity_pct=60)
    assert assessment["risk_level"] == "safe"
    assert assessment["safety_level"] == "red_alert"

    result = asyncio.run(AlertDispatcher.dispatch(assessment))
    assert result["skipped"] is False
    assert result["line"] is True
    assert "倒木" in captured["line"]
    assert "作業安全預警" in captured["line"]


def test_line_message_lists_hazard_actions(captured):
    assessment = build_assessment(
        rain_1h_mm=35, rain_3h_mm=60, rain_24h_mm=400,
        flow_accum_max=8000, wind_speed_ms=20.0,
        temperature_c=34, humidity_pct=85,
    )
    asyncio.run(AlertDispatcher.dispatch(assessment))
    message = captured["line"]
    for expected in ("崩塌", "溪流暴漲", "倒木", "熱危害", "現場條件"):
        assert expected in message


def test_no_repeat_notification_at_same_level(captured):
    assessment = build_assessment(wind_speed_ms=20.0)
    first = asyncio.run(AlertDispatcher.dispatch(assessment))
    second = asyncio.run(AlertDispatcher.dispatch(assessment))
    assert first["skipped"] is False
    assert second["skipped"] is True


def test_email_template_renders(monkeypatch):
    """Render the email body end-to-end to catch template errors."""
    rendered = {}

    class FakeEmailService:
        admin_email = "admin@example.com"

        def send_email(self, to, subject, html, text):
            rendered["to"] = to
            rendered["subject"] = subject
            rendered["html"] = html
            rendered["text"] = text
            return True

    import sys
    import types

    fake_module = types.ModuleType("app.email_service")
    fake_module.EmailService = FakeEmailService
    monkeypatch.setitem(sys.modules, "app.email_service", fake_module)

    assessment = build_assessment(
        rain_1h_mm=35, rain_3h_mm=60, rain_24h_mm=400,
        flow_accum_max=8000, wind_speed_ms=20.0,
        temperature_c=34, humidity_pct=85,
    )
    assert EmailAlertNotifier.send(assessment, recipients=["worker@example.com"]) is True

    assert "作業安全預警" in rendered["subject"]
    assert "危害項目與建議作為" in rendered["html"]
    assert "現場條件" in rendered["html"]
    assert "倒木" in rendered["html"]
    assert "危害項目與建議作為" in rendered["text"]
    # No unresolved template placeholders
    assert "{" not in rendered["subject"]
