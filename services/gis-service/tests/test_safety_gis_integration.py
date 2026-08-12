"""
Tests for GIS-AHP safety index integration in gis-service.

Covers:
- AHP module imports and basic mechanics
- safety-index-map endpoint (static + composite modes)
- terrain-profile endpoint
- ahp-weights audit endpoint
- compute-static-risk dual-score output
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# AHP module unit tests (pure Python, no DB needed)
# ---------------------------------------------------------------------------

class TestAHPSafetyModules:
    """Verify the ported AHP modules work in gis-service context."""

    def test_assess_safety_index_taiwan(self):
        from app.modules.safety.safety_index import assess_safety_index
        result = assess_safety_index(
            "taiwan",
            slope_deg=38,
            twi=5.2,
            flow_accumulation=8000,
            elevation_m=1800,
            ndvi=0.72,
            geomorphon=5,
            aspect_deg=180,
        )
        assert 1.0 <= result["composite_score"] <= 5.0
        assert 1.0 <= result["static_score"] <= 5.0
        assert result["static_class"] in {"very_low", "low", "moderate", "high", "very_high"}
        assert result["static_class_label"]
        assert "consistency_ratios" in result

    def test_assess_safety_index_japan_different_weights(self):
        from app.modules.safety.safety_index import assess_safety_index
        tw = assess_safety_index("taiwan", slope_deg=40, elevation_m=1500, twi=6, flow_accumulation=5000, ndvi=0.6, geomorphon=3, aspect_deg=90)
        jp = assess_safety_index("japan", slope_deg=40, elevation_m=1500, twi=6, flow_accumulation=5000, ndvi=0.6, geomorphon=3, aspect_deg=90)
        # Different regions should produce different composite scores
        assert tw["composite_score"] != jp["composite_score"]

    def test_static_score_excludes_weather(self):
        from app.modules.safety.safety_index import assess_safety_index
        calm = assess_safety_index("taiwan", slope_deg=35, elevation_m=1500, twi=5, flow_accumulation=3000, ndvi=0.7, geomorphon=5, aspect_deg=180, rt_mm=0, wind_speed_ms=0, wbgt_c=0)
        storm = assess_safety_index("taiwan", slope_deg=35, elevation_m=1500, twi=5, flow_accumulation=3000, ndvi=0.7, geomorphon=5, aspect_deg=180, rt_mm=200, wind_speed_ms=15, wbgt_c=32)
        # Static score must be identical regardless of weather
        assert calm["static_score"] == storm["static_score"]
        # Composite should differ with weather
        assert calm["composite_score"] <= storm["composite_score"]

    def test_consistency_ratios_within_threshold(self):
        from app.modules.safety.safety_index import assess_safety_index
        for region in ("taiwan", "japan", "indonesia"):
            result = assess_safety_index(region)
            cr = result["consistency_ratios"]
            for matrix_name, ratio in cr.items():
                assert ratio <= 0.10, f"{region} / {matrix_name} CR={ratio} exceeds 0.10"

    def test_classify_index_boundaries(self):
        from app.modules.safety.safety_index import classify_index
        assert classify_index(1.0)[0] == "very_low"
        assert classify_index(2.0)[0] == "low"
        assert classify_index(3.0)[0] == "moderate"
        assert classify_index(4.0)[0] == "high"
        assert classify_index(5.0)[0] == "very_high"

    def test_ahp_weights_audit_data(self):
        from app.modules.safety.safety_criteria import get_profile, REGION_PROFILES
        assert "taiwan" in REGION_PROFILES
        assert "japan" in REGION_PROFILES
        assert "indonesia" in REGION_PROFILES
        p = get_profile("taiwan")
        assert p.code == "taiwan"
        assert p.rainfall_reference_mm > 0
        assert p.rainfall_authority


# ---------------------------------------------------------------------------
# Endpoint tests (mock DB)
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture
def mock_terrain_rows():
    """Simulate terrain_risk_grid rows returned by SQLAlchemy."""
    return [
        # id, slope, aspect, geomorphon, twi, catchment_m2, elevation, ndvi, canopy_height
        (1, 38.0, 180.0, 5, 5.2, 7200.0, 1800.0, 0.72, 12.0),
        (2, 15.0, 90.0, 1, 2.1, 900.0, 1100.0, 0.85, 18.0),
        (3, 45.0, 270.0, 9, 8.5, 45000.0, 2200.0, 0.35, 3.0),
    ]


class TestSafetyIndexMapEndpoint:
    """Test /disaster/safety-index-map endpoint."""

    def test_static_map_returns_geojson(self, client, mock_terrain_rows):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = mock_terrain_rows
        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_conn.execute.return_value.fetchone.return_value = None
        # Build a more careful mock
        mock_execute_result = MagicMock()
        mock_execute_result.fetchall.return_value = [
            (
                '{"type":"Polygon","coordinates":[[[121.01,24.13],[121.011,24.13],[121.011,24.131],[121.01,24.131],[121.01,24.13]]]}',
                38.0, 180.0, 5, 5.2, 7200.0, 1800.0, 0.72, 12.0,
            ),
            (
                '{"type":"Polygon","coordinates":[[[121.02,24.14],[121.021,24.14],[121.021,24.141],[121.02,24.141],[121.02,24.14]]]}',
                15.0, 90.0, 1, 2.1, 900.0, 1100.0, 0.85, 18.0,
            ),
            (
                '{"type":"Polygon","coordinates":[[[121.03,24.15],[121.031,24.15],[121.031,24.151],[121.03,24.151],[121.03,24.15]]]}',
                45.0, 270.0, 9, 8.5, 45000.0, 2200.0, 0.35, 3.0,
            ),
        ]
        mock_conn.execute.return_value = mock_execute_result

        with patch("app.api.endpoints.disaster._engine") as mock_engine:
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            resp = client.get("/api/v1/baxianshan/disaster/safety-index-map", params={"region": "taiwan", "max_features": 10})

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 3
        meta = data["metadata"]
        assert meta["total_cells"] == 3
        assert "static_class_distribution" in meta
        for f in data["features"]:
            props = f["properties"]
            assert 1.0 <= props["static_score"] <= 5.0
            assert props["static_class"] in {"very_low", "low", "moderate", "high", "very_high"}
            assert "slope_deg" in props
            assert "twi" in props

    def test_composite_map_includes_weather_scores(self, client):
        mock_conn = MagicMock()
        mock_execute_result = MagicMock()
        mock_execute_result.fetchall.return_value = [
            (
                '{"type":"Polygon","coordinates":[[[121.01,24.13],[121.011,24.13],[121.011,24.131],[121.01,24.131],[121.01,24.13]]]}',
                38.0, 180.0, 5, 5.2, 7200.0, 1800.0, 0.72, 12.0,
            ),
        ]
        mock_conn.execute.return_value = mock_execute_result

        with patch("app.api.endpoints.disaster._engine") as mock_engine:
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            resp = client.get(
                "/api/v1/baxianshan/disaster/safety-index-map",
                params={
                    "region": "taiwan",
                    "max_features": 10,
                    "include_composite": True,
                    "rt_mm": 150,
                    "wind_speed_ms": 12,
                    "wbgt_c": 30,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        meta = data["metadata"]
        assert "composite_class_distribution" in meta
        assert "weather_input" in meta
        props = data["features"][0]["properties"]
        assert "composite_score" in props
        assert "class" in props
        assert "class_label" in props

    def test_empty_grid_returns_empty_collection(self, client):
        mock_conn = MagicMock()
        mock_execute_result = MagicMock()
        mock_execute_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_execute_result

        with patch("app.api.endpoints.disaster._engine") as mock_engine:
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            resp = client.get("/api/v1/baxianshan/disaster/safety-index-map")

        assert resp.status_code == 200
        data = resp.json()
        assert data["features"] == []
        assert data["metadata"]["total_cells"] == 0


class TestTerrainProfileEndpoint:
    """Test /disaster/terrain-profile endpoint."""

    def test_terrain_profile_found(self, client):
        mock_conn = MagicMock()
        mock_execute_result = MagicMock()
        mock_execute_result.fetchone.return_value = (
            38.0, 180.0, 5, 5.2, 7200.0, 1800.0, 0.72, 12.0, 0.0001,
        )
        mock_conn.execute.return_value = mock_execute_result

        with patch("app.api.endpoints.disaster._engine") as mock_engine:
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            resp = client.get(
                "/api/v1/baxianshan/disaster/terrain-profile",
                params={"lon": 121.02, "lat": 24.14},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        t = data["terrain"]
        assert t["slope_deg"] == 38.0
        assert t["twi"] == 5.2
        assert t["elevation_m"] == 1800.0
        assert "flow_accumulation" in t
        assert "soil_type" in t

    def test_terrain_profile_not_found(self, client):
        mock_conn = MagicMock()
        mock_execute_result = MagicMock()
        mock_execute_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_execute_result

        with patch("app.api.endpoints.disaster._engine") as mock_engine:
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            resp = client.get(
                "/api/v1/baxianshan/disaster/terrain-profile",
                params={"lon": 999.0, "lat": 999.0},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is False


class TestAHPWeightsAuditEndpoint:
    """Test /disaster/ahp-weights endpoint."""

    def test_ahp_weights_returns_audit_data(self, client):
        resp = client.get(
            "/api/v1/baxianshan/disaster/ahp-weights",
            params={"region": "taiwan"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["region"] == "taiwan"
        assert "consistency_ratios" in data
        assert "groups" in data
        assert "criteria" in data
        assert data["rainfall_reference_mm"] > 0
        assert "available_regions" in data
        assert "taiwan" in data["available_regions"]

    def test_ahp_weights_japan(self, client):
        resp = client.get(
            "/api/v1/baxianshan/disaster/ahp-weights",
            params={"region": "japan"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["region"] == "japan"
        assert data["region_name"]
