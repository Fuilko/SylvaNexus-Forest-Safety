# SylvaNexus (HiiForest) GIS-AHP Forest Operations Safety Platform

A real-time, GIS-based multi-hazard occupational safety early-warning platform for mountain forestry in Taiwan, integrating the Analytic Hierarchy Process (AHP) composite safety index from Rahmawati et al. (2025).

---

## What the Platform Does

SylvaNexus combines weather data, terrain analysis, and an AHP-weighted safety model to assess forest-work risk before crews enter the field. It is designed for Taiwan mountain forestry but is built to be extensible to Pacific-Asia contexts.

**Core capabilities**

- **Multi-hazard risk scoring** — landslide, stream surge, treefall, heat stress
- **AHP-weighted safety index** — 14 criteria, auditable pairwise weights, Saaty consistency check
- **Real-time weather** — Central Weather Administration (CWA) and Japan Meteorological Agency (JMA)
- **Spatial GIS mapping** — PostGIS `terrain_risk_grid` with DEM-derived factors (slope, TWI, geomorphon, elevation, NDVI, canopy height)
- **Alerting** — LINE Notify and email for yellow / red advisories
- **Web + mobile ready** — FastAPI backend with nginx, frontend served as static SPA

---

## Information Flow

```
Terrain data (PostGIS) ──────────┐
DEM / NDVI / canopy (GEE/S3) ────┼──▶ GIS service ──▶ AHP safety index
Real-time weather (CWA/JMA) ─────┤    (FastAPI)
                                  │
                                  ▼
Backend (FastAPI)  ◀──▶  PostgreSQL/PostGIS  ◀──▶  Scheduled risk checks
  │                                                      (APScheduler)
  ├─▶ /landslide-risk          ├─▶ terrain_risk_grid
  ├─▶ /safety-index-map        └─▶ alert_logs
  ├─▶ /weather/forecast
  └─▶ LINE / Email notifier

Frontend (nginx / SPA)
  ├─▶ Web GIS map with color-coded risk grid
  ├─▶ Advisory dashboard
  └─▶ Mobile-friendly alert view
```

1. **Ingest** — CWA/JMA weather, GEE satellite indices, and local DEM/terrain layers are loaded into PostGIS.
2. **Compute** — GIS service computes per-cell AHP safety scores; backend runs scheduled multi-hazard checks.
3. **Notify** — When risk reaches a threshold, LINE and email alerts are sent in the local language.
4. **Visualize** — Web and mobile frontends show the risk grid, forecast, and active advisories.

---

## AHP Safety Index

The 14 criteria, grouped by domain, are scored and weighted by AHP:

| Domain | Criteria |
|--------|----------|
| Terrain | slope, elevation, aspect, TWI, flow accumulation, geomorphon |
| Vegetation | NDVI, canopy height, abnormal trees |
| Weather | rainfall (RT), wind speed, WBGT heat stress |
| Operations | soil type, accessibility (minutes), wildlife |

The weight vector is derived from a pairwise comparison matrix with a consistency-ratio check. Any matrix exceeding the Saaty threshold is rejected, so weights are always auditably consistent.

Two scores are produced:

- **Composite score** — dynamic, including weather variables (1–5, 5 = highest risk)
- **Static score** — terrain-only, used for baseline spatial mapping and long-term zoning

---

## PostGIS & Spatial Stack

- `terrain_risk_grid` — raster/vector grid storing DEM-derived parameters per cell
- `baxianshan` schema — Taiwan demo site with slope, TWI, geomorphon, elevation, NDVI, canopy height
- PostGIS functions — point-in-polygon lookup, GeoJSON generation, nearest-cell queries
- GIS endpoints:
  - `POST /safety-index-map` — returns GeoJSON of AHP scores
  - `POST /terrain-profile` — returns terrain parameters at a given coordinate
  - `GET  /ahp-weights` — returns the AHP weight audit trail

---

## AWS & Runtime Environment

| Layer | Technology |
|-------|------------|
| Compute | AWS EC2 (Tokyo `ap-northeast-1`) running Docker Compose |
| Reverse proxy | nginx + ALB with HTTPS (ACM) |
| Object storage | S3 (`hiiforest-assets`) for photos, GIS exports, research PDFs |
| Database | PostgreSQL 15 + PostGIS on EC2 |
| Secrets | AWS Secrets Manager / host-mounted key files |
| CI/CD | GitHub Actions → SSM Run Command on EC2 |
| Monitoring | Uptime monitor (GitHub Actions) + Sentry error tracking |

The production stack is containerized with `docker-compose.yml` and includes:

- `backend` (FastAPI)
- `gis-service` (FastAPI + PostGIS)
- `frontend` (nginx static)
- `postgres` (PostGIS)
- `minio` (S3-compatible local object store for dev)

---

## Other Pipelines & Features

Beyond the AHP safety module, the full SylvaNexus platform operates multiple services on a single AWS EC2 instance (Tokyo `ap-northeast-1`) via Docker Compose:

| Service | Status | Description |
|---------|--------|-------------|
| **Forest Safety & AHP** | Production | Multi-hazard early-warning with AHP composite index |
| **Forest Photo Pipeline** | Production | EXIF GPS photos → S3 → map markers |
| **Summit Registration** | Production | Academic event portal with paper upload & payment |
| **Editorial CMS** | Production | Forest-science articles and news publishing |
| **GEE Satellite Bridge** | Production | NDVI, canopy height, SAR landslide detection, InSAR deformation |
| **GIS Service (PostGIS)** | Production | Spatial analysis, terrain_risk_grid, compartment boundaries |
| **Unified File Registry** | Production | Cross-module S3 file management with presigned URLs |
| **Monitoring & Alerting** | Production | Uptime monitor + Sentry + LINE/Email advisories |
| **Scenario Engine** | Dev | Forest management simulation (growth, harvest, carbon) |
| **Knowledge MCP / RAG** | Dev | Document Q&A over forestry laws and research papers |

**Full architecture diagram:** see [`platform_concept_en.html`](platform_concept_en.html) for a visual overview of all services, AWS infrastructure, and the future vision.

---

## From Reference to Real-Time Field Action

The AHP framework by Rahmawati et al. (2025) provides a critical reference for assessing natural-information risks to human life and property in forest operations. However, translating a static assessment into **real-time, actionable field support** requires bridging the gap between the GIS platform and the crew on the mountain.

Our goal is to make the GIS platform not just a map or a report, but a **living, interactive real-time assistance tool**.

### Future Roadmap

- **Mobile app** — native app for forest workers: real-time risk map with GPS, push alerts, check-in/check-out, offline tiles, photo reporting, crew safety status
- **Low-orbit satellite integration** — Starlink/Iridium connectivity for remote mountains without cellular: real-time weather sync, SOS signals, IoT sensor uplink, edge-cached risk grid refresh
- **Edge computing** — lightweight inference on field gateways; local cache of AHP risk grid for off-network use; on-device threshold alerts; federated sync when connectivity returns
- **Real-time IoT** — rain gauges, soil moisture, wind masts feeding directly into the AHP score
- **LLM advisory** — Gemini-generated, locale-aware safety briefings (CN/JP/EN) with voice interface for hands-free field use
- **Multi-project expansion** — Japan (Shikoku/Kumamoto) and other Taiwan sites beyond Baxianshan

---

## Repository Contents

```
backend/app/weather/
  ahp.py                 # AHP weight derivation
  safety_criteria.py     # 14 criteria definitions
  safety_index.py        # Composite and static safety scoring
  router.py              # API routes including /landslide-risk
  providers.py           # Multi-hazard weather & terrain providers
  notifier.py            # Alert dispatch
  scheduler.py           # Scheduled risk checks

services/gis-service/app/
  modules/safety/        # Ported AHP modules for gis-service
  api/endpoints/disaster.py  # GIS safety endpoints

tests/                   # pytest suite (90 tests)
project_showcase_en.html # English project overview for the paper author
diff_report.html         # Code-change diff report
```

---

## Citation / Collaboration

This work extends the AHP forest-work-safety framework of **Rahmawati et al. (2025)** with real-time GIS, weather data, and operational alerting. We welcome research collaboration, validation of the AHP criteria in other Pacific-Asia forestry contexts, and feedback on multi-hazard expansion.

- **Platform**: https://hiiforest.com
- **Full source**: https://github.com/Fuilko/SaaSDocker
- **Public share (this repo)**: https://github.com/Fuilko/SylvaNexus-Forest-Safety
