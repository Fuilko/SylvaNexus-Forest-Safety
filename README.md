# SylvaNexus GIS-AHP Forest Operations Safety Index

A GIS-based multi-hazard occupational safety early-warning platform for mountain forestry in Taiwan, integrating the Analytic Hierarchy Process (AHP) composite safety index from Rahmawati et al. (2025).

## Core Components

- **AHP decision engine** — pairwise comparison, Saaty consistency check, geometric-mean eigenvector weights
- **14 safety criteria** — slope, TWI, geomorphon, elevation, NDVI, canopy height, rainfall, wind, WBGT, soil, accessibility, wildlife, abnormal trees, flow accumulation
- **Multi-hazard assessment** — landslide, stream surge, treefall, heat stress
- **GIS spatial mapping** — cell-by-cell AHP safety index as GeoJSON from `terrain_risk_grid`
- **FastAPI endpoints** — `/safety-index-map`, `/terrain-profile`, `/ahp-weights`, `/compute-static-risk`
- **LINE / Email notifications** — multi-lingual advisory messages

## File Map

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
project_showcase_en.html # English project overview
```

## Citation / Collaboration

This work extends the AHP forest-work-safety framework of Rahmawati et al. (2025) with real-time GIS and weather data. We welcome research collaboration and feedback.
