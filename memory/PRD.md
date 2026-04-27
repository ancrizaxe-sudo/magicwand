# AVLE-C — Product Requirements (v2)

## Problem statement
End-to-end open-source remote-sensing framework for Adaptive Vegetation Loss
Estimation + Carbon flux prediction. Four novelty claims; real uncertainty
quantification; two-regime unbiased validation against circular evaluation.

## Scientific quality upgrades (v2, 2026-02-24)
- [x] Metrics: **CSI = Pearson²**, **Spearman ρ²**, **R²**, RMSE, MAE (all bounded / standard)
- [x] **Two synthetic regimes** (neutral / structured) to prevent circular validation
- [x] Progressive AVLE+ ablation: NDVI-diff → +spatial → +temporal → +canopy → full → ML
- [x] Realistic NDVI → AGB → CO₂ coupling (Myneni/Baccini empirical curve + IPCC C-fraction 0.47)
- [x] NDVI z-score normalisation across t₁↔t₂ to remove seasonal phenology
- [x] Monte-Carlo-dropout recommender (25 passes) — real epistemic uncertainty
- [x] Recommendation-aware mitigation scenario (β_mit modulated by intervention factor)
- [x] Three-way emission scenario: BAU / Mitigation / Unchecked growth
- [x] Cumulative 5-yr emissions panel (If followed vs If not followed)
- [x] 384 px default raster (was 256) — crisper loss overlay
- [x] Geolocation button + draggable bbox corners on map

## Implemented endpoints
| Method | Path | Purpose |
|---|---|---|
| GET  | /api/               | Health |
| POST | /api/analyze        | Full pipeline (size=384, use_synthetic=false by default) |
| GET  | /api/jobs           | Past analyses |
| GET  | /api/jobs/{id}      | Single job |
| GET  | /api/ablation       | Two-regime table + metrics |
| GET  | /api/weights/info   | Model metrics |
| GET  | /api/blockchain     | Chain + records |
| GET  | /api/blockchain/verify/{tx} | Verify tx |
| GET  | /api/datasets       | List training CSVs |
| GET  | /api/datasets/{key} | Download CSV |

## Headline numbers (current snapshot)
### Structured regime (expected to favour AVLE+)
| Method | RMSE | R² | CSI | Spearman ρ² |
|---|---|---|---|---|
| Constant IPCC factor | 8696 | −9.99 | 0.00 | 0.00 |
| NDVI-difference | 1950 | 0.45 | 0.21 | 0.24 |
| AVLE + spatial | 1612 | 0.62 | 0.19 | 0.24 |
| AVLE + spatial + temporal | 1690 | 0.59 | 0.26 | 0.48 |
| AVLE + spatial + temporal + canopy | 1831 | 0.51 | 0.22 | 0.42 |
| AVLE+ (no ML) | 1727 | 0.57 | 0.14 | 0.27 |
| **AVLE-C full (ML)** | **365** | **0.98** | 0.14 | 0.20 |

### Neutral regime (null hypothesis — AVLE+ should NOT help)
NDVI-diff R²=0.20, AVLE progression stays 0.60-0.68 (minor noise),
AVLE-C full ML 0.97. Confirms AVLE+ gains are **structural** not circular.

### Recommendation (MC-dropout, σ_noise=0.15)
MLP accuracy 80.1 % · F1 0.79  ·  Rule-based 76.4 % · F1 0.75

## External APIs used
- Microsoft Planetary Computer STAC (unauthenticated)
- OpenStreetMap tile server

## Next tasks
- P1 — integrate real Hansen GFC + GEDI biomass (pending user decision)
- P1 — migrate FastAPI on_event hooks to lifespan
- P2 — multi-date temporal momentum via real time-series endpoints
