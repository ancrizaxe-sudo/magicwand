# AVLE-C — Research Paper Guide

## Title
AVLE-C: Adaptive Vegetation Loss Estimator for Carbon — a research-grade end-to-end remote-sensing framework

## Four novelty claims
1. **AVLE+ index** — spatially (K-means), temporally (momentum) and
   canopy-density weighted loss score that beats NDVI-difference on
   structured data, verified to **not** beat it on a random-NDVI null
   regime (unbiased validation).
2. **ML-regressed carbon estimation** — Random Forest trained on 50 k
   synthetic IPCC Tier-2 samples with realistic NDVI → AGB (Myneni/Baccini
   empirical relation) → CO₂ (IPCC carbon fraction 0.47) coupling,
   reported against constant-factor, NDVI-diff, NDVI-smoothing,
   NDVI-clustering, and NDVI-temporal-averaging baselines.
3. **Neural recommendation classifier** — PyTorch MLP with 25-pass
   Monte-Carlo dropout producing calibrated mean probabilities and
   per-class std, evaluated against a rule-based IF-ELSE baseline on the
   same test split.
4. **XGBoost scenario-conditioned projection** — BAU, Mitigation
   (intervention-aware β modulation tied to the recommender's class) and
   quantile 2.5 %/97.5 % CI, with linear-regression and ARIMA baselines
   and an explicit Scenario Separation Score (SSS).

## Engineering components (cited, NOT claimed as novel)
* U-Net architecture (Ronneberger et al.)
* NDVI
* XGBoost
* STAC satellite retrieval (Planetary Computer)
* FastAPI / React stack
* Blockchain logging (eth-tester / py-evm)

## Table 1 (progressive ablation, structured regime)
| # | Method | Family | RMSE | R² | Spearman ρ² |
|---|---|---|---|---|---|
| 1 | Constant IPCC factor | baseline | ↑↑↑ | <0 | 0 |
| 2 | NDVI-difference | baseline | moderate | 0.45 | 0.24 |
| 3 | NDVI + smoothing | strong baseline | moderate | similar | similar |
| 4 | NDVI + clustering | strong baseline | moderate | slight ↑ | similar |
| 5 | NDVI + temporal avg | strong baseline | moderate | slight ↑ | higher |
| 6 | AVLE + spatial | avle progression | ↓ | 0.62 | 0.24 |
| 7 | AVLE + spatial + temporal | avle progression | ↓ | 0.59 | 0.48 |
| 8 | AVLE + … + canopy | avle progression | ~ | 0.51 | 0.42 |
| 9 | AVLE+ (no ML) | avle progression | ~ | 0.57 | 0.27 |
| **10** | **AVLE-C full (ML-regressed)** | **full system** | **↓↓↓** | **≈ 0.98** | — |

## Table 2 (neutral regime — null hypothesis)
AVLE+ ladder is expected to **flatten** relative to NDVI-diff. If it does
not, the model is suspected of circular validation. Report both tables
side-by-side.

## Table 3 (segmentation)
| Method | IoU |
|---|---|
| NDVI threshold (baseline) | IoU_NDVI |
| U-Net (trained on WorldCover + Sentinel-2) | IoU_UNet |
| Δ (U-Net − NDVI) | IoU_UNet − IoU_NDVI |

## Table 4 (projection)
XGBoost RMSE / SSS vs Linear-regression RMSE / SSS vs ARIMA RMSE / SSS.

## Figures
* NDVI t₁, t₂ (z-scored), loss-overlay
* BAU / Mitigation / Unchecked annual + cumulative
* Recommendation posterior with MC-dropout error bars
* Segmentation IoU scatter (NDVI vs U-Net per tile)

## API surface
```
POST /api/analyze
GET  /api/ablation
GET  /api/weights/info
GET  /api/blockchain, /api/blockchain/verify/{tx}
GET  /api/datasets, /api/datasets/{key}
GET  /api/jobs, /api/jobs/{id}
```

## Reproducibility
All randomness is seeded (`np.random.default_rng(42)` etc.).  All weights
are < 100 MB, committed to `/app/backend/avle/weights/`.  All synthetic
data is regenerable from the same seeds.
