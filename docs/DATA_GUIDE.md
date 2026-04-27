# AVLE-C — Data Guide

## Satellite imagery
**Sentinel-2 L2A (surface reflectance)** via Microsoft Planetary Computer:
* Collection: `sentinel-2-l2a`
* Bands used: B04 (red, 665 nm), B08 (NIR, 842 nm)
* Resolution: 10 m native, fetched at 192–384 px preview PNGs
* Access: `pystac-client` + `planetary-computer` (unauthenticated)
* Fallback: Copernicus Open Access Hub

## Segmentation labels
**ESA WorldCover 2021** via the same Planetary Computer catalogue:
* Collection: `esa-worldcover`
* Native: 10 m global land-cover, 11 classes
* Mapping used: value < 40 (class 10 "Tree cover" + class 20 "Shrubland") → 1, else 0

## Synthetic training data
Four CSVs are regenerated on every training run and downloadable via the API (no longer shown in the top nav but still at `/api/datasets`):

| key | rows | description |
|---|---|---|
| `carbon` | 50 000 | 13-feature vector → target tCO₂, structured regime |
| `recommendation` | 30 000 | 6-feature vector → 4-class remediation label |
| `xgb_bau` | ≈ 10 000 | Lag-feature supervised rows for XGBoost BAU |
| `xgb_sequences` | 500 | Raw 24-month carbon sequences (one per row) |

### Regimes
* `structured` — spatial clusters + canopy-density coupling + temporal drift. Used to train the shipped models.
* `neutral` — random NDVI without structure. Used at evaluation time only to confirm AVLE+ does not gain where no structure exists.

### Generating distributions
Realistic NDVI → AGB coupling:
```
AGB_Mg_ha ≈ biome_ceiling · NDVI^1.8         (Myneni 2001 / Baccini 2012)
emissions_tCO2 = AGB × loss_area_ha × fraction_lost × 0.47 × 44/12
```
Biome ceilings from IPCC 2006 Guidelines, Vol. 4 Ch. 4 (Tier 2 ranges).

## Mitigation drift (β)
| Biome | β (tCO₂/km²/year reduction) | Source |
|---|---|---|
| Tropical moist | 8.5 | Bonner et al. 2013 |
| Tropical dry / cerrado | 4.2 | Silva et al. 2018 |
| Temperate | 3.1 | Laganière et al. 2010 |
| Boreal | 2.4 | (interpolated) |
| Grassland | 1.1 | (interpolated) |

## Intervention factor (recommendation → mitigation slope)
| Class | Intervention factor |
|---|---|
| monitoring_only | 0.00 |
| targeted_replanting | 0.20 |
| assisted_regeneration | 0.45 |
| critical_afforestation | 0.75 |

Final `β_mit_effective = β_biome · (0.3 + 1.4 · intervention_factor)` — the recommender's class literally shapes the mitigation scenario.

## Regional presets (Analyze page)
| Preset | bbox | Biome |
|---|---|---|
| Amazon · Pará | −54.9, −9.3, −54.7, −9.1 | tropical moist |
| Mato Grosso Cerrado | −55.2, −9.8, −54.8, −9.4 | tropical dry |
| Western Ghats · India | 75.4, 14.2, 75.6, 14.4 | tropical moist / monsoon |
| Sumatra peatland | 101.2, 0.3, 101.4, 0.5 | tropical peat |
| Congo Basin · CAR | 18.4, 4.3, 18.6, 4.5 | tropical moist |

## Geographic suitability gate
The pipeline rejects regions where the NDVI distribution indicates open water, snow/ice or desert:
* `water_fraction > 0.65` → "predominantly water"
* `nonveg_fraction (NDVI<0.15) > 0.80` → "predominantly non-vegetated"
* `mean_ndvi < 0.10` → "no vegetation to track"

When triggered, the UI shows a yellow warning banner, carbon is forced to 0, and the on-chain receipt still logs the failure so the attempt remains auditable.
