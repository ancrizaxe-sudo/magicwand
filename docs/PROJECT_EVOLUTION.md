# AVLE-C — Project Evolution

## v1 (initial)
* U-Net trained on **synthetic** patches (no real imagery)
* Random Forest carbon regressor on 50 k synthetic samples
* MLP 4-class recommender on 30 k samples, σ_noise = 0.05 (too confident)
* XGBoost BAU / Mitigation (ARIMA baseline)
* 6-row ablation with **CSI** (Pearson variance ratio — unbounded)
* Single-regime evaluation (risk of circular validation)
* NDVI threshold for vegetation detection
* Click-twice bbox picker

## v2 (scientific-quality upgrades)
* **CSI fixed** — Pearson² ∈ [0,1]; added R², Spearman ρ²
* **Two-regime ablation** (neutral vs structured) — unbiased
* **Realistic AGB coupling** (Myneni/Baccini NDVI^1.8 × biome ceiling)
* **NDVI z-score normalisation** across t₁↔t₂ (remove phenology)
* **MC-dropout recommender** (25 passes, realistic confidence)
* **Recommendation-aware mitigation** (XGBoost β modulated by class)
* **Three scenarios**: BAU / Mitigation / Unchecked compound growth
* **Cumulative emissions** area chart
* **Draggable bbox + geolocation**

## v3 (current — real computer vision + deeper science)
* **Real U-Net training** on ESA WorldCover + Sentinel-2 patches from
  Microsoft Planetary Computer (`train_segmentation_real.py`, 16 tiles
  fetched + cached).  NDVI-threshold kept only as offline fallback.
* **CSI removed** from results — keep RMSE / MAE / R² / Spearman ρ²
  for carbon, IoU for segmentation only.
* **Strong baselines** added: NDVI + smoothing, + clustering, + temporal
  averaging, U-Net-only.
* **Segmentation ablation** panel — NDVI-threshold IoU vs trained U-Net.
* **Linear regression** projection baseline with real slope-change
  mitigation, alongside XGBoost + ARIMA.
* **Loss-map cleanup**: median filter → morphological opening →
  connected components with `min_cluster_px=16`.
* **Zero-carbon edge case** enforced (no loss ⇒ 0 tCO₂).
* **Region suitability gate** (blocks water / snow / desert based on NDVI distribution)
  with a visible UI warning banner.
* **Region presets** — Amazon, Mato Grosso, Western Ghats, Sumatra, Congo.
* **Local EVM blockchain** receipts (eth-tester/py-evm) kept.
* **/docs** section in the UI + `/docs/*.md` in the repository.

## File-level deltas v2 → v3
```
backend/avle/
  + train_segmentation_real.py   # NEW real S2 + WorldCover U-Net trainer
  ~ evaluate.py                  # CSI removed, strong baselines + segmentation IoU
  ~ pipeline.py                  # cleaner loss mask, suitability gate, zero-carbon
  ~ utils.py                     # clean_loss_mask, assess_region_suitability
frontend/src/
  ~ components/NavBar.jsx        # removed Datasets link, added Docs
  ~ pages/Analyze.jsx            # presets + suitability warning
  ~ pages/Ablation.jsx           # Family column, IoU panel, linreg baseline
  + pages/Docs.jsx               # in-app browser of /docs/*.md
```
