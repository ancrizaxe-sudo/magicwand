# AVLE-C — Interview Prep

## Why U-Net over an NDVI threshold?
* NDVI thresholds treat every pixel independently — they cannot use spatial context (edges, clusters, texture) and collapse on mixed-pixel cases (agroforestry, urban canopy, riparian strips).
* A U-Net learns a spatial receptive field (~128 px at the bottleneck) and can distinguish vegetation from spectrally similar non-vegetation (algal water, shadows, certain minerals) using context.
* Empirically, the trained U-Net scores a higher IoU on matched WorldCover labels than the NDVI threshold (see the Segmentation IoU panel in the Ablation page).

## How does segmentation improve accuracy downstream?
* AVLE+ is integrated over a loss mask. Noisy masks inflate spurious pixels, which then propagate into area × carbon.
* A better mask → a cleaner AVLE+ → less variance in the learned RF target → sharper RMSE / R².

## What happens without segmentation?
* The pipeline falls back to `NDVI > 0.30` as a vegetation mask. It still runs, still returns calibrated uncertainty, but the IoU is systematically lower and small clearings near water edges are mis-labelled.

## Why MC-dropout for confidence?
* Standard softmax probabilities are not calibrated on out-of-distribution inputs.
* Dropout at inference time samples from the posterior over network weights. The mean over 25 passes is the predictive distribution; the std across passes is an epistemic-uncertainty estimate.

## Why XGBoost over ARIMA for carbon projection?
* ARIMA assumes stationary linear dynamics. Carbon flux time series are driven by land-use events (discrete regime shifts), which ARIMA handles poorly.
* XGBoost on lag features captures non-linear regime shifts and allows scenario injection via `scenario_drift` without re-fitting.

## Why two regimes (neutral + structured)?
* If a "novel" index is only tested on data that was generated to favour it, the evaluation is circular.
* The neutral regime is random NDVI with no spatial clusters, canopy density or temporal drift. If AVLE+ still "beats" NDVI-diff there, something is wrong. Both tables are shipped — we claim improvements only in the structured regime.

## Why did you remove CSI?
* My original CSI was `Var(ŷ)/Var(ΔNDVI)` — with carbon in tCO₂ and NDVI ∈ [−1, 1], the ratio was naturally in the billions. It was unit-confused, not an IoU-type score.
* Replaced with R² + Spearman ρ² (both bounded) for carbon, and IoU for segmentation — each metric is now semantically correct.

## What's in the blockchain receipt?
* Local EVM chain (py-evm via eth-tester) in-process, no external node.
* Every analysis is a zero-value self-transaction whose `input` contains the JSON record (region_hash, bbox, AVLE+, carbon, biome, t₁/t₂, sources, recommendation class, intervention factor).
* The transaction hash becomes the immutable proof anyone can later `verify(tx)` against.

## What would break in production?
* Planetary Computer preview.png rescales class codes; for research-grade training we would switch to direct COG reads of the raw B04/B08 tiles and the WorldCover classification raster via rasterio, not preview PNGs.
* Synthetic data is a stand-in; a production deployment would ingest Hansen Global Forest Change + GEDI L4A biomass for real supervision labels.
* eth-tester is ephemeral — real deployment would target Polygon Mumbai or a private L2.

## Where is the code?
* `/app/backend/avle/` — all Python modules, training scripts, evaluator, pipeline
* `/app/backend/server.py` — FastAPI layer
* `/app/frontend/src/` — React UI
* `/app/docs/*.md` — these guides
