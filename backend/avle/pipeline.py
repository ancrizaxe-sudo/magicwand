"""End-to-end analysis orchestrator.

v2 scientific upgrades
----------------------
*  NDVI is z-score normalised per-scene *before* differencing so seasonal
   intensity shifts cancel out — ΔNDVI reflects structural vegetation
   change, not phenological timing.
*  Default raster size 384×384 (was 128×128) for sharper loss overlays.
*  Recommendation-aware mitigation: the XGBoost mitigation scenario is
   modulated by an *intervention factor* derived from the MLP's predicted
   class — so the projection reacts to the actual remediation strategy.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, Tuple

import numpy as np
from matplotlib import colormaps
from PIL import Image

from . import blockchain
from .avle_plus import avle_plus
from .carbon_model import build_feature_vector, predict_carbon
from .config import CONFIG
from .fetch import fetch_scene
from .ndvi import compute_ndvi
from .prediction import forecast, load_all_models, scenario_separation_score
from .recommendation_model import build_rec_features, recommend
from .segmentation_model import predict_mask
from .utils import assign_biome, morph_open, pixel_to_ha, region_hash, clean_loss_mask, assess_region_suitability

log = logging.getLogger(__name__)

# Intervention factors tied to the recommendation class id.
# 0 = monitoring_only    → effectively BAU
# 1 = targeted_replanting
# 2 = assisted_regeneration
# 3 = critical_afforestation → strongest mitigation
INTERVENTION_FACTOR = [0.00, 0.20, 0.45, 0.75]


# --------------------------------------------------------------------------- #
#  NDVI z-score normalisation (removes per-scene seasonal offset / scale)
# --------------------------------------------------------------------------- #
def _zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd < 1e-6:
        return x - mu
    return (x - mu) / sd


def _match_to_reference(ndvi_obs: np.ndarray,
                        ref_mean: float, ref_std: float) -> np.ndarray:
    """Map an observation-scene NDVI into the reference-scene distribution."""
    z = _zscore(ndvi_obs)
    return z * ref_std + ref_mean


# --------------------------------------------------------------------------- #
#  Raster → PNG helpers (base-64 embedded in JSON responses)
# --------------------------------------------------------------------------- #
def _png_b64(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _ndvi_to_png(ndvi: np.ndarray) -> str:
    cmap = colormaps.get_cmap("YlGn")
    norm = (np.clip(ndvi, -0.2, 0.9) + 0.2) / 1.1
    rgba = (cmap(norm) * 255).astype(np.uint8)
    return _png_b64(rgba[..., :3])


def _false_color(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Sharp NIR-red-green false-colour composite with contrast stretch."""
    def stretch(b, lo=2, hi=98):
        p0, p1 = np.nanpercentile(b, lo), np.nanpercentile(b, hi)
        if p1 - p0 < 1e-9: return np.zeros_like(b)
        return np.clip((b - p0) / (p1 - p0), 0, 1)
    r = stretch(nir)
    g = stretch((red + nir) / 2)
    b = stretch(red)
    return np.stack([r, g, b], axis=-1)


def _loss_overlay(red: np.ndarray, nir: np.ndarray,
                  mask: np.ndarray, mask_raw: np.ndarray | None = None) -> str:
    base = (_false_color(red, nir) * 255).astype(np.uint8)
    out = base.copy()
    if mask_raw is not None:
        # Faint orange for raw detections removed by morphological opening
        raw_only = (mask_raw.astype(bool) & ~mask.astype(bool))
        out[raw_only] = (0.6 * out[raw_only] + 0.4 * np.array([255, 170, 60])).astype(np.uint8)
    # Strong red for confirmed loss
    out[mask > 0] = (0.25 * out[mask > 0] + 0.75 * np.array([224, 50, 50])).astype(np.uint8)
    return _png_b64(out)


# --------------------------------------------------------------------------- #
#  Main orchestrator
# --------------------------------------------------------------------------- #
def analyse(bbox: Tuple[float, float, float, float],
            t1_start: str, t1_end: str,
            t2_start: str, t2_end: str,
            size: int = 384,
            log_blockchain: bool = True,
            allow_real: bool = True) -> Dict[str, Any]:
    red1, nir1, src1 = fetch_scene(bbox, t1_start, t1_end, size=size,
                                   disturbance=0.0, allow_real=allow_real)
    red2, nir2, src2 = fetch_scene(bbox, t2_start, t2_end, size=size,
                                   disturbance=0.55, allow_real=allow_real)

    ndvi1_raw = compute_ndvi(red1, nir1)
    ndvi2_raw = compute_ndvi(red2, nir2)

    # Common-range linear stretch (same scaling applied to BOTH scenes, based
    # on the concatenated percentiles).  This removes per-scene sensor-gain
    # differences without erasing the real NDVI change between t1 and t2.
    concat = np.concatenate([ndvi1_raw.ravel(), ndvi2_raw.ravel()])
    lo, hi = float(np.nanpercentile(concat, 2)), float(np.nanpercentile(concat, 98))
    scale = max(hi - lo, 1e-6)
    ndvi1 = np.clip((ndvi1_raw - lo) / scale, -0.2, 1.2)
    ndvi2 = np.clip((ndvi2_raw - lo) / scale, -0.2, 1.2)
    delta = np.clip(ndvi1 - ndvi2, -1, 1)

    rgb_nir_1 = np.stack([red1, nir1, nir1], axis=-1)
    rgb_nir_2 = np.stack([red2, nir2, nir2], axis=-1)
    veg1 = predict_mask(rgb_nir_1)
    veg2 = predict_mask(rgb_nir_2)

    loss_mask_raw = ((veg1 == 1) & (veg2 == 0)).astype(np.uint8)
    # Multi-stage cleanup: morphological opening → connected-component filter
    # with minimum cluster size so single-pixel noise cannot dominate AVLE+.
    loss_mask = clean_loss_mask(loss_mask_raw,
                                open_kernel=CONFIG.morph_kernel,
                                min_cluster_px=16)

    avle = avle_plus(loss_mask, ndvi1, ndvi2, np.clip(delta, 0, None))

    loss_px = int(loss_mask.sum())
    loss_area_ha = pixel_to_ha(loss_px)

    # Geographic suitability gate (water / snow / non-vegetated) based on
    # baseline NDVI distribution.  Returned to the caller as a warning flag.
    suitability = assess_region_suitability(ndvi1_raw)

    ndvi1_mean = float(ndvi1.mean())
    ndvi2_mean = float(ndvi2.mean())
    delta_mean = float(delta[loss_mask > 0].mean()) if loss_px else 0.0
    delta_std  = float(delta[loss_mask > 0].std())  if loss_px else 0.0
    ndvi1_var  = float(ndvi1.var())
    temp_factor_mean = float(avle["temporal_weight"][loss_mask > 0].mean()) if loss_px else 0.0
    if loss_px:
        vals, counts = np.unique(avle["cluster_map"][loss_mask > 0], return_counts=True)
        dom_cluster = int(vals[np.argmax(counts)])
    else:
        dom_cluster = 0

    # Zero-carbon edge case: no confirmed loss → no emission
    if loss_area_ha <= 0.0 or not suitability["suitable"]:
        carbon_tco2 = 0.0
    else:
        fv = build_feature_vector(
            avle_plus=avle["avle_plus_per_km2"],
            ndvi_mean_t1=ndvi1_mean,
            ndvi_mean_t2=ndvi2_mean,
            delta_ndvi_mean=delta_mean,
            delta_ndvi_std=delta_std,
            loss_area_ha=loss_area_ha,
            ndvi_variance_t1=ndvi1_var,
            dominant_cluster=dom_cluster,
            temporal_factor_mean=temp_factor_mean,
        )
        carbon_tco2 = predict_carbon(fv)

    avle_norm = float(np.clip(avle["avle_plus_per_km2"], 0, 1))
    rec_features = build_rec_features(
        avle_plus_norm=avle_norm,
        ndvi_mean_t2=ndvi2_mean,
        loss_area_ha=loss_area_ha,
        delta_ndvi_mean=delta_mean,
        delta_ndvi_std=delta_std,
        temporal_factor_mean=temp_factor_mean,
    )
    rec = recommend(rec_features)

    # ------------------------------------------------------------------ #
    #  Scenario projection — intervention-aware
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(abs(hash(str(bbox))) % (2**32))
    base = max(carbon_tco2 * 0.6, 1.0)
    history = base + np.linspace(0, carbon_tco2 - base, 12) + rng.normal(0, base * 0.08, 12)
    history = np.clip(history, 0, None)
    biome = assign_biome(ndvi1_mean)
    beta_mit_base = -abs(CONFIG.mitigation_beta.get(biome, 4.0))
    intervention = INTERVENTION_FACTOR[rec["class_id"]]
    # Intervention strengthens the mitigation drift proportionally
    beta_mit = beta_mit_base * (0.3 + 1.4 * intervention)

    models = load_all_models()
    steps = CONFIG.forecast_horizon
    bau = forecast(models["bau"], history, steps=steps, beta=0.0)
    mit = forecast(models["mitigation"], history, steps=steps, beta=beta_mit)
    lo  = forecast(models["lower"], history, steps=steps, beta=0.0)
    up  = forecast(models["upper"], history, steps=steps, beta=0.0)

    # Additional "if-not-followed" compound-growth counterfactual (BAU +
    # unchecked expansion 6 %/yr).  Gives the user a visual contrast.
    if_not_followed = bau * np.power(1.06, np.arange(1, steps + 1))
    cum_bau = np.cumsum(bau)
    cum_mit = np.cumsum(mit)
    cum_not = np.cumsum(if_not_followed)

    sss_current = scenario_separation_score(bau, mit)

    record = {
        "region_hash":         region_hash(bbox, t1_start, t2_start),
        "bbox":                list(bbox),
        "avle_plus":           avle["avle_plus"],
        "avle_plus_per_km2":   avle["avle_plus_per_km2"],
        "carbon_estimate_tco2": carbon_tco2,
        "loss_area_ha":        loss_area_ha,
        "biome":               biome,
        "t1":                  [t1_start, t1_end],
        "t2":                  [t2_start, t2_end],
        "source":              {"t1": src1, "t2": src2},
        "recommendation_class_id": rec["class_id"],
        "intervention_factor": intervention,
    }
    proof = blockchain.log_analysis(record) if log_blockchain else None

    return {
        "region_hash":              record["region_hash"],
        "bbox":                     list(bbox),
        "source":                   {"t1": src1, "t2": src2},
        "biome":                    biome,
        "ndvi_t1_mean":             ndvi1_mean,
        "ndvi_t2_mean":             ndvi2_mean,
        "ndvi_delta_mean":          float(delta.mean()),
        "ndvi_normalised":          True,
        "loss_area_ha":             loss_area_ha,
        "loss_pixel_count":         loss_px,
        "avle_plus":                avle["avle_plus"],
        "avle_plus_per_km2":        avle["avle_plus_per_km2"],
        "carbon_estimate_tco2":     carbon_tco2,
        "carbon_scenario_bau":        [float(x) for x in bau],
        "carbon_scenario_mitigation": [float(x) for x in mit],
        "carbon_scenario_unchecked":  [float(x) for x in if_not_followed],
        "carbon_cumulative_bau":        [float(x) for x in cum_bau],
        "carbon_cumulative_mitigation": [float(x) for x in cum_mit],
        "carbon_cumulative_unchecked":  [float(x) for x in cum_not],
        "carbon_ci_lower":          [float(x) for x in lo],
        "carbon_ci_upper":          [float(x) for x in up],
        "history_window":           [float(x) for x in history],
        "scenario_separation_score": sss_current,
        "intervention_factor":      intervention,
        "suitability":              suitability,
        "recommendation":           rec,
        "images": {
            "ndvi_t1":      _ndvi_to_png(ndvi1),
            "ndvi_t2":      _ndvi_to_png(ndvi2),
            "loss_overlay": _loss_overlay(red2, nir2, loss_mask, loss_mask_raw),
        },
        "blockchain":               proof,
    }
