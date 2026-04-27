"""Central configuration for AVLE-C.

All thresholds, paths and hyper-parameters live here. No hard-coded secrets;
environment variables are read where relevant.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
for _d in (WEIGHTS_DIR, DATA_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class AVLEConfig:
    # ------------- IPCC Tier-2 biome carbon stocks (tC / ha) ------------- #
    # Source: IPCC 2006 Guidelines, Vol. 4, Ch. 4 (Forest Land) Table 4.7/4.8
    biome_carbon_range: Dict[str, tuple] = field(
        default_factory=lambda: {
            "tropical_moist":      (120.0, 210.0),
            "tropical_dry":        (50.0, 120.0),
            "temperate_broadleaf": (50.0, 140.0),
            "boreal":              (30.0, 90.0),
            "grassland":           (5.0, 30.0),
        }
    )
    # Mitigation drift (tCO2/km2/year reduction) per biome
    mitigation_beta: Dict[str, float] = field(
        default_factory=lambda: {
            "tropical_moist":      8.5,
            "tropical_dry":        4.2,
            "temperate_broadleaf": 3.1,
            "boreal":              2.4,
            "grassland":           1.1,
        }
    )
    # ------------- Model I/O ------------- #
    segmentation_weights: Path = WEIGHTS_DIR / "segmentation_unet.pth"
    segmentation_config:  Path = WEIGHTS_DIR / "segmentation_config.json"
    carbon_rf_model:      Path = WEIGHTS_DIR / "carbon_rf_model.pkl"
    carbon_lin_model:     Path = WEIGHTS_DIR / "carbon_lin_model.pkl"
    carbon_gb_model:      Path = WEIGHTS_DIR / "carbon_gb_model.pkl"
    carbon_scaler:        Path = WEIGHTS_DIR / "carbon_scaler.pkl"
    carbon_metrics:       Path = WEIGHTS_DIR / "carbon_metrics.json"
    rec_mlp_weights:      Path = WEIGHTS_DIR / "recommendation_mlp.pth"
    rec_scaler:           Path = WEIGHTS_DIR / "recommendation_scaler.pkl"
    rec_config:           Path = WEIGHTS_DIR / "recommendation_config.json"
    rec_metrics:          Path = WEIGHTS_DIR / "recommendation_metrics.json"
    xgb_bau:              Path = WEIGHTS_DIR / "xgb_carbon_bau.json"
    xgb_mitigation:       Path = WEIGHTS_DIR / "xgb_carbon_mitigation.json"
    xgb_lower:            Path = WEIGHTS_DIR / "xgb_carbon_lower.json"
    xgb_upper:            Path = WEIGHTS_DIR / "xgb_carbon_upper.json"
    xgb_metrics:          Path = WEIGHTS_DIR / "xgb_metrics.json"
    ablation_results:     Path = RESULTS_DIR / "ablation.json"

    # ------------- AVLE+ weights / thresholds ------------- #
    kmeans_k: int = 5
    morph_kernel: int = 3
    ndvi_cloud_min: float = 0.02
    temporal_saturation: float = 0.30

    # ------------- Recommendation classes ------------- #
    rec_classes: List[str] = field(
        default_factory=lambda: [
            "monitoring_only",
            "targeted_replanting",
            "assisted_regeneration",
            "critical_afforestation",
        ]
    )
    rec_human_labels: List[str] = field(
        default_factory=lambda: [
            "Monitoring only — marginal loss detected",
            "Targeted replanting — high-recovery potential",
            "Assisted natural regeneration with soil stabilization",
            "Large-scale afforestation required",
        ]
    )
    rec_priority: List[str] = field(
        default_factory=lambda: ["low", "moderate", "high", "critical"]
    )

    # ------------- Forecast horizon ------------- #
    forecast_horizon: int = 5
    forecast_lags:    int = 4

    # ------------- Carbon conversion ------------- #
    # tons of carbon → tons of CO2 equivalent
    c_to_co2: float = 44.0 / 12.0


CONFIG = AVLEConfig()
