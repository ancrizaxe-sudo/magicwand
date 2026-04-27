"""Neural-network recommendation engine (Novelty Claim #3).

MLP classifier (PyTorch) trained on 30 000 synthetic samples generated from a
rule-based decision matrix perturbed with Gaussian boundary noise so the
network must learn a probabilistic decision surface rather than memorise
deterministic thresholds.
"""
from __future__ import annotations

import json
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn

from .config import CONFIG


REC_FEATURES = [
    "avle_plus_normalized",
    "ndvi_mean_t2",
    "loss_area_ha_log",
    "delta_ndvi_mean",
    "delta_ndvi_std",
    "temporal_factor_mean",
]
N_FEATURES = len(REC_FEATURES)
N_CLASSES = 4
MC_DROPOUT_PASSES = 25          # Monte-Carlo forward passes for uncertainty


# --------------------------------------------------------------------------- #
#  Architecture
# --------------------------------------------------------------------------- #
class RecommendationNet(nn.Module):
    def __init__(self, input_dim: int = N_FEATURES, num_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
#  Synthetic dataset — rule-based labels + Gaussian boundary noise
# --------------------------------------------------------------------------- #
def generate_recommendation_dataset(n: int = 30_000, seed: int = 42
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Rule-based labels perturbed with heavier Gaussian boundary noise
    (σ = 0.15, up from 0.05).  Higher σ prevents the MLP from memorising
    hard thresholds and forces a calibrated probabilistic surface."""
    rng = np.random.default_rng(seed)
    avle     = rng.uniform(0, 1, n)
    ndvi     = rng.uniform(0.1, 0.9, n)
    area_ha  = rng.exponential(50, n).clip(0.1, 500)
    d_ndvi   = rng.uniform(0, 0.8, n)
    d_std    = rng.uniform(0, 0.3, n)
    temp_f   = rng.uniform(0, 1, n)

    label = np.zeros(n, dtype=np.int64)
    noise = rng.normal(0, 0.15, n)   # ← was 0.05

    label[avle + noise > 0.75] = 3
    mask2 = ((avle + noise > 0.45) & (ndvi < 0.4))
    label[mask2 & (label == 0)] = 2
    mask1 = ((area_ha < 10) & (ndvi > 0.5))
    label[mask1 & (label == 0)] = 1

    mask2b = (d_ndvi > 0.45) & (avle > 0.35) & (avle < 0.75)
    label[mask2b & (label == 0)] = 2

    features = np.stack([
        avle, ndvi, np.log1p(area_ha),
        d_ndvi, d_std, temp_f,
    ], axis=1).astype(np.float32)
    return features, label


# --------------------------------------------------------------------------- #
#  Rule-based baseline (for ablation)
# --------------------------------------------------------------------------- #
def rule_based_recommendation(features: np.ndarray) -> np.ndarray:
    """Deterministic IF-ELSE engine — used as the Claim #3 ablation baseline."""
    avle   = features[:, 0]
    ndvi   = features[:, 1]
    area   = np.expm1(features[:, 2])
    out = np.zeros(len(features), dtype=np.int64)
    out[avle > 0.75] = 3
    out[(out == 0) & (avle > 0.45) & (ndvi < 0.4)] = 2
    out[(out == 0) & (area < 10) & (ndvi > 0.5)] = 1
    return out


# --------------------------------------------------------------------------- #
#  Inference
# --------------------------------------------------------------------------- #
_CACHE: Dict[str, object] = {}


def _load():
    if _CACHE:
        return _CACHE
    scaler = joblib.load(CONFIG.rec_scaler)
    model = RecommendationNet()
    state = torch.load(CONFIG.rec_mlp_weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    _CACHE["m"] = model
    _CACHE["s"] = scaler
    return _CACHE


def recommend(features: np.ndarray, mc_passes: int = MC_DROPOUT_PASSES) -> dict:
    """Monte-Carlo-dropout inference.

    We run `mc_passes` forward passes with dropout *enabled* (train-mode),
    then take the mean probability vector as the predictive distribution and
    the std across passes as an epistemic-uncertainty estimate.  This gives
    genuine confidence — not an over-confident softmax from a calibrated net.
    """
    cache = _load()
    x = cache["s"].transform(features.reshape(1, -1)).astype(np.float32)
    t = torch.from_numpy(x)

    model = cache["m"]
    # Keep BatchNorm in eval-mode (it can't cope with batch-size 1 in train-mode)
    # but force every Dropout layer into train-mode so it remains stochastic.
    model.eval()
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()
    probs_list = []
    with torch.no_grad():
        for _ in range(mc_passes):
            logits = model(t)
            probs_list.append(torch.softmax(logits, dim=1).numpy()[0])
    model.eval()
    probs_arr = np.stack(probs_list, axis=0)         # (passes, 4)
    mean_p = probs_arr.mean(axis=0)
    std_p  = probs_arr.std(axis=0)

    pred = int(np.argmax(mean_p))
    # Predictive entropy (nats) — higher = more uncertain
    entropy = float(-np.sum(mean_p * np.log(mean_p + 1e-9)))
    entropy_max = float(np.log(N_CLASSES))
    # Normalised epistemic uncertainty ∈ [0,1]
    epistemic = float(std_p.mean() / 0.5)
    return {
        "class_id":            pred,
        "action":              CONFIG.rec_human_labels[pred],
        "priority":            CONFIG.rec_priority[pred],
        "confidence":          float(mean_p[pred]),
        "class_probabilities": [float(p) for p in mean_p],
        "class_probabilities_std": [float(s) for s in std_p],
        "predictive_entropy":  entropy,
        "entropy_normalised":  entropy / entropy_max,
        "epistemic_uncertainty": min(1.0, epistemic),
        "mc_passes":           mc_passes,
        "class_names":         CONFIG.rec_classes,
    }


def build_rec_features(avle_plus_norm: float,
                       ndvi_mean_t2: float,
                       loss_area_ha: float,
                       delta_ndvi_mean: float,
                       delta_ndvi_std: float,
                       temporal_factor_mean: float) -> np.ndarray:
    return np.array([
        avle_plus_norm, ndvi_mean_t2, float(np.log1p(loss_area_ha)),
        delta_ndvi_mean, delta_ndvi_std, temporal_factor_mean,
    ], dtype=np.float32)


def save_config(extra: dict | None = None):
    data = {
        "architecture": "MLP 128→64→32→4",
        "input_features": REC_FEATURES,
        "classes": CONFIG.rec_classes,
    }
    if extra:
        data.update(extra)
    CONFIG.rec_config.write_text(json.dumps(data, indent=2))
