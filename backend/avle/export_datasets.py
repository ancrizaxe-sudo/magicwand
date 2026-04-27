"""Utility: export the synthetic training datasets to CSV files.

Runs once during training orchestration so the data that was *actually* used
to fit the models is downloadable from /api/datasets.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .carbon_model import FEATURE_NAMES, generate_carbon_dataset
from .config import CONFIG
from .prediction import FEATURE_COLS, build_supervised, generate_carbon_sequences
from .recommendation_model import REC_FEATURES, generate_recommendation_dataset


def _out(name: str) -> Path:
    p = CONFIG.ablation_results.parent / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def export_all() -> dict:
    out = {}

    # ---- carbon ---- #
    X, y = generate_carbon_dataset(n=50_000)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target_carbon_tco2"] = y
    p = _out("carbon_training_data.csv")
    df.to_csv(p, index=False, float_format="%.6g")
    out["carbon_training_data.csv"] = {"rows": len(df), "size_bytes": p.stat().st_size}

    # ---- recommendation ---- #
    Xr, yr = generate_recommendation_dataset(n=30_000)
    dfr = pd.DataFrame(Xr, columns=REC_FEATURES)
    dfr["label_id"] = yr
    dfr["label_name"] = [CONFIG.rec_classes[int(i)] for i in yr]
    p = _out("recommendation_training_data.csv")
    dfr.to_csv(p, index=False, float_format="%.6g")
    out["recommendation_training_data.csv"] = {"rows": len(dfr), "size_bytes": p.stat().st_size}

    # ---- XGBoost BAU supervised rows ---- #
    seqs = generate_carbon_sequences(n_series=500, length=24)
    Xp, yp = build_supervised(seqs, beta=0.0)
    dfp = pd.DataFrame(Xp, columns=FEATURE_COLS)
    dfp["target_carbon_tco2"] = yp
    p = _out("xgb_bau_training_data.csv")
    dfp.to_csv(p, index=False, float_format="%.6g")
    out["xgb_bau_training_data.csv"] = {"rows": len(dfp), "size_bytes": p.stat().st_size}

    # ---- raw time-series (the sequences themselves) ---- #
    max_len = max(len(s) for s in seqs)
    matrix = np.full((len(seqs), max_len), np.nan)
    for i, s in enumerate(seqs):
        matrix[i, :len(s)] = s
    dft = pd.DataFrame(matrix, columns=[f"t{t}" for t in range(max_len)])
    dft.insert(0, "sequence_id", range(len(seqs)))
    p = _out("xgb_raw_sequences.csv")
    dft.to_csv(p, index=False, float_format="%.6g")
    out["xgb_raw_sequences.csv"] = {"rows": len(dft), "size_bytes": p.stat().st_size}

    return out


if __name__ == "__main__":
    info = export_all()
    for name, meta in info.items():
        print(f"{name:40s}  {meta['rows']:>7d} rows  {meta['size_bytes']/1024:.1f} KB")
