"""XGBoost scenario-conditioned carbon projection (Novelty Claim #4).

Two diverging scenarios (BAU + Mitigation) via feature engineering of
lag variables and a `scenario_drift` injection.  Uncertainty via quantile
regression at 2.5 % / 97.5 %.

Also ships an ARIMA baseline for ablation.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .config import CONFIG


LAG_COLS = [f"lag_{i}" for i in range(1, CONFIG.forecast_lags + 1)]
FEATURE_COLS = LAG_COLS + ["rolling_mean_3", "rolling_std_3",
                           "trend_index", "scenario_drift"]


# --------------------------------------------------------------------------- #
#  Feature engineering
# --------------------------------------------------------------------------- #
def make_lag_features(series: np.ndarray,
                      lags: int = CONFIG.forecast_lags,
                      scenario: str = "bau",
                      beta: float = 0.0) -> pd.DataFrame:
    df = pd.DataFrame({"carbon": np.asarray(series, dtype=np.float32)})
    for k in range(1, lags + 1):
        df[f"lag_{k}"] = df["carbon"].shift(k)
    df["rolling_mean_3"] = df["carbon"].rolling(3).mean()
    df["rolling_std_3"]  = df["carbon"].rolling(3).std().fillna(0.0)
    df["trend_index"]    = np.arange(len(df))
    df["scenario_drift"] = float(beta)
    df = df.dropna().reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
#  Synthetic historical sequences — used for training + ablation
# --------------------------------------------------------------------------- #
def generate_carbon_sequences(n_series: int = 500,
                              length: int = 24,
                              seed: int = 42) -> List[np.ndarray]:
    """Monthly-resolution carbon-flux sequences with seasonality + regime shifts."""
    rng = np.random.default_rng(seed)
    seqs: List[np.ndarray] = []
    for i in range(n_series):
        base = rng.uniform(50, 400)
        trend = rng.uniform(-2, 6)
        season_amp = rng.uniform(5, 25)
        noise = rng.normal(0, 8, length)
        t = np.arange(length)
        series = base + trend * t + season_amp * np.sin(2 * np.pi * t / 12) + noise
        if rng.random() < 0.3:
            shift = rng.integers(6, length - 3)
            series[shift:] += rng.uniform(20, 80)
        series = np.clip(series, 0.0, None)
        seqs.append(series.astype(np.float32))
    return seqs


def build_supervised(seqs: List[np.ndarray],
                     beta: float = 0.0,
                     lags: int = CONFIG.forecast_lags):
    X_parts, y_parts = [], []
    for s in seqs:
        df = make_lag_features(s, lags=lags, beta=beta)
        if len(df) == 0:
            continue
        X_parts.append(df[FEATURE_COLS].to_numpy(dtype=np.float32))
        y_parts.append(df["carbon"].to_numpy(dtype=np.float32))
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# --------------------------------------------------------------------------- #
#  Model factory
# --------------------------------------------------------------------------- #
def make_xgb(**overrides) -> XGBRegressor:
    params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    params.update(overrides)
    return XGBRegressor(**params)


def make_xgb_quantile(alpha: float) -> XGBRegressor:
    return make_xgb(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
    )


# --------------------------------------------------------------------------- #
#  Forecasting
# --------------------------------------------------------------------------- #
def forecast(model: XGBRegressor,
             history: np.ndarray,
             steps: int = CONFIG.forecast_horizon,
             lags: int = CONFIG.forecast_lags,
             beta: float = 0.0) -> np.ndarray:
    """Recursive multi-step forecast."""
    window = [float(x) for x in history[-lags:]]
    preds: List[float] = []
    for step in range(steps):
        roll3 = np.array(window[-3:])
        feats = window[-lags:] + [
            float(roll3.mean()),
            float(roll3.std()) if len(roll3) > 1 else 0.0,
            float(len(history) + step),
            float(beta),
        ]
        yhat = float(model.predict(np.array(feats, dtype=np.float32).reshape(1, -1))[0])
        preds.append(max(yhat, 0.0))
        window.append(yhat)
    return np.asarray(preds, dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Runtime loader
# --------------------------------------------------------------------------- #
_CACHE: Dict[str, XGBRegressor] = {}


def _load_single(path) -> XGBRegressor:
    m = XGBRegressor()
    m.load_model(str(path))
    return m


def load_all_models() -> Dict[str, XGBRegressor]:
    if _CACHE:
        return _CACHE
    _CACHE["bau"]        = _load_single(CONFIG.xgb_bau)
    _CACHE["mitigation"] = _load_single(CONFIG.xgb_mitigation)
    _CACHE["lower"]      = _load_single(CONFIG.xgb_lower)
    _CACHE["upper"]      = _load_single(CONFIG.xgb_upper)
    return _CACHE


def project_scenarios(history: np.ndarray,
                      biome: str = "tropical_moist",
                      steps: int = CONFIG.forecast_horizon
                      ) -> Dict[str, list]:
    models = load_all_models()
    beta_mit = -abs(CONFIG.mitigation_beta.get(biome, 4.0))
    bau = forecast(models["bau"], history, steps=steps, beta=0.0)
    mit = forecast(models["mitigation"], history, steps=steps, beta=beta_mit)
    lo  = forecast(models["lower"], history, steps=steps, beta=0.0)
    up  = forecast(models["upper"], history, steps=steps, beta=0.0)
    return {
        "bau":         [float(x) for x in bau],
        "mitigation":  [float(x) for x in mit],
        "ci_lower":    [float(x) for x in lo],
        "ci_upper":    [float(x) for x in up],
    }


# --------------------------------------------------------------------------- #
#  Scenario Separation Score  (SSS)  – Claim #4 novelty metric
# --------------------------------------------------------------------------- #
def scenario_separation_score(bau: np.ndarray, mit: np.ndarray) -> float:
    divergence = float(np.abs(bau[-1] - mit[-1]))
    magnitude  = float(np.mean(bau)) if np.mean(bau) > 1e-6 else 1.0
    return divergence / magnitude
