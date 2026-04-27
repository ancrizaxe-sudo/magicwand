"""Train the XGBoost carbon-projection models (Claim #4).

Produces four XGBoost boosters:
    - bau         : point forecast, scenario_drift=0
    - mitigation  : point forecast, scenario_drift=β<0 (biome dependent)
    - lower       : 2.5 %-quantile regressor (95 % CI lower bound)
    - upper       : 97.5 %-quantile regressor (95 % CI upper bound)

Also fits an ARIMA baseline for the ablation.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import CONFIG
from .prediction import (build_supervised, forecast, generate_carbon_sequences,
                         make_xgb, make_xgb_quantile, scenario_separation_score)

warnings.filterwarnings("ignore")


def _train_arima_baseline(seqs, horizon):
    """Fit an ARIMA(p,d,q) per sequence via AIC search and recursive forecast.

    Returns (rmse, mae, sss_score_list) averaged across series.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception:
        return {"rmse": None, "mae": None, "sss": None, "error": "statsmodels not installed"}
    rmses, maes, sss = [], [], []
    for s in seqs[:80]:  # ARIMA is slow; sub-sample for the baseline
        if len(s) < horizon + 6:
            continue
        train, test = s[:-horizon], s[-horizon:]
        best_aic, best_order = np.inf, (1, 1, 1)
        for p in (0, 1, 2):
            for d in (0, 1):
                for q in (0, 1, 2):
                    try:
                        m = ARIMA(train, order=(p, d, q)).fit()
                        if m.aic < best_aic:
                            best_aic, best_order = m.aic, (p, d, q)
                    except Exception:
                        continue
        try:
            m = ARIMA(train, order=best_order).fit()
            yp = np.asarray(m.forecast(steps=horizon))
        except Exception:
            continue
        yp = np.clip(yp, 0, None)
        rmses.append(float(np.sqrt(mean_squared_error(test, yp))))
        maes.append(float(mean_absolute_error(test, yp)))
        # ARIMA has no scenario concept → pseudo-mitigation = yp * 0.9
        sss.append(float(abs(yp[-1] - yp[-1] * 0.9) / max(yp.mean(), 1e-6)))
    return {"rmse": float(np.mean(rmses) if rmses else 0),
            "mae":  float(np.mean(maes)  if maes  else 0),
            "sss":  float(np.mean(sss)   if sss   else 0),
            "n":    len(rmses)}


def main():
    print("[xgb] Generating 500 synthetic sequences …")
    seqs = generate_carbon_sequences(n_series=500, length=24)

    # ---------------- Build supervised datasets ---------------- #
    X_bau, y_bau = build_supervised(seqs, beta=0.0)
    # Mitigation: same histories, but supervised targets shifted down
    beta_mit = -abs(CONFIG.mitigation_beta["tropical_moist"])
    mit_seqs = [s + beta_mit * np.arange(len(s)) / 12.0 for s in seqs]
    mit_seqs = [np.clip(s, 0, None) for s in mit_seqs]
    X_mit, y_mit = build_supervised(mit_seqs, beta=beta_mit)

    # Train/test split (row-wise) for evaluation
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_bau))
    cut = int(0.85 * len(idx))
    tr, te = idx[:cut], idx[cut:]

    print(f"[xgb] Fitting BAU on {len(tr)} rows …")
    m_bau = make_xgb(); m_bau.fit(X_bau[tr], y_bau[tr])
    print("[xgb] Fitting Mitigation …")
    m_mit = make_xgb(); m_mit.fit(X_mit[tr], y_mit[tr])
    print("[xgb] Fitting quantile 2.5 % …")
    m_lo  = make_xgb_quantile(0.025); m_lo.fit(X_bau[tr], y_bau[tr])
    print("[xgb] Fitting quantile 97.5 % …")
    m_up  = make_xgb_quantile(0.975); m_up.fit(X_bau[tr], y_bau[tr])

    # ---------------- Eval on held-out rows ---------------- #
    yp = m_bau.predict(X_bau[te])
    rmse = float(np.sqrt(mean_squared_error(y_bau[te], yp)))
    mae  = float(mean_absolute_error(y_bau[te], yp))

    # ---------------- Recursive forecast SSS ---------------- #
    hs = CONFIG.forecast_horizon
    sss_list = []
    for s in seqs[:50]:
        history = s[:-hs]
        bau_f = forecast(m_bau, history, steps=hs, beta=0.0)
        mit_f = forecast(m_mit, history, steps=hs, beta=beta_mit)
        sss_list.append(scenario_separation_score(bau_f, mit_f))
    xgb_sss = float(np.mean(sss_list))

    # ---------------- ARIMA baseline ---------------- #
    print("[xgb] Fitting ARIMA baseline (sub-sample) …")
    arima = _train_arima_baseline(seqs, horizon=hs)

    # Save models
    m_bau.save_model(str(CONFIG.xgb_bau))
    m_mit.save_model(str(CONFIG.xgb_mitigation))
    m_lo.save_model(str(CONFIG.xgb_lower))
    m_up.save_model(str(CONFIG.xgb_upper))

    metrics = {
        "xgb":   {"rmse": rmse, "mae": mae, "sss": xgb_sss,
                  "n_train": int(cut), "n_test": int(len(idx) - cut)},
        "arima": arima,
        "config": {"forecast_horizon": hs, "lags": CONFIG.forecast_lags},
    }
    CONFIG.xgb_metrics.write_text(json.dumps(metrics, indent=2))
    print(f"[xgb] ✓ RMSE={rmse:.2f}  MAE={mae:.2f}  SSS={xgb_sss:.3f}  (ARIMA SSS={arima.get('sss')})")


if __name__ == "__main__":
    main()
