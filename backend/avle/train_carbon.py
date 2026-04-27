"""Train the carbon-flux Random Forest regressor (Claim #2)."""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .carbon_model import build_baselines, generate_carbon_dataset
from .config import CONFIG


def carbon_sensitivity_index(preds: np.ndarray, inputs: np.ndarray) -> float:
    """CSI = Var(model output) / Var(ΔNDVI).  delta_ndvi_mean is feature idx 3."""
    d_ndvi = inputs[:, 3]
    vin = float(np.var(d_ndvi)) or 1e-9
    vout = float(np.var(preds))
    return vout / vin


def main():
    print("[carbon] Generating 50 000 synthetic training samples …")
    X, y = generate_carbon_dataset(n=50_000)

    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.2, random_state=42)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=42)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xval_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xval), scaler.transform(Xte)

    models = build_baselines()
    report: dict = {}
    for name, model in models.items():
        print(f"[carbon] Fitting {name} …")
        model.fit(Xtr_s, ytr)
        yp = model.predict(Xte_s)
        rmse = float(np.sqrt(mean_squared_error(yte, yp)))
        mae = float(mean_absolute_error(yte, yp))
        r2 = float(r2_score(yte, yp))
        csi = carbon_sensitivity_index(yp, Xte)
        report[name] = {"rmse": rmse, "mae": mae, "r2": r2, "csi": csi,
                        "n_train": len(Xtr), "n_test": len(Xte)}
        print(f"           RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}  CSI={csi:.3f}")

    joblib.dump(models["rf"],     CONFIG.carbon_rf_model)
    joblib.dump(models["linear"], CONFIG.carbon_lin_model)
    joblib.dump(models["gb"],     CONFIG.carbon_gb_model)
    joblib.dump(scaler,           CONFIG.carbon_scaler)
    CONFIG.carbon_metrics.write_text(json.dumps(report, indent=2))
    print(f"[carbon] ✓ Weights saved to {CONFIG.carbon_rf_model.parent}")


if __name__ == "__main__":
    main()
