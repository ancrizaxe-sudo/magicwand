"""Train the MLP recommendation classifier (Claim #3)."""
from __future__ import annotations

import json

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import CONFIG
from .recommendation_model import (N_CLASSES, RecommendationNet, save_config,
                                   generate_recommendation_dataset,
                                   rule_based_recommendation)


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    freq = np.bincount(y, minlength=n_classes).astype(np.float32) + 1e-6
    inv = 1.0 / freq
    w = inv / inv.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)


def main(epochs: int = 40, batch_size: int = 256, lr: float = 3e-4):
    print("[rec] Generating 30 000 synthetic samples …")
    X, y = generate_recommendation_dataset(n=30_000)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=42, stratify=ytmp)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xval_s, Xte_s = (scaler.transform(a).astype(np.float32) for a in (Xtr, Xval, Xte))

    dl_tr  = DataLoader(TensorDataset(torch.from_numpy(Xtr_s),  torch.from_numpy(ytr)),
                         batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(TensorDataset(torch.from_numpy(Xval_s), torch.from_numpy(yval)),
                         batch_size=batch_size)

    model = RecommendationNet()
    cls_w = _class_weights(ytr, N_CLASSES)
    crit = nn.CrossEntropyLoss(weight=cls_w)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    best_val = float("inf")
    patience_left = 10
    best_state = None
    history = []
    for ep in range(epochs):
        model.train()
        tr_losses = []
        for xb, yb in dl_tr:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            val_losses = [float(crit(model(xb), yb)) for xb, yb in dl_val]
        val_loss = float(np.mean(val_losses))
        sched.step(val_loss)
        history.append({"epoch": ep, "train": float(np.mean(tr_losses)), "val": val_loss})
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = 10
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[rec] early-stop @ epoch {ep}")
                break
        if ep % 5 == 0:
            print(f"[rec] epoch {ep:3d}  train={history[-1]['train']:.4f}  val={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    # Evaluation on test set
    with torch.no_grad():
        logits_te = model(torch.from_numpy(Xte_s))
        yp_mlp = logits_te.argmax(dim=1).numpy()

    yp_rule = rule_based_recommendation(Xte)
    def _metrics(yt, yp):
        p, r, f, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        acc = float((yt == yp).mean())
        return {"accuracy": acc, "precision_macro": float(p), "recall_macro": float(r),
                "f1_macro": float(f)}
    metrics = {
        "mlp":        _metrics(yte, yp_mlp),
        "rule_based": _metrics(yte, yp_rule),
        "confusion_mlp":        confusion_matrix(yte, yp_mlp).tolist(),
        "confusion_rule_based": confusion_matrix(yte, yp_rule).tolist(),
        "per_class_mlp":        classification_report(yte, yp_mlp, output_dict=True,
                                                      zero_division=0),
        "n_train": len(Xtr), "n_val": len(Xval), "n_test": len(Xte),
        "history": history,
        "class_names": CONFIG.rec_classes,
    }
    torch.save(model.state_dict(), CONFIG.rec_mlp_weights)
    joblib.dump(scaler, CONFIG.rec_scaler)
    save_config({"best_val_loss": best_val})
    CONFIG.rec_metrics.write_text(json.dumps(metrics, indent=2))
    print(f"[rec] ✓ MLP acc={metrics['mlp']['accuracy']:.3f}  F1={metrics['mlp']['f1_macro']:.3f}")
    print(f"[rec] ✓ Rule acc={metrics['rule_based']['accuracy']:.3f}  F1={metrics['rule_based']['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
