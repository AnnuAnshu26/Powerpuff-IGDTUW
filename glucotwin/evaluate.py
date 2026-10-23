"""Evaluation: MAE/RMSE, Clarke Error Grid zones, low-glucose detection, AUROC."""
import numpy as np
from sklearn.metrics import roc_auc_score
from .features import HORIZONS


def clarke_zone(r: float, p: float) -> str:
    """Clarke Error Grid zone for reference r and prediction p (mg/dL)."""
    if (p <= 70 and r <= 70) or (0.8 * r <= p <= 1.2 * r): return "A"
    if (r >= 180 and p <= 70) or (r <= 70 and p >= 180): return "E"
    if (r >= 240 and 70 <= p <= 180) or (r <= 175 / 3 and 70 <= p <= 180) or (175 / 3 <= r <= 70 and p >= 1.2 * r): return "D"
    if (70 <= r <= 290 and p >= r + 110) or (130 <= r <= 180 and p <= 1.4 * r - 182): return "C"
    return "B"


def baselines(X) -> dict:
    """Reference forecasters: persistence ('no change') and damped linear trend."""
    g = X["glucose"].to_numpy(); slope = X["d30"].to_numpy() / 6
    return {"Persistence (no change)": {m: g.copy() for m in HORIZONS},
            "Damped linear trend": {m: np.clip(g + 0.5 * slope * h, 30, 400) for m, h in HORIZONS.items()}}


def score(test, preds: dict, probs: dict | None = None) -> dict:
    """preds: {name: {minutes: absolute forecast}}. Returns metrics per model."""
    out = {}
    for name, p in preds.items():
        o = {}
        for m in HORIZONS:
            y, yh = test[f"abs{m}"].to_numpy(), p[m]
            o[f"mae_{m}"] = float(np.mean(np.abs(y - yh))); o[f"rmse_{m}"] = float(np.sqrt(np.mean((y - yh) ** 2)))
            if m in (60, 120):
                z = np.array([clarke_zone(a, b) for a, b in zip(y, yh)])
                o[f"clarke_A_{m}"] = float(np.mean(z == "A")); o[f"clarke_AB_{m}"] = float(np.mean(np.isin(z, ["A", "B"])))
                o[f"clarke_DE_{m}"] = float(np.mean(np.isin(z, ["D", "E"])))
                tp = np.sum((y < 70) & (yh < 70)); fn = np.sum((y < 70) & (yh >= 70)); fp = np.sum((y >= 70) & (yh < 70))
                o[f"low_sens_{m}"] = float(tp / (tp + fn)) if tp + fn else None
                o[f"low_prec_{m}"] = float(tp / (tp + fp)) if tp + fp else None
        if probs and name in probs:
            o["auroc_low_60"] = float(roc_auc_score((test["abs60"] < 70).astype(int), probs[name]))
        out[name] = o
    return out


def coverage(test, model) -> dict:
    X = test; f = model.forecast(X)
    return {m: float(np.mean(np.abs(test[f"abs{m}"].to_numpy() - f[m]) <= model.q[m])) for m in HORIZONS}
