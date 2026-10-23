"""Counterfactual what-if: edit the event log at 'now', rebuild features, re-run the trained model.
Hypothetical only: it never recommends insulin doses or treatment."""
import pandas as pd
from .features import dynamic_features
from .ehr import STATIC_COLS


def _row(df, static_row):
    f = dynamic_features(df).iloc[[-1]].copy()
    for c in STATIC_COLS: f[c] = static_row[c]
    return f


def what_if(series: pd.DataFrame, t: int, static_row, model, carbs_g: float = 0.0,
            insulin_scale: float = 1.0, walk_steps: float = 0.0) -> dict:
    window = series.iloc[max(0, t - 600): t + 1].copy()
    base = model.forecast(_row(window, static_row))
    w = window.copy()
    w.iloc[-1, w.columns.get_loc("carbs")] += carbs_g
    w.iloc[-36:, w.columns.get_loc("insulin")] *= insulin_scale
    w.iloc[-6:, w.columns.get_loc("steps")] += walk_steps / 6
    scen = model.forecast(_row(w, static_row))
    return {"baseline": {m: float(v[0]) for m, v in base.items()}, "scenario": {m: float(v[0]) for m, v in scen.items()},
            "label": "HYPOTHETICAL scenario. Not treatment advice."}
