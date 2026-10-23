"""Feature engineering. Every dynamic feature at time t uses only data up to t (no leakage)."""
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from .ehr import STATIC_COLS

HORIZONS = {30: 6, 60: 12, 120: 24}          # minutes -> 5-min steps
DYN_COLS = ["glucose", "d15", "d30", "d60", "mean_1h", "std_1h", "min_1h", "carbs_on_board",
            "insulin_on_board", "steps_30m", "steps_2h", "hr_30m", "min_since_meal",
            "min_since_insulin", "sleeping", "tod_sin", "tod_cos"]
DYN_LABELS = {"glucose": "Current glucose", "d15": "15-min trend", "d30": "30-min trend", "d60": "60-min trend",
              "mean_1h": "1-hour average glucose", "std_1h": "Recent glucose variability", "min_1h": "Lowest glucose in last hour",
              "carbs_on_board": "Carbs on board", "insulin_on_board": "Insulin on board", "steps_30m": "Walking, last 30 min",
              "steps_2h": "Walking, last 2 h", "hr_30m": "Heart rate, last 30 min", "min_since_meal": "Minutes since last meal",
              "min_since_insulin": "Minutes since last insulin", "sleeping": "Asleep", "tod_sin": "Time of day (a)", "tod_cos": "Time of day (b)"}


def _since(x, cap=360):
    out = np.full(len(x), float(cap)); last = -10 ** 9
    for i, v in enumerate(x):
        if v > 0: last = i
        out[i] = min(cap, (i - last) * 5)
    return out


def dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df["glucose"]; f = pd.DataFrame(index=df.index)
    f["glucose"] = g
    for k, s in [("d15", 3), ("d30", 6), ("d60", 12)]:
        f[k] = g - g.shift(s)
    r = g.rolling(12)
    f["mean_1h"], f["std_1h"], f["min_1h"] = r.mean(), r.std(), r.min()
    f["carbs_on_board"] = lfilter([1], [1, -np.exp(-1 / 18)], df["carbs"].to_numpy())
    f["insulin_on_board"] = lfilter([1], [1, -np.exp(-1 / 30)], df["insulin"].to_numpy())
    f["steps_30m"] = df["steps"].rolling(6, min_periods=1).sum() / 100
    f["steps_2h"] = df["steps"].rolling(24, min_periods=1).sum() / 100
    f["hr_30m"] = df["heart_rate"].rolling(6, min_periods=1).mean()
    f["min_since_meal"] = _since(df["carbs"].to_numpy()); f["min_since_insulin"] = _since(df["insulin"].to_numpy())
    f["sleeping"] = df["sleeping"]
    f["tod_sin"], f["tod_cos"] = np.sin(2 * np.pi * df["tod"]), np.cos(2 * np.pi * df["tod"])
    return f[DYN_COLS]


def build_table(cohort: pd.DataFrame, series: dict, stride: int = 1) -> pd.DataFrame:
    """Join static EHR (stream 1) with dynamic features (stream 2) and add future targets."""
    parts = []
    for _, row in cohort.iterrows():
        df = series[row.patient_id]; f = dynamic_features(df)
        for m, h in HORIZONS.items():
            f[f"y{m}"] = df["glucose"].shift(-h) - df["glucose"]
            f[f"abs{m}"] = df["glucose"].shift(-h)
        f["patient_id"] = row.patient_id; f["t"] = np.arange(len(df))
        for c in STATIC_COLS: f[c] = row[c]
        f = f.iloc[48:-max(HORIZONS.values())].dropna()
        parts.append(f.iloc[::stride])
    return pd.concat(parts, ignore_index=True)
