"""Fused forecaster: gradient-boosted trees over [static EHR features + dynamic wearable features]."""
import joblib
import numpy as np
import xgboost as xgb
from .features import DYN_COLS, HORIZONS
from .ehr import STATIC_COLS

PARAMS = dict(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=10, early_stopping_rounds=30, tree_method="hist", n_jobs=1)


class GlucoTwinModel:
    """use_static=True is the digital twin (two streams fused). False is the dynamic-only ablation."""

    def __init__(self, use_static: bool = True, seed: int = 0):
        self.use_static, self.seed = use_static, seed
        self.cols = DYN_COLS + (STATIC_COLS if use_static else [])
        self.reg, self.q, self.clf = {}, {}, None

    def fit(self, train, val):
        for m in HORIZONS:
            reg = xgb.XGBRegressor(random_state=self.seed, **PARAMS)
            reg.fit(train[self.cols], train[f"y{m}"], eval_set=[(val[self.cols], val[f"y{m}"])], verbose=False)
            self.reg[m] = reg
            # split-conformal half-width: 90% of validation errors fall inside +/- q
            self.q[m] = float(np.quantile(np.abs(val[f"y{m}"] - reg.predict(val[self.cols])), 0.9))
        clf = xgb.XGBClassifier(random_state=self.seed, **PARAMS)
        clf.fit(train[self.cols], (train["abs60"] < 70).astype(int),
                eval_set=[(val[self.cols], (val["abs60"] < 70).astype(int))], verbose=False)
        self.clf = clf
        return self

    def forecast(self, X) -> dict:
        """Absolute glucose forecasts (mg/dL) keyed by horizon in minutes."""
        return {m: np.clip(X["glucose"].to_numpy() + self.reg[m].predict(X[self.cols]), 30, 400) for m in HORIZONS}

    def low_probability(self, X):
        return self.clf.predict_proba(X[self.cols])[:, 1]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
