"""Explainability: SHAP feature attributions, grouped by data stream, with plain-language text."""
import numpy as np
import shap
from .ehr import STATIC_COLS, STATIC_LABELS
from .features import DYN_COLS, DYN_LABELS

LABELS = {**DYN_LABELS, **STATIC_LABELS}


class Explainer:
    def __init__(self, model, horizon: int = 60):
        self.model, self.h = model, horizon
        self.tree = shap.TreeExplainer(model.reg[horizon])

    def values(self, X):
        return self.tree.shap_values(X[self.model.cols])

    def stream_share(self, X) -> dict:
        """Share of total mean |SHAP| carried by static EHR features vs dynamic wearable features."""
        sv = np.abs(self.values(X)).mean(0); cols = self.model.cols
        st = sum(v for c, v in zip(cols, sv) if c in STATIC_COLS); dy = sum(v for c, v in zip(cols, sv) if c in DYN_COLS)
        return {"static_ehr": float(st / (st + dy)), "dynamic_wearable": float(dy / (st + dy))}

    def explain_row(self, X_row, top: int = 4) -> list:
        sv = self.tree.shap_values(X_row[self.model.cols])[0]; items = []
        for c, s in sorted(zip(self.model.cols, sv), key=lambda a: -abs(a[1]))[:top]:
            v = float(X_row[c].iloc[0]); stream = "EHR" if c in STATIC_COLS else "wearable"
            verb = "lowers" if s < 0 else "raises"
            items.append({"feature": c, "label": LABELS[c], "value": round(v, 1), "shap_mgdl": round(float(s), 1), "stream": stream,
                          "text": f"{LABELS[c]} ({v:.0f}) {verb} the {self.h}-minute forecast by about {abs(s):.0f} mg/dL."})
        return items
