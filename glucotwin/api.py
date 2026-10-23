"""REST API for the clinician dashboard.  Run:  uvicorn glucotwin.api:app --reload
Endpoints: GET /health, GET /patients, GET /patients/{id}/ehr, GET /forecast/{id}?t=..., POST /whatif"""
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .ehr import generate_cohort, to_fhir_bundle
from .cgm import simulate_patient
from .features import dynamic_features
from .ehr import STATIC_COLS
from .model import GlucoTwinModel
from .explain import Explainer
from .whatif import what_if

SEED, N_PAT, DAYS = 7, 60, 14           # must match run_pipeline.py so the demo cohort is reproducible
app = FastAPI(title="GlucoTwin AI", version="1.0.0", description="Research prototype. Synthetic data only. Not a medical device.")
S = {}


def _load():
    if S: return
    root = Path(__file__).resolve().parent.parent
    S["model"] = GlucoTwinModel.load(root / "models" / "glucotwin_fused.joblib")
    S["cohort"] = generate_cohort(N_PAT, SEED).set_index("patient_id", drop=False)
    S["series"] = {r.patient_id: simulate_patient(r, DAYS, SEED * 1000 + i) for i, r in enumerate(S["cohort"].itertuples())}
    S["expl"] = Explainer(S["model"], 60)


def _patient(pid):
    _load()
    if pid not in S["cohort"].index: raise HTTPException(404, "unknown patient")
    return S["cohort"].loc[pid], S["series"][pid]


def _row(series, static, t):
    f = dynamic_features(series.iloc[max(0, t - 600): t + 1]).iloc[[-1]].copy()
    for c in STATIC_COLS: f[c] = static[c]
    return f


@app.get("/health")
def health(): return {"status": "ok", "synthetic_data_only": True}


@app.get("/patients")
def patients():
    _load(); return S["cohort"][["patient_id", "age", "hba1c", "egfr"]].to_dict("records")


@app.get("/patients/{pid}/ehr")
def ehr(pid: str):
    st, _ = _patient(pid); return {"record": st.to_dict(), "fhir": to_fhir_bundle(st)}


@app.get("/forecast/{pid}")
def forecast(pid: str, t: int = 1000):
    st, s = _patient(pid)
    if not 60 <= t < len(s) - 1: raise HTTPException(422, "t out of range")
    m, X = S["model"], _row(s, st, t); f = m.forecast(X); p = float(m.low_probability(X)[0])
    return {"patient": pid, "t": t, "current_glucose": float(X.glucose.iloc[0]),
            "forecast": {str(k): {"mgdl": float(v[0]), "low": float(v[0] - m.q[k]), "high": float(v[0] + m.q[k])} for k, v in f.items()},
            "p_low_60min": p, "explanation": S["expl"].explain_row(X, 5),
            "disclaimer": "Advisory model output from synthetic data. Not medical advice."}


class WhatIf(BaseModel):
    patient: str; t: int = 1000
    carbs_g: float = Field(0, ge=0, le=120); insulin_scale: float = Field(1, ge=0, le=1.5); walk_steps: float = Field(0, ge=0, le=6000)


@app.post("/whatif")
def whatif(q: WhatIf):
    st, s = _patient(q.patient)
    return what_if(s, q.t, st, S["model"], q.carbs_g, q.insulin_scale, q.walk_steps)
