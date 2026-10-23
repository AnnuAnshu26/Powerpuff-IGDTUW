"""Robustness check: repeat the fused-vs-dynamic-only ablation on 5 different synthetic cohorts."""
import json
import numpy as np
from glucotwin.ehr import generate_cohort
from glucotwin.cgm import simulate_patient
from glucotwin.features import build_table
from glucotwin.model import GlucoTwinModel
from glucotwin.evaluate import score

rows = []
for seed in range(11, 16):
    co = generate_cohort(60, seed); ser = {r.patient_id: simulate_patient(r, 14, seed * 1000 + i) for i, r in co.iterrows()}
    ids = co.patient_id.tolist(); T = lambda s, k: build_table(co[co.patient_id.isin(s)], ser, k)
    tr, va, te = T(ids[:36], 2), T(ids[36:44], 2), T(ids[44:], 1)
    fu, dy = GlucoTwinModel(True).fit(tr, va), GlucoTwinModel(False).fit(tr, va)
    m = score(te, {"d": dy.forecast(te), "f": fu.forecast(te)})
    rows.append({"seed": seed, **{f"{k}_{n}": m[k][n] for k in "df" for n in ("mae_60", "mae_120", "rmse_60")}})
    print(rows[-1])
d = lambda a, b: np.array([r[a] - r[b] for r in rows])
out = {"runs": rows, "mae60_gain_mean": float(-d("f_mae_60", "d_mae_60").mean()), "mae120_gain_mean": float(-d("f_mae_120", "d_mae_120").mean()),
       "fusion_better_in_runs_mae60": int((d("f_mae_60", "d_mae_60") < 0).sum()), "n_runs": len(rows)}
json.dump(out, open("results/ablation_seeds.json", "w"), indent=1); print(out)
