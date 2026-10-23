"""End-to-end: synthetic EHR + wearable streams -> train -> evaluate -> explain -> save artefacts.
Usage:  python run_pipeline.py            (about 2 to 4 minutes on a laptop CPU)
"""
import json, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glucotwin.ehr import generate_cohort, to_fhir_bundle
from glucotwin.cgm import simulate_patient
from glucotwin.features import build_table, HORIZONS
from glucotwin.model import GlucoTwinModel
from glucotwin.evaluate import baselines, score, coverage
from glucotwin.explain import Explainer, LABELS
from glucotwin.whatif import what_if

OUT = Path("results"); OUT.mkdir(exist_ok=True); Path("models").mkdir(exist_ok=True)
SEED, N_PAT, DAYS = 7, 60, 14


def main():
    t0 = time.time()
    cohort = generate_cohort(N_PAT, SEED)
    series = {r.patient_id: simulate_patient(r, DAYS, seed=SEED * 1000 + i) for i, r in cohort.iterrows()}
    cohort.to_csv(OUT / "synthetic_ehr.csv", index=False)
    (OUT / "example_fhir_bundle.json").write_text(json.dumps(to_fhir_bundle(cohort.iloc[0]), indent=1))
    ids = cohort.patient_id.tolist()                       # patient-level split: test patients are never seen in training
    tr_ids, va_ids, te_ids = ids[:36], ids[36:44], ids[44:]
    tab = lambda s, k=1: build_table(cohort[cohort.patient_id.isin(s)], series, stride=k)
    train, val, test = tab(tr_ids, 2), tab(va_ids, 2), tab(te_ids, 1)
    print(f"train {len(train)} rows / {len(tr_ids)} patients | val {len(val)} / {len(va_ids)} | test {len(test)} / {len(te_ids)} (unseen patients)")

    fused = GlucoTwinModel(use_static=True).fit(train, val); print("fused model trained", round(time.time() - t0), "s")
    dyn = GlucoTwinModel(use_static=False).fit(train, val); print("dynamic-only model trained", round(time.time() - t0), "s")
    fused.save("models/glucotwin_fused.joblib"); dyn.save("models/glucotwin_dynamic_only.joblib")

    preds = {**baselines(test), "Dynamic-only XGBoost": dyn.forecast(test), "GlucoTwin (EHR + wearable fusion)": fused.forecast(test)}
    probs = {"Dynamic-only XGBoost": dyn.low_probability(test), "GlucoTwin (EHR + wearable fusion)": fused.low_probability(test)}
    metrics = score(test, preds, probs)
    alert = {}
    for name, mdl in [("Dynamic-only XGBoost", dyn), ("GlucoTwin (EHR + wearable fusion)", fused)]:
        alert[name] = alert_rule(mdl, val, test, len(te_ids))
    ex = Explainer(fused, 60); share = ex.stream_share(test.sample(3000, random_state=0))
    cov = coverage(test, fused)
    summary = {"split": {"train_patients": len(tr_ids), "val_patients": len(va_ids), "test_patients": len(te_ids), "test_rows": len(test)},
               "metrics": metrics, "alert_rule_p_low": alert, "interval_coverage_90": {str(k): v for k, v in cov.items()}, "shap_stream_share_60min": share,
               "low_event_rate_60min_test": float((test["abs60"] < 70).mean())}
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=1))
    write_tables(metrics, share, cov, summary)
    plot_example(test, series, cohort, fused, dyn, ex)
    print(open(OUT / "ablation.md").read()); print("done in", round(time.time() - t0), "s")


def alert_rule(model, val, test, n_pat, target_sens=0.8):
    """Alert when P(low in 60 min) >= tau. tau is chosen on validation patients (best precision at >=80% sensitivity), then fixed for test."""
    yv = (val["abs60"] < 70).to_numpy(); pv = model.low_probability(val)
    best = 0.5
    for tau in np.linspace(0.05, 0.95, 91):
        s = ((pv >= tau) & yv).sum() / max(1, yv.sum())
        if s >= target_sens: best = tau
    yt = (test["abs60"] < 70).to_numpy(); pt = model.low_probability(test); fl = pt >= best
    tp, fp, fn = (fl & yt).sum(), (fl & ~yt).sum(), (~fl & yt).sum()
    days = len(test) / n_pat / 288
    return {"tau": float(best), "sensitivity": float(tp / (tp + fn)), "precision": float(tp / max(1, tp + fp)), "false_alert_readings_per_patient_day": float(fp / n_pat / days)}


def write_tables(metrics, share, cov, summary):
    f = lambda x: "n/a" if x is None else f"{100 * x:.0f}%"
    L = ["| Model | MAE 30 | MAE 60 | MAE 120 | RMSE 60 | Clarke A+B 60 | Clarke D+E 60 | Low sens. 60 | Low prec. 60 | AUROC low 60 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, o in metrics.items():
        L.append(f"| {n} | {o['mae_30']:.1f} | {o['mae_60']:.1f} | {o['mae_120']:.1f} | {o['rmse_60']:.1f} | {f(o['clarke_AB_60'])} | {f(o['clarke_DE_60'])} | {f(o['low_sens_60'])} | {f(o['low_prec_60'])} | {o.get('auroc_low_60', float('nan')):.2f} |")
    L += ["", f"MAE in mg/dL on {summary['split']['test_patients']} unseen synthetic patients ({summary['split']['test_rows']} forecasts). "
          f"Low-glucose event rate at 60 min in the test set: {100 * summary['low_event_rate_60min_test']:.1f}%.",
          f"Share of SHAP importance (60-min model): static EHR {100 * share['static_ehr']:.0f}%, dynamic wearable {100 * share['dynamic_wearable']:.0f}%.",
          "", "Alert rule P(low in 60 min) >= tau, tau tuned on validation patients for >=80% sensitivity:",
          *[f"- {n}: tau {a['tau']:.2f}, sensitivity {f(a['sensitivity'])}, precision {f(a['precision'])}, false-alert readings per patient-day {a['false_alert_readings_per_patient_day']:.1f}" for n, a in summary["alert_rule_p_low"].items()], "",
          "90% interval coverage (fused): " + ", ".join(f"{k} min {100 * v:.0f}%" for k, v in cov.items()) + "."]
    (OUT / "ablation.md").write_text("\n".join(L))


def plot_example(test, series, cohort, fused, dyn, ex):
    """Find a held-out moment where glucose looks fine, a low follows, and EHR fusion catches it."""
    f60, d60 = fused.forecast(test)[60], dyn.forecast(test)[60]; y = test["abs60"].to_numpy(); g = test["glucose"].to_numpy()
    ok = (g >= 95) & (y < 65) & (f60 < 70) & (d60 >= 70)
    idx = np.where(ok)[0] if ok.any() else np.where((g >= 90) & (y < 70) & (f60 < 75))[0]
    if len(idx) == 0: print("no showcase event found"); return
    i = idx[len(idx) // 2]; row = test.iloc[[i]]; pid, t = row.patient_id.iloc[0], int(row.t.iloc[0])
    s = series[pid]; static = cohort[cohort.patient_id == pid].iloc[0]
    fx, dx = fused.forecast(row), dyn.forecast(row); expl = ex.explain_row(row, 5)
    wi = what_if(s, t, static, fused, carbs_g=15)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    hx = np.arange(-36, 1) * 5; ax[0].plot(hx, s["glucose"].iloc[t - 36: t + 1], "k", lw=2, label="CGM reading")
    fut = np.arange(0, 25) * 5; ax[0].plot(fut[:25], s["glucose"].iloc[t: t + 25], color="grey", ls=":", label="what actually happened")
    mins = [0, 30, 60, 120]; ax[0].plot(mins, [row.glucose.iloc[0]] + [fx[m][0] for m in (30, 60, 120)], "o--", color="#1d5fd1", label="GlucoTwin forecast (EHR + wearable)")
    ax[0].plot(mins, [row.glucose.iloc[0]] + [dx[m][0] for m in (30, 60, 120)], "s--", color="#b7791f", label="dynamic-only forecast")
    ax[0].plot(mins, [row.glucose.iloc[0]] + [wi["scenario"][m] for m in (30, 60, 120)], "^-", color="#2a9d6f", label="what-if: +15 g carbs (hypothetical)")
    ax[0].fill_between(mins[1:], [fx[m][0] - fused.q[m] for m in (30, 60, 120)], [fx[m][0] + fused.q[m] for m in (30, 60, 120)], color="#1d5fd1", alpha=.12, label="90% interval")
    ax[0].axhline(70, color="r", ls="--", lw=1); ax[0].set(title=f"{pid}: reading looks fine, a low follows", xlabel="minutes from now", ylabel="glucose (mg/dL)"); ax[0].legend(fontsize=7.5, loc="lower left")
    names = [e["label"] for e in expl][::-1]; vals = [e["shap_mgdl"] for e in expl][::-1]
    ax[1].barh(names, vals, color=["#cf3b3b" if v < 0 else "#1d5fd1" for v in vals]); ax[1].axvline(0, color="k", lw=.8)
    ax[1].set(title="Why (SHAP, mg/dL on 60-min forecast)")
    plt.tight_layout(); plt.savefig(OUT / "example_forecast.png", dpi=150); plt.close()
    (OUT / "example_explanation.json").write_text(json.dumps({"patient": pid, "t": t, "current": float(row.glucose.iloc[0]), "fused_60": float(fx[60][0]),
        "dynamic_only_60": float(dx[60][0]), "actual_60": float(row.abs60.iloc[0]), "explanation": expl, "what_if": wi}, indent=1))


if __name__ == "__main__":
    main()
