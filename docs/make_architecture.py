"""Draws docs/architecture.pdf and docs/architecture.png"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 9)); ax.set_xlim(0, 160); ax.set_ylim(0, 90); ax.axis("off")
def box(x, y, w, h, title, lines=(), fc="#eef3fb", ec="#1d5fd1", tc="#0f1b2d"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=tc)
    for i, l in enumerate(lines): ax.text(x + w / 2, y + h - 6.6 - i * 3.3, l, ha="center", va="top", fontsize=8.2, color="#33475f")
def arrow(a, b, c="#33475f"): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.6, color=c))
ax.text(80, 87, "GlucoTwin AI: architecture (all data synthetic, processed locally)", ha="center", fontsize=15, fontweight="bold", color="#0f1b2d")
box(3, 54, 38, 27, "Stream 1: static / historical", ["Synthetic EHR (Synthea-style, FHIR R4 bundle)", "Demographics: age, sex, BMI", "Diagnoses: duration, HTN, CKD, neuropathy", "Labs: HbA1c, eGFR, fasting glucose, LDL", "Genetic markers: TCF7L2, KCNJ11 alleles", "Medication: sulfonylurea"], "#e9f6f1", "#16915c")
box(3, 24, 38, 24, "Stream 2: dynamic / real-time", ["Simulated CGM glucose (5-min)", "Smartwatch: steps, heart rate, sleep flag", "Logbook: meals (carbs), insulin doses", "Streamed live to the UI (WebSocket-ready)", "Replaceable by OhioT1DM / Shanghai CGM"], "#fdf3e3", "#d97706")
box(50, 38, 34, 32, "Feature engineering", ["Static: 14 EHR features", "Dynamic: 17 features (trends, variability,", "carbs / insulin on board, activity, HR,", "time since meal / insulin, time of day)", "Leakage-tested: features use data up to now", "Patient-level train / validation / test split"], "#eef3fb", "#1d5fd1")
box(93, 50, 30, 30, "Fusion forecaster", ["XGBoost on [EHR + wearable] features", "Horizons: +30, +60, +120 min", "P(glucose < 70 mg/dL in 60 min)", "Split-conformal 90% interval", "Ablation: dynamic-only model", "Baselines: persistence, damped trend"], "#eef3fb", "#1d5fd1")
box(93, 12, 30, 32, "Explain and simulate", ["SHAP attributions per forecast", "Share of EHR vs wearable evidence", "Plain-language reasons", "Counterfactual what-if: edit meals,", "insulin, activity; re-run the model", "Hypothetical only, no dosing advice"], "#f5eefb", "#7c3aed")
box(50, 6, 34, 24, "Evaluation", ["MAE / RMSE vs baselines", "Clarke Error Grid zones", "Low-event sensitivity, precision, AUROC", "False alerts per patient-day", "5-cohort robustness check"], "#f2f5fa", "#56677a")
box(131, 28, 27, 44, "Clinician dashboard", ["Patient profile (EHR)", "Live glucose + forecast cone", "Alert with reasons", "What-if panel", "Trust panel (accuracy)", "FHIR Observation export", "", "FastAPI: /forecast /whatif /ehr", "Browser demo: web/index.html"], "#fff", "#0f1b2d")
arrow((41, 68), (50, 62)); arrow((41, 36), (50, 46)); arrow((84, 58), (93, 62)); arrow((108, 50), (108, 44)); arrow((123, 62), (131, 58)); arrow((123, 28), (131, 36)); arrow((84, 54), (84, 30), "#56677a")
ax.text(80, 2, "Privacy by design: synthetic data only, no identifiers, local inference, no external AI API calls. Research prototype, not a medical device.", ha="center", fontsize=9, color="#56677a")
plt.savefig("docs/architecture.pdf", bbox_inches="tight"); plt.savefig("docs/architecture.png", dpi=150, bbox_inches="tight")
