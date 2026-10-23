"""Stream 1 (static / historical): synthetic Electronic Health Records.

Everything here is generated. No real patient data is used. The relationships between
EHR fields and physiology are *modelling assumptions* made so the fusion can be tested;
they are not clinical findings.
"""
import numpy as np
import pandas as pd

STATIC_COLS = ["age", "sex_male", "bmi", "duration_y", "hypertension", "neuropathy",
               "sulfonylurea", "ckd_stage", "hba1c", "egfr", "fasting_glucose", "ldl",
               "tcf7l2_risk_alleles", "kcnj11_risk_alleles"]

STATIC_LABELS = {
    "age": "Age", "sex_male": "Sex (male)", "bmi": "BMI", "duration_y": "Years with diabetes",
    "hypertension": "Hypertension diagnosis", "neuropathy": "Neuropathy diagnosis",
    "sulfonylurea": "Sulfonylurea medication", "ckd_stage": "Kidney disease stage",
    "hba1c": "HbA1c (lab)", "egfr": "Kidney function eGFR (lab)", "fasting_glucose": "Fasting glucose (lab)",
    "ldl": "LDL cholesterol (lab)", "tcf7l2_risk_alleles": "TCF7L2 rs7903146 risk alleles",
    "kcnj11_risk_alleles": "KCNJ11 rs5219 risk alleles"}


def generate_cohort(n: int = 60, seed: int = 7) -> pd.DataFrame:
    """Return one row per synthetic Type 2 diabetes patient."""
    r = np.random.default_rng(seed)
    age = r.integers(35, 81, n)
    male = r.random(n) < 0.55
    bmi = np.clip(r.normal(28.5, 4.5, n), 19, 45)
    dur = np.clip(r.gamma(2.5, 3.0, n), 0.5, 30)
    hba1c = np.clip(6.3 + 0.04 * dur + 0.03 * (bmi - 28) + r.normal(0, 0.8, n), 5.8, 12)
    egfr = np.clip(100 - 0.7 * (age - 35) - 0.6 * dur + r.normal(0, 12, n), 15, 120)
    sig = lambda x: 1 / (1 + np.exp(-x))
    htn = r.random(n) < sig(0.05 * (age - 55) + 0.04 * (bmi - 28))
    neuro = r.random(n) < sig(0.12 * (dur - 8))
    su = r.random(n) < 0.35
    fasting = np.clip(90 + 14 * (hba1c - 6) + r.normal(0, 10, n), 70, 300)
    ldl = np.clip(r.normal(105, 28, n), 50, 200)
    ckd = np.select([egfr >= 90, egfr >= 60, egfr >= 30], [1, 2, 3], 4)
    df = pd.DataFrame({
        "patient_id": [f"SYN-{i:03d}" for i in range(n)], "age": age, "sex_male": male.astype(int),
        "bmi": bmi.round(1), "duration_y": dur.round(1), "hypertension": htn.astype(int),
        "neuropathy": neuro.astype(int), "sulfonylurea": su.astype(int), "ckd_stage": ckd,
        "hba1c": hba1c.round(1), "egfr": egfr.round(0), "fasting_glucose": fasting.round(0),
        "ldl": ldl.round(0), "tcf7l2_risk_alleles": r.choice([0, 1, 2], n, p=[.45, .42, .13]),
        "kcnj11_risk_alleles": r.choice([0, 1, 2], n, p=[.5, .4, .1])})
    return df


def physiology(row, rng):
    """Hidden physiological parameters implied by a patient's EHR (simulation assumptions)."""
    ig = (17 * (1 - 0.015 * (row.bmi - 28)) * (1 + max(0, 90 - row.egfr) / 200)
          * (1.12 if row.sulfonylurea else 1.0) * rng.normal(1, .05))      # mg/dL per insulin unit
    tau_ins = 18 * (1 + max(0, 60 - row.egfr) / 60)                         # slower clearance in CKD (5-min steps)
    cg = 3.4 * (1 + 0.07 * row.tcf7l2_risk_alleles + 0.03 * (row.hba1c - 7)) * rng.normal(1, .05)
    base = 104 + 20 * (row.hba1c - 6)
    icr = 10 * (1 + 0.02 * (row.bmi - 28))                                  # nominal regimen ignores kidney function
    return dict(ig=ig, tau_ins=tau_ins, cg=cg, base=base, icr=icr)


def to_fhir_bundle(row) -> dict:
    """FHIR R4-style bundle (Patient, Conditions, Observations) for one synthetic patient."""
    pid = row["patient_id"]
    obs = lambda code, name, val, unit: {"resourceType": "Observation", "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": name}]},
        "subject": {"reference": f"Patient/{pid}"}, "valueQuantity": {"value": float(val), "unit": unit}}
    res = [{"resourceType": "Patient", "id": pid, "gender": "male" if row["sex_male"] else "female",
            "extension": [{"url": "synthetic", "valueBoolean": True}]},
           {"resourceType": "Condition", "subject": {"reference": f"Patient/{pid}"},
            "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": "E11", "display": "Type 2 diabetes mellitus"}]}},
           obs("4548-4", "HbA1c", row["hba1c"], "%"), obs("62238-1", "eGFR", row["egfr"], "mL/min/1.73m2"),
           obs("1558-6", "Fasting glucose", row["fasting_glucose"], "mg/dL"), obs("13457-7", "LDL cholesterol", row["ldl"], "mg/dL"),
           obs("39-1", "BMI (placeholder code)", row["bmi"], "kg/m2")]
    for gene, col in [("TCF7L2 rs7903146", "tcf7l2_risk_alleles"), ("KCNJ11 rs5219", "kcnj11_risk_alleles")]:
        res.append({"resourceType": "Observation", "status": "final", "code": {"text": f"{gene} risk alleles (simulated)"},
                    "subject": {"reference": f"Patient/{pid}"}, "valueInteger": int(row[col])})
    return {"resourceType": "Bundle", "type": "collection", "entry": [{"resource": x} for x in res]}
