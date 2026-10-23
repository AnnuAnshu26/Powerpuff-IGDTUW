import numpy as np, pytest
from glucotwin.ehr import generate_cohort, to_fhir_bundle, STATIC_COLS
from glucotwin.cgm import simulate_patient
from glucotwin.features import dynamic_features, build_table, DYN_COLS
from glucotwin.evaluate import clarke_zone
from glucotwin.whatif import what_if


@pytest.fixture(scope="module")
def world():
    co = generate_cohort(6, 1); s = {r.patient_id: simulate_patient(r, 6, i) for i, r in co.iterrows()}
    return co, s


def test_reproducible():
    a, b = generate_cohort(5, 3), generate_cohort(5, 3)
    assert a.equals(b)


def test_no_future_leakage(world):
    """Dynamic features at time t must not change if later glucose is altered."""
    co, s = world; df = s[co.patient_id[0]].copy(); f1 = dynamic_features(df)
    df2 = df.copy(); df2.loc[df2.index[500:], "glucose"] += 50; f2 = dynamic_features(df2)
    assert np.allclose(f1.iloc[:500].fillna(0), f2.iloc[:500].fillna(0))


def test_table_has_both_streams(world):
    co, s = world; t = build_table(co, s, stride=5)
    assert set(DYN_COLS) <= set(t.columns) and set(STATIC_COLS) <= set(t.columns) and not t.isna().any().any()


def test_clarke_known_points():
    assert clarke_zone(100, 105) == "A" and clarke_zone(60, 200) == "E" and clarke_zone(200, 60) == "E"


def test_fhir_bundle(world):
    b = to_fhir_bundle(world[0].iloc[0]); kinds = {e["resource"]["resourceType"] for e in b["entry"]}
    assert {"Patient", "Condition", "Observation"} <= kinds


def test_whatif_changes_forecast(world):
    from glucotwin.model import GlucoTwinModel
    co, s = world; tr = build_table(co, s, 3)
    m = GlucoTwinModel(True).fit(tr, tr.iloc[::7])
    r = what_if(s[co.patient_id[0]], 800, co.iloc[0], m, carbs_g=30)
    assert r["scenario"][60] > r["baseline"][60] and "HYPOTHETICAL" in r["label"]
