import pytest
from pathlib import Path
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(not Path("models/glucotwin_fused.joblib").exists(), reason="run run_pipeline.py first")


def test_endpoints():
    from glucotwin.api import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    pid = c.get("/patients").json()[45]["patient_id"]
    f = c.get(f"/forecast/{pid}?t=1200").json()
    assert set(f["forecast"]) == {"30", "60", "120"} and 0 <= f["p_low_60min"] <= 1 and f["explanation"]
    w = c.post("/whatif", json={"patient": pid, "t": 1200, "carbs_g": 20}).json()
    assert "HYPOTHETICAL" in w["label"]
    assert c.get("/forecast/NOPE").status_code == 404
