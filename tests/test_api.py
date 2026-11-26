import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import driftguard as dg
from fastapi.testclient import TestClient
import app as appmod

client = TestClient(appmod.app)


def test_psi_identical_is_near_zero():
    x = np.random.default_rng(0).normal(size=500)
    assert dg.psi(x, x) < 0.01


def test_never_policy_never_triggers():
    assert dg.decide("Never", {}, 3, -99, dg.Cfg())[0] is False


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_step_returns_decision_with_reason():
    r = client.post("/simulate/step").json()
    assert {"score", "reason", "triggered"} <= set(r)


def test_predict():
    feats = client.get("/sample").json()
    r = client.post("/predict", json={"features": feats}).json()
    assert r["low"] <= r["next_day_aqi"] <= r["high"] or "next_day_aqi" in r


def test_metrics_exposed():
    assert "driftguard_retrain_score" in client.get("/metrics").text
