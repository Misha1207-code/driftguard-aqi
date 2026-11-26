"""DriftGuard-AQI service: FastAPI + Prometheus metrics."""
import os
import threading
import time

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

from stream import Stream

app = FastAPI(title="DriftGuard-AQI")
S, lock = Stream(), threading.Lock()

GAUGES = {k: Gauge(f"driftguard_{k}", d) for k, d in {
    "drift_score": "Data drift vs same weeks last year (0-1)",
    "error_score": "Rolling error rise vs baseline (0-1)",
    "interval_score": "Prediction-interval widening (0-1)",
    "seasonal_match": "Share of drift explained by season (0-1)",
    "retrain_score": "Combined retrain score",
    "weekly_rmse": "RMSE of the latest replayed week",
    "champion_version": "Version of the current champion model",
    "replay_week": "Replay week number"}.items()}
TRIGGERED = Counter("driftguard_retrains_triggered", "Retrains triggered")
PROMOTED = Counter("driftguard_challengers_promoted", "Challengers promoted")
REJECTED = Counter("driftguard_challengers_rejected", "Challengers rejected by the gate")
PREDICTIONS = Counter("driftguard_predictions", "Predictions served")


def do_step():
    with lock:
        if S.done:
            S.reset()
        r = S.step()
    for k, g in (("drift_score", "D"), ("error_score", "E"), ("interval_score", "U"),
                 ("seasonal_match", "S"), ("retrain_score", "score")):
        GAUGES[k].set(r[g])
    GAUGES["weekly_rmse"].set(r["weekly_rmse"])
    GAUGES["champion_version"].set(r["version"])
    GAUGES["replay_week"].set(r["week"])
    if r["triggered"]:
        TRIGGERED.inc()
        (PROMOTED if r["promoted"] else REJECTED).inc()
    return r


@app.on_event("startup")
def start_auto():
    secs = float(os.getenv("AUTO_SIM_SECONDS", "0"))
    if secs > 0:                       # keeps Grafana moving during the demo
        def loop():
            while True:
                do_step()
                time.sleep(secs)
        threading.Thread(target=loop, daemon=True).start()


class PredictIn(BaseModel):
    features: dict


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html"))


@app.get("/health")
def health():
    return {"status": "ok", "champion_version": S.version, "week": S.i}


@app.get("/sample")
def sample():
    return S.sample_row()


@app.post("/predict")
def predict(body: PredictIn):
    missing = [k for k in S.feats if k not in body.features]
    if missing:
        raise HTTPException(422, f"missing features: {missing}")
    p, lo, hi = S.predict(body.features)
    PREDICTIONS.inc()
    return {"next_day_aqi": p, "low": lo, "high": hi, "champion_version": S.version}


@app.post("/simulate/step")
def simulate_step():
    return do_step()


@app.post("/simulate/run")
def simulate_run(weeks: int = 10):
    return [do_step() for _ in range(max(1, min(weeks, 60)))]


@app.post("/simulate/reset")
def simulate_reset():
    with lock:
        S.reset()
    return {"status": "reset"}


@app.get("/decisions")
def decisions(limit: int = 20):
    return S.log[-limit:]


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
