# DriftGuard-AQI

Drift-aware, season-aware retraining with a challenger gate.
FastAPI service + Prometheus metrics + Grafana dashboard + Jenkins CI/CD.

## Run without Docker (fastest check)
    python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    pytest -q
    AUTO_SIM_SECONDS=5 uvicorn app:app --port 8000        # Windows: set AUTO_SIM_SECONDS=5 first
    open http://localhost:8000/docs

## Run the full stack
    docker compose up -d --build
    App http://localhost:8000/docs | Prometheus http://localhost:9090 | Grafana http://localhost:3000
    (Grafana: no login; dashboard "DriftGuard-AQI" is pre-loaded)

## Demo calls
    curl -X POST localhost:8000/simulate/run?weeks=10     # advance the stream 10 weeks
    curl localhost:8000/decisions?limit=5                  # retrain/skip decisions with reasons
    curl localhost:8000/sample                             # a feature row
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"features": {...}}'

## Jenkins (host install, needs Java 17, Python 3, Docker)
    java -jar jenkins.war --httpPort=8080
    New Item > Pipeline > Pipeline script from SCM > your Git repo > Script Path: Jenkinsfile
    Build with parameters (tick RUN_REPLAY to run the full experiment and archive results).

## Files
driftguard.py replay engine + policy comparison | stream.py live weekly stepper
app.py API + metrics | Jenkinsfile CI/CD | monitoring/ Prometheus + Grafana
Data is synthetic (demo). Replace with real data via driftguard.py --csv.
