"""End-to-end: load real data -> weekly features -> tune on validation stream -> compare on test stream.

  python run_experiment.py --csv data/raw/city_day.csv --city Delhi
"""
import argparse, json, os, sys
import joblib, pandas as pd
from driftguard_aqi.data import load_city_day, fetch_weather, attach_weather
from driftguard_aqi.experiment import prepare, tune, compare
from driftguard_aqi.policy import Config
from driftguard_aqi.report import make_plots

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/raw/city_day.csv")
ap.add_argument("--city", default="Delhi")
ap.add_argument("--weather", action="store_true", help="download ERA5 weather from Open-Meteo (needs internet)")
ap.add_argument("--no-tune", action="store_true", help="skip the grid search (fast smoke test)")
ap.add_argument("--review-every", type=int, default=2, help="2 = biweekly review, 4 = monthly")
ap.add_argument("--drift-ref", default="train", choices=["train", "same_season"])
ap.add_argument("--retrain-cost", type=float, default=0.3)
ap.add_argument("--out", default="results")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); os.makedirs("models", exist_ok=True)

daily = load_city_day(a.csv, a.city)
print(f"{a.city}: {len(daily)} daily rows, {daily.index.min().date()} -> {daily.index.max().date()}")
if a.weather:
    cache = f"data/raw/weather_{a.city}.csv"
    w = pd.read_csv(cache, parse_dates=[0], index_col=0) if os.path.exists(cache) else \
        fetch_weather(a.city, str(daily.index.min().date()), str(daily.index.max().date()), cache)
    daily = attach_weather(daily, w)
    print("weather attached:", [c for c in ("temp", "precip", "wind") if c in daily.columns])

X, y, s_val, s_test, n = prepare(daily)
print(f"weeks: train {s_val} | validation stream {s_test - s_val} | test stream {n - s_test}")
print(f"  validation {X.index[s_val].date()} -> {X.index[s_test-1].date()};  test {X.index[s_test].date()} -> {X.index[-1].date()}")

base = Config(drift_ref=a.drift_ref, review_every=a.review_every)
if a.no_tune:
    tuned = base
else:
    print("Tuning on validation stream ...")
    tuned, table = tune(X, y, daily, s_val, s_test, base, a.retrain_cost)
    table.to_csv(f"{a.out}/tuning_grid.csv", index=False)
print("Chosen config:", tuned.to_dict())

logs, metrics, champ = compare(X, y, daily, s_test, n, tuned)
metrics.round(3).to_csv(f"{a.out}/metrics.csv")
for k, v in logs.items():
    v.to_csv(f"{a.out}/log_{k}.csv", index=False)
json.dump(tuned.to_dict(), open(f"{a.out}/chosen_config.json", "w"), indent=2)
joblib.dump(dict(model=champ, features=list(X.columns), city=a.city), "models/champion.joblib")
X.tail(1).to_csv(f"{a.out}/latest_features.csv")
make_plots(logs, metrics, a.out)
try:  # optional experiment tracking / model registry (deck: "MLflow registers models")
    import mlflow
    mlflow.set_experiment("driftguard-aqi")
    for name, row in metrics.iterrows():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v) for k, v in (tuned.to_dict() if name.startswith("driftguard") else {"mode": row["mode"]}).items()})
            mlflow.log_metrics({k: float(row[k]) for k in ("mae", "rmse", "coverage", "model_fits", "promotions", "rejections")})
            mlflow.log_artifact(f"{a.out}/log_{name}.csv")
            if name == "driftguard_full":
                mlflow.log_artifact("models/champion.joblib"); mlflow.log_artifact(f"{a.out}/timeline.png")
    print("Logged to MLflow: run `mlflow ui` to browse.")
except ImportError:
    print("mlflow not installed - skipped tracking.")

print("\n", metrics[["mae", "rmse", "coverage", "model_fits", "promotions", "rejections"]].round(2).to_string())
