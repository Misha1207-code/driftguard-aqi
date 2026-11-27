"""SYNTHETIC data in the city_day.csv layout. ONLY for testing the code path. Never report results from it."""
import numpy as np, pandas as pd

def make(path="data/raw/synthetic_city_day.csv", seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2015-01-01", "2020-07-01", freq="D")
    doy = d.dayofyear.values
    season = 1 + 0.8 * np.cos(2 * np.pi * (doy - 340) / 365)          # winter peak
    base = 110 * season
    shock = np.where((d >= "2019-11-01") & (d < "2020-02-01"), 1.35, 1.0) * np.where(d >= "2020-03-25", 0.5, 1.0)
    ar = np.zeros(len(d)); 
    for i in range(1, len(d)): ar[i] = 0.8 * ar[i-1] + rng.normal(0, 18)
    pm25 = np.clip(base * shock + ar, 10, None)
    df = pd.DataFrame({"City": "Delhi", "Date": d, "PM2.5": pm25, "PM10": pm25 * 1.9 + rng.normal(0, 15, len(d)),
                       "NO2": 40 + 0.15 * pm25 + rng.normal(0, 6, len(d)), "SO2": 15 + rng.normal(0, 3, len(d)),
                       "CO": 1.2 + rng.normal(0, .2, len(d)), "O3": 35 + rng.normal(0, 8, len(d))})
    df["AQI"] = np.clip(pm25 * 1.7 + rng.normal(0, 12, len(d)), 30, 500)
    df.to_csv(path, index=False)
    return path

if __name__ == "__main__":
    print(make())
