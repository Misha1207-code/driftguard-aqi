"""Build the real-data CSV that driftguard.py / run_real.py read.

Input : Kaggle 'Air Quality Data in India (2015-2020)' -> city_day.csv  (CPCB data)
Weather: Open-Meteo ERA5 archive (free, no key). If the download fails the script continues
         with pollutants only and tells you.
Output : data/real_<city>.csv with columns  date, aqi, pm25, pm10, no2 [, temp, humidity, wind]

  python prepare_real_data.py --csv data/raw/city_day.csv --city Delhi
"""
import argparse, json, os, urllib.request, urllib.error
import pandas as pd

COORDS = {"Delhi": (28.61, 77.21), "Gurugram": (28.46, 77.03), "Lucknow": (26.85, 80.95),
          "Patna": (25.59, 85.14), "Ahmedabad": (23.03, 72.58), "Mumbai": (19.08, 72.88),
          "Kolkata": (22.57, 88.36), "Chennai": (13.08, 80.27), "Hyderabad": (17.38, 78.48),
          "Bengaluru": (12.97, 77.59), "Jaipur": (26.91, 75.79), "Chandigarh": (30.73, 76.78)}
# daily variable sets to try, in order (humidity is optional)
WEATHER_TRIES = [
    {"temp": "temperature_2m_mean", "humidity": "relative_humidity_2m_mean", "wind": "wind_speed_10m_max"},
    {"temp": "temperature_2m_mean", "wind": "wind_speed_10m_max"},
]


def fetch_weather(lat, lon, start, end):
    for mapping in WEATHER_TRIES:
        url = ("https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
               f"&daily={','.join(mapping.values())}&timezone=Asia%2FKolkata")
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                d = json.load(r)["daily"]
            w = pd.DataFrame({"date": pd.to_datetime(d["time"])})
            for short, long in mapping.items():
                w[short] = d[long]
            return w.set_index("date")
        except (urllib.error.URLError, KeyError, ValueError) as e:
            print(f"  weather attempt with {list(mapping.values())} failed: {e}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/city_day.csv")
    ap.add_argument("--city", default="Delhi")
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    raw = pd.read_csv(a.csv, parse_dates=["Date"])
    if a.city not in set(raw["City"]):
        raise SystemExit(f"City {a.city!r} not found. Available: {sorted(raw['City'].unique())}")
    d = raw[raw["City"] == a.city].rename(columns={"Date": "date", "AQI": "aqi", "PM2.5": "pm25",
                                                   "PM10": "pm10", "NO2": "no2"})
    d = d[["date", "aqi", "pm25", "pm10", "no2"]].sort_values("date").set_index("date")
    print(f"{a.city}: {len(d)} rows, {d.index.min().date()} -> {d.index.max().date()}")
    print("missing values per column:\n", d.isna().sum().to_string())

    if not a.no_weather and a.city in COORDS:
        w = fetch_weather(*COORDS[a.city], str(d.index.min().date()), str(d.index.max().date()))
        if w is not None:
            d = d.join(w, how="left")
            print("weather columns added:", list(w.columns))
        else:
            print("WARNING: weather unavailable, continuing with pollutants only (say so in your report).")

    out = a.out or f"data/real_{a.city.lower()}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.reset_index().to_csv(out, index=False)
    print(f"Saved {out}  (features: {[c for c in d.columns if c != 'aqi']})")


if __name__ == "__main__":
    main()
