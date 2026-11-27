"""Load real CPCB-derived daily AQI data, optionally add ERA5 weather, aggregate weekly."""
from __future__ import annotations
import json
import urllib.request
import numpy as np
import pandas as pd

POLLUTANTS = ["PM25", "PM10", "NO2", "SO2", "CO", "O3"]
CITY_COORDS = {  # lat, lon for the optional weather download
    "Delhi": (28.61, 77.21), "Ahmedabad": (23.03, 72.58), "Lucknow": (26.85, 80.95),
    "Patna": (25.59, 85.14), "Mumbai": (19.08, 72.88), "Kolkata": (22.57, 88.36),
    "Chennai": (13.08, 80.27), "Hyderabad": (17.38, 78.48), "Bengaluru": (12.97, 77.59),
}
WEATHER_COLS = ["temp", "precip", "wind"]


def load_city_day(path: str, city: str = "Delhi") -> pd.DataFrame:
    """Kaggle 'Air Quality Data in India (2015-2020)' city_day.csv -> clean daily frame."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["City"] == city].copy()
    if df.empty:
        raise ValueError(f"City {city!r} not in file. Available: {sorted(pd.read_csv(path)['City'].unique())}")
    df = df.rename(columns={"PM2.5": "PM25"}).set_index("Date").sort_index()
    keep = [c for c in POLLUTANTS + ["AQI"] if c in df.columns]
    df = df[keep].asfreq("D")
    df[keep] = df[keep].interpolate(limit=3)          # fill only short sensor gaps
    return df.dropna(subset=["AQI", "PM25", "PM10"])  # a real AQI needs these


def fetch_weather(city: str, start: str, end: str, cache: str | None = None) -> pd.DataFrame:
    """Free Open-Meteo ERA5 archive (no key). Returns daily temp, precip, wind."""
    lat, lon = CITY_COORDS[city]
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
           "&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max&timezone=Asia%2FKolkata")
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.load(r)["daily"]
    w = pd.DataFrame({"Date": pd.to_datetime(d["time"]), "temp": d["temperature_2m_mean"],
                      "precip": d["precipitation_sum"], "wind": d["wind_speed_10m_max"]}).set_index("Date")
    if cache:
        w.to_csv(cache)
    return w


def attach_weather(daily: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    if weather is None:
        return daily
    return daily.join(weather[WEATHER_COLS], how="left").interpolate(limit=3)


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly (Mon-Sun) means; precip is summed. Weeks with <4 observed days are dropped."""
    agg = {c: "mean" for c in daily.columns}
    if "precip" in agg:
        agg["precip"] = "sum"
    wk = daily.resample("W-SUN").agg(agg)
    wk["n_days"] = daily["AQI"].resample("W-SUN").count()
    wk = wk[wk["n_days"] >= 4].drop(columns="n_days")
    wk = wk.asfreq("W-SUN").interpolate(limit=2).dropna(subset=["AQI"])
    return wk
