"""Weekly supervised table. Row t uses information up to week t and predicts AQI of week t+1."""
import numpy as np
import pandas as pd


def build_features(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    f = pd.DataFrame(index=weekly.index)
    for c in weekly.columns:
        f[c] = weekly[c]
    for lag in (1, 2, 3):
        f[f"AQI_lag{lag}"] = weekly["AQI"].shift(lag)
        f[f"PM25_lag{lag}"] = weekly["PM25"].shift(lag)
    f["AQI_roll4"] = weekly["AQI"].rolling(4).mean()
    woy = weekly.index.isocalendar().week.astype(float).values
    f["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    f["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    y = weekly["AQI"].shift(-1).rename("target")          # next week's AQI
    ok = f.notna().all(axis=1)
    return f[ok], y[ok]   # final row has NaN target (the future); caller drops it
