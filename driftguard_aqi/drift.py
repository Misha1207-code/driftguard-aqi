"""Data-drift measures (PSI, KS) and the seasonal-match term."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_CAP = 0.25          # conventional "significant shift" level; used to normalise to 0-1


def psi(ref, cur, bins: int = 5, autocorr: float = 3.0) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) < 20 or len(cur) < 5:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    raw = float(np.sum((c - r) * np.log(c / r)))
    # small-sample PSI is biased upwards by about (bins-1)(1/n_ref + 1/n_cur); daily data is autocorrelated,
    # so use an effective sample size n/autocorr. Without this, a 4-week window always looks "drifted".
    bias = (len(edges) - 2) * (autocorr / len(ref) + autocorr / len(cur))
    return max(0.0, raw - bias)


def drift_score(ref_daily: pd.DataFrame, cur_daily: pd.DataFrame, cols) -> tuple[float, dict]:
    """0-1 score: mean over features of min(PSI/0.25,1), halved when the KS test is not significant."""
    parts = {}
    for c in cols:
        r, u = ref_daily[c].dropna(), cur_daily[c].dropna()
        p = psi(r, u)
        pv = ks_2samp(r, u).pvalue if len(u) > 3 and len(r) > 3 else 1.0
        parts[c] = min(p / PSI_CAP, 1.0) * (1.0 if pv < 0.05 else 0.5)
    return float(np.mean(list(parts.values()))), parts


def seasonal_match(history_daily: pd.DataFrame, cur_daily: pd.DataFrame, cols, pad_days: int = 21) -> float:
    """1 = current window resembles the same weeks last year (drift is expected seasonality)."""
    if len(cur_daily) == 0:
        return 0.0
    start = cur_daily.index.min() - pd.Timedelta(days=365 + pad_days)
    end = cur_daily.index.max() - pd.Timedelta(days=365 - pad_days)
    ly = history_daily.loc[start:end]
    if len(ly) < 20:
        return 0.0
    vals = [min(psi(ly[c], cur_daily[c]) / PSI_CAP, 1.0) for c in cols]
    return float(1.0 - np.mean(vals))
