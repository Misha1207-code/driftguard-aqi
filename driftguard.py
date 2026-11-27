"""DriftGuard-AQI: drift-aware, season-aware retraining with a challenger gate.

Replays a daily AQI series as a weekly stream and compares retraining policies:
never, weekly, monthly, 12-month, drift-only, error-only, DriftGuard without
the gate, and DriftGuard (score + challenger gate + decision log).

Run:  python driftguard.py                      (synthetic demo data)
      python driftguard.py --csv my.csv --date-col date --target aqi \
             --features pm25,pm10,no2,temp,humidity,wind
"""
import argparse
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

POLICIES = ["Never", "Weekly", "Biweekly", "Monthly", "12-month", "Drift-only",
            "Error-only", "DriftGuard (no gate)", "DriftGuard (no season)", "DriftGuard"]


@dataclass
class Cfg:
    start: int = 540        # first replayed day (also initial training window)
    win: int = 540          # sliding training window (days)
    holdout: int = 28       # most recent labelled days used by the gate
    delay: int = 7          # days until verified labels are available
    step: int = 7           # replay step (weekly)
    margin: float = 0.02    # challenger must beat champion MAE by this fraction
    tau: float = 0.35       # DriftGuard retrain-score threshold
    cooldown: int = 4       # weeks between retrain attempts
    review: int = 1         # review cadence in weeks: 2 = biweekly, 4 = monthly. Adaptive policies only act at reviews
    fade: bool = True       # seasonal discount fades as measured error rises: season can excuse expected drift, never proven damage
    season: bool = True     # False = ablation: drift measured against training data, no seasonal discount
    w: tuple = (0.35, 0.35, 0.20, 0.30)  # drift, error, interval, seasonal
    drift_thr: float = 0.5  # drift-only baseline threshold (normalised PSI)
    err_thr: float = 0.5    # error-only baseline threshold
    iters: int = 80


# ----------------------------------------------------------------- data
def make_synthetic(n=1600, seed=0):
    """Dehradun-style daily data: winter peaks, recurring Nov stubble spike,
    a sensor (covariate) drift at day 950 and a concept drift at day 1150."""
    rng = np.random.default_rng(seed)
    t = np.arange(n); doy = t % 365
    winter = np.cos(2 * np.pi * (doy - 15) / 365)
    stubble = np.exp(-((doy - 305) / 12) ** 2)
    temp = 18 - 10 * winter + rng.normal(0, 2, n)
    hum = 60 + 15 * np.sin(2 * np.pi * (doy - 180) / 365) + rng.normal(0, 6, n)
    wind = np.clip(2.5 - 0.5 * winter + rng.normal(0, .6, n), 0.3, None)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 8)
    pm25 = np.clip(45 + 35 * winter + 90 * stubble + noise, 5, None)
    pm10 = pm25 * 1.6 + rng.normal(0, 10, n)
    no2 = 25 + 0.2 * pm25 + rng.normal(0, 5, n)
    aqi = 0.9 * pm25 + 0.25 * pm10 + 15 - 6 * wind + rng.normal(0, 8, n)
    aqi = np.where(t >= 1150, aqi + 0.35 * pm25 + 20, aqi)   # concept drift
    pm10 = np.where(t >= 950, pm10 * 1.4, pm10)              # sensor drift
    df = pd.DataFrame(dict(date=pd.date_range("2021-01-01", periods=n),
                           pm25=pm25, pm10=pm10, no2=no2, temp=temp,
                           humidity=hum, wind=wind, aqi=aqi))
    events = {"sensor drift": df.date[950], "concept drift": df.date[1150]}
    return df, events


def load_csv(path, date_col, target, features):
    df = pd.read_csv(path)
    df = df.rename(columns={date_col: "date", target: "aqi"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "aqi"] + features].sort_values("date")
    df = df.set_index("date").asfreq("D").interpolate(limit=3).reset_index()
    return df.dropna().reset_index(drop=True)


def build_features(df, raw):
    F = df[["date"] + raw].copy()
    a = df["aqi"]
    F["aqi_0"], F["aqi_1"], F["aqi_2"], F["aqi_6"] = a, a.shift(1), a.shift(2), a.shift(6)
    F["aqi_m7"] = a.rolling(7).mean()
    doy = df["date"].dt.dayofyear
    F["doy_s"] = np.sin(2 * np.pi * doy / 365.25)
    F["doy_c"] = np.cos(2 * np.pi * doy / 365.25)
    F["y"] = a.shift(-1)                       # next-day AQI is the target
    F = F.dropna().reset_index(drop=True)
    feats = [c for c in F.columns if c not in ("date", "y")]
    return F, feats


# ------------------------------------------------------------- monitoring
def psi(ref, cur, bins=5, autocorr=3.0):
    """Population Stability Index, corrected for small-sample bias.

    Raw PSI between a short window and a reference is biased upwards by about
    (bins-1)*(1/n_ref + 1/n_cur) even when nothing changed; daily AQI is autocorrelated,
    so the effective sample is n/autocorr. Without the correction, a 56-day window looks
    'drifted' almost every week and the drift signal saturates at 1.
    """
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) < 20 or len(cur) < 5:
        return 0.0
    q = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return 0.0
    q[0], q[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, q)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, q)[0] / len(cur), 1e-4, None)
    raw = float(np.sum((c - r) * np.log(c / r)))
    bias = (len(q) - 2) * (autocorr / len(ref) + autocorr / len(cur))
    return max(0.0, raw - bias)


def norm_psi(ref, cur):
    """Mean PSI over features, scaled so 0.25 (major shift) maps to 1."""
    p = np.mean([psi(ref[:, j], cur[:, j]) for j in range(ref.shape[1])])
    return float(min(1.0, p / 0.25))


class Bundle:
    """A model (plus 10-90% quantile models for the interval signal)."""

    def __init__(self, F, lo, hi, cfg, feats, drift_cols, intervals):
        lo = max(0, lo)
        tr, ho = F.iloc[lo:hi - cfg.holdout], F.iloc[hi - cfg.holdout:hi]
        kw = dict(max_iter=cfg.iters, learning_rate=0.08, random_state=0)
        self.feats, self.intervals, self.ho = feats, intervals, ho
        self.model = HGB(**kw).fit(tr[feats], tr["y"])
        if intervals:
            self.lo = HGB(loss="quantile", quantile=0.1, **kw).fit(tr[feats], tr["y"])
            self.hi = HGB(loss="quantile", quantile=0.9, **kw).fit(tr[feats], tr["y"])
        self.ref = tr[drift_cols].values
        mid, a, b = self.predict(ho)
        self.base_mae = float(np.mean(np.abs(mid - ho["y"].values)))
        self.base_width = float(np.mean(b - a)) if intervals else np.nan

    def predict(self, X):
        X = X[self.feats]
        mid = self.model.predict(X)
        if not self.intervals:
            return mid, np.full(len(X), np.nan), np.full(len(X), np.nan)
        lo, hi = self.lo.predict(X), self.hi.predict(X)
        return mid, np.minimum(lo, hi), np.maximum(lo, hi)


def signals(F, champ, pred, width, since, known, e, cfg, drift_cols):
    y = F["y"].values
    clip = lambda v: float(np.clip(v, 0, 1))
    cur = F[drift_cols].iloc[max(0, e - 56):e].values
    rp = lambda r, c: float(np.mean([psi(r[:, j], c[:, j]) for j in range(r.shape[1])]))
    raw_ref = rp(champ.ref, cur)
    if not cfg.season:          # ablation: plain drift vs the training window, no seasonal discount
        D, S = min(1.0, raw_ref / 0.25), 0.0
    elif e - 56 - 365 >= 0:     # drift beyond what the same weeks looked like last year
        prev = F[drift_cols].iloc[e - 56 - 365:e - 365].values
        raw_adj = rp(prev, cur)
        D = min(1.0, raw_adj / 0.25)                       # label-free
        S = clip(1 - raw_adj / raw_ref) if raw_ref > 0 else 0.0   # share explained by season
    else:
        D, S = min(1.0, raw_ref / 1.0), 0.0
    idx = np.arange(max(since, known - 28), known)      # labelled, by this champion
    E = 0.0
    if len(idx) >= 14:
        mae = np.mean(np.abs(pred[idx] - y[idx]))
        E = clip((mae / champ.base_mae - 1) / 0.5)
    U = 0.0                                             # label-free interval widening
    if champ.intervals:
        wi = width[max(since, e - 28):e]
        if len(wi) >= 7:
            U = clip((wi.mean() / champ.base_width - 1) / 0.5)
    w1, w2, w3, w4 = cfg.w
    score = max(0.0, w1 * D + w2 * E + w3 * U - w4 * S * ((1 - E) if cfg.fade else 1.0))
    return dict(D=D, E=E, U=U, S=S, score=score)


def decide(name, sg, i, last, cfg):
    cool = (i - last) >= cfg.cooldown
    at_review = (i + 1) % cfg.review == 0
    if name == "Never":
        return False, "never retrains"
    if name == "Weekly":
        return True, "weekly schedule"
    if name == "Biweekly":
        return ((i + 1) % 2 == 0), "biweekly schedule"
    if name == "Monthly":
        return (i > 0 and i % 4 == 0), "monthly schedule"
    if name == "12-month":
        return (i > 0 and i % 52 == 0), "12-month schedule"
    if name == "Drift-only":
        return (sg["D"] >= cfg.drift_thr and cool and at_review), f"drift {sg['D']:.2f} vs {cfg.drift_thr}"
    if name == "Error-only":
        return (sg["E"] >= cfg.err_thr and cool and at_review), f"error {sg['E']:.2f} vs {cfg.err_thr}"
    parts = (f"drift {sg['D']:.2f}, error {sg['E']:.2f}, interval {sg['U']:.2f}, "
             f"seasonal match {sg['S']:.2f}")
    if sg["score"] < cfg.tau:
        note = " (seasonal match discounted drift)" if sg["S"] > 0.5 and sg["D"] > 0.3 else ""
        return False, f"skip: score {sg['score']:.2f} < {cfg.tau}; {parts}{note}"
    if not at_review:
        return False, f"skip: above threshold, waiting for next review; score {sg['score']:.2f}"
    if not cool:
        return False, f"skip: cooldown; score {sg['score']:.2f}"
    return True, f"retrain: score {sg['score']:.2f} >= {cfg.tau}; {parts}"


# ----------------------------------------------------------------- replay
def run_policy(F, name, cfg, feats, drift_cols):
    n, y = len(F), F["y"].values
    if name == "DriftGuard (no season)":
        cfg = replace(cfg, season=False)
    gated = name in ("DriftGuard", "DriftGuard (no season)")
    intervals = name.startswith("DriftGuard")
    mk = lambda hi: Bundle(F, hi - cfg.win, hi, cfg, feats, drift_cols, intervals)
    champ = mk(cfg.start - cfg.delay)
    pred, width = np.full(n, np.nan), np.full(n, np.nan)
    since, last, log, weekly = cfg.start, -10 ** 6, [], []
    for i, d in enumerate(range(cfg.start, n, cfg.step)):
        e = min(d + cfg.step, n)
        mid, lo, hi = champ.predict(F.iloc[d:e])
        pred[d:e], width[d:e] = mid, hi - lo
        weekly.append((F["date"].iloc[d], float(np.sqrt(np.mean((mid - y[d:e]) ** 2)))))
        known = e - cfg.delay
        sg = signals(F, champ, pred, width, since, known, e, cfg, drift_cols)
        trig, reason = decide(name, sg, i, last, cfg)
        rec = dict(policy=name, week=i, date=F["date"].iloc[e - 1], **sg,
                   triggered=False, useful=False, promoted=False,
                   champ_mae=np.nan, chal_mae=np.nan, reason=reason)
        if trig and e < n:
            last = i
            chal = mk(known)
            ho = chal.ho
            cm = float(np.mean(np.abs(champ.predict(ho)[0] - ho["y"].values)))
            useful = chal.base_mae < cm * (1 - cfg.margin)
            promote = useful if gated else True
            rec.update(triggered=True, useful=useful, promoted=promote,
                       champ_mae=cm, chal_mae=chal.base_mae)
            if gated:
                rec["reason"] += (" -> PROMOTED" if promote else
                                  f" -> REJECTED (challenger MAE {chal.base_mae:.1f} "
                                  f"not better than champion {cm:.1f})")
            if promote:
                champ, since = chal, e
        log.append(rec)
    log = pd.DataFrame(log)
    err = pred[cfg.start:] - y[cfg.start:]
    summ = dict(Policy=name, Retrains=int(log.triggered.sum()),
                Promoted=int(log.promoted.sum()),
                Unneeded=int((log.triggered & ~log.useful).sum()),
                RMSE=float(np.sqrt(np.mean(err ** 2))), MAE=float(np.mean(np.abs(err))))
    return summ, pd.DataFrame(weekly, columns=["date", "rmse"]), log


def run_all(F, cfg, feats, drift_cols, policies=POLICIES, progress=None):
    summ, weekly, logs = [], {}, []
    for k, p in enumerate(policies):
        s, w, l = run_policy(F, p, cfg, feats, drift_cols)
        summ.append(s); weekly[p] = w; logs.append(l)
        if progress:
            progress((k + 1) / len(policies), p)
    return pd.DataFrame(summ).round(2), weekly, pd.concat(logs, ignore_index=True)


def plot(weekly, log, events, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    for p in [q for q in ["Never", "Biweekly", "Monthly", "Drift-only", "DriftGuard"] if q in weekly]:
        w = weekly[p]
        ax.plot(w.date, w.rmse.rolling(4, min_periods=1).mean(), label=p,
                lw=2.5 if p == "DriftGuard" else 1.3)
    for _, r in log[(log.policy == "DriftGuard") & log.promoted].iterrows():
        ax.axvline(r.date, color="green", alpha=.35, lw=1)
    for k, v in (events or {}).items():
        ax.axvline(v, color="red", ls="--"); ax.text(v, ax.get_ylim()[1] * .95, k, color="red", rotation=90, va="top")
    ax.set_ylabel("4-week rolling RMSE"); ax.legend(); ax.set_title("Green lines = DriftGuard promotions")
    fig.tight_layout(); fig.savefig(path, dpi=150)


def prepare(df, cfg_over=None):
    raw = [c for c in df.columns if c not in ("date", "aqi")]
    F, feats = build_features(df, raw)
    cfg = Cfg(**(cfg_over or {}))
    if len(F) < cfg.start + 200:                  # short real datasets
        cfg.start = cfg.win = len(F) // 2
    return F, feats, raw, cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--date-col", default="date")
    ap.add_argument("--target", default="aqi")
    ap.add_argument("--features", default="pm25,pm10,no2,temp,humidity,wind")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    import os; os.makedirs(a.out, exist_ok=True)
    if a.csv:
        df, events = load_csv(a.csv, a.date_col, a.target, a.features.split(",")), None
    else:
        df, events = make_synthetic()
    F, feats, raw, cfg = prepare(df)
    summ, weekly, log = run_all(F, cfg, feats, raw,
                                progress=lambda f, p: print(f"  done: {p}"))
    summ.to_csv(f"{a.out}/summary.csv", index=False)
    log.to_csv(f"{a.out}/decision_log.csv", index=False)
    plot(weekly, log, events, f"{a.out}/rmse_timeline.png")
    print("\n" + summ.to_string(index=False))
    print(f"\nSaved to {a.out}/ (summary.csv, decision_log.csv, rmse_timeline.png)")
