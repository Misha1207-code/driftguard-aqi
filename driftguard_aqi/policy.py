"""The DriftGuard decision policy and the baselines it is compared with, run as a weekly stream.

Every week T:  predict week T+1  ->  compute monitoring signals using ONLY what is known at T
(actual AQI for week t arrives `label_delay` weeks after it happens)  ->  decide  ->  log reason.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from .drift import drift_score, seasonal_match
from .models import Forecaster


@dataclass
class Config:
    mode: str = "driftguard"          # never | fixed | error_only | driftguard
    fixed_k: int = 13                 # weeks between retrains for mode=fixed
    w: tuple = (1.0, 1.0, 0.5, 0.5)   # w1 drift, w2 error rise, w3 interval warning, w4 seasonal match
    threshold: float = 0.6
    review_every: int = 2             # review cadence in weeks: 2 = biweekly, 4 = monthly. Retrains only happen at reviews.
    cooldown: int = 2                 # minimum weeks between retrain attempts
    margin: float = 0.05              # challenger must beat champion by this fraction
    gate_weeks: int = 8               # most recent labelled weeks used for the gate
    min_gate_weeks: int = 4           # fewer labelled weeks than this: gate cannot judge, defer
    label_delay: int = 1              # extra weeks before the true AQI is usable
    use_season: bool = True
    use_gate: bool = True
    drift_days: int = 28
    err_window: int = 4
    err_trigger: float = 1.5          # error_only: retrain when MAE > 1.5 x reference MAE
    max_train_weeks: int = 156
    drift_cols: tuple = ("PM25", "PM10", "NO2")
    drift_ref: str = "train"          # "train" = whole training window (deck default); "same_season" = training rows within +-5 weeks of the calendar date

    def to_dict(self):
        return asdict(self)


def _clip01(x):
    return float(min(max(x, 0.0), 1.0))


def run_stream(X: pd.DataFrame, y: pd.Series, daily: pd.DataFrame, start: int, end: int, cfg: Config,
               verbose: bool = False):
    """Simulate weeks [start, end). Returns (log DataFrame, metrics dict, final champion)."""
    Xv, yv, idx = X.values.astype(float), y.values.astype(float), X.index
    cols = [c for c in cfg.drift_cols if c in daily.columns]
    d = cfg.label_delay

    def fit_on(n_lab):
        lo = max(0, n_lab - cfg.max_train_weeks)
        return Forecaster().fit(Xv[lo:n_lab], yv[lo:n_lab])

    def ref_window(n_lab):
        end_d = idx[n_lab - 1]
        return daily.loc[end_d - pd.Timedelta(days=7 * cfg.max_train_weeks): end_d]

    n0 = start - d
    champ, champ_id, ref_daily = fit_on(n0), 0, ref_window(n0)
    last_attempt, n_fits, n_prom, n_rej = start, 0, 0, 0
    rows = []

    for T in range(start, end):
        yhat, lo, hi, width = (v[0] for v in champ.predict(Xv[T]))
        rec = dict(week=T, date=idx[T], pred=yhat, lo=lo, hi=hi, width=width, actual=yv[T], champion=champ_id,
                   drift=np.nan, err_ratio=np.nan, interval=np.nan, season=np.nan, score=np.nan,
                   action="skip", reason="")
        rows.append(rec)

        # ---- monitoring signals (information available at week T only) ----
        mine = [r for r in rows if r["champion"] == champ_id]
        labelled = [r for r in mine if r["week"] + 1 + d <= T][-cfg.err_window:]
        err_ratio = (np.mean([abs(r["pred"] - r["actual"]) for r in labelled]) / max(champ.ref_mae_, 1e-6)
                     if len(labelled) >= 2 else 1.0)
        rec["err_ratio"] = err_ratio
        if cfg.mode == "never":
            continue

        trigger, reason = False, ""
        since = T - last_attempt
        at_review = (T - start + 1) % cfg.review_every == 0
        if cfg.mode == "fixed":
            if (T - start + 1) % cfg.fixed_k == 0:
                trigger, reason = True, f"fixed schedule every {cfg.fixed_k} weeks"
        elif cfg.mode == "error_only":
            if err_ratio > cfg.err_trigger and since >= cfg.cooldown and at_review:
                trigger, reason = True, f"rolling MAE {err_ratio:.2f}x reference > {cfg.err_trigger}x"
        else:  # driftguard
            cur = daily.loc[idx[T] - pd.Timedelta(days=cfg.drift_days - 1): idx[T]]
            ref = ref_daily
            if cfg.drift_ref == "same_season":
                doy_c = cur.index.dayofyear.values.mean()
                diff = np.abs(ref_daily.index.dayofyear.values - doy_c)
                ref = ref_daily[np.minimum(diff, 365 - diff) <= 35]
            D, parts = drift_score(ref, cur, cols)
            E = _clip01(err_ratio - 1.0)
            I = _clip01(np.mean([r["width"] for r in mine[-4:]]) / champ.ref_width_ - 1.0)
            S = seasonal_match(daily.loc[:idx[T]], cur, cols) if cfg.use_season else 0.0
            w1, w2, w3, w4 = cfg.w
            score = w1 * D + w2 * E + w3 * I - w4 * S
            rec.update(drift=D, interval=I, season=S, score=score)
            top = max(parts, key=parts.get)
            reason = (f"score {score:.2f} (drift {D:.2f} [{top}], error {E:.2f}, interval {I:.2f}, "
                      f"season-match {S:.2f}) vs threshold {cfg.threshold:.2f}")
            if score > cfg.threshold and since >= cfg.cooldown and at_review:
                trigger = True
            elif score > cfg.threshold and not at_review:
                reason += " - above threshold, waiting for next review"
            elif score > cfg.threshold:
                reason += f" - blocked by cooldown ({since}/{cfg.cooldown} weeks)"
                rec["action"] = "cooldown"
            else:
                reason += " - below threshold"
        rec["reason"] = reason
        if not trigger:
            continue

        # ---- retrain attempt ----
        n_lab = T - d
        gated = cfg.mode == "driftguard" and cfg.use_gate
        promote = True
        if gated:
            # Shadow evaluation: compare what we actually SERVED (logged live forecasts, genuinely
            # out-of-sample when made) with a challenger trained only on data before those weeks.
            lab = [r for r in rows if r["week"] + 1 + d <= T][-cfg.gate_weeks:]
            if len(lab) < cfg.min_gate_weeks:
                rec["reason"] += "; gate cannot judge yet (too few labelled weeks), deferred"
                continue
            wks = [r["week"] for r in lab]
            lo_i = max(0, wks[0] - cfg.max_train_weeks)
            chal = Forecaster().fit(Xv[lo_i:wks[0]], yv[lo_i:wks[0]])
            mae_chal = float(np.mean(np.abs(chal.predict(Xv[wks])[0] - yv[wks])))
            mae_champ = float(np.mean([abs(r["pred"] - r["actual"]) for r in lab]))
            promote = mae_chal < mae_champ * (1 - cfg.margin)
            rec["reason"] += (f"; gate over last {len(lab)} weeks: challenger MAE {mae_chal:.1f} vs served "
                              f"{mae_champ:.1f} ({100 * (mae_chal / max(mae_champ, 1e-6) - 1):+.0f}%, need <-{100 * cfg.margin:.0f}%)")
            n_fits += 1
        last_attempt = T
        if promote:
            champ, champ_id = fit_on(n_lab), champ_id + 1
            ref_daily = ref_window(n_lab)
            n_fits += 1
            n_prom += 1
            rec["action"] = "retrain_promote"
            rec["reason"] += " -> PROMOTED"
        else:
            n_rej += 1
            rec["action"] = "retrain_reject"
            rec["reason"] += " -> REJECTED, champion kept"
        if verbose:
            print(f"[{rec['date'].date()}] {rec['action']}: {rec['reason']}")

    log = pd.DataFrame(rows)
    err = log["pred"] - log["actual"]
    metrics = dict(mode=cfg.mode, mae=float(err.abs().mean()), rmse=float(np.sqrt((err ** 2).mean())),
                   coverage=float(((log["actual"] >= log["lo"]) & (log["actual"] <= log["hi"])).mean()),
                   model_fits=n_fits, promotions=n_prom, rejections=n_rej, weeks=len(log))
    return log, metrics, champ
