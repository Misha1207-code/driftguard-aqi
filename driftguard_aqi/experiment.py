"""Chronological split, weight tuning on the validation stream, final comparison on the test stream."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from .features import build_features
from .data import to_weekly
from .policy import Config, run_stream


def prepare(daily: pd.DataFrame, train_frac=0.35, val_frac=0.30):
    weekly = to_weekly(daily)
    X, y = build_features(weekly)
    X, y = X.iloc[:-1], y.iloc[:-1]          # drop the final row (its target is the future)
    n = len(X)
    s_val = int(n * train_frac)
    s_test = int(n * (train_frac + val_frac))
    return X, y, s_val, s_test, n


def tune(X, y, daily, s_val, s_test, base: Config, retrain_cost=0.3, verbose=True):
    """Grid search on the validation stream: minimise MAE + retrain_cost * (#model fits)."""
    grid = itertools.product([0.5, 0.75, 1.0], [0.0, 0.05], [0.5, 1.0, 1.5], [0.5, 1.0])  # thr, margin, w4, w1
    best, table = None, []
    for thr, mar, w4, w1 in grid:
        cfg = Config(**{**base.to_dict(), "threshold": thr, "margin": mar,
                        "w": (w1, base.w[1], base.w[2], w4)})
        _, m, _ = run_stream(X, y, daily, s_val, s_test, cfg)
        cost = m["mae"] + retrain_cost * m["model_fits"]
        table.append(dict(threshold=thr, margin=mar, w1=w1, w4=w4, mae=m["mae"], fits=m["model_fits"], cost=cost))
        if best is None or cost < best[0]:
            best = (cost, cfg)
        if verbose:
            print(f"  thr={thr} margin={mar} w1={w1} w4={w4}: MAE {m['mae']:.2f}, fits {m['model_fits']}, cost {cost:.2f}")
    return best[1], pd.DataFrame(table).sort_values("cost")


def compare(X, y, daily, s_test, n, tuned: Config):
    """All policies on the untouched test stream."""
    variants = {
        "never_retrain": Config(mode="never"),
        "fixed_biweekly(2w)": Config(mode="fixed", fixed_k=2),
        "fixed_monthly(4w)": Config(mode="fixed", fixed_k=4),
        "fixed_quarterly(13w)": Config(mode="fixed", fixed_k=13),
        "error_threshold_only": Config(mode="error_only", review_every=tuned.review_every),
        "driftguard_no_season": Config(**{**tuned.to_dict(), "use_season": False}),
        "driftguard_no_gate": Config(**{**tuned.to_dict(), "use_gate": False}),
        "driftguard_full": tuned,
    }
    logs, metrics, champ = {}, [], None
    for name, cfg in variants.items():
        log, m, c = run_stream(X, y, daily, s_test, n, cfg)
        m["policy"] = name
        logs[name], metrics = log, metrics + [m]
        if name == "driftguard_full":
            champ = c
    return logs, pd.DataFrame(metrics).set_index("policy"), champ
