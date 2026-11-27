import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from driftguard_aqi.drift import psi
from driftguard_aqi.policy import Config, run_stream
from driftguard_aqi.data import load_city_day
from driftguard_aqi.experiment import prepare


def test_psi_small_when_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=500), rng.normal(size=28)) < 0.25


def test_psi_large_when_shifted():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=500), rng.normal(3, 1, size=60)) > 0.5


def test_stream_logs_every_week_and_promotions_passed_gate():
    import make_synthetic
    daily = load_city_day(make_synthetic.make("/tmp/syn.csv"))
    X, y, s_val, s_test, n = prepare(daily)
    log, m, _ = run_stream(X, y, daily, s_test, n, Config())
    assert len(log) == n - s_test
    assert (log["actual"].values == y.values[s_test:n]).all()
    assert all("PROMOTED" in r for r in log.loc[log.action == "retrain_promote", "reason"])
