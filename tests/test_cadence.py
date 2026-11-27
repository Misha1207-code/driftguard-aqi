import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import driftguard as dg


def test_psi_bias_corrected_no_false_drift():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=500)
    # short window from the SAME distribution must not look drifted
    assert np.mean([dg.psi(ref, rng.normal(size=56)) for _ in range(50)]) < 0.1


def test_psi_detects_real_shift():
    rng = np.random.default_rng(1)
    assert dg.psi(rng.normal(size=500), rng.normal(2, 1, size=56)) > 0.5


def test_biweekly_baseline_fires_every_second_week():
    c = dg.Cfg()
    fires = [dg.decide("Biweekly", {}, i, -99, c)[0] for i in range(6)]
    assert fires == [False, True, False, True, False, True]


def test_adaptive_policy_waits_for_review():
    c = dg.Cfg(review=4, cooldown=4)
    sg = dict(D=1, E=1, U=1, S=0, score=2.0)
    assert dg.decide("DriftGuard", sg, 1, -99, c)[0] is False   # not a review week
    assert dg.decide("DriftGuard", sg, 3, -99, c)[0] is True    # review week
