"""Forecaster = point model + label-free uncertainty (RF tree spread) + normalised split conformal."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:   # minimal installs
    HAS_XGB = False

EPS = 1.0


def make_point_model(seed=0):
    if HAS_XGB:
        return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, random_state=seed, verbosity=0)
    return GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=seed)


def _rf(seed):
    return RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=seed, n_jobs=1)


def _spread(rf: RandomForestRegressor, X) -> np.ndarray:
    return np.std(np.stack([t.predict(X) for t in rf.estimators_]), axis=0)


class Forecaster:
    """Point forecast + 90% interval whose width grows when inputs look unfamiliar."""

    def __init__(self, cal_weeks: int = 16, alpha: float = 0.1, seed: int = 0):
        self.cal_weeks, self.alpha, self.seed = cal_weeks, alpha, seed

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        k = min(self.cal_weeks, max(8, len(X) // 5))
        Xtr, ytr, Xc, yc = X[:-k], y[:-k], X[-k:], y[-k:]
        pm = make_point_model(self.seed).fit(Xtr, ytr)
        rf = _rf(self.seed).fit(Xtr, ytr)
        resid = np.abs(yc - pm.predict(Xc))
        sig = _spread(rf, Xc)
        self.q_ = float(np.quantile(resid / (sig + EPS), 1 - self.alpha))
        self.ref_mae_ = float(resid.mean())
        self.ref_width_ = float(np.mean(2 * self.q_ * (sig + EPS)))
        # refit on all data after calibration (documented simplification)
        self.pm_ = make_point_model(self.seed).fit(X, y)
        self.rf_ = _rf(self.seed).fit(X, y)
        self.n_train_ = len(X)
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        yhat = self.pm_.predict(X)
        half = self.q_ * (_spread(self.rf_, X) + EPS)
        return yhat, yhat - half, yhat + half, 2 * half
