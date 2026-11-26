"""Live, step-by-step version of the DriftGuard replay (one call = one week)."""
import numpy as np
import driftguard as dg


class Stream:
    def __init__(self, df=None, cfg_over=None):
        if df is None:
            df, _ = dg.make_synthetic()
        self.F, self.feats, self.raw, self.cfg = dg.prepare(df, cfg_over)
        self.n, self.y = len(self.F), self.F["y"].values
        self.reset()

    def _bundle(self, hi):
        c = self.cfg
        return dg.Bundle(self.F, hi - c.win, hi, c, self.feats, self.raw, True)

    def reset(self):
        c = self.cfg
        self.champ = self._bundle(c.start - c.delay)
        self.pred = np.full(self.n, np.nan)
        self.width = np.full(self.n, np.nan)
        self.d, self.i, self.since = c.start, 0, c.start
        self.last, self.version, self.log = -10 ** 6, 1, []

    @property
    def done(self):
        return self.d >= self.n

    def step(self):
        c, d = self.cfg, self.d
        e = min(d + c.step, self.n)
        mid, lo, hi = self.champ.predict(self.F.iloc[d:e])
        self.pred[d:e], self.width[d:e] = mid, hi - lo
        rmse = float(np.sqrt(np.mean((mid - self.y[d:e]) ** 2)))
        known = e - c.delay                      # labels arrive with a delay
        sg = dg.signals(self.F, self.champ, self.pred, self.width,
                        self.since, known, e, c, self.raw)
        trig, reason = dg.decide("DriftGuard", sg, self.i, self.last, c)
        rec = dict(week=self.i, date=str(self.F["date"].iloc[e - 1].date()), **sg,
                   weekly_rmse=rmse, triggered=bool(trig and e < self.n),
                   promoted=False, champ_mae=None, chal_mae=None, reason=reason)
        if rec["triggered"]:
            self.last = self.i
            chal = self._bundle(known)           # challenger
            ho = chal.ho
            cm = float(np.mean(np.abs(self.champ.predict(ho)[0] - ho["y"].values)))
            ok = bool(chal.base_mae < cm * (1 - c.margin))   # the gate
            rec.update(promoted=ok, champ_mae=cm, chal_mae=float(chal.base_mae))
            rec["reason"] += (" -> PROMOTED" if ok else
                              f" -> REJECTED (challenger MAE {chal.base_mae:.1f} "
                              f"not better than champion {cm:.1f})")
            if ok:
                self.champ, self.since, self.version = chal, e, self.version + 1
        rec["version"] = self.version
        self.log.append(rec)
        self.i, self.d = self.i + 1, e
        return rec

    def sample_row(self):
        row = self.F.iloc[min(self.d, self.n - 1)]
        return {k: float(row[k]) for k in self.feats}

    def predict(self, features):
        import pandas as pd
        X = pd.DataFrame([{k: features[k] for k in self.feats}])
        mid, lo, hi = self.champ.predict(X)
        return float(mid[0]), float(lo[0]), float(hi[0])
