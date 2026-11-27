"""Real-data experiment: tune on a validation stream, then compare every policy on the untouched test stream.

  python run_real.py --csv data/real_delhi.csv --review 2 --out results_real_biweekly
  python run_real.py --csv data/real_delhi.csv --review 4 --out results_real_monthly

review = how often DriftGuard looks at the system (weeks). 2 = biweekly, 4 = monthly.
The fixed-schedule baselines (Biweekly / Monthly) retrain at those cadences no matter what;
DriftGuard may only act at the same review points and only if the score says so and the gate agrees.
"""
import argparse, itertools, json, os
from dataclasses import asdict, replace
import pandas as pd
import driftguard as dg

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/real_delhi.csv")
ap.add_argument("--review", type=int, default=2)
ap.add_argument("--val-frac", type=float, default=0.4, help="share of the replay used for tuning")
ap.add_argument("--retrain-cost", type=float, default=0.2, help="RMSE points charged per retrain attempt when tuning")
ap.add_argument("--no-tune", action="store_true")
ap.add_argument("--out", default="results_real")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

cols = [c for c in pd.read_csv(a.csv, nrows=1).columns if c not in ("date", "aqi")]
df = dg.load_csv(a.csv, "date", "aqi", cols)
F, feats, raw, cfg0 = dg.prepare(df, dict(review=a.review, cooldown=a.review))
n, start0 = len(F), cfg0.start
val_end = start0 + int(a.val_frac * (n - start0))
print(f"{len(df)} usable days {F.date.iloc[0].date()} -> {F.date.iloc[-1].date()} | features: {raw}")
print(f"initial training window ends {F.date.iloc[start0].date()} | validation stream to {F.date.iloc[val_end].date()} "
      f"| test stream {F.date.iloc[val_end].date()} -> {F.date.iloc[-1].date()} ({(n - val_end) // 7} weeks)")

best = replace(cfg0, start=start0)
if not a.no_tune:
    Fv, rows, best_cost = F.iloc[:val_end], [], None
    weight_sets = [(0.35, 0.35, 0.20, 0.30), (0.50, 0.30, 0.20, 0.30), (0.30, 0.50, 0.20, 0.20), (0.25, 0.60, 0.15, 0.30)]
    for tau, margin, w in itertools.product([0.15, 0.25, 0.35, 0.50], [0.0, 0.02, 0.05], weight_sets):
        c = replace(cfg0, tau=tau, margin=margin, w=w)
        s, _, _ = dg.run_policy(Fv, "DriftGuard", c, feats, raw)
        cost = s["RMSE"] + a.retrain_cost * s["Retrains"]
        rows.append(dict(tau=tau, margin=margin, w=w, retrains=s["Retrains"], promoted=s["Promoted"], rmse=s["RMSE"], cost=cost))
        print(f"  tau={tau} margin={margin} w={w}: RMSE {s['RMSE']:.2f}, retrains {s['Retrains']}, cost {cost:.2f}")
        if best_cost is None or cost < best_cost:
            best, best_cost = c, cost
    pd.DataFrame(rows).sort_values("cost").to_csv(f"{a.out}/tuning_grid.csv", index=False)

test_cfg = replace(best, start=val_end)
print("\nChosen:", {k: v for k, v in asdict(test_cfg).items() if k in ("tau", "margin", "w", "review", "cooldown", "start")})
summ, weekly, log = dg.run_all(F, test_cfg, feats, raw, progress=lambda f, p: print(f"  done: {p}"))
summ.to_csv(f"{a.out}/summary.csv", index=False)
log.to_csv(f"{a.out}/decision_log.csv", index=False)
cfg_json = asdict(test_cfg); cfg_json["w"] = list(cfg_json["w"])
json.dump(cfg_json, open(f"{a.out}/chosen_config.json", "w"), indent=2)

events = {}
lock = pd.Timestamp("2020-03-25")
if F.date.iloc[val_end] < lock < F.date.iloc[-1]:
    events["COVID-19 lockdown"] = lock
dg.plot(weekly, log, events, f"{a.out}/rmse_timeline.png")

print("\n" + summ.to_string(index=False))
s = summ.set_index("Policy")
fixed = "Biweekly" if a.review == 2 else "Monthly"
print(f"\nHEADLINE (review every {a.review} weeks): DriftGuard retrained {s.loc['DriftGuard','Retrains']} times "
      f"({s.loc['DriftGuard','Promoted']} promoted) vs {s.loc[fixed,'Retrains']} for the fixed {fixed.lower()} schedule; "
      f"RMSE {s.loc['DriftGuard','RMSE']} vs {s.loc[fixed,'RMSE']} (never retrain: {s.loc['Never','RMSE']}).")
print(f"Saved to {a.out}/")
