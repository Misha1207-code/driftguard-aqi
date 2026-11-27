"""Figures for the slides."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_plots(logs, metrics, out):
    g = logs["driftguard_full"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(g["date"], g["actual"], "k", lw=1.2, label="actual weekly AQI")
    ax[0].plot(g["date"], g["pred"], color="tab:blue", lw=1.2, label="DriftGuard forecast")
    ax[0].fill_between(g["date"], g["lo"], g["hi"], color="tab:blue", alpha=0.15, label="90% interval")
    for _, r in g[g.action == "retrain_promote"].iterrows():
        ax[0].axvline(r["date"], color="green", ls="--", lw=1)
    for _, r in g[g.action == "retrain_reject"].iterrows():
        ax[0].axvline(r["date"], color="red", ls=":", lw=1)
    ax[0].set_ylabel("AQI"); ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("Test stream: green dashed = promoted retrain, red dotted = rejected by gate")
    ax[1].plot(g["date"], g["drift"], label="data drift"); ax[1].plot(g["date"], g["season"], label="seasonal match")
    ax[1].plot(g["date"], g["interval"], label="interval warning"); ax[1].legend(fontsize=8, ncol=3); ax[1].set_ylabel("0-1 terms")
    ax[2].plot(g["date"], g["score"], color="purple"); ax[2].set_ylabel("retrain score")
    plt.tight_layout(); plt.savefig(f"{out}/timeline.png", dpi=160); plt.close()

    m = metrics.sort_values("mae")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].barh(m.index, m["mae"], color="tab:blue"); ax[0].set_xlabel("MAE (AQI points, lower is better)")
    ax[1].barh(m.index, m["model_fits"], color="tab:orange"); ax[1].set_xlabel("model fits (retraining effort)")
    ax[1].set_yticklabels([])
    plt.tight_layout(); plt.savefig(f"{out}/comparison.png", dpi=160); plt.close()
