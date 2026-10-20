"""
scenarios.py
Where does the EHR help? Each scenario trains the same model WITHOUT and WITH the static EHR features
and reports the paired test-AUROC difference (patient-level bootstrap).
Scenarios (chosen after seeing that EHR adds ~nothing when 24h of CGM is available):
  - no clock         : remove time-of-day features (guards against the "habitual meal time" artefact)
  - no CGM           : low-cost wearable only (HR, HRV, steps, sleep)
  - cold start       : CGM history limited to the last ~2 hours (new sensor / new patient)
"""
import csv
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import baselines as B

N_BOOT = 100
SEED = 21


def main():
    d, X, names, feat, sn = B.load()
    y, split, pidx = d["y_hyper"], d["split"], d["pidx"]
    tr, va, te = split == 0, split == 1, split == 2

    cgm = [n for n in feat if n.startswith("cgm_")]
    time_ = ["hour_sin", "hour_cos", "weekend"]
    other = [n for n in feat if n not in cgm and n not in time_]
    short = ["cgm_now", "cgm_slope_15m", "cgm_slope_30m", "cgm_slope_60m", "cgm_mean_1h",
             "cgm_max_2h", "cgm_min_2h"]
    scen = [
        ("No clock | CGM only",                     cgm),
        ("No clock | CGM + wearables",              cgm + other),
        ("No CGM | wearables + clock",              other + time_),
        ("No CGM, no clock | wearables only",       other),
        ("Cold start | <=2h CGM + clock",           short + time_),
        ("Cold start | <=2h CGM + wearables + clock", short + other + time_),
        ("Cold start, no clock | <=2h CGM + wearables", short + other),
    ]

    yt, pt = y[te], pidx[te]
    pats = np.unique(pt)
    by_pat = {p: np.nonzero(pt == p)[0] for p in pats}
    rng = np.random.default_rng(SEED)
    boots = [np.concatenate([by_pat[p] for p in rng.choice(pats, len(pats), replace=True)]) for _ in range(N_BOOT)]

    rows = []
    print(f"\n{'scenario':48s} {'without EHR':>12s} {'with EHR':>10s} {'AUROC gain [95% CI]':>26s} {'AUPRC gain':>11s}")
    for name, fn in scen:
        preds = []
        for use_ehr in (False, True):
            cols = B.pick(fn + (sn if use_ehr else []), names)
            clf, k, _ = B.fit_gbt(X[tr][:, cols], y[tr], X[va][:, cols], y[va])
            preds.append(B.predict_at(clf, X[te][:, cols], k))
        a0, a1 = roc_auc_score(yt, preds[0]), roc_auc_score(yt, preds[1])
        p0, p1 = average_precision_score(yt, preds[0]), average_precision_score(yt, preds[1])
        dauc = np.array([roc_auc_score(yt[i], preds[1][i]) - roc_auc_score(yt[i], preds[0][i]) for i in boots])
        lo, hi = np.percentile(dauc, [2.5, 97.5])
        rows.append(dict(scenario=name, auroc_without_ehr=a0, auroc_with_ehr=a1, auroc_gain=a1 - a0,
                         ci_lo=lo, ci_hi=hi, auprc_without=p0, auprc_with=p1, auprc_gain=p1 - p0))
        print(f"{name:48s} {a0:12.3f} {a1:10.3f} {a1 - a0:+10.3f} [{lo:+.3f},{hi:+.3f}] {p1 - p0:+11.3f}")

    with open(B.OUT / "scenario_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    yy = np.arange(len(rows))
    ax.barh(yy - 0.2, [r["auroc_without_ehr"] for r in rows], 0.4, label="without EHR")
    ax.barh(yy + 0.2, [r["auroc_with_ehr"] for r in rows], 0.4, label="with EHR")
    ax.set_yticks(yy); ax.set_yticklabels([r["scenario"] for r in rows], fontsize=8)
    ax.set_xlim(0.5, 1.0); ax.invert_yaxis(); ax.set_xlabel("Test AUROC"); ax.legend()
    ax.set_title("Does the EHR help? By data-availability scenario")
    plt.tight_layout(); plt.savefig(B.OUT / "scenarios.png", dpi=130)
    print("\nSaved results/scenario_results.csv and results/scenarios.png")


if __name__ == "__main__":
    main()
