"""
baselines.py
Ablation study with gradient-boosted trees on the summary features:
  EHR only | CGM only | CGM rule | HR/HRV/steps/sleep only | CGM + wearables | CGM + EHR | Fused (all)
Everything is evaluated on TEST patients that the models never saw.
Reports AUROC, AUPRC, calibration, alert rate and lead time, with patient-level bootstrap CIs.
"""
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             brier_score_loss, log_loss)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 11
FPR_TARGET = 0.10          # operating point: 10% false alarms among normal moments
N_BOOT = 100
MAX_ITER = 300
OUT = Path("results")
MODELS = Path("models")
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


def load():
    d = np.load("data/processed/samples.npz", allow_pickle=True)
    feat = list(d["feature_names"])
    sn = list(d["static_names"])
    X_dyn = d["X_tab"]
    X_sta = d["static_X"][d["pidx"]]
    allnames = feat + sn
    X = np.concatenate([X_dyn, X_sta], axis=1)
    return d, X, allnames, feat, sn


def pick(names, all_names):
    return [all_names.index(n) for n in names]


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return tot


def fit_gbt(Xtr, ytr, Xva, yva):
    clf = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=MAX_ITER, max_depth=6, max_leaf_nodes=31,
        min_samples_leaf=1500, l2_regularization=5.0, early_stopping=False, random_state=SEED)
    clf.fit(Xtr, ytr)
    best, best_k, best_p = 1e9, 1, None
    for k, p in enumerate(clf.staged_predict_proba(Xva), 1):   # early stopping on VALIDATION patients
        ll = log_loss(yva, p[:, 1], labels=[0, 1])
        if ll < best:
            best, best_k, best_p = ll, k, p[:, 1].copy()
    return clf, best_k, best_p


def predict_at(clf, X, k):
    for i, p in enumerate(clf.staged_predict_proba(X), 1):
        if i == k:
            return p[:, 1]


def main():
    d, X, names, feat, sn = load()
    y, split, tte, pidx = d["y_hyper"], d["split"], d["tte_min"], d["pidx"]
    tr, va, te = split == 0, split == 1, split == 2
    print(f"samples: train {tr.sum():,} | val {va.sum():,} | test {te.sum():,}")

    cgm = [n for n in feat if n.startswith("cgm_")]
    time_ = ["hour_sin", "hour_cos", "weekend"]
    other = [n for n in feat if n not in cgm and n not in time_]
    specs = {
        "EHR only":                    sn,
        "CGM only":                    cgm + time_,
        "HR/HRV/steps/sleep only":     other + time_,
        "CGM + wearables":             cgm + other + time_,
        "CGM + EHR":                   cgm + time_ + sn,
        "Fused (CGM + wearables + EHR)": feat + sn,
    }

    pred_val, pred_te, info, fused_model = {}, {}, {}, None
    # --- simple rule baseline: logistic regression on current glucose and its trend ---
    cols = pick(["cgm_now", "cgm_slope_30m"], names)
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500))
    lr.fit(X[tr][:, cols], y[tr])
    pred_val["CGM rule (now + trend)"] = lr.predict_proba(X[va][:, cols])[:, 1]
    pred_te["CGM rule (now + trend)"] = lr.predict_proba(X[te][:, cols])[:, 1]

    for name, fnames in specs.items():
        c = pick(fnames, names)
        clf, k, pv = fit_gbt(X[tr][:, c], y[tr], X[va][:, c], y[va])
        pred_val[name] = pv
        pred_te[name] = predict_at(clf, X[te][:, c], k)
        info[name] = dict(n_features=len(c), best_iter=int(k))
        print(f"  {name:32s} features {len(c):2d}  best iteration {k:3d}  val AUROC {roc_auc_score(y[va], pv):.3f}")
        joblib.dump((clf, k, fnames), MODELS / (name.split(" (")[0].replace(" ", "_").replace("/", "-") + ".joblib"))
        if name.startswith("Fused"):
            fused_model, fused_cols, fused_k = clf, c, k

    # isotonic recalibration of the fused model (fit on validation patients only)
    fname = "Fused (CGM + wearables + EHR)"
    iso = IsotonicRegression(out_of_bounds="clip").fit(pred_val[fname], y[va])
    pred_te["Fused, calibrated"] = iso.predict(pred_te[fname])
    pred_val["Fused, calibrated"] = iso.predict(pred_val[fname])
    joblib.dump(iso, MODELS / "fused_isotonic.joblib")

    # --- evaluation on TEST patients ---
    yt, tt, pt = y[te], tte[te], pidx[te]
    pats = np.unique(pt)
    by_pat = {p: np.nonzero(pt == p)[0] for p in pats}
    n_test_patient_days = len(pats) * 30
    rng = np.random.default_rng(SEED)
    order = list(pred_te.keys())
    boot_auc = np.zeros((N_BOOT, len(order)))
    boot_ap = np.zeros((N_BOOT, len(order)))
    for b in range(N_BOOT):
        idx = np.concatenate([by_pat[p] for p in rng.choice(pats, len(pats), replace=True)])
        for j, m in enumerate(order):
            boot_auc[b, j] = roc_auc_score(yt[idx], pred_te[m][idx])
            boot_ap[b, j] = average_precision_score(yt[idx], pred_te[m][idx])

    rows = []
    print(f"\n{'model':32s} {'AUROC':>18s} {'AUPRC':>18s} {'sens@10%FPR':>12s} {'alerts/day':>11s} {'med lead':>9s} {'Brier':>7s} {'ECE':>6s}")
    for j, m in enumerate(order):
        p = pred_te[m]
        fpr, tpr, thr = roc_curve(y[va], pred_val[m])
        thresh = thr[np.searchsorted(fpr, FPR_TARGET)]          # threshold chosen on VALIDATION
        alert = p >= thresh
        sens = alert[yt == 1].mean()
        fa_per_day = (alert & (yt == 0)).sum() / n_test_patient_days
        caught = alert & (yt == 1)
        lead = np.median(tt[caught]) if caught.any() else np.nan
        ge30 = (tt[caught] >= 30).mean() if caught.any() else np.nan
        auc, ap = roc_auc_score(yt, p), average_precision_score(yt, p)
        alo, ahi = np.percentile(boot_auc[:, j], [2.5, 97.5])
        plo, phi = np.percentile(boot_ap[:, j], [2.5, 97.5])
        rows.append(dict(model=m, auroc=auc, auroc_lo=alo, auroc_hi=ahi, auprc=ap, auprc_lo=plo, auprc_hi=phi,
                         sensitivity=sens, false_alerts_per_patient_day=fa_per_day, median_lead_min=lead,
                         share_caught_with_30min_lead=ge30, brier=brier_score_loss(yt, p), ece=ece(yt, p),
                         base_rate=yt.mean()))
        print(f"{m:32s} {auc:.3f} [{alo:.3f},{ahi:.3f}] {ap:.3f} [{plo:.3f},{phi:.3f}] {sens:11.1%} "
              f"{fa_per_day:11.2f} {lead:9.0f} {brier_score_loss(yt, p):7.3f} {ece(yt, p):6.3f}")

    # paired bootstrap differences (same resampled patients for both models)
    def diff(a, b, arr):
        v = arr[:, order.index(a)] - arr[:, order.index(b)]
        return v.mean(), *np.percentile(v, [2.5, 97.5])
    print("\nPaired AUROC differences (mean, 95% CI) -- does fusion help?")
    for a, b in [(fname, "CGM only"), (fname, "EHR only"), (fname, "CGM + wearables"),
                 ("CGM + EHR", "CGM only"), ("CGM + wearables", "CGM only")]:
        m_, lo, hi = diff(a, b, boot_auc)
        print(f"  {a:32s} minus {b:18s}: {m_:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    import csv
    with open(OUT / "baseline_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    np.savez_compressed(OUT / "baseline_predictions.npz", y_test=yt, tte_test=tt, pidx_test=pt,
                        sample_index_test=np.nonzero(te)[0], **{"pred_" + k: v for k, v in pred_te.items()})
    json.dump(info, open(OUT / "baseline_info.json", "w"), indent=2)

    # --- permutation importance for the fused model (validation subsample) ---
    refit = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=fused_k, max_depth=6, max_leaf_nodes=31,
        min_samples_leaf=1500, l2_regularization=5.0, early_stopping=False, random_state=SEED)
    refit.fit(X[tr][:, fused_cols], y[tr])
    sub = np.random.default_rng(SEED).choice(np.nonzero(va)[0], min(15000, va.sum()), replace=False)
    Xs, ys = X[sub][:, fused_cols], y[sub]
    base_auc = roc_auc_score(ys, refit.predict_proba(Xs)[:, 1])
    prng = np.random.default_rng(SEED)
    fn = feat + sn
    imp = []
    for j, nm in enumerate(fn):
        drops = []
        for _ in range(2):
            Z = Xs.copy()
            Z[:, j] = prng.permutation(Z[:, j])
            drops.append(base_auc - roc_auc_score(ys, refit.predict_proba(Z)[:, 1]))
        imp.append((nm, float(np.mean(drops))))
    imp.sort(key=lambda z: -z[1])
    print("\nTop features by permutation importance (drop in validation AUROC when shuffled):")
    for n_, v in imp[:12]:
        print(f"  {n_:20s} {v:.4f}")
    json.dump(imp, open(OUT / "feature_importance.json", "w"), indent=2)

    # --- plots ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["EHR only", "CGM rule (now + trend)", "CGM only", "CGM + EHR", "CGM + wearables", fname]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for m in show:
        f_, t_, _ = roc_curve(yt, pred_te[m]); ax[0].plot(f_, t_, label=f"{m} ({roc_auc_score(yt, pred_te[m]):.3f})")
        pr, rc, _ = precision_recall_curve(yt, pred_te[m]); ax[1].plot(rc, pr, label=m)
    ax[0].plot([0, 1], [0, 1], "k:", lw=.7); ax[0].set_xlabel("False alarm rate"); ax[0].set_ylabel("Sensitivity")
    ax[0].set_title("ROC (test patients)"); ax[0].legend(fontsize=7)
    ax[1].axhline(yt.mean(), c="k", ls=":", lw=.7); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-recall (test patients)")
    for m, c in [(fname, "C0"), ("Fused, calibrated", "C1")]:
        p = pred_te[m]; q = np.quantile(p, np.linspace(0, 1, 11))
        mids, obs = [], []
        for lo, hi in zip(q[:-1], q[1:]):
            s = (p >= lo) & (p <= hi)
            if s.any(): mids.append(p[s].mean()); obs.append(yt[s].mean())
        ax[2].plot(mids, obs, "o-", c=c, label=m)
    ax[2].plot([0, 1], [0, 1], "k:", lw=.7); ax[2].set_xlabel("Predicted probability"); ax[2].set_ylabel("Observed frequency")
    ax[2].set_title("Calibration"); ax[2].legend()
    plt.tight_layout(); plt.savefig(OUT / "baseline_curves.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ms = [r["model"] for r in rows if r["model"] != "Fused, calibrated"]
    v = [r["auroc"] for r in rows if r["model"] != "Fused, calibrated"]
    e = [[r["auroc"] - r["auroc_lo"] for r in rows if r["model"] != "Fused, calibrated"],
         [r["auroc_hi"] - r["auroc"] for r in rows if r["model"] != "Fused, calibrated"]]
    ax.barh(ms, v, xerr=e, color="C0"); ax.set_xlim(0.5, 1.0); ax.invert_yaxis()
    ax.set_xlabel("Test AUROC (95% patient-level bootstrap CI)"); ax.set_title("Ablation: what each data stream adds")
    plt.tight_layout(); plt.savefig(OUT / "ablation.png", dpi=130); plt.close()
    print("\nSaved: results/baseline_results.csv, baseline_curves.png, ablation.png, models/*.joblib")


if __name__ == "__main__":
    main()
