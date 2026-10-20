"""
final_eval.py
1. Picks alert thresholds on VALIDATION patients (2%, 5%, 10% false-alarm rates) and saves them.
2. Saves reference values (training medians) used to explain individual predictions.
3. Subgroup performance on TEST patients (insulin use, HbA1c band, sex, age).
4. Event-level evaluation on TEST patients with continuous 5-minute monitoring:
   of all real high-glucose episodes, how many had an alert 30-120 minutes earlier,
   and how many alerts per day does that cost?
"""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
import baselines as B
import twin_core as TC

TARGETS = {"fpr_2": 0.02, "fpr_5": 0.05, "fpr_10": 0.10}
REFRACT = 24            # after an alert, stay quiet for 2 hours (limits alert fatigue)
LEAD_MIN_BINS = 6       # an alert must come at least 30 minutes before the episode
HORIZON = 24            # ... and at most 2 hours before
CHUNK = 50
BURN_IN = 1
OUT = Path("results")


def subgroups(d, X, pt_pred, te, thr):
    y = d["y_hyper"][te]
    pid = d["pidx"][te]
    stat = d["static_X"][pid]
    sn = list(d["static_names"])
    c = lambda n: stat[:, sn.index(n)] if n in sn else None
    defs = [("All test patients", np.ones(len(y), bool))]
    if c("insulin_active") is not None:
        defs += [("On insulin", c("insulin_active") == 1), ("Not on insulin", c("insulin_active") == 0)]
    if c("hba1c") is not None:
        h = c("hba1c")
        defs += [("HbA1c < 5.5", h < 5.5), ("HbA1c 5.5-6.5", (h >= 5.5) & (h < 6.5)),
                 ("HbA1c 6.5-7.5", (h >= 6.5) & (h < 7.5)), ("HbA1c >= 7.5", h >= 7.5)]
    if c("male") is not None:
        defs += [("Men", c("male") == 1), ("Women", c("male") == 0)]
    if c("age") is not None:
        a = c("age")
        defs += [("Age < 60", a < 60), ("Age 60-70", (a >= 60) & (a < 70)), ("Age >= 70", a >= 70)]
    rows = []
    for name, m in defs:
        if m.sum() < 200 or len(np.unique(y[m])) < 2:
            continue
        alert = pt_pred[m] >= thr
        rows.append(dict(group=name, patients=int(np.unique(pid[m]).size), samples=int(m.sum()),
                         event_rate=float(y[m].mean()), auroc=roc_auc_score(y[m], pt_pred[m]),
                         auprc=average_precision_score(y[m], pt_pred[m]),
                         sensitivity=float(alert[y[m] == 1].mean()),
                         false_alarm_rate=float(alert[y[m] == 0].mean())))
    return rows


def greedy_refractory(idx, gap):
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= gap:
            out.append(i)
            last = i
    return np.array(out, dtype=int)


def event_level(d, model):
    w = np.load("data/processed/wearables.npz", allow_pickle=True)
    g = np.load("data/processed/glucose_sim.npz", allow_pickle=True)
    beh = np.load("data/processed/behavior.npz", allow_pickle=True)
    test_pat = np.nonzero(d["patient_split"] == 2)[0]
    stat_all = d["static_X"]
    T = w["cgm"].shape[1]
    acc = {k: dict(alerts=0, tp_alerts=0, episodes=0, det30=0, det_any=0, leads=[]) for k in TARGETS}

    for c0 in range(0, len(test_pat), CHUNK):
        ids = test_pat[c0:c0 + CHUNK]
        cgm = w["cgm"][ids]
        sleep = {k: beh[k][ids][:, BURN_IN:] for k in ["tst_h", "sleep_eff", "deep_frac", "rem_frac"]}
        F = TC.compute_dynamic_features(cgm, w["hr"][ids], w["hrv"][ids], w["steps"][ids].astype(np.float32),
                                        w["sleep_stage"][ids], sleep, beh["weekday0"][ids])
        t = np.arange(T)[None, :]
        valid = (t >= 288) & ~np.isnan(cgm) & (cgm >= 70) & (cgm <= 180)
        pi = np.nonzero(valid)[0]
        Xv = np.concatenate([F[valid], stat_all[ids][pi]], axis=1)
        P = np.full(cgm.shape, np.nan, np.float32)
        P[valid] = model.proba(Xv)
        del F, Xv

        G = g["glucose_true"][ids]
        hi = G > TC.L.HYPER
        quiet = TC.L.wsum(hi.astype(np.float32), 12)          # highs in the previous hour
        start_mask = hi[:, 1:] & (quiet[:, :-1] == 0)
        for i in range(len(ids)):
            starts = np.nonzero(start_mask[i])[0] + 1
            starts = starts[starts >= 288]
            for key, thr in model.thresholds.items():
                a = greedy_refractory(np.nonzero(P[i] >= thr)[0], REFRACT)
                A = acc[key]
                A["alerts"] += len(a)
                A["episodes"] += len(starts)
                if len(a) and len(starts):
                    j = np.searchsorted(starts, a, side="right")
                    nxt = starts[np.minimum(j, len(starts) - 1)]
                    A["tp_alerts"] += int(((j < len(starts)) & (nxt <= a + HORIZON)).sum())
                if len(starts):
                    lo = np.searchsorted(a, starts - HORIZON, side="left")
                    hi30 = np.searchsorted(a, starts - LEAD_MIN_BINS, side="right")
                    det = hi30 > lo
                    A["det30"] += int(det.sum())
                    A["leads"] += list((starts[det] - a[lo[det]]) * 5)
                    hi_any = np.searchsorted(a, starts - 1, side="right")
                    A["det_any"] += int((hi_any > lo).sum())
        print(f"  event-level: {min(c0 + CHUNK, len(test_pat))}/{len(test_pat)} test patients")

    days = len(test_pat) * (T - 288) / 288
    rows = []
    for key, A in acc.items():
        rows.append(dict(operating_point=key, threshold=model.thresholds[key],
                         alerts_per_patient_day=A["alerts"] / days,
                         alert_precision=A["tp_alerts"] / max(A["alerts"], 1),
                         episodes=A["episodes"], episodes_per_patient_day=A["episodes"] / days,
                         share_caught_with_30min_lead=A["det30"] / max(A["episodes"], 1),
                         share_caught_any_lead=A["det_any"] / max(A["episodes"], 1),
                         median_lead_min=float(np.median(A["leads"])) if A["leads"] else float("nan")))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    d, X, names, feat, sn = B.load()
    y, split = d["y_hyper"], d["split"]
    tr, va, te = split == 0, split == 1, split == 2
    model = TC.RiskModel()
    assert names == model.fnames, "feature order differs from the saved model"

    pv, pt = model.proba(X[va]), model.proba(X[te])
    fpr, _, thr = roc_curve(y[va], pv)
    thresholds = {k: float(thr[np.searchsorted(fpr, f)]) for k, f in TARGETS.items()}
    json.dump(thresholds, open("models/thresholds.json", "w"), indent=2)
    reference = {n: float(np.nanmedian(X[tr][:, j])) for j, n in enumerate(names)}
    json.dump(reference, open("models/reference.json", "w"), indent=2)
    model.thresholds = thresholds
    print("Thresholds (chosen on validation patients):", {k: round(v, 3) for k, v in thresholds.items()})
    print(f"Test AUROC {roc_auc_score(y[te], pt):.3f} | AUPRC {average_precision_score(y[te], pt):.3f}")

    print("\n=== Subgroups (test patients, 5% false-alarm threshold) ===")
    rows = subgroups(d, X, pt, te, thresholds["fpr_5"])
    print(f"{'group':20s} {'pts':>5s} {'event rate':>10s} {'AUROC':>7s} {'AUPRC':>7s} {'sens':>7s} {'false alarm':>12s}")
    for r in rows:
        print(f"{r['group']:20s} {r['patients']:5d} {r['event_rate']:10.1%} {r['auroc']:7.3f} {r['auprc']:7.3f} "
              f"{r['sensitivity']:7.1%} {r['false_alarm_rate']:12.1%}")
    with open(OUT / "subgroups.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n=== Event-level evaluation (continuous 5-minute monitoring, test patients) ===")
    ev = event_level(d, model)
    print(f"{'operating point':16s} {'alerts/day':>11s} {'alert precision':>16s} {'episodes/day':>13s} "
          f"{'caught >=30min early':>21s} {'caught any':>11s} {'median lead (min)':>18s}")
    for r in ev:
        print(f"{r['operating_point']:16s} {r['alerts_per_patient_day']:11.2f} {r['alert_precision']:16.1%} "
              f"{r['episodes_per_patient_day']:13.2f} {r['share_caught_with_30min_lead']:21.1%} "
              f"{r['share_caught_any_lead']:11.1%} {r['median_lead_min']:18.0f}")
    with open(OUT / "event_level.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ev[0].keys()))
        w.writeheader(); w.writerows(ev)
    print("\nSaved models/thresholds.json, models/reference.json, results/subgroups.csv, results/event_level.csv")


if __name__ == "__main__":
    main()
