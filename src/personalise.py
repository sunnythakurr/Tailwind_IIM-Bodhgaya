"""
personalise.py
Does calibrating the alert threshold to each patient's own first week help?
  - Calibration window: days 1-7 (the twin watches the patient, comparing its risk with what the CGM then showed)
  - Evaluation window : days 8-30 only, so there is no peeking
Policies compared at the 2%, 5%, 10% false-alarm operating points:
  global     : one threshold for everyone (chosen on validation patients)
  personal   : threshold set so that THIS patient's false-alarm rate in week 1 hits the target
  blended    : average of global and personal (shrinks noisy personal estimates toward the population)
Calibration labels use the OBSERVED CGM (what a real deployment would have), not the hidden true glucose.
Also saves the risk series for all test patients (results/test_risk_series.npz) for the dashboard bundle.
"""
import csv
import json
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
import twin_core as TC
from final_eval import greedy_refractory, REFRACT, LEAD_MIN_BINS, HORIZON, TARGETS

BINS = 288
CAL_START, CAL_END = BINS, 8 * BINS
EVAL_START = CAL_END
MIN_CAL = 200
OUT = Path("results")


def score(Pi, thr, starts, t0):
    idx = np.nonzero(Pi >= thr)[0]
    a = greedy_refractory(idx[idx >= t0], REFRACT)
    r = dict(alerts=len(a), tp=0, episodes=len(starts), det30=0, det_any=0, leads=[])
    if len(a) and len(starts):
        j = np.searchsorted(starts, a, side="right")
        nxt = starts[np.minimum(j, len(starts) - 1)]
        r["tp"] = int(((j < len(starts)) & (nxt <= a + HORIZON)).sum())
    if len(starts):
        lo = np.searchsorted(a, starts - HORIZON, side="left")
        h30 = np.searchsorted(a, starts - LEAD_MIN_BINS, side="right")
        det = h30 > lo
        r["det30"] = int(det.sum())
        r["leads"] = list((starts[det] - a[lo[det]]) * 5)
        r["det_any"] = int((np.searchsorted(a, starts - 1, side="right") > lo).sum())
    return r


def main():
    OUT.mkdir(exist_ok=True)
    d = np.load("data/processed/samples.npz", allow_pickle=True)
    w = np.load("data/processed/wearables.npz", allow_pickle=True)
    g = np.load("data/processed/glucose_sim.npz", allow_pickle=True)
    beh = np.load("data/processed/behavior.npz", allow_pickle=True)
    model = TC.RiskModel()
    glob = model.thresholds
    test_pat = np.nonzero(d["patient_split"] == 2)[0]
    ids = d["patient_id"][test_pat]
    sn = list(d["static_names"])
    insulin = d["static_X"][test_pat][:, sn.index("insulin_active")] == 1
    n = len(test_pat)

    print("Computing risk for every 5-minute step of the test patients ...")
    P = TC.risk_series(model, test_pat, w, beh, d["static_X"])
    T = P.shape[1]
    np.savez_compressed(OUT / "test_risk_series.npz", P=P, patient_idx=test_pat, patient_id=ids)

    cgm, G = w["cgm"][test_pat], g["glucose_true"][test_pat]
    mx = np.fmax.reduce(swv(cgm, 24, axis=1)[:, 1:], axis=-1)
    y_obs = np.full(cgm.shape, np.nan, np.float32)
    y_obs[:, :T - 24] = np.where(np.isnan(mx), np.nan, (mx > TC.L.HYPER).astype(np.float32))

    personal = {k: np.full(n, np.nan) for k in TARGETS}
    for i in range(n):
        p, yy = P[i, CAL_START:CAL_END], y_obs[i, CAL_START:CAL_END]
        m = ~np.isnan(p) & (yy == 0)
        if m.sum() >= MIN_CAL:
            for k, f in TARGETS.items():
                personal[k][i] = np.quantile(p[m], 1 - f)
    thr = {}
    for k in TARGETS:
        pe = np.where(np.isnan(personal[k]), glob[k], personal[k])
        thr[("global", k)] = np.full(n, glob[k])
        thr[("personal", k)] = pe
        thr[("blended", k)] = 0.5 * (pe + glob[k])

    hi = G > TC.L.HYPER
    quiet = TC.L.wsum(hi.astype(np.float32), 12)
    start_mask = hi[:, 1:] & (quiet[:, :-1] == 0)
    starts = []
    for i in range(n):
        s = np.nonzero(start_mask[i])[0] + 1
        starts.append(s[s >= EVAL_START + HORIZON])

    per = {key: [score(P[i], thr[key][i], starts[i], EVAL_START) for i in range(n)] for key in thr}
    days = (T - EVAL_START) / BINS
    groups = {"All patients": np.ones(n, bool), "On insulin": insulin, "Not on insulin": ~insulin}
    rows = []
    for (policy, k), res in per.items():
        for gname, gm in groups.items():
            R = [r for r, m in zip(res, gm) if m]
            al, tp = sum(r["alerts"] for r in R), sum(r["tp"] for r in R)
            ep, d30 = sum(r["episodes"] for r in R), sum(r["det30"] for r in R)
            dany = sum(r["det_any"] for r in R)
            leads = [x for r in R for x in r["leads"]]
            rows.append(dict(group=gname, operating_point=k, policy=policy, patients=int(gm.sum()),
                             alerts_per_patient_day=al / (gm.sum() * days), alert_precision=tp / max(al, 1),
                             episodes=ep, share_caught_with_30min_lead=d30 / max(ep, 1),
                             share_caught_any_lead=dany / max(ep, 1),
                             median_lead_min=float(np.median(leads)) if leads else float("nan")))
    with open(OUT / "personalised_event_level.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    hdr = f"{'group':15s} {'op':7s} {'policy':9s} {'alerts/day':>10s} {'precision':>9s} {'>=30min':>8s} {'any':>6s} {'lead':>5s}"
    print(f"\nEvaluation on days 8-30 only (thresholds set from days 1-7). Episodes per group in CSV.\n{hdr}")
    for gname in ("All patients", "On insulin"):
        for op in ("fpr_2", "fpr_5", "fpr_10"):
            if gname == "On insulin" and op == "fpr_2":
                continue
            for pol in ("global", "personal", "blended"):
                r = next(x for x in rows if x["group"] == gname and x["operating_point"] == op and x["policy"] == pol)
                print(f"{r['group']:15s} {op:7s} {pol:9s} {r['alerts_per_patient_day']:10.2f} "
                      f"{r['alert_precision']:9.1%} {r['share_caught_with_30min_lead']:8.1%} "
                      f"{r['share_caught_any_lead']:6.1%} {r['median_lead_min']:5.0f}")
        print()

    out = {}
    for i, pid in enumerate(ids):
        out[str(pid)] = {f"{pol}_{k}": float(thr[(pol, k)][i]) for pol in ("personal", "blended") for k in TARGETS}
    json.dump(out, open("models/personal_thresholds.json", "w"), indent=1)
    print("\nSaved results/personalised_event_level.csv, results/test_risk_series.npz, models/personal_thresholds.json")


if __name__ == "__main__":
    main()
