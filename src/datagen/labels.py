"""
labels.py
Builds the model dataset from the simulated sensors:
  * label  : glucose > 180 mg/dL at any point in the next 2 hours (read off the simulated trace)
  * samples: only time points where glucose is CURRENTLY in range (70-180), so the task is not trivial
  * split  : by PATIENT (70/15/15, stratified by HbA1c band), never by time window
  * features: summary features from the sensor history (for XGBoost / baselines) and
              an index table (patient, time) so the neural net can slice raw windows on the fly
Meals are NOT features: a real wearable does not know what the patient ate.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

BINS = 288
BURN_IN = 1
H = 24                  # 2 hours of 5-min bins
HYPER, HYPO = 180.0, 70.0
STRIDE = 12             # one candidate sample per hour (random offset per patient)
MIN_T = 288             # need a full day of history
SEED = 7
CHUNK = 100

STATIC_CANDIDATES = ["age", "bmi", "hba1c", "glucose", "sbp", "dbp", "chol_total", "triglycerides",
                     "hdl", "ldl", "years_since_dx", "hypertension", "ckd", "retinopathy",
                     "neuropathy", "obesity", "cvd", "metformin_active", "insulin_active", "prs"]


def wsum(x, w):
    """Trailing window sum (window of w bins ending at t, inclusive), shape preserved."""
    cs = np.cumsum(x, axis=1, dtype=np.float64)
    cs = np.pad(cs, ((0, 0), (w, 0)))
    return cs[:, w:w + x.shape[1]] - cs[:, :x.shape[1]]


def wmean_nan(x, w):
    ok = ~np.isnan(x)
    s = wsum(np.where(ok, x, 0.0), w)
    n = wsum(ok.astype(np.float64), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= max(2, w // 4), s / n, np.nan)


def wsd_nan(x, w):
    ok = ~np.isnan(x)
    n = wsum(ok.astype(np.float64), w)
    s1 = wsum(np.where(ok, x, 0.0), w)
    s2 = wsum(np.where(ok, x * x, 0.0), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = s2 / n - (s1 / n) ** 2
        return np.where(n >= max(2, w // 4), np.sqrt(np.maximum(var, 0)), np.nan)


def trailing_extreme(x, w, fn):
    xp = np.concatenate([np.full((x.shape[0], w - 1), np.nan, x.dtype), x], axis=1)
    return fn.reduce(swv(xp, w, axis=1), axis=-1)


def ffill(x, max_gap=12):
    n, T = x.shape
    idx = np.where(~np.isnan(x), np.arange(T)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.take_along_axis(x, idx, axis=1)
    out[(np.arange(T)[None, :] - idx) > max_gap] = np.nan
    return out


def lag(x, k):
    out = np.full_like(x, np.nan)
    out[:, k:] = x[:, :-k]
    return out


def patient_split(hba1c, rng):
    n = len(hba1c)
    split = np.zeros(n, np.int8)
    order = np.argsort(hba1c)
    for block in np.array_split(order, 5):                 # stratify by HbA1c quintile
        block = rng.permutation(block)
        a, b = int(round(0.70 * len(block))), int(round(0.85 * len(block)))
        split[block[a:b]] = 1                              # validation
        split[block[b:]] = 2                               # test
    return split


def run_all():
    rng = np.random.default_rng(SEED)
    w = np.load("data/processed/wearables.npz", allow_pickle=True)
    g = np.load("data/processed/glucose_sim.npz", allow_pickle=True)
    beh = np.load("data/processed/behavior.npz", allow_pickle=True)
    ids = w["patient_id"]
    assert (g["patient_id"] == ids).all() and (beh["patient_id"] == ids).all(), "patient order mismatch"
    cohort = pd.read_csv("data/processed/t2d_cohort_with_prs.csv").set_index("patient_id").loc[ids]

    cgm_all, hr_all, hrv_all = w["cgm"], w["hr"], w["hrv"]
    steps_all, stage_all = w["steps"].astype(np.float32), w["sleep_stage"]
    Gp_all = g["glucose_true"]                        # ground truth, used ONLY to make labels
    N, T = cgm_all.shape
    D = T // BINS

    # per-day sleep summaries from the night that ended on that day (observable from a wearable)
    sl = {k: beh[k][:, BURN_IN:] for k in ["tst_h", "sleep_eff", "deep_frac", "rem_frac"]}
    wd0 = beh["weekday0"]

    split = patient_split(cohort["hba1c"].to_numpy(), rng)
    static_cols = [c for c in STATIC_CANDIDATES if c in cohort.columns]
    static = cohort[static_cols].astype(np.float32).copy()
    static["male"] = (cohort["GENDER"] == "M").astype(np.float32)
    static_names = list(static.columns)

    feat_names = ["cgm_now", "cgm_slope_15m", "cgm_slope_30m", "cgm_slope_60m", "cgm_mean_1h", "cgm_mean_3h",
                  "cgm_mean_6h", "cgm_max_2h", "cgm_min_2h", "cgm_sd_6h", "cgm_mean_24h", "cgm_sd_24h",
                  "cgm_cv_24h", "cgm_tar_24h", "hr_now", "hr_mean_1h", "hr_mean_6h", "hrv_mean_1h",
                  "hrv_mean_6h", "steps_1h", "steps_3h", "steps_24h", "asleep_now", "sleep_tst_h",
                  "sleep_eff", "sleep_deep", "sleep_rem", "hour_sin", "hour_cos", "weekend"]

    P, Tt, Yh, Yl, TTE, XF = [], [], [], [], [], []
    for c0 in range(0, N, CHUNK):
        sl_p = slice(c0, min(c0 + CHUNK, N))
        cgm, hr, hrv = cgm_all[sl_p], hr_all[sl_p], hrv_all[sl_p]
        steps, stage, Gp = steps_all[sl_p], stage_all[sl_p], Gp_all[sl_p]
        n = cgm.shape[0]

        # ---- labels from the true trace ----
        v = swv(Gp, H, axis=1)[:, 1:]                       # window t+1 .. t+H for each t
        fut_max, fut_min = v.max(-1), v.min(-1)             # length T-H
        first_hi = np.where((v > HYPER).any(-1), (v > HYPER).argmax(-1) + 1, 0)
        L = T - H
        cur_true = Gp[:, :L]

        # ---- sample times: hourly stride with random per-patient offset ----
        t_idx = np.arange(L)[None, :]
        off = rng.integers(0, STRIDE, (n, 1))
        on_grid = ((t_idx - off) % STRIDE == 0) & (t_idx >= MIN_T)
        cgm_ff = ffill(cgm)
        cur_obs = cgm_ff[:, :L]
        eligible = (on_grid & ~np.isnan(cgm[:, :L]) & (cur_obs >= HYPO) & (cur_obs <= HYPER)
                    & (cur_true >= HYPO) & (cur_true <= HYPER))
        pi, ti = np.nonzero(eligible)

        # ---- summary features (computed on full arrays, then sampled) ----
        hr_ff, hrv_ff = ffill(hr), ffill(hrv)
        fe = {}
        fe["cgm_now"] = cgm_ff
        fe["cgm_slope_15m"] = (cgm_ff - lag(cgm_ff, 3)) / 15.0
        fe["cgm_slope_30m"] = (cgm_ff - lag(cgm_ff, 6)) / 30.0
        fe["cgm_slope_60m"] = (cgm_ff - lag(cgm_ff, 12)) / 60.0
        fe["cgm_mean_1h"] = wmean_nan(cgm, 12)
        fe["cgm_mean_3h"] = wmean_nan(cgm, 36)
        fe["cgm_mean_6h"] = wmean_nan(cgm, 72)
        fe["cgm_max_2h"] = trailing_extreme(cgm, 24, np.fmax)
        fe["cgm_min_2h"] = trailing_extreme(cgm, 24, np.fmin)
        fe["cgm_sd_6h"] = wsd_nan(cgm, 72)
        fe["cgm_mean_24h"] = wmean_nan(cgm, BINS)
        fe["cgm_sd_24h"] = wsd_nan(cgm, BINS)
        with np.errstate(invalid="ignore", divide="ignore"):
            fe["cgm_cv_24h"] = fe["cgm_sd_24h"] / fe["cgm_mean_24h"]
        fe["cgm_tar_24h"] = wmean_nan(np.where(np.isnan(cgm), np.nan, (cgm > HYPER).astype(np.float32)), BINS)
        fe["hr_now"] = hr_ff
        fe["hr_mean_1h"] = wmean_nan(hr, 12)
        fe["hr_mean_6h"] = wmean_nan(hr, 72)
        fe["hrv_mean_1h"] = wmean_nan(hrv, 12)
        fe["hrv_mean_6h"] = wmean_nan(hrv, 72)
        fe["steps_1h"] = wsum(steps, 12)
        fe["steps_3h"] = wsum(steps, 36)
        fe["steps_24h"] = wsum(steps, BINS)
        fe["asleep_now"] = (stage > 0).astype(np.float32)

        cols = []
        for name in feat_names[:23]:
            cols.append(fe[name][pi, ti])
        # last completed night's sleep: today's summary only after midday, else yesterday's
        day = ti // BINS
        use_day = np.where((ti % BINS) >= 150, day, np.maximum(day - 1, 0))
        gi = pi + c0
        for k in ["tst_h", "sleep_eff", "deep_frac", "rem_frac"]:
            cols.append(sl[k][gi, use_day])
        hour = (ti % BINS) / 12.0
        cols.append(np.sin(2 * np.pi * hour / 24))
        cols.append(np.cos(2 * np.pi * hour / 24))
        cols.append((((wd0[gi] + day + BURN_IN) % 7) >= 5).astype(np.float32))   # day 0 of behavior is the burn-in day
        XF.append(np.stack(cols, axis=1).astype(np.float32))

        P.append(gi.astype(np.int32)); Tt.append(ti.astype(np.int32))
        Yh.append((fut_max[pi, ti] > HYPER).astype(np.int8))
        Yl.append((fut_min[pi, ti] < HYPO).astype(np.int8))
        TTE.append((first_hi[pi, ti] * 5).astype(np.int16))   # minutes to first >180, 0 if none
        print(f"  patients {c0}-{c0 + n - 1}: {len(pi)} samples")

    pidx, tidx = np.concatenate(P), np.concatenate(Tt)
    y_hyper, y_hypo, tte = np.concatenate(Yh), np.concatenate(Yl), np.concatenate(TTE)
    X = np.concatenate(XF)
    sp = split[pidx]
    np.savez_compressed("data/processed/samples.npz", pidx=pidx, t=tidx, y_hyper=y_hyper, y_hypo=y_hypo,
                        tte_min=tte, split=sp, X_tab=X, feature_names=np.array(feat_names),
                        static_X=static.to_numpy(np.float32), static_names=np.array(static_names),
                        patient_split=split, patient_id=ids)

    # ---------- validation ----------
    print(f"\nTotal samples: {len(pidx):,} from {N} patients ({len(pidx) / N:.0f} per patient)")
    for s, nm in enumerate(["train", "val", "test"]):
        m = sp == s
        print(f"  {nm:5s}: {np.unique(pidx[m]).size:4d} patients, {m.sum():8,d} samples, "
              f"hyper rate {y_hyper[m].mean():.1%}, hypo rate {y_hypo[m].mean():.2%}")
    pos = tte[y_hyper == 1]
    print(f"\nLead time among positives (min to first >180): median {np.median(pos):.0f}, "
          f"p10 {np.percentile(pos, 10):.0f}, p90 {np.percentile(pos, 90):.0f}")
    print(f"Patients with zero events: {(np.bincount(pidx[y_hyper == 1], minlength=N) == 0).mean():.1%}")
    nan_rate = np.isnan(X).mean(0)
    bad = [(feat_names[j], round(float(nan_rate[j]), 3)) for j in range(len(feat_names)) if nan_rate[j] > 0.001]
    print("Features with >0.1% missing:", bad if bad else "none")
    print(f"Static features ({len(static_names)}): {static_names}")
    base = {}
    for nm, col in [("cgm_now", 0), ("cgm_slope_30m", 2)]:
        base[nm] = X[:, col]
    ok = ~np.isnan(base["cgm_now"])
    from numpy import corrcoef
    print(f"Sanity: corr(cgm_now, label) = {corrcoef(base['cgm_now'][ok], y_hyper[ok])[0, 1]:.2f}")


if __name__ == "__main__":
    run_all()
