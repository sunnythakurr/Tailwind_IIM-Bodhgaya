"""
twin_core.py
Shared logic for the digital twin: computing model features from raw sensor streams,
the RiskModel (probability, alert, explanation). The dashboard and evaluation both import this.
"""
import json
import sys
from pathlib import Path
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent / "datagen"))
import labels as L          # reuse the exact rolling-window helpers used to build the training set

BINS = 288
FEATURE_NAMES = ["cgm_now", "cgm_slope_15m", "cgm_slope_30m", "cgm_slope_60m", "cgm_mean_1h", "cgm_mean_3h",
                 "cgm_mean_6h", "cgm_max_2h", "cgm_min_2h", "cgm_sd_6h", "cgm_mean_24h", "cgm_sd_24h",
                 "cgm_cv_24h", "cgm_tar_24h", "hr_now", "hr_mean_1h", "hr_mean_6h", "hrv_mean_1h",
                 "hrv_mean_6h", "steps_1h", "steps_3h", "steps_24h", "asleep_now", "sleep_tst_h",
                 "sleep_eff", "sleep_deep", "sleep_rem", "hour_sin", "hour_cos", "weekend"]

# Groups used to explain a prediction in plain language
GROUPS = {
    "Current glucose and its recent trend": ["cgm_now", "cgm_slope_15m", "cgm_slope_30m", "cgm_slope_60m",
                                             "cgm_mean_1h", "cgm_max_2h", "cgm_min_2h"],
    "Glucose control over the past day":   ["cgm_mean_3h", "cgm_mean_6h", "cgm_sd_6h", "cgm_mean_24h",
                                             "cgm_sd_24h", "cgm_cv_24h", "cgm_tar_24h"],
    "Heart rate and HRV":                  ["hr_now", "hr_mean_1h", "hr_mean_6h", "hrv_mean_1h", "hrv_mean_6h"],
    "Physical activity":                   ["steps_1h", "steps_3h", "steps_24h"],
    "Sleep":                               ["asleep_now", "sleep_tst_h", "sleep_eff", "sleep_deep", "sleep_rem"],
    "Time of day":                         ["hour_sin", "hour_cos", "weekend"],
    "Health record (EHR)":                 "STATIC",
}


def compute_dynamic_features(cgm, hr, hrv, steps, stage, sleep, wd0):
    """
    cgm/hr/hrv/steps/stage: (n, T) arrays on the 5-minute grid (NaN where missing).
    sleep: dict of (n, D) per-day summaries with keys tst_h, sleep_eff, deep_frac, rem_frac.
    wd0: (n,) weekday of day 0.  Returns (n, T, 30) float32 in FEATURE_NAMES order.
    """
    n, T = cgm.shape
    cgm_ff, hr_ff = L.ffill(cgm), L.ffill(hr)
    fe = {
        "cgm_now": cgm_ff,
        "cgm_slope_15m": (cgm_ff - L.lag(cgm_ff, 3)) / 15.0,
        "cgm_slope_30m": (cgm_ff - L.lag(cgm_ff, 6)) / 30.0,
        "cgm_slope_60m": (cgm_ff - L.lag(cgm_ff, 12)) / 60.0,
        "cgm_mean_1h": L.wmean_nan(cgm, 12),
        "cgm_mean_3h": L.wmean_nan(cgm, 36),
        "cgm_mean_6h": L.wmean_nan(cgm, 72),
        "cgm_max_2h": L.trailing_extreme(cgm, 24, np.fmax),
        "cgm_min_2h": L.trailing_extreme(cgm, 24, np.fmin),
        "cgm_sd_6h": L.wsd_nan(cgm, 72),
        "cgm_mean_24h": L.wmean_nan(cgm, BINS),
        "cgm_sd_24h": L.wsd_nan(cgm, BINS),
        "cgm_tar_24h": L.wmean_nan(np.where(np.isnan(cgm), np.nan, (cgm > L.HYPER).astype(np.float32)), BINS),
        "hr_now": hr_ff,
        "hr_mean_1h": L.wmean_nan(hr, 12),
        "hr_mean_6h": L.wmean_nan(hr, 72),
        "hrv_mean_1h": L.wmean_nan(hrv, 12),
        "hrv_mean_6h": L.wmean_nan(hrv, 72),
        "steps_1h": L.wsum(steps.astype(np.float32), 12),
        "steps_3h": L.wsum(steps.astype(np.float32), 36),
        "steps_24h": L.wsum(steps.astype(np.float32), BINS),
        "asleep_now": (stage > 0).astype(np.float32),
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        fe["cgm_cv_24h"] = fe["cgm_sd_24h"] / fe["cgm_mean_24h"]
    t = np.arange(T)
    day = t // BINS
    use_day = np.where((t % BINS) >= 150, day, np.maximum(day - 1, 0))   # no peeking at tonight's sleep
    for out_name, key in [("sleep_tst_h", "tst_h"), ("sleep_eff", "sleep_eff"),
                          ("sleep_deep", "deep_frac"), ("sleep_rem", "rem_frac")]:
        fe[out_name] = sleep[key][:, use_day]
    hour = (t % BINS) / 12.0
    fe["hour_sin"] = np.broadcast_to(np.sin(2 * np.pi * hour / 24), (n, T))
    fe["hour_cos"] = np.broadcast_to(np.cos(2 * np.pi * hour / 24), (n, T))
    fe["weekend"] = (((np.asarray(wd0)[:, None] + day[None, :] + L.BURN_IN) % 7) >= 5).astype(np.float32)
    return np.stack([np.asarray(fe[k], dtype=np.float32) for k in FEATURE_NAMES], axis=-1)


GROUP_ORDER = list(GROUPS.keys())


def risk_series(model, ids, w, beh, static_all, chunk=50, burn_in=1):
    """Risk at every 5-minute step for the given patients (NaN when no prediction applies:
    first day, missing CGM, or glucose outside 70-180). Returns (n, T) float32."""
    T = w["cgm"].shape[1]
    out = np.full((len(ids), T), np.nan, np.float32)
    for c0 in range(0, len(ids), chunk):
        sel = ids[c0:c0 + chunk]
        cgm = w["cgm"][sel]
        sleep = {k: beh[k][sel][:, burn_in:] for k in ["tst_h", "sleep_eff", "deep_frac", "rem_frac"]}
        F = compute_dynamic_features(cgm, w["hr"][sel], w["hrv"][sel], w["steps"][sel].astype(np.float32),
                                     w["sleep_stage"][sel], sleep, beh["weekday0"][sel])
        valid = (np.arange(T)[None, :] >= BINS) & ~np.isnan(cgm) & (cgm >= L.HYPO) & (cgm <= L.HYPER)
        pi = np.nonzero(valid)[0]
        Xv = np.concatenate([F[valid], static_all[sel][pi]], axis=1)
        block = np.full(cgm.shape, np.nan, np.float32)
        block[valid] = model.proba(Xv)
        out[c0:c0 + chunk] = block
    return out


def explain_batch(model, X):
    """Vectorised explain(): returns (base_prob (m,), contributions (m, len(GROUP_ORDER)))."""
    X = np.asarray(X, dtype=np.float32)
    ref = np.array([model.reference.get(n, np.nan) for n in model.fnames], dtype=np.float32)
    base = model.proba(X)
    contrib = np.zeros((len(X), len(GROUP_ORDER)), np.float32)
    for gi, g in enumerate(GROUP_ORDER):
        names = model.static_names if GROUPS[g] == "STATIC" else GROUPS[g]
        idx = [model.fnames.index(n) for n in names if n in model.fnames]
        if not idx:
            continue
        Z = X.copy()
        Z[:, idx] = ref[idx]
        contrib[:, gi] = base - model.proba(Z)
    return base, contrib


class RiskModel:
    """Fused model: sensor-history features + EHR -> probability of glucose >180 mg/dL within 2 hours."""

    def __init__(self, model_dir="models"):
        md = Path(model_dir)
        self.clf, self.k, self.fnames = joblib.load(md / "Fused.joblib")
        self.fnames = list(self.fnames)
        self.dyn_names = self.fnames[:len(FEATURE_NAMES)]
        self.static_names = self.fnames[len(FEATURE_NAMES):]
        assert self.dyn_names == FEATURE_NAMES, "feature order in the saved model has changed"
        self.thresholds = json.load(open(md / "thresholds.json")) if (md / "thresholds.json").exists() else {}
        self.reference = json.load(open(md / "reference.json")) if (md / "reference.json").exists() else {}

    def proba(self, X, chunk=200_000):
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), np.float32)
        for a in range(0, len(X), chunk):
            for i, p in enumerate(self.clf.staged_predict_proba(X[a:a + chunk]), 1):
                if i == self.k:
                    out[a:a + chunk] = p[:, 1]
                    break
        return out

    def explain(self, x):
        """Approximate attribution for one feature row: how much does risk fall if a group is replaced
        by the typical (median) patient value? Positive = that group pushes risk UP."""
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        ref = np.array([self.reference.get(n, np.nan) for n in self.fnames], dtype=np.float32)
        rows, labels = [x[0]], ["full"]
        for g, members in GROUPS.items():
            names = self.static_names if members == "STATIC" else members
            idx = [self.fnames.index(n) for n in names if n in self.fnames]
            if not idx:
                continue
            z = x[0].copy()
            z[idx] = ref[idx]
            rows.append(z)
            labels.append(g)
        p = self.proba(np.stack(rows))
        base = float(p[0])
        res = [(g, base - float(pp)) for g, pp in zip(labels[1:], p[1:])]
        return base, sorted(res, key=lambda r: -abs(r[1]))

    def status(self, cgm_now, prob, threshold):
        """The model was trained on moments when glucose is in range, so say so when it is not."""
        if cgm_now is None or np.isnan(cgm_now):
            return "NO SENSOR DATA"
        if cgm_now > L.HYPER:
            return "ALREADY HIGH"
        if cgm_now < L.HYPO:
            return "LOW GLUCOSE"
        return "ALERT" if prob >= threshold else "OK"
