"""
sensors.py
Turns the simulated latent state into what a wearable / CGM would actually record:
  - CGM       : lagged interstitial glucose + correlated noise, sensor bias, compression lows, dropouts
  - Heart rate: resting HR from EHR (age, BMI, hypertension) + activity, meals, sleep stage, poor sleep
  - HRV (RMSSD): lower with age/BMI, higher in deep sleep, lower after activity, poor sleep and hyperglycemia
  - Steps, sleep stage: passed through from behavior.py
Only these sensor channels (plus the EHR) are model inputs. Plasma glucose and hidden
parameters are NOT saved here.
"""
import numpy as np
import pandas as pd

BINS = 288
BURN_IN = 1
SEED = 2029
STAGE_HR = np.array([0.0, -0.08, -0.13, -0.03])        # awake, light, deep, REM (fraction of resting HR)
STAGE_HRV = np.array([1.0, 1.20, 1.45, 0.90])


def ar1(rng, n, t, phi):
    """Row-wise AR(1) noise, unit variance, shape (n, t)."""
    z = rng.standard_normal((n, t)).astype(np.float32)
    out = np.empty_like(z)
    e = z[:, 0].copy()
    out[:, 0] = e
    s = np.sqrt(1 - phi ** 2)
    for k in range(1, t):
        e = phi * e + s * z[:, k]
        out[:, k] = e
    return out


def run_all():
    rng = np.random.default_rng(SEED)
    gl = np.load("data/processed/glucose_sim.npz", allow_pickle=True)
    beh = np.load("data/processed/behavior.npz", allow_pickle=True)
    ids = gl["patient_id"]
    assert (beh["patient_id"] == ids).all(), "patient order mismatch"
    cohort = pd.read_csv("data/processed/t2d_cohort_with_prs.csv").set_index("patient_id").loc[ids]

    B0 = BURN_IN * BINS
    Gp = gl["glucose_true"]                      # used only to drive HR/HRV physiology, not saved
    Gi = gl["glucose_interstitial"]
    steps = beh["steps"][:, B0:].astype(np.float32)
    carbs = beh["carbs"][:, B0:].astype(np.float32)
    stage = beh["stage"][:, B0:].astype(np.int8)
    N, T = Gp.shape
    D = T // BINS
    poor = np.repeat(beh["poor_sleep"][:, BURN_IN:], BINS, axis=1).astype(np.float32)

    age = cohort["age"].to_numpy()[:, None]
    bmi = cohort["bmi"].to_numpy()[:, None]
    htn = cohort["hypertension"].to_numpy()[:, None]

    # ---------- state variables with memory (activity and meals) ----------
    act_target = np.clip(0.42 * steps / 5.0, 0, 60)          # bpm above rest from walking cadence
    act = np.empty((N, T), np.float32)
    meal = np.empty((N, T), np.float32)
    a = np.zeros(N, np.float32)
    m = np.zeros(N, np.float32)
    for t in range(T):
        a = a + (act_target[:, t] - a) * 0.5                 # ~10 min response lag
        m = m * 0.93 + carbs[:, t] * 0.07                    # post-meal rise decays over ~1 h
        act[:, t], meal[:, t] = a, m

    # ---------- heart rate ----------
    rhr = 64 + 0.5 * (bmi - 28.8) - 0.12 * (age - 62) + 2.5 * htn + rng.normal(0, 4, (N, 1))
    gl_term = 0.02 * np.clip(Gp - 140, 0, 200) + 0.2 * np.clip(70 - Gp, 0, 40)
    hr = rhr * (1 + STAGE_HR[stage]) + act + meal + 2.0 * poor + gl_term
    hr = hr + 2.5 * ar1(rng, N, T, 0.8)
    hr = np.clip(hr, 40, 185)

    # ---------- HRV (RMSSD, ms) ----------
    hrv0 = 30 * np.exp(-0.012 * (age - 62)) * np.exp(-0.015 * (bmi - 28.8)) * np.exp(0.25 * rng.standard_normal((N, 1)))
    hrv = (hrv0 * STAGE_HRV[stage] * np.exp(-0.012 * act) * (1 - 0.18 * poor)
           * np.exp(-0.004 * np.clip(Gp - 140, 0, 200)) * np.exp(0.12 * ar1(rng, N, T, 0.7)))
    hrv = np.clip(hrv, 5, 120)

    # ---------- CGM ----------
    sess = np.arange(T) // (14 * BINS)                       # a new sensor every 14 days
    n_sess = sess.max() + 1
    bias = rng.normal(0, 0.04, (N, n_sess))[:, sess]
    drift = rng.normal(0, 0.03, (N, n_sess))[:, sess] * ((np.arange(T) % (14 * BINS)) / (14 * BINS))
    cgm = Gi * (1 + bias + drift + 0.11 * ar1(rng, N, T, 0.85))     # ~9-10% MARD
    for i, d in np.argwhere(rng.random((N, D)) < 0.04):             # overnight compression lows
        s = d * BINS + int(rng.integers(12, 70))
        cgm[i, s:s + int(rng.integers(4, 10))] -= rng.uniform(15, 40)
    cgm = np.round(np.clip(cgm, 40, 400)).astype(np.float32)

    # dropouts: short gaps, occasional 1-4 h gaps
    for i, t in np.argwhere(rng.random((N, T)) < 0.004):
        cgm[i, t:t + int(rng.geometric(0.2))] = np.nan
    for i, d in np.argwhere(rng.random((N, D)) < 0.03):
        s = d * BINS + int(rng.integers(0, BINS - 48))
        cgm[i, s:s + int(rng.integers(12, 48))] = np.nan
    # watch off (charging) and optical-sensor artefacts
    for i, d in np.argwhere(rng.random((N, D)) < 0.08):
        s = d * BINS + int(rng.integers(96, 240))
        L = int(rng.integers(12, 24))
        hr[i, s:s + L] = np.nan
        hrv[i, s:s + L] = np.nan
    art = rng.random((N, T)) < 0.01
    hr[art] = np.nan
    hrv[art] = np.nan

    np.savez_compressed("data/processed/wearables.npz", patient_id=ids,
                        cgm=cgm, hr=hr.astype(np.float32), hrv=hrv.astype(np.float32),
                        steps=steps.astype(np.int16), sleep_stage=stage)

    # ---------- validation ----------
    asleep = stage > 0
    print(f"Saved wearables for {N} patients x {D} days ({T} five-minute samples each)")
    print(f"CGM missing: {np.isnan(cgm).mean():.1%} | HR missing: {np.isnan(hr).mean():.1%}")
    ok = ~np.isnan(cgm)
    print(f"CGM MARD vs plasma glucose: {np.mean(np.abs(cgm[ok] - Gp[ok]) / Gp[ok]):.1%}")
    awake_hr, sleep_hr = np.nanmean(hr[~asleep]), np.nanmean(hr[asleep])
    print(f"HR awake {awake_hr:.0f} bpm | asleep {sleep_hr:.0f} bpm (dip {1 - sleep_hr / awake_hr:.0%})")
    print(f"HRV awake {np.nanmean(hrv[~asleep]):.0f} ms | asleep {np.nanmean(hrv[asleep]):.0f} ms")
    v = ~np.isnan(hr)
    print(f"corr(steps, HR): {np.corrcoef(steps[v], hr[v])[0, 1]:.2f}")
    print(f"Mean steps/day: {steps.sum() / (N * D):.0f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        i, S = 0, slice(0, 2 * BINS)
        x = np.arange(2 * BINS) / 12
        fig, ax = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        ax[0].plot(x, cgm[i, S]); ax[0].axhline(180, ls="--", c="r", lw=.7); ax[0].set_ylabel("CGM mg/dL")
        ax[1].plot(x, hr[i, S]); ax[1].set_ylabel("HR bpm")
        ax[2].plot(x, hrv[i, S]); ax[2].set_ylabel("HRV ms")
        ax[3].bar(x, steps[i, S], width=0.08); ax[3].set_ylabel("steps")
        ax[4].step(x, stage[i, S]); ax[4].set_ylabel("sleep stage"); ax[4].set_xlabel("hours")
        plt.tight_layout(); plt.savefig("data/processed/wearables_example.png", dpi=120)
        print("Saved plot -> data/processed/wearables_example.png")
    except ImportError:
        pass


if __name__ == "__main__":
    run_all()