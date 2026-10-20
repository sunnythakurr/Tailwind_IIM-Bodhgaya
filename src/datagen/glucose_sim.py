"""
glucose_sim.py
Vectorised glucose-insulin simulator (Bergman-style minimal model + gut model + exercise,
sleep/stress insulin-sensitivity modulation, dawn phenomenon, human 70/30 insulin).
Simulates ALL patients at once at 1-minute resolution, records every 5 minutes.
Labels are NOT created here; adverse events are read off the simulated trace later.
"""
import numpy as np
import pandas as pd

BINS = 288
N_DAYS = 30
BURN_IN = 1
TOTAL_DAYS = N_DAYS + BURN_IN
SEED = 2028

# ---- global physiological constants (population level) ----
P2 = 0.02            # 1/min, decay of remote insulin action
P3 = 1.2e-5 * 4      # 1/min^2 per uU/mL, scaled by each patient's si_true
N_CLR = 0.09         # 1/min, plasma insulin clearance
IB = 8.0             # uU/mL, basal insulin
GAMMA = 0.020        # uU/mL/min per mg/dL, secretion gain, scaled by bcell
CARB_BIO = 0.90      # fraction of carbs absorbed
V_G = 230.0          # dL, effective glucose distribution volume
TAU_CGM = 6.0        # min, blood -> interstitial lag
EX_UPTAKE = 0.35     # mg/dL/min of insulin-independent uptake at 500 steps / 5 min
EX_SI_GAIN = 0.8     # insulin-sensitivity boost per unit of recent exercise
EX_DECAY = 1.0 / 360 # 1/min, exercise effect fades over ~6 h
COUNTERREG = 0.04    # 1/min, glucagon/adrenaline rescue below 65 mg/dL
KA_FAST, KA_SLOW = 0.020, 0.0045   # 1/min, 70/30 insulin: regular and NPH parts
INS_UNIT_TO_CONC = 80.0            # uU/mL added to a depot per unit injected
BASE_DOSE_UNITS = 8.0              # typical total daily dose at dose_scale = 1


def build_insulin_schedule(meals, pid_to_idx, params, rng):
    """Units of insulin injected per 5-min bin (twice daily with breakfast and dinner)."""
    N, T = len(params), TOTAL_DAYS * BINS
    inj = np.zeros((N, T), np.float32)
    scale = params["insulin_dose_scale"].to_numpy()
    m = meals[meals["kind"].isin(["breakfast", "dinner"])]
    for pid, kind, b in zip(m["patient_id"], m["kind"], m["bin"]):
        i = pid_to_idx[pid]
        if scale[i] <= 0 or rng.random() < 0.08:       # not on insulin, or missed dose
            continue
        frac = 0.6 if kind == "breakfast" else 0.4
        inj[i, b] += BASE_DOSE_UNITS * scale[i] * frac * rng.lognormal(0, 0.10)
    return inj


def simulate(params, carbs, steps, si_mult, stress_mult, inj, gb, n_bins, record_from=0):
    """Returns plasma and interstitial (CGM-like) glucose, shape (N, n_bins - record_from)."""
    N = len(params)
    si = params["si_true"].to_numpy()
    bcell = params["bcell"].to_numpy()
    met = params["metformin_hgo_reduction"].to_numpy()
    kg = params["k_gut"].to_numpy()
    dawn = params["dawn_amp"].to_numpy()

    si_base = si * (1.0 + 1.0 * met)            # metformin improves sensitivity a little
    G = gb.copy()
    Gi = gb.copy()
    X = np.zeros(N)
    I = np.full(N, IB)
    Q1 = np.zeros(N)
    Q2 = np.zeros(N)
    Sf = np.zeros(N)
    Ss = np.zeros(N)
    ex = np.zeros(N)                            # recent-exercise state
    out_G = np.zeros((N, n_bins - record_from), np.float32)
    out_Gi = np.zeros_like(out_G)

    for b in range(n_bins):
        day, binofday = divmod(b, BINS)
        hour = binofday / 12.0
        dawn_bump = np.exp(-0.5 * ((hour - 6.0) / 1.5) ** 2)    # dawn phenomenon around 06:00
        gb_eff = gb + dawn * dawn_bump + (1.0 - stress_mult[:, day]) * 60.0
        si_day = si_base * si_mult[:, day]
        e_in = steps[:, b] / 500.0              # ~1 at brisk walking

        for m in range(5):                      # 1-minute internal steps
            if m == 0:
                Q1 = Q1 + carbs[:, b] * 1000.0 * CARB_BIO
                u = inj[:, b]
                Sf = Sf + 0.3 * u * INS_UNIT_TO_CONC
                Ss = Ss + 0.7 * u * INS_UNIT_TO_CONC
            ex = ex + (e_in / 5.0) * 0.05 - EX_DECAY * ex
            ra = kg * Q2                                          # mg/min
            Q1_new = Q1 - kg * Q1
            Q2 = Q2 + kg * Q1 - kg * Q2
            Q1 = Q1_new
            I_exo = KA_FAST * Sf + KA_SLOW * Ss
            Sf = Sf - KA_FAST * Sf
            Ss = Ss - KA_SLOW * Ss
            sec = GAMMA * bcell * np.maximum(G - gb_eff, 0.0)
            I = I + (-N_CLR * (I - IB) + sec + I_exo)
            si_eff = si_day * (1.0 + EX_SI_GAIN * np.minimum(ex, 2.0))
            X = X + (-P2 * X + P3 * si_eff * np.maximum(I - IB, 0.0))
            dG = (-0.03 * (G - gb_eff) - X * G + ra / V_G
                  - EX_UPTAKE * e_in + COUNTERREG * np.maximum(65.0 - G, 0.0))
            G = np.clip(G + dG, 25.0, 600.0)
            Gi = Gi + (G - Gi) / TAU_CGM
        if b >= record_from:
            out_G[:, b - record_from] = G
            out_Gi[:, b - record_from] = Gi
    return out_G, out_Gi


def run_all():
    rng = np.random.default_rng(SEED)
    params = pd.read_csv("data/processed/sim_params.csv")
    cohort = pd.read_csv("data/processed/t2d_cohort_with_prs.csv").set_index("patient_id").loc[params["patient_id"]]
    beh = np.load("data/processed/behavior.npz", allow_pickle=True)
    assert (beh["patient_id"] == params["patient_id"].to_numpy()).all(), "patient order mismatch"
    N, T = len(params), TOTAL_DAYS * BINS

    carbs = beh["carbs"].astype(np.float32)
    steps = beh["steps"].astype(np.float32)
    si_mult = beh["hidden_si_mult"]                      # HIDDEN
    stress_mult = beh["hidden_stress_mult"]              # HIDDEN
    meals = pd.read_csv("data/processed/meals_log.csv")
    a1c = cohort["hba1c"].clip(4.5, 14.0).to_numpy()
    # Clinicians titrate insulin to control: well-controlled patients get lower doses.
    titration = np.clip(0.75 + 0.30 * (a1c - 5.0), 0.4, 1.4)
    params = params.copy()
    params["insulin_dose_scale"] = params["insulin_dose_scale"] * titration
    inj = build_insulin_schedule(meals, {p: i for i, p in enumerate(params["patient_id"])}, params, rng)

    # Calibrate each patient's baseline so mean glucose matches their HbA1c.
    # HbA1c-to-glucose is imperfect in real life, so the target carries ~5% individual noise.
    eag = 28.7 * a1c - 46.7
    target = np.maximum(eag * (1 + 0.05 * rng.standard_normal(N)), 100.0)
    gb = np.clip(0.8 * eag, 80, 260)
    for it in range(5):
        G, _ = simulate(params, carbs, steps, si_mult, stress_mult, inj, gb, 4 * BINS, record_from=BINS)
        err = target - G.mean(1)
        gb = np.clip(gb + 0.9 * err, 60, 300)
        print(f"calibration {it + 1}/5: mean |error| {np.abs(err).mean():.1f} mg/dL")

    print("Running full simulation...")
    G, Gi = simulate(params, carbs, steps, si_mult, stress_mult, inj, gb, T, record_from=BURN_IN * BINS)
    np.savez_compressed("data/processed/glucose_sim.npz", patient_id=params["patient_id"].to_numpy(),
                        glucose_true=G, glucose_interstitial=Gi, gb_calibrated=gb, hba1c=a1c,
                        insulin_units=inj[:, BURN_IN * BINS:])

    # ---- validation ----
    mean_g = G.mean(1)
    ins = params["insulin_dose_scale"].to_numpy() > 0
    tir = ((G >= 70) & (G <= 180)).mean(1)
    tbr = (G < 70).mean(1)
    tar = (G > 180).mean(1)
    print("\n=== Generator validation ===")
    print(f"Mean glucose: {mean_g.mean():.0f} mg/dL | corr(mean glucose, HbA1c) = {np.corrcoef(mean_g, a1c)[0, 1]:.2f}")
    print(f"Time in range 70-180: mean {tir.mean():.0%} (p10 {np.percentile(tir, 10):.0%}, p90 {np.percentile(tir, 90):.0%})")
    print(f"Time above 180: {tar.mean():.1%} | below 70: {tbr.mean():.1%} "
          f"(insulin users {tbr[ins].mean():.1%}, others {tbr[~ins].mean():.1%})")
    for lo, hi in [(0, 5.5), (5.5, 6.5), (6.5, 7.5), (7.5, 20)]:
        s = (a1c >= lo) & (a1c < hi)
        print(f"  HbA1c {lo}-{hi}: n={s.sum()}, TIR {tir[s].mean():.0%}, TAR {tar[s].mean():.1%}")
    print(f"Glucose CV (mean across patients): {(G.std(1) / mean_g).mean():.0%}")
    print(f"Min / max glucose: {G.min():.0f} / {G.max():.0f}")

    H = 24   # 2 hours of 5-min bins
    ok = (G[:, :-H] >= 70) & (G[:, :-H] <= 180)
    fut = np.stack([G[:, k:G.shape[1] - H + k] for k in range(1, H + 1)])
    print(f"P(>180 within 2h | in range now): {(fut.max(0)[ok] > 180).mean():.1%}")
    print(f"P(<70 within 2h | in range now):  {(fut.min(0)[ok] < 70).mean():.1%}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        picks = [int(np.argmax(ins & (tbr > 0.02))), int(np.argsort(a1c)[-5]), int(np.argsort(a1c)[N // 2])]
        fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        x = np.arange(3 * BINS) / 12
        for a, i in zip(ax, picks):
            a.plot(x, G[i, :3 * BINS], label="plasma")
            a.plot(x, Gi[i, :3 * BINS], label="CGM-like", alpha=.7)
            a.axhline(70, ls="--", c="r", lw=.7); a.axhline(180, ls="--", c="r", lw=.7)
            cb = carbs[i, BURN_IN * BINS:(BURN_IN + 3) * BINS]
            a.vlines(x[cb > 0], 40, 40 + cb[cb > 0], color="g", lw=2)
            a.set_ylabel(f"mg/dL\nA1c {a1c[i]:.1f}, insulin={bool(ins[i])}")
        ax[0].legend(loc="upper right"); ax[2].set_xlabel("hours (green bars = carbs)")
        plt.tight_layout(); plt.savefig("data/processed/glucose_example.png", dpi=120)
        print("Saved plot -> data/processed/glucose_example.png")
    except ImportError:
        pass


if __name__ == "__main__":
    run_all()