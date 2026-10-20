"""
make_demo_bundle.py
Builds the small data bundle the Streamlit dashboard reads (demo/demo_bundle.npz + demo/patients.json).
Picks 12 TEST patients (never seen in training) in three archetypes:
  A = poorly controlled, not on insulin   (frequent alerts)
  B = insulin users                       (under-alerted by a global threshold; hypo exposure)
  C = well controlled                     (the twin should stay quiet)
For each: the last 7 days of sensor data, risk + explanation at every 5-minute step,
what actually happened next (for the replay overlay), and the inputs needed for the what-if simulator.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import twin_core as TC
import glucose_sim as GS          # src/datagen is on sys.path via twin_core

BINS = 288
DISPLAY_DAYS = 7
SIM_DAYS = DISPLAY_DAYS + 2       # two warm-up days for the what-if simulator
PER_GROUP = 4
OUT = Path("demo")


def main():
    OUT.mkdir(exist_ok=True)
    d = np.load("data/processed/samples.npz", allow_pickle=True)
    w = np.load("data/processed/wearables.npz", allow_pickle=True)
    g = np.load("data/processed/glucose_sim.npz", allow_pickle=True)
    beh = np.load("data/processed/behavior.npz", allow_pickle=True)
    sim_params = pd.read_csv("data/processed/sim_params.csv")
    model = TC.RiskModel()
    pers = json.load(open("models/personal_thresholds.json"))
    sn = list(d["static_names"])
    static_all = d["static_X"]
    ids = d["patient_id"]
    N, T = w["cgm"].shape
    T0 = T - DISPLAY_DAYS * BINS
    S0 = T - SIM_DAYS * BINS

    test = d["patient_split"] == 2
    pidx, y = d["pidx"], d["y_hyper"]
    rate = np.bincount(pidx[y == 1], minlength=N) / np.maximum(np.bincount(pidx, minlength=N), 1)
    ins = static_all[:, sn.index("insulin_active")] == 1
    a1c = static_all[:, sn.index("hba1c")]
    tbr = (g["glucose_true"] < 70).mean(1)

    def top(mask, score, k=PER_GROUP):
        cand = np.nonzero(mask)[0]
        return list(cand[np.argsort(-score[cand])][:k])

    A = top(test & ~ins & (a1c >= 6.5), rate)
    if len(A) < PER_GROUP:
        A = top(test & ~ins & (a1c >= 6.0), rate)
    B = top(test & ins & (a1c >= 5.5), tbr)          # realistic HbA1c only (not the clipped floor value)
    if len(B) < PER_GROUP:
        B = top(test & ins & (a1c >= 5.0), tbr)
    C = top(test & ~ins & (a1c < 5.8), -rate)
    chosen = [("A", "Poorly controlled, not on insulin", A), ("B", "On insulin", B),
              ("C", "Well controlled, not on insulin", C)]

    arrays = {k: [] for k in ["cgm", "hr", "hrv", "steps", "stage", "risk", "contrib", "feats", "event_next2h",
                              "episode_start", "sim_carbs", "sim_steps", "sim_inj", "sim_si_mult",
                              "sim_stress_mult", "sim_gb", "ref_cgm_interstitial"]}
    meta = []
    for letter, label, members in chosen:
        for n_, i in enumerate(members, 1):
            sel = np.array([i])
            sleep = {k: beh[k][sel][:, 1:] for k in ["tst_h", "sleep_eff", "deep_frac", "rem_frac"]}
            F = TC.compute_dynamic_features(w["cgm"][sel], w["hr"][sel], w["hrv"][sel],
                                            w["steps"][sel].astype(np.float32), w["sleep_stage"][sel],
                                            sleep, beh["weekday0"][sel])[0]
            cg = w["cgm"][i, T0:]
            valid = ~np.isnan(cg) & (cg >= 70) & (cg <= 180)
            Xv = np.concatenate([F[T0:][valid], np.tile(static_all[i], (int(valid.sum()), 1))], axis=1)
            base, contrib = TC.explain_batch(model, Xv)
            risk = np.full(len(cg), np.nan, np.float32); risk[valid] = base
            con = np.full((len(cg), len(TC.GROUP_ORDER)), np.nan, np.float32); con[valid] = contrib

            G = g["glucose_true"][i]
            hi = G > 180
            fut = np.full(T, -1, np.int8)
            fm = np.max(np.lib.stride_tricks.sliding_window_view(G, 24)[1:], axis=-1)
            fut[:T - 24] = (fm > 180).astype(np.int8)
            quiet = TC.L.wsum(hi[None, :].astype(np.float32), 12)[0]
            ep = np.zeros(T, np.int8)
            ep[1:] = (hi[1:] & (quiet[:-1] == 0)).astype(np.int8)

            # what-if inputs (days T-9 .. T) and a reproducibility check of the re-simulation
            sl = slice(S0, T)
            d0 = S0 // BINS
            p1 = sim_params[sim_params["patient_id"] == ids[i]].reset_index(drop=True)
            si_m = beh["hidden_si_mult"][i, 1 + d0:1 + d0 + SIM_DAYS][None, :]
            st_m = beh["hidden_stress_mult"][i, 1 + d0:1 + d0 + SIM_DAYS][None, :]
            gb = g["gb_calibrated"][[i]]
            _, gi = GS.simulate(p1, beh["carbs"][[i], BINS + S0:BINS + T].astype(np.float32),
                                w["steps"][[i], S0:T].astype(np.float32), si_m, st_m,
                                g["insulin_units"][[i], S0:T], gb, SIM_DAYS * BINS, record_from=2 * BINS)
            ref = g["glucose_interstitial"][i, T0:]
            mae = float(np.mean(np.abs(gi[0] - ref)))

            for k, v in [("cgm", cg), ("hr", w["hr"][i, T0:]), ("hrv", w["hrv"][i, T0:]),
                         ("steps", w["steps"][i, T0:]), ("stage", w["sleep_stage"][i, T0:]), ("risk", risk),
                         ("contrib", con), ("feats", F[T0:]), ("event_next2h", fut[T0:]), ("episode_start", ep[T0:]),
                         ("sim_carbs", beh["carbs"][i, BINS + S0:BINS + T].astype(np.float32)),
                         ("sim_steps", w["steps"][i, S0:T]), ("sim_inj", g["insulin_units"][i, S0:T]),
                         ("sim_si_mult", si_m[0]), ("sim_stress_mult", st_m[0]), ("sim_gb", gb[0]),
                         ("ref_cgm_interstitial", ref)]:
                arrays[k].append(v)
            key = f"{letter}{n_}"
            tir = float(np.mean((cg[~np.isnan(cg)] >= 70) & (cg[~np.isnan(cg)] <= 180)))
            meta.append(dict(
                key=key, archetype=label, source_index=int(i),
                ehr={n: float(static_all[i, j]) for j, n in enumerate(sn)},
                thresholds={"global": {k: float(v) for k, v in model.thresholds.items()},
                            "personal": {k: pers[str(ids[i])][f"personal_{k}"] for k in model.thresholds},
                            "blended": {k: pers[str(ids[i])][f"blended_{k}"] for k in model.thresholds}},
                window=dict(episodes=int(ep[T0:].sum()), time_in_range=tir, mean_cgm=float(np.nanmean(cg)),
                            weekday_of_first_day=int((beh["weekday0"][i] + 1 + T0 // BINS) % 7)),
                sim_params={c: float(p1[c][0]) for c in ["si_true", "bcell", "metformin_hgo_reduction",
                                                         "k_gut", "dawn_amp"]},
                resim_mae_mgdl=mae))
            print(f"  {key}: {label:34s} HbA1c {a1c[i]:.1f} | episodes in window {int(ep[T0:].sum()):2d} | "
                  f"time in range {tir:.0%} | what-if re-simulation error {mae:.1f} mg/dL")

    np.savez_compressed(OUT / "demo_bundle.npz", **{k: np.stack(v) for k, v in arrays.items()},
                        feature_names=np.array(TC.FEATURE_NAMES), group_names=np.array(TC.GROUP_ORDER),
                        static_names=np.array(sn))
    json.dump(dict(patients=meta, display_days=DISPLAY_DAYS, sim_days=SIM_DAYS, warmup_days=2,
                   bins_per_day=BINS, group_names=TC.GROUP_ORDER), open(OUT / "patients.json", "w"), indent=1)
    size = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"\nSaved demo/demo_bundle.npz and demo/patients.json ({size:.1f} MB total)")


if __name__ == "__main__":
    main()
