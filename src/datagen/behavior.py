"""
behavior.py
Generates daily behavior per patient on a 5-minute grid:
meals (Indian carb profiles), steps, sleep stages, stress/illness days.
Day 0 is a burn-in day (dropped after simulation).
HIDDEN arrays (prefixed hidden_) are for the simulator only, never model features.

v2: meal timing is irregular (per-patient chronotype, per-day shifts, ~10% disrupted days,
    wider per-meal spread) so the model cannot simply learn the clock.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BINS = 288                      # 5-min bins per day
N_DAYS = 30
BURN_IN = 1
TOTAL_DAYS = N_DAYS + BURN_IN
SEED = 2027
AWAKE, LIGHT, DEEP, REM = 0, 1, 2, 3

# (food, mean carbs g, sd g, probability)
MEAL_MENU = {
    "breakfast": [("idli_sambar", 45, 10, .25), ("dosa", 55, 12, .15), ("poha_upma", 50, 10, .20),
                  ("paratha", 60, 12, .15), ("bread_egg_oats", 40, 10, .15), ("puri_bhaji", 75, 15, .10)],
    "lunch":     [("rice_thali", 95, 20, .35), ("roti_sabzi_dal", 70, 15, .25), ("biryani", 110, 20, .10),
                  ("khichdi", 65, 12, .10), ("light_meal", 40, 12, .20)],
    "dinner":    [("roti_sabzi", 65, 15, .35), ("rice_dal", 85, 18, .30), ("dosa_idli", 60, 12, .15),
                  ("light_meal", 40, 12, .20)],
    "snack":     [("chai_biscuit", 20, 6, .35), ("fruit", 25, 8, .15), ("samosa_pakora", 30, 8, .20),
                  ("mithai", 35, 12, .10), ("namkeen", 15, 5, .20)],
}


def pick(menu, rng):
    i = rng.choice(len(menu), p=[m[3] for m in menu])
    return menu[i][0], menu[i][1], menu[i][2]


def gen_meals(day, weekend, carb_pref, rng, chrono=0.0):
    shift = 0.5 if weekend else 0.0
    # (kind, mean hour, sd hours, probability the meal happens)
    plan = [("breakfast", 8.0 + shift, 0.75, 0.90), ("snack", 11.0, 0.6, 0.35),
            ("lunch", 13.25 + shift, 0.75, 0.98), ("snack", 17.0, 0.7, 0.75),
            ("dinner", 20.5 + 0.5 * shift, 1.0, 0.98)]
    festival = rng.random() < 0.04
    day_shift = rng.normal(0, 0.6)                   # the whole day runs early or late
    if rng.random() < 0.10:                          # shift work, travel, fasting-type day
        day_shift += rng.choice([-2.0, 2.5, 3.0])
    out = []
    for kind, mu_h, sd_h, p in plan:
        if rng.random() > p:
            continue
        hour = float(np.clip(rng.normal(mu_h + chrono + day_shift, 1.4 * sd_h), 5.5, 23.5))
        food, mu, sd = pick(MEAL_MENU[kind], rng)
        carbs = rng.normal(mu, sd) * carb_pref * rng.lognormal(0, 0.08)
        if weekend and kind == "lunch":
            carbs *= 1.1
        if festival and kind in ("lunch", "dinner"):
            carbs += rng.normal(40, 15)
            food += "+sweets"
        out.append((day * BINS + int(hour * 12), float(np.clip(carbs, 8, 180)), kind, food))
    return out


def gen_sleep_night(night_day, wd0, stressed, rng):
    """Night starting on evening of night_day (may be -1 = night before day 0)."""
    late = ((wd0 + night_day) % 7) in (4, 5)           # Fri/Sat nights are later
    bed_h = float(np.clip(rng.normal(23.2 + (0.7 if late else 0.0), 1.0), 20.5, 28.0))
    dur_h = float(np.clip(rng.normal(6.9 + (0.5 if late else 0.0) - (0.6 if stressed else 0.0), 1.0), 3.5, 9.5))
    n = int(dur_h * 12)
    t = np.arange(n)
    cycle, pos = t // 18, (t % 18) / 18                  # 90-min sleep cycles
    p_deep = np.where((pos > 0.1) & (pos < 0.5), np.maximum(0.0, 0.65 - 0.17 * cycle), 0.0)
    p_rem = np.where(pos >= 0.55, np.minimum(0.65, 0.20 + 0.12 * cycle), 0.0)
    st = np.full(n, LIGHT, dtype=np.int8)
    st[rng.random(n) < p_deep] = DEEP                    # deep sleep early in the night
    st[(rng.random(n) < p_rem) & (st == LIGHT)] = REM    # REM late in the night
    frag = rng.beta(2, 6) + (0.10 if stressed else 0.0)
    st[rng.random(n) < 0.02 + 0.05 * frag] = AWAKE       # night-time awakenings
    st[: min(int(rng.exponential(2.5)), 8)] = AWAKE      # sleep-onset latency
    asleep = st != AWAKE
    m = dict(tst_h=asleep.sum() / 12, eff=asleep.mean(),
             deep=(st == DEEP).sum() / max(asleep.sum(), 1),
             rem=(st == REM).sum() / max(asleep.sum(), 1))
    return int(night_day * BINS + bed_h * 12), st, m


def gen_steps(in_bed_day, weekend, activity, ill, meal_bins, rng):
    awake = np.where(~in_bed_day)[0]
    steps = np.zeros(BINS)
    steps[awake] = rng.poisson(8 + 25 * activity, len(awake))       # light background movement

    def bout(start, length, cadence):
        end = min(start + length, BINS)
        if end > start:
            steps[start:end] += np.clip(rng.normal(cadence * 5, 40, end - start), 100, None)

    for _ in range(rng.poisson(2 + 5 * activity)):                   # random walking bouts
        bout(int(rng.choice(awake)), int(rng.integers(1, 7)), rng.normal(95, 12))
    for b, kind in meal_bins:                                        # post-meal walks
        if kind in ("lunch", "dinner") and rng.random() < 0.10 + 0.50 * activity:
            bout(b + int(rng.integers(2, 5)), int(rng.integers(2, 5)), rng.normal(90, 10))

    target = (2500 + 7500 * activity) * rng.lognormal(0, 0.25) * (0.5 if ill else 1.0)
    if weekend:
        target *= rng.choice([0.9, 1.0, 1.15])
    steps *= np.clip(target / max(steps.sum(), 1), 0.5, 2.0)
    steps[in_bed_day] = 0
    return np.clip(steps, 0, 900).astype(np.int16)


def generate_patient(p, seed):
    rng = np.random.default_rng(seed)
    T = TOTAL_DAYS * BINS
    wd0 = int(rng.integers(0, 7))                                    # weekday of day 0 (0=Mon)
    chrono = float(rng.normal(0, 0.7))                               # habitual early/late eating

    # day-level hidden states
    stress = rng.random(TOTAL_DAYS) < 0.07
    illness = np.zeros(TOTAL_DAYS, bool)
    for d in range(1, TOTAL_DAYS):
        if rng.random() < 0.015:
            illness[d:d + 3] = True
    stress_mult = np.ones(TOTAL_DAYS)
    stress_mult[stress] = rng.uniform(0.80, 0.92, stress.sum())
    stress_mult[illness] = rng.uniform(0.60, 0.80, illness.sum())

    # sleep
    stage = np.zeros(T, np.int8)
    in_bed = np.zeros(T, bool)
    tst, eff, deep, rem = (np.full(TOTAL_DAYS, np.nan) for _ in range(4))
    for night in range(-1, TOTAL_DAYS):
        flag = bool((stress | illness)[night]) if night >= 0 else False
        start, st, m = gen_sleep_night(night, wd0, flag, rng)
        s0, e0 = max(start, 0), min(start + len(st), T)
        if e0 > s0:
            stage[s0:e0] = st[s0 - start:e0 - start]
            in_bed[s0:e0] = True
        wake_day = night + 1                                         # this night affects the next day
        if wake_day < TOTAL_DAYS:
            tst[wake_day], eff[wake_day], deep[wake_day], rem[wake_day] = m["tst_h"], m["eff"], m["deep"], m["rem"]
    poor = np.clip(0.5 * np.maximum(0, (7.0 - tst) / 2.5) + 0.5 * np.maximum(0, (0.90 - eff) / 0.15), 0, 1)
    si_mult = (1 - p["sleep_si_penalty"] * poor) * stress_mult       # HIDDEN: used by simulator

    # meals + steps
    carbs = np.zeros(T, np.float32)
    steps = np.zeros(T, np.int16)
    rows = []
    for d in range(TOTAL_DAYS):
        weekend = ((wd0 + d) % 7) >= 5
        meals = gen_meals(d, weekend, p["carb_pref"], rng, chrono)
        for b, c, kind, food in meals:
            carbs[b] += c
            rows.append((p["patient_id"], d, b, round(c, 1), kind, food))
        sl = slice(d * BINS, (d + 1) * BINS)
        steps[sl] = gen_steps(in_bed[sl], weekend, p["activity_level"], illness[d],
                              [(b - d * BINS, k) for b, _, k, _ in meals], rng)

    return dict(carbs=carbs, steps=steps, stage=stage, tst_h=tst, sleep_eff=eff, deep_frac=deep,
                rem_frac=rem, poor_sleep=poor, hidden_stress_mult=stress_mult,
                hidden_illness=illness, hidden_si_mult=si_mult, weekday0=wd0), rows


if __name__ == "__main__":
    params = pd.read_csv("data/processed/sim_params.csv")
    N, T = len(params), TOTAL_DAYS * BINS
    keys = ["carbs", "steps", "stage", "tst_h", "sleep_eff", "deep_frac", "rem_frac",
            "poor_sleep", "hidden_stress_mult", "hidden_illness", "hidden_si_mult"]
    store, all_rows, wd0 = {}, [], np.zeros(N, int)
    for i, (_, p) in enumerate(params.iterrows()):
        res, rows = generate_patient(p, SEED + i)
        for k in keys:
            if k not in store:
                store[k] = np.zeros((N,) + res[k].shape, dtype=res[k].dtype)
            store[k][i] = res[k]
        wd0[i] = res["weekday0"]
        all_rows += rows
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{N} patients done")

    np.savez_compressed("data/processed/behavior.npz", patient_id=params["patient_id"].to_numpy(),
                        weekday0=wd0, **store)
    meals = pd.DataFrame(all_rows, columns=["patient_id", "day", "bin", "carbs_g", "kind", "food"])
    meals.to_csv("data/processed/meals_log.csv", index=False)

    # ---- sanity summary (burn-in day excluded) ----
    s = store["steps"].reshape(N, TOTAL_DAYS, BINS).sum(2)[:, BURN_IN:]
    c = store["carbs"].reshape(N, TOTAL_DAYS, BINS).sum(2)[:, BURN_IN:]
    print(f"\nSaved behavior for {N} patients x {N_DAYS} days (+1 burn-in)")
    print(f"Steps/day   mean {s.mean():.0f}  sd {s.std():.0f}  (5th-95th pct: {np.percentile(s, 5):.0f}-{np.percentile(s, 95):.0f})")
    print(f"Carbs/day   mean {c.mean():.0f} g  sd {c.std():.0f}")
    print(f"Meals/day   {len(meals[meals.day >= BURN_IN]) / (N * N_DAYS):.2f}")
    print(f"Sleep       {np.nanmean(store['tst_h']):.1f} h | efficiency {np.nanmean(store['sleep_eff']):.0%} "
          f"| deep {np.nanmean(store['deep_frac']):.0%} | REM {np.nanmean(store['rem_frac']):.0%}")
    print(f"Poor-sleep days (score>0.3): {(store['poor_sleep'] > 0.3).mean():.0%}")
    print(f"Stress/illness days: {(store['hidden_stress_mult'] < 1).mean():.1%}")
    bk = meals[(meals.kind == "breakfast") & (meals.day >= BURN_IN)]
    lu = meals[(meals.kind == "lunch") & (meals.day >= BURN_IN)]
    print(f"Breakfast time: mean {(bk['bin'] % BINS).mean() / 12:.1f} h, sd {(bk['bin'] % BINS).std() / 12:.2f} h | "
          f"Lunch: mean {(lu['bin'] % BINS).mean() / 12:.1f} h, sd {(lu['bin'] % BINS).std() / 12:.2f} h")
    print("\nFood mix:\n", meals["food"].value_counts().head(8))

    # ---- optional picture of one patient's 3 days ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sl = slice(BURN_IN * BINS, (BURN_IN + 3) * BINS)
        x = np.arange(3 * BINS) / 12
        fig, ax = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
        ax[0].bar(x, store["carbs"][0][sl], width=0.2); ax[0].set_ylabel("carbs (g)")
        ax[1].plot(x, store["steps"][0][sl]); ax[1].set_ylabel("steps / 5 min")
        ax[2].step(x, store["stage"][0][sl]); ax[2].set_ylabel("sleep stage"); ax[2].set_xlabel("hours")
        plt.tight_layout(); plt.savefig("data/processed/behavior_example.png", dpi=120)
        print("\nSaved plot -> data/processed/behavior_example.png")
    except ImportError:
        print("\n(matplotlib not installed, skipping plot: pip install matplotlib)")
