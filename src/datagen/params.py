"""
params.py
Maps each patient's EHR fields to hidden physiological parameters for the simulator.
IMPORTANT: sim_params.csv contains HIDDEN variables (si_true etc.).
They must NEVER be used as model features, only by the simulator.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 2026
INDIA_ADJUST = True   # South Asians develop insulin resistance at lower BMI
rng = np.random.default_rng(SEED)

df = pd.read_csv("data/processed/t2d_cohort.csv")
n = len(df)

# ---------- fields from the EHR ----------
a1c = df["hba1c"].clip(4.5, 14.0).to_numpy()
bmi = df["bmi"].to_numpy()
age = df["age"].to_numpy()
yrs = df["years_since_dx"].to_numpy()
met = df["metformin_active"].to_numpy()
ins = df["insulin_active"].to_numpy()

# ---------- 1. baseline glucose ----------
# Estimated average glucose from HbA1c: eAG = 28.7*A1c - 46.7 (mg/dL).
# Mean glucose includes meal excursions, so the fasting baseline sits lower.
eag = 28.7 * a1c - 46.7
gb = np.clip(0.80 * eag, 80, 260)

# ---------- 2. insulin sensitivity (SI) ----------
# SI falls with adiposity, age and poorer control, plus a HIDDEN random component.
eff_bmi = bmi + (2.5 if INDIA_ADJUST else 0.0)     # same adiposity, more resistance
si_ehr = (
    1.0
    * np.exp(-0.05 * (eff_bmi - 25))
    * np.exp(-0.15 * (a1c - 6.0))
    * np.exp(-0.008 * (age - 50))
    * (0.90 if INDIA_ADJUST else 1.0)
)
hidden_z = rng.standard_normal(n)                  # what the EHR cannot see
si_true = si_ehr * np.exp(0.30 * hidden_z)         # lognormal patient noise

# ---------- 3. beta-cell function (0..1) ----------
bcell = (
    np.exp(-0.04 * yrs)
    * np.exp(-0.20 * (a1c - 6.0))
    * np.exp(0.20 * rng.standard_normal(n))
)
bcell = np.clip(bcell, 0.10, 1.0)

# ---------- 4. synthetic polygenic risk score ("genetic marker") ----------
# Partly informative about the hidden insulin resistance, so the genetic field
# carries real but imperfect signal. Clearly synthetic; state this in the README.
rho = 0.45
prs = -rho * hidden_z + np.sqrt(1 - rho**2) * rng.standard_normal(n)

# ---------- 5. medication effects ----------
metformin_hgo_reduction = 0.15 * met               # lowers liver glucose output
insulin_dose_scale = np.where(ins == 1, rng.uniform(0.5, 1.5, n), 0.0)

# ---------- 6. other per-patient randomness ----------
k_gut = np.exp(np.log(0.035) + 0.25 * rng.standard_normal(n))   # gastric emptying, 1/min
dawn_amp = rng.uniform(0, 25, n)                  # mg/dL early-morning rise
sleep_si_penalty = rng.uniform(0.10, 0.20, n)     # SI drop after poor sleep
carb_pref = rng.uniform(0.8, 1.3, n)              # usual meal-size multiplier
activity_level = rng.beta(2, 3, n)                # 0 sedentary .. 1 active

out = pd.DataFrame({
    "patient_id": df["patient_id"],
    "gb": gb.round(1),
    "si_true": si_true,
    "si_ehr_expected": si_ehr,
    "bcell": bcell.round(3),
    "metformin_hgo_reduction": metformin_hgo_reduction,
    "insulin_dose_scale": insulin_dose_scale.round(2),
    "k_gut": k_gut,
    "dawn_amp": dawn_amp.round(1),
    "sleep_si_penalty": sleep_si_penalty.round(3),
    "carb_pref": carb_pref.round(2),
    "activity_level": activity_level.round(3),
})
out.to_csv("data/processed/sim_params.csv", index=False)

# the genetic field IS an EHR-side feature, so it goes back with the cohort
df["prs"] = prs.round(3)
df.to_csv("data/processed/t2d_cohort_with_prs.csv", index=False)

print(f"Saved {n} patients")
print(out.drop(columns="patient_id").describe().round(3).T[["mean", "std", "min", "max"]])
print("\ncorr(PRS, log si_true):", np.corrcoef(prs, np.log(si_true))[0, 1].round(2))
print("corr(log si_ehr, log si_true):", np.corrcoef(np.log(si_ehr), np.log(si_true))[0, 1].round(2))