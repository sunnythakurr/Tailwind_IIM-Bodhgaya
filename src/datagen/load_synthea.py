"""
load_synthea.py
Extracts the Type 2 Diabetes cohort from Synthea CSVs into one row per patient.
Observations.csv is large, so it is read in chunks.
"""
from pathlib import Path
import pandas as pd

RAW = Path("output/csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
REF_DATE = pd.Timestamp("2026-10-10")  # date the Synthea run was made

T2D_DESC = "Diabetes mellitus type 2 (disorder)"

OBS_CODES = {
    "4548-4":  "hba1c",
    "39156-5": "bmi",
    "2339-0":  "glucose",
    "8480-6":  "sbp",
    "8462-4":  "dbp",
    "2093-3":  "chol_total",
    "2571-8":  "triglycerides",
    "2085-9":  "hdl",
    "18262-6": "ldl",
}

# 1. Conditions -> T2D patients + comorbidity flags
cond = pd.read_csv(RAW / "conditions.csv", usecols=["START", "PATIENT", "DESCRIPTION"],
                   parse_dates=["START"])
t2d = cond[cond["DESCRIPTION"] == T2D_DESC]
dx = t2d.groupby("PATIENT")["START"].min().rename("t2d_dx_date")
t2d_ids = set(dx.index)
print(f"Patients with T2D: {len(t2d_ids)}")

c = cond[cond["PATIENT"].isin(t2d_ids)]
def flag(pattern):
    hit = c[c["DESCRIPTION"].str.contains(pattern, case=False, na=False)]
    return c["PATIENT"].drop_duplicates().isin(hit["PATIENT"]).set_axis(
        c["PATIENT"].drop_duplicates()).astype(int)

flags = pd.DataFrame({
    "hypertension": flag("hypertension"),
    "ckd":          flag("chronic kidney disease|kidney due to diabetes"),
    "retinopathy":  flag("retinopathy"),
    "neuropathy":   flag("neuropathy"),
    "obesity":      flag("obesity"),
    "cvd":          flag("myocardial infarction|coronary|heart failure|stroke|cardiac arrest"),
}).reindex(list(t2d_ids)).fillna(0).astype(int)

# 2. Patients -> demographics (PII columns are dropped)
pat = pd.read_csv(RAW / "patients.csv",
                  usecols=["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "RACE", "ETHNICITY", "INCOME"],
                  parse_dates=["BIRTHDATE", "DEATHDATE"])
pat = pat[pat["Id"].isin(t2d_ids)].set_index("Id")
pat["age"] = ((REF_DATE - pat["BIRTHDATE"]).dt.days / 365.25).round(1)
pat["alive"] = pat["DEATHDATE"].isna().astype(int)

# 3. Observations (chunked) -> keep only our codes for T2D patients
parts = []
for chunk in pd.read_csv(RAW / "observations.csv", chunksize=500_000,
                         usecols=["DATE", "PATIENT", "CODE", "VALUE"],
                         dtype={"CODE": str, "VALUE": str}):
    chunk = chunk[chunk["PATIENT"].isin(t2d_ids) & chunk["CODE"].isin(OBS_CODES)]
    parts.append(chunk)
obs = pd.concat(parts, ignore_index=True)
obs["DATE"] = pd.to_datetime(obs["DATE"], errors="coerce").dt.tz_localize(None)
obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")
obs["name"] = obs["CODE"].map(OBS_CODES)
obs = obs.dropna(subset=["VALUE", "DATE"]).sort_values("DATE")
obs.to_csv(OUT / "obs_t2d_long.csv", index=False)  # full history kept for later

latest = obs.groupby(["PATIENT", "name"])["VALUE"].last().unstack()
first_a1c = obs[obs["name"] == "hba1c"].groupby("PATIENT")["VALUE"].first().rename("hba1c_first")

# 4. Medications -> active/ever flags
med = pd.read_csv(RAW / "medications.csv", usecols=["START", "STOP", "PATIENT", "DESCRIPTION"])
med = med[med["PATIENT"].isin(t2d_ids)]
def med_flag(pattern, active_only):
    m = med[med["DESCRIPTION"].str.contains(pattern, case=False, na=False)]
    if active_only:
        m = m[m["STOP"].isna()]
    return pd.Series(1, index=m["PATIENT"].unique())

meds = pd.DataFrame({
    "metformin_active": med_flag("metformin", True),
    "insulin_active":   med_flag("insulin", True),
    "metformin_ever":   med_flag("metformin", False),
    "insulin_ever":     med_flag("insulin", False),
}).reindex(list(t2d_ids)).fillna(0).astype(int)

# 5. Assemble
cohort = (pat.join(dx).join(flags).join(meds).join(latest).join(first_a1c))
cohort["years_since_dx"] = ((REF_DATE - cohort["t2d_dx_date"]).dt.days / 365.25).round(1)
cohort.index.name = "patient_id"
cohort = cohort.drop(columns=["BIRTHDATE", "DEATHDATE"])
cohort = cohort[cohort["alive"] == 1].copy()          
cohort["hba1c"] = cohort["hba1c"].clip(4.5, 14.0)    
cohort.to_csv(OUT / "t2d_cohort.csv")

print(f"\nSaved {len(cohort)} patients -> {OUT / 't2d_cohort.csv'}")
print("\nMissing values per column:")
print(cohort.isna().sum()[cohort.isna().sum() > 0])
print("\nSummary:")
print(cohort[["age", "years_since_dx", "bmi", "hba1c", "glucose"]].describe().round(1))
print("\nGender:\n", cohort["GENDER"].value_counts())
print("\nAlive:", cohort["alive"].sum())
print("On metformin:", cohort["metformin_active"].sum(),
      "| On insulin:", cohort["insulin_active"].sum())