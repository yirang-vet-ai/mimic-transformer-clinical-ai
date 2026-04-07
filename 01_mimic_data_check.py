from pathlib import Path
import pandas as pd

BASE_DIR = Path(r"C:\workspace_mimic_transformer_env\mimic-iv-clinical-database-demo-2.2")
HOSP_DIR = BASE_DIR / "hosp"
ICU_DIR = BASE_DIR / "icu"

print("=" * 80)
print("1. PATH CHECK")
print("=" * 80)

print("BASE_DIR exists:", BASE_DIR.exists())
print("HOSP_DIR exists:", HOSP_DIR.exists())
print("ICU_DIR exists:", ICU_DIR.exists())

print("\n" + "=" * 80)
print("2. LOAD CORE TABLES")
print("=" * 80)

patients = pd.read_csv(HOSP_DIR / "patients.csv")
admissions = pd.read_csv(HOSP_DIR / "admissions.csv")
icustays = pd.read_csv(ICU_DIR / "icustays.csv")
d_items = pd.read_csv(ICU_DIR / "d_items.csv")

print("patients shape:", patients.shape)
print("admissions shape:", admissions.shape)
print("icustays shape:", icustays.shape)
print("d_items shape:", d_items.shape)

print("\n" + "=" * 80)
print("3. CHARTEVENTS SAMPLE")
print("=" * 80)

chartevents_sample = pd.read_csv(ICU_DIR / "chartevents.csv", nrows=5000)

print("chartevents sample shape:", chartevents_sample.shape)
print(chartevents_sample.head())

print("\n" + "=" * 80)
print("4. MERGE TEST")
print("=" * 80)

chartevents_sample = chartevents_sample.merge(
    d_items[["itemid", "label"]],
    on="itemid",
    how="left"
)

print(chartevents_sample[["itemid", "label"]].head(20))

print("\nDone.")