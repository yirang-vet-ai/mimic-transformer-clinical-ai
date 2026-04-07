from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)

BASE_DIR = Path(r"C:\workspace_mimic_transformer_env\mimic-iv-clinical-database-demo-2.2")
HOSP_DIR = BASE_DIR / "hosp"
ICU_DIR = BASE_DIR / "icu"
OUTPUT_DIR = Path(r"C:\workspace_mimic_transformer_env\outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("1. LOAD CORE FILES")
print("=" * 80)

icustays = pd.read_csv(ICU_DIR / "icustays.csv")
d_items = pd.read_csv(ICU_DIR / "d_items.csv")

print("icustays shape :", icustays.shape)
print("d_items shape  :", d_items.shape)

print("\n[icustays columns]")
print(icustays.columns.tolist())

print("\n[d_items columns]")
print(d_items.columns.tolist())

print("\n" + "=" * 80)
print("2. SELECT ONE ICU STAY")
print("=" * 80)

print("\n[icustays head]")
print(icustays.head())

target_row = icustays.iloc[0].copy()

target_subject_id = target_row["subject_id"]
target_hadm_id = target_row["hadm_id"]
target_stay_id = target_row["stay_id"]

print("\nSelected ICU stay")
print("subject_id :", target_subject_id)
print("hadm_id    :", target_hadm_id)
print("stay_id    :", target_stay_id)

print("\n" + "=" * 80)
print("3. LOAD CHARTEVENTS")
print("=" * 80)

chartevents = pd.read_csv(ICU_DIR / "chartevents.csv")

print("chartevents shape:", chartevents.shape)
print("\n[chartevents columns]")
print(chartevents.columns.tolist())

print("\n" + "=" * 80)
print("4. FILTER ONLY THE SELECTED STAY")
print("=" * 80)

stay_events = chartevents[chartevents["stay_id"] == target_stay_id].copy()

print("stay_events shape:", stay_events.shape)
print("\n[stay_events head]")
print(stay_events.head())

print("\n" + "=" * 80)
print("5. MERGE WITH D_ITEMS")
print("=" * 80)

merge_cols = ["itemid"]
for col in ["label", "abbreviation", "category", "unitname", "linksto"]:
    if col in d_items.columns:
        merge_cols.append(col)

stay_events = stay_events.merge(
    d_items[merge_cols].drop_duplicates(),
    on="itemid",
    how="left"
)

print("merged stay_events shape:", stay_events.shape)

preview_cols = [
    col for col in [
        "subject_id", "hadm_id", "stay_id", "charttime",
        "itemid", "label", "category", "value", "valuenum", "unitname"
    ]
    if col in stay_events.columns
]

print("\n[merged stay_events head(20)]")
print(stay_events[preview_cols].head(20))

print("\n" + "=" * 80)
print("6. CONVERT TIME COLUMN")
print("=" * 80)

stay_events["charttime"] = pd.to_datetime(stay_events["charttime"], errors="coerce")
stay_events = stay_events.sort_values("charttime").reset_index(drop=True)

print(stay_events[preview_cols].head(20))

print("\n" + "=" * 80)
print("7. CHECK AVAILABLE LABELS IN THIS STAY")
print("=" * 80)

label_counts = (
    stay_events["label"]
    .fillna("UNKNOWN")
    .value_counts()
    .reset_index()
)
label_counts.columns = ["label", "count"]

print(label_counts.head(50))

label_counts.to_csv(OUTPUT_DIR / "one_stay_label_counts.csv", index=False, encoding="utf-8-sig")

print("\nSaved:", OUTPUT_DIR / "one_stay_label_counts.csv")

print("\n" + "=" * 80)
print("8. KEEP VITAL SIGN CANDIDATES")
print("=" * 80)

vital_keywords = [
    "Heart Rate",
    "Respiratory Rate",
    "SpO2",
    "O2 saturation",
    "Blood Pressure",
    "Arterial Pressure",
    "NBP",
    "Temperature",
    "Pulse"
]

pattern = "|".join(vital_keywords)

vital_events = stay_events[
    stay_events["label"].astype(str).str.contains(pattern, case=False, na=False)
].copy()

vital_events = vital_events.sort_values(["charttime", "label"]).reset_index(drop=True)

vital_preview_cols = [
    col for col in [
        "subject_id", "hadm_id", "stay_id", "charttime",
        "itemid", "label", "value", "valuenum", "unitname"
    ]
    if col in vital_events.columns
]

print("vital_events shape:", vital_events.shape)
print("\n[vital_events head(50)]")
print(vital_events[vital_preview_cols].head(50))

vital_events.to_csv(OUTPUT_DIR / "one_stay_vital_events_raw.csv", index=False, encoding="utf-8-sig")
print("\nSaved:", OUTPUT_DIR / "one_stay_vital_events_raw.csv")

print("\n" + "=" * 80)
print("9. MAKE A SIMPLE TIME-SERIES TABLE")
print("=" * 80)

vital_numeric = vital_events.copy()
vital_numeric["valuenum"] = pd.to_numeric(vital_numeric["valuenum"], errors="coerce")
vital_numeric = vital_numeric.dropna(subset=["charttime", "label", "valuenum"])

timeseries_wide = vital_numeric.pivot_table(
    index="charttime",
    columns="label",
    values="valuenum",
    aggfunc="mean"
).sort_index()

print("timeseries_wide shape:", timeseries_wide.shape)
print("\n[timeseries_wide head(20)]")
print(timeseries_wide.head(20))

timeseries_wide.to_csv(OUTPUT_DIR / "one_stay_timeseries_wide.csv", encoding="utf-8-sig")
print("\nSaved:", OUTPUT_DIR / "one_stay_timeseries_wide.csv")

print("\n" + "=" * 80)
print("10. SUMMARY")
print("=" * 80)
print("Selected stay_id:", target_stay_id)
print("Total events in selected stay:", len(stay_events))
print("Vital-sign events:", len(vital_events))
print("Time-series rows:", len(timeseries_wide))
print("Time-series columns:", len(timeseries_wide.columns))
print("\nDone.")