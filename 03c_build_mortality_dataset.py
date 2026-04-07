from pathlib import Path
import pandas as pd
import numpy as np
import torch

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

print("=" * 80)
print("1. PATH SETUP")
print("=" * 80)

BASE_DIR = Path(r"C:\workspace_mimic_transformer_env")
MIMIC_DIR = BASE_DIR / "mimic-iv-clinical-database-demo-2.2"
HOSP_DIR = MIMIC_DIR / "hosp"
ICU_DIR = MIMIC_DIR / "icu"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ADMISSIONS_PATH = HOSP_DIR / "admissions.csv"
ICUSTAYS_PATH = ICU_DIR / "icustays.csv"
D_ITEMS_PATH = ICU_DIR / "d_items.csv"
CHARTEVENTS_PATH = ICU_DIR / "chartevents.csv"

print("ADMISSIONS_PATH exists :", ADMISSIONS_PATH.exists())
print("ICUSTAYS_PATH exists   :", ICUSTAYS_PATH.exists())
print("D_ITEMS_PATH exists    :", D_ITEMS_PATH.exists())
print("CHARTEVENTS_PATH exists:", CHARTEVENTS_PATH.exists())

print("\n" + "=" * 80)
print("2. LOAD TABLES")
print("=" * 80)

admissions = pd.read_csv(ADMISSIONS_PATH)
icustays = pd.read_csv(ICUSTAYS_PATH)
d_items = pd.read_csv(D_ITEMS_PATH)
chartevents = pd.read_csv(CHARTEVENTS_PATH)

print("admissions shape :", admissions.shape)
print("icustays shape   :", icustays.shape)
print("d_items shape    :", d_items.shape)
print("chartevents shape:", chartevents.shape)

print("\nAdmissions columns:")
print(admissions.columns.tolist())

print("\n" + "=" * 80)
print("3. BUILD MORTALITY LABEL")
print("=" * 80)

# admissions.csv 구조에 따라 label 생성
if "hospital_expire_flag" in admissions.columns:
    admissions["mortality_label"] = admissions["hospital_expire_flag"].fillna(0).astype(int)
elif "deathtime" in admissions.columns:
    admissions["mortality_label"] = admissions["deathtime"].notna().astype(int)
else:
    raise ValueError("admissions.csv 에서 사망 라벨 컬럼을 찾을 수 없습니다.")

label_df = admissions[["subject_id", "hadm_id", "mortality_label"]].drop_duplicates()

print(label_df["mortality_label"].value_counts(dropna=False))

print("\n" + "=" * 80)
print("4. MERGE LABEL TO ICU STAYS")
print("=" * 80)

stay_label_df = icustays.merge(
    label_df,
    on=["subject_id", "hadm_id"],
    how="left"
)

stay_label_df["mortality_label"] = stay_label_df["mortality_label"].fillna(0).astype(int)

print(stay_label_df[["subject_id", "hadm_id", "stay_id", "mortality_label"]].head())
print("\nMortality by stay:")
print(stay_label_df["mortality_label"].value_counts(dropna=False))

print("\n" + "=" * 80)
print("5. MERGE LABELS INTO CHARTEVENTS")
print("=" * 80)

chartevents = chartevents.merge(
    d_items[["itemid", "label"]].drop_duplicates(),
    on="itemid",
    how="left"
)

target_labels = [
    "Heart Rate",
    "Respiratory Rate",
    "O2 saturation pulseoxymetry",
    "Arterial Blood Pressure systolic",
    "Arterial Blood Pressure diastolic",
    "Arterial Blood Pressure mean",
    "Temperature Fahrenheit"
]

vital_events = chartevents[chartevents["label"].isin(target_labels)].copy()

vital_events["charttime"] = pd.to_datetime(vital_events["charttime"], errors="coerce")
vital_events["valuenum"] = pd.to_numeric(vital_events["valuenum"], errors="coerce")

vital_events = vital_events.dropna(subset=["stay_id", "charttime", "label", "valuenum"])
vital_events = vital_events.sort_values(["stay_id", "charttime", "label"]).reset_index(drop=True)

print("vital_events shape:", vital_events.shape)
print("available labels:", sorted(vital_events["label"].dropna().unique().tolist()))

print("\n" + "=" * 80)
print("6. BUILD STAY-LEVEL SEQUENCES")
print("=" * 80)

SEQ_LEN = 100

all_X = []
all_y = []
meta_rows = []

valid_stay_count = 0
skipped_stay_count = 0

stay_ids = sorted(vital_events["stay_id"].dropna().unique().tolist())
print("Total stay_ids in vital_events:", len(stay_ids))

for idx, stay_id in enumerate(stay_ids, start=1):
    stay_df = vital_events[vital_events["stay_id"] == stay_id].copy()

    wide_df = stay_df.pivot_table(
        index="charttime",
        columns="label",
        values="valuenum",
        aggfunc="mean"
    ).sort_index()

    for col in target_labels:
        if col not in wide_df.columns:
            wide_df[col] = np.nan

    wide_df = wide_df[target_labels]

    wide_df = wide_df.ffill()
    wide_df = wide_df.fillna(wide_df.mean())

    if wide_df.isna().sum().sum() > 0:
        skipped_stay_count += 1
        continue

    if len(wide_df) < SEQ_LEN:
        skipped_stay_count += 1
        continue

    # stay별 z-score normalization
    mean_vals = wide_df.mean()
    std_vals = wide_df.std().replace(0, 1.0)
    wide_norm = (wide_df - mean_vals) / std_vals

    arr = wide_norm.values.astype(np.float32)

    # 가장 단순한 방식: stay당 첫 100개 시점 사용
    X_seq = arr[:SEQ_LEN]

    label_row = stay_label_df[stay_label_df["stay_id"] == stay_id]
    if len(label_row) == 0:
        skipped_stay_count += 1
        continue

    mortality_label = int(label_row["mortality_label"].iloc[0])

    all_X.append(X_seq)
    all_y.append(mortality_label)

    meta_rows.append({
        "stay_id": stay_id,
        "subject_id": int(label_row["subject_id"].iloc[0]),
        "hadm_id": int(label_row["hadm_id"].iloc[0]),
        "mortality_label": mortality_label,
        "num_timepoints_used": SEQ_LEN,
        "original_num_rows": len(wide_df)
    })

    valid_stay_count += 1

    if idx % 10 == 0 or idx == len(stay_ids):
        print(f"[{idx}/{len(stay_ids)}] processed")

print("\n" + "=" * 80)
print("7. CONVERT TO TENSOR")
print("=" * 80)

X = np.array(all_X, dtype=np.float32)
y = np.array(all_y, dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("valid_stay_count :", valid_stay_count)
print("skipped_stay_count:", skipped_stay_count)

if len(y) > 0:
    unique, counts = np.unique(y, return_counts=True)
    print("label distribution:", dict(zip(unique.tolist(), counts.tolist())))

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

print("X_tensor shape:", X_tensor.shape)
print("y_tensor shape:", y_tensor.shape)

print("\n" + "=" * 80)
print("8. SAVE")
print("=" * 80)

torch.save(X_tensor, OUTPUT_DIR / "mortality_X_tensor.pt")
torch.save(y_tensor, OUTPUT_DIR / "mortality_y_tensor.pt")

meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(OUTPUT_DIR / "mortality_meta.csv", index=False, encoding="utf-8-sig")

summary_df = pd.DataFrame({
    "metric": [
        "num_total_stays_in_vitals",
        "valid_stay_count",
        "skipped_stay_count",
        "num_samples",
        "sequence_length",
        "num_features",
        "num_positive_labels",
        "num_negative_labels"
    ],
    "value": [
        len(stay_ids),
        valid_stay_count,
        skipped_stay_count,
        len(X),
        SEQ_LEN,
        len(target_labels),
        int((y == 1).sum()) if len(y) > 0 else 0,
        int((y == 0).sum()) if len(y) > 0 else 0
    ]
})
summary_df.to_csv(OUTPUT_DIR / "mortality_dataset_summary.csv", index=False, encoding="utf-8-sig")

print("Saved:", OUTPUT_DIR / "mortality_X_tensor.pt")
print("Saved:", OUTPUT_DIR / "mortality_y_tensor.pt")
print("Saved:", OUTPUT_DIR / "mortality_meta.csv")
print("Saved:", OUTPUT_DIR / "mortality_dataset_summary.csv")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)