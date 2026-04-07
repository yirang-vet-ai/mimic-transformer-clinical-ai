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
ICU_DIR = MIMIC_DIR / "icu"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHARTEVENTS_PATH = ICU_DIR / "chartevents.csv"
ICUSTAYS_PATH = ICU_DIR / "icustays.csv"
D_ITEMS_PATH = ICU_DIR / "d_items.csv"

print("CHARTEVENTS_PATH exists:", CHARTEVENTS_PATH.exists())
print("ICUSTAYS_PATH exists   :", ICUSTAYS_PATH.exists())
print("D_ITEMS_PATH exists    :", D_ITEMS_PATH.exists())

print("\n" + "=" * 80)
print("2. LOAD CORE TABLES")
print("=" * 80)

icustays = pd.read_csv(ICUSTAYS_PATH)
d_items = pd.read_csv(D_ITEMS_PATH)
chartevents = pd.read_csv(CHARTEVENTS_PATH)

print("icustays shape   :", icustays.shape)
print("d_items shape    :", d_items.shape)
print("chartevents shape:", chartevents.shape)

print("\n" + "=" * 80)
print("3. MERGE LABELS")
print("=" * 80)

use_item_cols = ["itemid", "label"]
chartevents = chartevents.merge(
    d_items[use_item_cols].drop_duplicates(),
    on="itemid",
    how="left"
)

print("Merged chartevents shape:", chartevents.shape)

print("\n" + "=" * 80)
print("4. KEEP ONLY TRUE VITAL SIGNS")
print("=" * 80)

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

print("vital_events shape:", vital_events.shape)
print("Available labels:")
print(sorted(vital_events["label"].dropna().unique().tolist()))

print("\n" + "=" * 80)
print("5. BASIC CLEANING")
print("=" * 80)

vital_events["charttime"] = pd.to_datetime(vital_events["charttime"], errors="coerce")
vital_events["valuenum"] = pd.to_numeric(vital_events["valuenum"], errors="coerce")

vital_events = vital_events.dropna(subset=["stay_id", "charttime", "label", "valuenum"])
vital_events = vital_events.sort_values(["stay_id", "charttime", "label"]).reset_index(drop=True)

print("After cleaning:", vital_events.shape)

print("\n" + "=" * 80)
print("6. BUILD WIDE TABLE PER STAY")
print("=" * 80)

SEQ_LEN = 100
MIN_ROWS_PER_STAY = 101   # X: 100, y: 1 이상 확보용

all_sequences = []
all_targets = []
valid_stay_count = 0
skipped_stay_count = 0

stay_ids = sorted(vital_events["stay_id"].dropna().unique().tolist())
print("Total stay_ids in filtered data:", len(stay_ids))

for idx, stay_id in enumerate(stay_ids, start=1):
    stay_df = vital_events[vital_events["stay_id"] == stay_id].copy()

    wide_df = stay_df.pivot_table(
        index="charttime",
        columns="label",
        values="valuenum",
        aggfunc="mean"
    ).sort_index()

    # 필요한 컬럼 순서 고정
    for col in target_labels:
        if col not in wide_df.columns:
            wide_df[col] = np.nan

    wide_df = wide_df[target_labels]

    # 결측값 처리
    wide_df = wide_df.ffill()
    wide_df = wide_df.fillna(wide_df.mean())

    # 여전히 NaN이면 전체 stay 제외
    if wide_df.isna().sum().sum() > 0:
        skipped_stay_count += 1
        continue

    # 너무 짧은 stay 제외
    if len(wide_df) < MIN_ROWS_PER_STAY:
        skipped_stay_count += 1
        continue

    # 정규화: stay별 z-score
    std = wide_df.std()
    std_replaced = std.replace(0, 1.0)
    wide_norm = (wide_df - wide_df.mean()) / std_replaced

    arr = wide_norm.values.astype(np.float32)

    # sequence 생성
    for i in range(len(arr) - SEQ_LEN):
        seq_x = arr[i:i+SEQ_LEN]
        target_y = arr[i+SEQ_LEN]
        all_sequences.append(seq_x)
        all_targets.append(target_y)

    valid_stay_count += 1

    if idx % 10 == 0 or idx == len(stay_ids):
        print(f"[{idx}/{len(stay_ids)}] processed")

print("\n" + "=" * 80)
print("7. CONVERT TO TENSOR")
print("=" * 80)

X = np.array(all_sequences, dtype=np.float32)
y = np.array(all_targets, dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("valid_stay_count :", valid_stay_count)
print("skipped_stay_count:", skipped_stay_count)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

print("X_tensor shape:", X_tensor.shape)
print("y_tensor shape:", y_tensor.shape)

print("\n" + "=" * 80)
print("8. SAVE")
print("=" * 80)

torch.save(X_tensor, OUTPUT_DIR / "multistay_X_tensor.pt")
torch.save(y_tensor, OUTPUT_DIR / "multistay_y_tensor.pt")

summary_df = pd.DataFrame({
    "metric": [
        "total_filtered_stays",
        "valid_stay_count",
        "skipped_stay_count",
        "num_samples",
        "sequence_length",
        "num_features"
    ],
    "value": [
        len(stay_ids),
        valid_stay_count,
        skipped_stay_count,
        len(X),
        SEQ_LEN,
        len(target_labels)
    ]
})
summary_df.to_csv(OUTPUT_DIR / "multistay_sequence_summary.csv", index=False, encoding="utf-8-sig")

print("Saved:", OUTPUT_DIR / "multistay_X_tensor.pt")
print("Saved:", OUTPUT_DIR / "multistay_y_tensor.pt")
print("Saved:", OUTPUT_DIR / "multistay_sequence_summary.csv")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)