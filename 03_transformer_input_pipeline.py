from pathlib import Path
import pandas as pd
import numpy as np
import torch

print("=" * 80)
print("1. LOAD TIMESERIES DATA")
print("=" * 80)

BASE_DIR = Path(r"C:\workspace_mimic_transformer_env")
DATA_PATH = BASE_DIR / "outputs" / "one_stay_timeseries_wide.csv"

df = pd.read_csv(DATA_PATH)

print("Loaded shape:", df.shape)
print(df.head())

# charttime index 복원
df["charttime"] = pd.to_datetime(df["charttime"])
df = df.set_index("charttime").sort_index()

print("\nAfter setting index:")
print(df.head())

print("=" * 80)
print("2. SELECT TRUE VITAL SIGNS ONLY")
print("=" * 80)

# 진짜 vital만 선택 (핵심)
selected_columns = [
    "Heart Rate",
    "Respiratory Rate",
    "O2 saturation pulseoxymetry",
    "Arterial Blood Pressure systolic",
    "Arterial Blood Pressure diastolic",
    "Arterial Blood Pressure mean",
    "Temperature Fahrenheit"
]

# 존재하는 컬럼만 선택
selected_columns = [col for col in selected_columns if col in df.columns]

df = df[selected_columns].copy()

print("Selected columns:", selected_columns)
print("Shape after selection:", df.shape)

print("=" * 80)
print("3. HANDLE MISSING VALUES")
print("=" * 80)

# forward fill → 이전 값으로 채움
df = df.fillna(method="ffill")

# 아직 남은 NaN은 평균값으로 채움
df = df.fillna(df.mean())

print("NaN remaining:", df.isna().sum().sum())

print("=" * 80)
print("4. NORMALIZATION")
print("=" * 80)

# z-score normalization
df_norm = (df - df.mean()) / df.std()

print(df_norm.head())

print("=" * 80)
print("5. CONVERT TO NUMPY")
print("=" * 80)

data_array = df_norm.values

print("Array shape:", data_array.shape)

print("=" * 80)
print("6. CREATE SEQUENCE DATA")
print("=" * 80)

SEQ_LEN = 100  # Transformer 입력 길이

sequences = []

for i in range(len(data_array) - SEQ_LEN):
    seq = data_array[i:i+SEQ_LEN]
    sequences.append(seq)

sequences = np.array(sequences)

print("Sequence shape:", sequences.shape)

print("=" * 80)
print("7. CONVERT TO TORCH TENSOR")
print("=" * 80)

tensor_data = torch.tensor(sequences, dtype=torch.float32)

print("Tensor shape:", tensor_data.shape)

print("=" * 80)
print("8. SAVE TENSOR")
print("=" * 80)

OUTPUT_PATH = BASE_DIR / "outputs" / "transformer_input_tensor.pt"

torch.save(tensor_data, OUTPUT_PATH)

print("Saved tensor:", OUTPUT_PATH)

print("=" * 80)
print("DONE")
print("=" * 80)