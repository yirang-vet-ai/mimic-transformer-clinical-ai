from pathlib import Path
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

print("=" * 80)
print("1. LOAD TENSOR")
print("=" * 80)

BASE_DIR = Path(r"C:\workspace_mimic_transformer_env")
INPUT_PATH = BASE_DIR / "outputs" / "transformer_input_tensor.pt"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu") #("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data = torch.load(INPUT_PATH)
print("Loaded tensor shape:", data.shape)   # expected: (N, 100, 7)

# 입력 X: 처음 99개 시점
# 정답 y: 마지막 1개 시점
X_all = data[:, :-1, :]   # (N, 99, 7)
y_all = data[:, -1, :]    # (N, 7)

print("X_all shape:", X_all.shape)
print("y_all shape:", y_all.shape)


print("\n" + "=" * 80)
print("2. DATASET")
print("=" * 80)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TimeSeriesDataset(X_all, y_all)

total_size = len(dataset)
train_size = int(total_size * 0.8)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Total dataset size:", total_size)
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))


print("\n" + "=" * 80)
print("3. POSITIONAL ENCODING")
print("=" * 80)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


print("\n" + "=" * 80)
print("4. TRANSFORMER MODEL")
print("=" * 80)

class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim=7,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        output_dim=7
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)              # (batch, seq_len, d_model)
        x = self.pos_encoder(x)             # positional encoding 추가
        x = self.transformer_encoder(x)     # (batch, seq_len, d_model)

        x_last = x[:, -1, :]                # 마지막 시점 representation
        out = self.regressor(x_last)        # (batch, output_dim)
        return out

model = TransformerRegressor(
    input_dim=X_all.shape[-1],
    output_dim=y_all.shape[-1]
).to(device)

print(model)


print("\n" + "=" * 80)
print("5. LOSS / OPTIMIZER")
print("=" * 80)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


print("\n" + "=" * 80)
print("6. TRAINING LOOP")
print("=" * 80)

EPOCHS = 30
best_val_loss = float("inf")

train_loss_history = []
val_loss_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_sum = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * X_batch.size(0)

    train_epoch_loss = train_loss_sum / len(train_loader.dataset)
    train_loss_history.append(train_epoch_loss)

    model.eval()
    val_loss_sum = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            val_loss_sum += loss.item() * X_batch.size(0)

    val_epoch_loss = val_loss_sum / len(val_loader.dataset)
    val_loss_history.append(val_epoch_loss)

    print(f"Epoch {epoch:02d} | Train Loss: {train_epoch_loss:.6f} | Val Loss: {val_epoch_loss:.6f}")

    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(), OUTPUT_DIR / "best_transformer_model.pt")
        print(f"  -> Best model saved. Val Loss: {best_val_loss:.6f}")


print("\n" + "=" * 80)
print("7. FINAL EVALUATION SAMPLE")
print("=" * 80)

model.load_state_dict(torch.load(OUTPUT_DIR / "best_transformer_model.pt", map_location=device))
model.eval()

with torch.no_grad():
    sample_X, sample_y = dataset[0]
    sample_X = sample_X.unsqueeze(0).to(device)   # (1, seq_len, input_dim)
    pred_y = model(sample_X).squeeze(0).cpu()

print("Sample target:")
print(sample_y)

print("\nSample prediction:")
print(pred_y)


print("\n" + "=" * 80)
print("8. SAVE LOSS HISTORY")
print("=" * 80)

loss_txt_path = OUTPUT_DIR / "transformer_training_log.txt"
with open(loss_txt_path, "w", encoding="utf-8") as f:
    for i, (tr, va) in enumerate(zip(train_loss_history, val_loss_history), start=1):
        f.write(f"Epoch {i:02d} | Train Loss: {tr:.6f} | Val Loss: {va:.6f}\n")

print("Saved training log:", loss_txt_path)
print("Saved best model   :", OUTPUT_DIR / "best_transformer_model.pt")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)