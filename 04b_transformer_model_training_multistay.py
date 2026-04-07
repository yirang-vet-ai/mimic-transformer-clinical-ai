from pathlib import Path
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

print("=" * 80)
print("1. LOAD MULTISTAY DATA")
print("=" * 80)

BASE_DIR = Path(r"C:\workspace_mimic_transformer_env")
OUTPUT_DIR = BASE_DIR / "outputs"

device = torch.device("cpu")
print("Device:", device)

X_all = torch.load(OUTPUT_DIR / "multistay_X_tensor.pt")
y_all = torch.load(OUTPUT_DIR / "multistay_y_tensor.pt")

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

train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Train:", len(train_dataset))
print("Val  :", len(val_dataset))

print("\n" + "=" * 80)
print("3. POSITIONAL ENCODING")
print("=" * 80)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

print("\n" + "=" * 80)
print("4. MODEL")
print("=" * 80)

class SmallTransformer(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()

        d_model = 32

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=64,
            dropout=0.3,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x[:, -1]
        return self.fc(x)

model = SmallTransformer(input_dim=X_all.shape[-1]).to(device)
print(model)

print("\n" + "=" * 80)
print("5. TRAIN SETUP")
print("=" * 80)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

print("\n" + "=" * 80)
print("6. TRAIN LOOP")
print("=" * 80)

EPOCHS = 50
PATIENCE = 5

best_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1:02d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model_multistay.pt")
        print("  -> saved best model")
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("Early stopping triggered")
        break

print("\n" + "=" * 80)
print("7. SAVE TRAINING LOG")
print("=" * 80)

with open(OUTPUT_DIR / "multistay_training_done.txt", "w", encoding="utf-8") as f:
    f.write(f"Best val loss: {best_loss:.6f}\n")

print("Saved:", OUTPUT_DIR / "best_model_multistay.pt")
print("Saved:", OUTPUT_DIR / "multistay_training_done.txt")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)