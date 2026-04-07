from pathlib import Path
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

print("=" * 80)
print("1. LOAD MORTALITY DATA")
print("=" * 80)

BASE_DIR = Path(r"C:\workspace_mimic_transformer_env")
OUTPUT_DIR = BASE_DIR / "outputs"

device = torch.device("cpu")
print("Device:", device)

X_all = torch.load(OUTPUT_DIR / "mortality_X_tensor.pt")
y_all = torch.load(OUTPUT_DIR / "mortality_y_tensor.pt")

print("X_all shape:", X_all.shape)
print("y_all shape:", y_all.shape)

print("\nLabel distribution:")
unique_vals, counts = torch.unique(y_all, return_counts=True)
for u, c in zip(unique_vals.tolist(), counts.tolist()):
    print(f"Label {int(u)}: {c}")

print("\n" + "=" * 80)
print("2. DATASET")
print("=" * 80)

class MortalityDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = MortalityDataset(X_all, y_all)

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

class MortalityTransformer(nn.Module):
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

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x[:, -1]
        logits = self.classifier(x).squeeze(1)
        return logits

model = MortalityTransformer(input_dim=X_all.shape[-1]).to(device)
print(model)

print("\n" + "=" * 80)
print("5. LOSS / OPTIMIZER")
print("=" * 80)

# class imbalance 대응
num_pos = (y_all == 1).sum().item()
num_neg = (y_all == 0).sum().item()

if num_pos == 0:
    raise ValueError("사망 label(1)이 0개입니다. demo 데이터에서 양성 샘플이 없는지 확인이 필요합니다.")

pos_weight_value = num_neg / max(num_pos, 1)
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

print("num_pos:", num_pos)
print("num_neg:", num_neg)
print("pos_weight:", pos_weight_value)

print("\n" + "=" * 80)
print("6. METRIC FUNCTION")
print("=" * 80)

def compute_binary_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

print("\n" + "=" * 80)
print("7. TRAIN LOOP")
print("=" * 80)

EPOCHS = 50
PATIENCE = 5

best_val_loss = float("inf")
patience_counter = 0

history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * X_batch.size(0)

    train_loss = train_loss_sum / len(train_loader.dataset)

    model.eval()
    val_loss_sum = 0.0

    all_val_logits = []
    all_val_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            val_loss_sum += loss.item() * X_batch.size(0)

            all_val_logits.append(logits.cpu())
            all_val_targets.append(y_batch.cpu())

    val_loss = val_loss_sum / len(val_loader.dataset)

    all_val_logits = torch.cat(all_val_logits)
    all_val_targets = torch.cat(all_val_targets)

    metrics = compute_binary_metrics(all_val_logits, all_val_targets, threshold=0.5)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss {train_loss:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {metrics['acc']:.4f} | "
        f"Val F1 {metrics['f1']:.4f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": metrics["acc"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_f1": metrics["f1"],
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), OUTPUT_DIR / "best_mortality_transformer.pt")
        print("  -> saved best model")
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("Early stopping triggered")
        break

print("\n" + "=" * 80)
print("8. SAVE HISTORY")
print("=" * 80)

import pandas as pd
history_df = pd.DataFrame(history)
history_df.to_csv(OUTPUT_DIR / "mortality_training_history.csv", index=False, encoding="utf-8-sig")

print("Saved:", OUTPUT_DIR / "best_mortality_transformer.pt")
print("Saved:", OUTPUT_DIR / "mortality_training_history.csv")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)