import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
DATA_PATH = "data/processed/climate_feature_engineered.csv"
MODEL_PATH = "models"
MODEL_FILE = os.path.join(MODEL_PATH, "gru_model.pth")
PREDICTIONS_PATH = "data/processed"

# Create directories if they don't exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PREDICTIONS_PATH, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

SEED = 42
WINDOW = 30
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5

torch.manual_seed(SEED)
np.random.seed(SEED)

# ================= OPTIMIZED FEATURES =================
SELECTED_FEATURES = [
    "T2M", "RH2M", "PRECTOTCORR", "WS2M", "PS",
    "T2M_lag1", "T2M_lag3", "T2M_lag7",
    "T2M_roll7_mean", "T2M_roll30_mean",
    "City_Code"
]

# ================= HELPERS =================
def add_time_features(df):
    if "Month" in df.columns:
        df["Month_Sin"] = np.sin(2*np.pi*df["Month"]/12)
        df["Month_Cos"] = np.cos(2*np.pi*df["Month"]/12)
    if "Day_of_Year" in df.columns:
        df["Day_Sin"] = np.sin(2*np.pi*df["Day_of_Year"]/365.25)
        df["Day_Cos"] = np.cos(2*np.pi*df["Day_of_Year"]/365.25)
    return df

def get_features(df):
    cyc = [c for c in ["Month_Sin","Month_Cos","Day_Sin","Day_Cos"] if c in df.columns]
    return [c for c in SELECTED_FEATURES if c in df.columns] + cyc

def create_sequences(df, features, target, window):
    X, y, meta = [], [], []
    
    for city in df["City"].unique():
        sub = df[df["City"] == city].sort_values("Date")
        
        values = sub[features].values
        targets = sub[target].values
        
        for i in range(window, len(sub)):
            X.append(values[i-window:i])
            y.append(targets[i])
            meta.append({
                "Date": sub.iloc[i]["Date"],
                "City": city,
            })
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.DataFrame(meta)

# ================= LIGHTWEIGHT MODEL =================
class LightGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        out = self.head(last_out)
        return out.squeeze(-1)

# ================= OPTIMIZED TRAINING LOOP =================
def run_epoch(loader, model, optimizer, criterion, train=True):
    if train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    preds, targets = [], []
    
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        if train:
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(train):
            out = model(X)
            loss = criterion(out, y)
            
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        total_loss += loss.item() * len(X)
        
        if not train:
            preds.append(out.detach().cpu().numpy())
            targets.append(y.cpu().numpy())
    
    if train:
        return total_loss / len(loader.dataset), None, None
    else:
        return total_loss / len(loader.dataset), np.concatenate(preds), np.concatenate(targets)

# ================= DATA PREPARATION =================
print("="*60)
print("GRU MODEL TRAINING")
print("="*60)

print("\n1. Loading data...")
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["City", "Date"])

# Add time features
df = add_time_features(df)

# Directly predict T2M
features = get_features(df)
target = "T2M"

# Split data
train = df[df["Date"] <= "2020-12-31"].copy()
val = df[(df["Date"] > "2020-12-31") & (df["Date"] <= "2022-12-31")].copy()
test = df[df["Date"] > "2022-12-31"].copy()

print(f"   Train: {len(train)} rows ({train['Date'].min()} to {train['Date'].max()})")
print(f"   Val: {len(val)} rows ({val['Date'].min()} to {val['Date'].max()})")
print(f"   Test: {len(test)} rows ({test['Date'].min()} to {test['Date'].max()})")

# Scale features
NUM_FEATURES = [f for f in features if f != "City_Code"]
fs = StandardScaler()
ts = StandardScaler()

fs.fit(train[NUM_FEATURES])
ts.fit(train[[target]])

for d in [train, val, test]:
    d.loc[:, NUM_FEATURES] = fs.transform(d[NUM_FEATURES])
    d.loc[:, [target]] = ts.transform(d[[target]])

# Create sequences
print("\n2. Creating sequences...")
X_train, y_train, m_train = create_sequences(train, features, target, WINDOW)
X_val, y_val, m_val = create_sequences(val, features, target, WINDOW)
X_test, y_test, m_test = create_sequences(test, features, target, WINDOW)

print(f"   Train sequences: {len(X_train)}")
print(f"   Val sequences: {len(X_val)}")
print(f"   Test sequences: {len(X_test)}")
print(f"   Features: {len(features)}")
print(f"   Input shape: ({WINDOW}, {len(features)})")

# Create data loaders
train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())
test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)

# ================= TRAIN =================
print("\n3. Training GRU model...")
model = LightGRU(len(features)).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

best_rmse = float("inf")
patience = 0
best_model_state = None

print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LR}")
print("\nTraining progress:")

for epoch in range(EPOCHS):
    # Training
    train_loss, _, _ = run_epoch(train_loader, model, optimizer, criterion, True)
    
    # Validation
    val_loss, vp, vt = run_epoch(val_loader, model, optimizer, criterion, False)
    
    # Inverse transform
    vp = ts.inverse_transform(vp.reshape(-1, 1)).flatten()
    vt = ts.inverse_transform(vt.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(vt, vp))
    mae = mean_absolute_error(vt, vp)
    
    print(f"   Epoch {epoch+1:2d}/{EPOCHS} | Loss: {train_loss:.4f} | Val RMSE: {rmse:.4f}°C | Val MAE: {mae:.4f}°C", end="")
    
    # Learning rate scheduling
    scheduler.step(rmse)
    
    # Early stopping
    if rmse < best_rmse:
        best_rmse = rmse
        patience = 0
        best_model_state = model.state_dict().copy()
        torch.save(model.state_dict(), MODEL_FILE)
        print(" ✓ (best)")
    else:
        patience += 1
        print(f" | Patience: {patience}/{PATIENCE}")
        
        if patience >= PATIENCE:
            print(f"\n   Early stopping triggered at epoch {epoch+1}")
            break

# ================= EVALUATION =================
print("\n4. Evaluating best model...")
model.load_state_dict(best_model_state)
model.eval()

# Validation evaluation
_, vp, vt = run_epoch(val_loader, model, optimizer, criterion, False)
vp = ts.inverse_transform(vp.reshape(-1, 1)).flatten()
vt = ts.inverse_transform(vt.reshape(-1, 1)).flatten()

# Test evaluation
_, tp, tt = run_epoch(test_loader, model, optimizer, criterion, False)
tp = ts.inverse_transform(tp.reshape(-1, 1)).flatten()
tt = ts.inverse_transform(tt.reshape(-1, 1)).flatten()

# Calculate metrics
val_rmse = np.sqrt(mean_squared_error(vt, vp))
val_mae = mean_absolute_error(vt, vp)
test_rmse = np.sqrt(mean_squared_error(tt, tp))
test_mae = mean_absolute_error(tt, tp)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Validation RMSE: {val_rmse:.4f}°C")
print(f"Validation MAE: {val_mae:.4f}°C")
print(f"Test RMSE: {test_rmse:.4f}°C")
print(f"Test MAE: {test_mae:.4f}°C")
print("="*60)

# ================= SAVE PREDICTIONS =================
print("\n5. Saving predictions...")

val_df = pd.DataFrame({
    "Date": m_val["Date"].dt.strftime("%Y-%m-%d"),
    "City": m_val["City"],
    "y_true": vt,
    "gru_pred": vp,
})

test_df = pd.DataFrame({
    "Date": m_test["Date"].dt.strftime("%Y-%m-%d"),
    "City": m_test["City"],
    "y_true": tt,
    "gru_pred": tp,
})

val_df.to_csv(f"{PREDICTIONS_PATH}/gru_val_preds.csv", index=False)
test_df.to_csv(f"{PREDICTIONS_PATH}/gru_test_preds.csv", index=False)

# Save metrics
metrics_df = pd.DataFrame({
    'model': ['GRU'],
    'val_rmse': [val_rmse],
    'val_mae': [val_mae],
    'test_rmse': [test_rmse],
    'test_mae': [test_mae]
})
metrics_df.to_csv(f"{PREDICTIONS_PATH}/gru_metrics.csv", index=False)

print(f"   ✓ Validation predictions: {PREDICTIONS_PATH}/gru_val_preds.csv")
print(f"   ✓ Test predictions: {PREDICTIONS_PATH}/gru_test_preds.csv")
print(f"   ✓ Model saved: {MODEL_FILE}")

# Print sample predictions
print("\nSample predictions (first 10 test samples):")
sample_df = test_df.head(10)[['Date', 'City', 'y_true', 'gru_pred']]
sample_df['error'] = sample_df['y_true'] - sample_df['gru_pred']
print(sample_df.to_string(index=False))

print("\n" + "="*60)
print("✅ GRU training completed successfully!")
print("="*60)