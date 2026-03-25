"""
FIXED GRU MODEL - Direct Temperature Prediction
No residual prediction, no baseline addition
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
DATA_PATH = "data/processed/climate_feature_engineered.csv"
MODEL_PATH = "models"
MODEL_FILE = os.path.join(MODEL_PATH, "gru_model_fixed.pth")
PREDICTIONS_PATH = "data/processed"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

SEED = 42
WINDOW = 30
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 15
LR = 0.001
WEIGHT_DECAY = 1e-5

torch.manual_seed(SEED)
np.random.seed(SEED)

# ================= FEATURES =================
SELECTED_FEATURES = [
    "T2M", "RH2M", "PRECTOTCORR", "WS2M", "PS",
    "T2M_lag1", "T2M_lag3", "T2M_lag7",
    "RH2M_lag1", "RH2M_lag3", "RH2M_lag7",
    "PRECTOTCORR_lag1", "PRECTOTCORR_lag3", "PRECTOTCORR_lag7",
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

# ================= GRU MODEL =================
class DirectGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        out = self.head(last_out)
        return out.squeeze(-1)

# ================= TRAINING =================
def train_epoch(loader, model, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(X)
    
    return total_loss / len(loader.dataset)

def validate_epoch(loader, model, criterion, target_scaler):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * len(X)
            
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Concatenate
    preds_scaled = np.concatenate(all_preds)
    targets_scaled = np.concatenate(all_targets)
    
    # Inverse transform to get actual temperatures
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    targets = target_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    
    return total_loss / len(loader.dataset), preds, targets, rmse, mae

# ================= MAIN =================
print("\n" + "="*60)
print("FIXED GRU - DIRECT TEMPERATURE PREDICTION")
print("="*60)

# Load data
print("\n1. Loading data...")
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["City", "Date"])

# Add time features
df = add_time_features(df)

# IMPORTANT: Predict T2M directly
features = get_features(df)
target = "T2M"  # Direct temperature prediction

print(f"Features: {len(features)}")
print(f"Target: {target}")

# Split data
train = df[df["Date"] <= "2020-12-31"].copy()
val = df[(df["Date"] > "2020-12-31") & (df["Date"] <= "2022-12-31")].copy()
test = df[df["Date"] > "2022-12-31"].copy()

print(f"Train: {len(train)} samples")
print(f"Val: {len(val)} samples")
print(f"Test: {len(test)} samples")

# Scale features and target
NUM_FEATURES = [f for f in features if f != "City_Code"]
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

# Fit scalers on training data only
feature_scaler.fit(train[NUM_FEATURES])
target_scaler.fit(train[[target]])

# Transform all datasets
for d in [train, val, test]:
    d.loc[:, NUM_FEATURES] = feature_scaler.transform(d[NUM_FEATURES])
    d.loc[:, [target]] = target_scaler.transform(d[[target]])

# Create sequences
print("\n2. Creating sequences...")
X_train, y_train, meta_train = create_sequences(train, features, target, WINDOW)
X_val, y_val, meta_val = create_sequences(val, features, target, WINDOW)
X_test, y_test, meta_test = create_sequences(test, features, target, WINDOW)

print(f"Train sequences: {len(X_train)}")
print(f"Val sequences: {len(X_val)}")
print(f"Test sequences: {len(X_test)}")

# Create data loaders
train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())
test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Train model
print("\n3. Training GRU model...")
model = DirectGRU(len(features)).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

best_rmse = float('inf')
best_model_state = None
patience_counter = 0

for epoch in range(EPOCHS):
    # Train
    train_loss = train_epoch(train_loader, model, optimizer, criterion)
    
    # Validate
    val_loss, val_preds, val_targets, val_rmse, val_mae = validate_epoch(
        val_loader, model, criterion, target_scaler
    )
    
    # Learning rate scheduling
    scheduler.step(val_rmse)
    
    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val RMSE: {val_rmse:.4f}°C | Val MAE: {val_mae:.4f}°C")
    
    # Early stopping
    if val_rmse < best_rmse:
        best_rmse = val_rmse
        # state_dict() is a dict of Tensors; use a deep copy so it doesn't drift
        # as training continues.
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        print(f"  ✓ New best model! RMSE: {val_rmse:.4f}°C")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

# Load best model
model.load_state_dict(best_model_state)

# Evaluate on validation
print("\n4. Evaluating on validation set...")
_, val_preds, val_targets, val_rmse, val_mae = validate_epoch(
    val_loader, model, criterion, target_scaler
)

# Evaluate on test
print("\n5. Evaluating on test set...")
_, test_preds, test_targets, test_rmse, test_mae = validate_epoch(
    test_loader, model, criterion, target_scaler
)

print("\n" + "="*60)

# -----------------------------------------------------------------------------
# Robust post-processing: ensure saved predictions are in Celsius
# -----------------------------------------------------------------------------
def maybe_unscale(values):
    """
    If `values` look like standardized z-scores (common range ~[-6, 6]) and the
    target scaler mean is far from 0, convert them back to Celsius.
    """
    values = np.asarray(values)
    if np.max(np.abs(values)) <= 10 and target_scaler.mean_[0] > 10:
        return target_scaler.inverse_transform(values.reshape(-1, 1)).flatten()
    return values

val_preds = maybe_unscale(val_preds)
val_targets = maybe_unscale(val_targets)
test_preds = maybe_unscale(test_preds)
test_targets = maybe_unscale(test_targets)

# Recompute metrics on the unscaled (Celsius) values
val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
val_mae = mean_absolute_error(val_targets, val_preds)
test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
test_mae = mean_absolute_error(test_targets, test_preds)

print("FINAL RESULTS")
print("="*60)
print(f"Validation RMSE: {val_rmse:.4f}°C")
print(f"Validation MAE: {val_mae:.4f}°C")
print(f"Test RMSE: {test_rmse:.4f}°C")
print(f"Test MAE: {test_mae:.4f}°C")

# Save predictions
print("\n6. Saving predictions...")

# Validation predictions
val_df = pd.DataFrame({
    "Date": meta_val["Date"].dt.strftime("%Y-%m-%d"),
    "City": meta_val["City"],
    "y_true": val_targets,
    "gru_pred": val_preds
})
val_df.to_csv(f"{PREDICTIONS_PATH}/gru_val_preds_fixed.csv", index=False)

# Test predictions
test_df = pd.DataFrame({
    "Date": meta_test["Date"].dt.strftime("%Y-%m-%d"),
    "City": meta_test["City"],
    "y_true": test_targets,
    "gru_pred": test_preds
})
test_df.to_csv(f"{PREDICTIONS_PATH}/gru_test_preds_fixed.csv", index=False)

# Save metrics
metrics_df = pd.DataFrame({
    'model': ['GRU_Fixed'],
    'val_rmse': [val_rmse],
    'val_mae': [val_mae],
    'test_rmse': [test_rmse],
    'test_mae': [test_mae]
})
metrics_df.to_csv(f"{PREDICTIONS_PATH}/gru_metrics_fixed.csv", index=False)

# Save model
torch.save(model.state_dict(), MODEL_FILE)

print(f"\n✓ Model saved to {MODEL_FILE}")
print(f"✓ Validation predictions saved to {PREDICTIONS_PATH}/gru_val_preds_fixed.csv")
print(f"✓ Test predictions saved to {PREDICTIONS_PATH}/gru_test_preds_fixed.csv")

# Show sample predictions
print("\n7. Sample predictions (first 10 test samples):")
sample = test_df.head(10)[['Date', 'City', 'y_true', 'gru_pred']].copy()
sample['error'] = sample['y_true'] - sample['gru_pred']
print(sample.to_string(index=False))

# Check if predictions are in correct range
print(f"\nPrediction range check:")
print(f"  y_true range: [{test_df['y_true'].min():.2f}, {test_df['y_true'].max():.2f}]")
print(f"  gru_pred range: [{test_df['gru_pred'].min():.2f}, {test_df['gru_pred'].max():.2f}]")

if test_df['gru_pred'].min() < 0:
    print("\n⚠️ WARNING: GRU predictions still have negative values!")
    print("   This suggests the model is still predicting residuals.")
else:
    print("\n✓ GRU predictions are in correct temperature range!")

print("\n" + "="*60)
print("✅ GRU training complete!")
print("="*60)