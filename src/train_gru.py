import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ================= CONFIG =================
DATA_PATH = "data/processed/climate_feature_engineered.csv"
MODEL_PATH = "models"
MODEL_FILE = os.path.join(MODEL_PATH, "gru_model.pth")
PREDICTIONS_PATH = "data/processed"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
WINDOW = 30
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5

torch.manual_seed(SEED)
np.random.seed(SEED)

# ================= FEATURES =================
SELECTED_FEATURES = [
    "T2M", "RH2M", "PRECTOTCORR", "WS2M", "PS",
    "T2M_lag1", "T2M_lag3", "T2M_lag7",
    "RH2M_lag1", "RH2M_lag3", "RH2M_lag7",
    "PRECTOTCORR_lag1", "PRECTOTCORR_lag3", "PRECTOTCORR_lag7",
    "T2M_roll7_mean", "T2M_roll30_mean", "T2M_roll7_std",
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
        baseline = sub["Baseline_T2M"].values

        for i in range(window, len(sub)):
            X.append(values[i-window:i])
            y.append(targets[i])
            meta.append({
                "Date": sub.iloc[i]["Date"],
                "City": city,
                "baseline": baseline[i]
            })

    return np.array(X), np.array(y), pd.DataFrame(meta)

# ================= MODEL =================
class GRUModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)

# ================= TRAIN LOOP =================
def run_epoch(loader, model, optimizer, criterion, train=True):
    model.train() if train else model.eval()

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
        preds.append(out.detach().cpu().numpy())
        targets.append(y.cpu().numpy())

    return total_loss/len(loader.dataset), np.concatenate(preds), np.concatenate(targets)

# ================= DATA =================
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["City","Date"])

df = add_time_features(df)

# 🔥 RESIDUAL TARGET
df["Residual"] = df["Target_T2M"] - df["T2M"]

features = get_features(df)
NUM_FEATURES = [f for f in features if f != "City_Code"]
target = "Residual"

train = df[df["Date"] <= "2020-12-31"].copy()
val = df[(df["Date"] > "2020-12-31") & (df["Date"] <= "2022-12-31")].copy()
test = df[df["Date"] > "2022-12-31"].copy()

for d in [train, val, test]:
    d["Baseline_T2M"] = d["T2M"].values

# scaling
fs = StandardScaler()
ts = StandardScaler()

fs.fit(train[NUM_FEATURES])
ts.fit(train[[target]])

for d in [train, val, test]:
    d.loc[:, NUM_FEATURES] = fs.transform(d[NUM_FEATURES])
    d.loc[:, [target]] = ts.transform(d[[target]])

X_train, y_train, _ = create_sequences(train, features, target, WINDOW)
X_val, y_val, m_val = create_sequences(val, features, target, WINDOW)
X_test, y_test, m_test = create_sequences(test, features, target, WINDOW)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float()), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float()), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float()), batch_size=BATCH_SIZE)

# ================= TRAIN =================
model = GRUModel(len(features)).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_rmse = float("inf")
patience = 0

for epoch in range(EPOCHS):
    train_loss, _, _ = run_epoch(train_loader, model, optimizer, criterion, True)
    val_loss, vp, vt = run_epoch(val_loader, model, optimizer, criterion, False)

    # inverse scale
    vp = ts.inverse_transform(vp.reshape(-1,1)).flatten()
    vt = ts.inverse_transform(vt.reshape(-1,1)).flatten()

    # add baseline back
    vp = vp + m_val["baseline"].values

    rmse = np.sqrt(mean_squared_error(vt + m_val["baseline"].values, vp))

    print(f"Epoch {epoch+1} | RMSE {rmse:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        patience = 0
        torch.save(model.state_dict(), MODEL_FILE)
    else:
        patience += 1
        if patience >= PATIENCE:
            print("Early stopping")
            break

# ================= EVAL =================
model.load_state_dict(torch.load(MODEL_FILE))

_, vp, vt = run_epoch(val_loader, model, optimizer, criterion, False)
_, tp, tt = run_epoch(test_loader, model, optimizer, criterion, False)

vp = ts.inverse_transform(vp.reshape(-1,1)).flatten() + m_val["baseline"].values
tp = ts.inverse_transform(tp.reshape(-1,1)).flatten() + m_test["baseline"].values

vt = ts.inverse_transform(vt.reshape(-1,1)).flatten() + m_val["baseline"].values
tt = ts.inverse_transform(tt.reshape(-1,1)).flatten() + m_test["baseline"].values

print("\nGRU Val RMSE:", np.sqrt(mean_squared_error(vt, vp)))
print("GRU Test RMSE:", np.sqrt(mean_squared_error(tt, tp)))

# ================= SAVE =================
val_df = pd.DataFrame({
    "Date": m_val["Date"].dt.strftime("%Y-%m-%d"),
    "City": m_val["City"],
    "y_true": vt,
    "gru_pred": vp
})

test_df = pd.DataFrame({
    "Date": m_test["Date"].dt.strftime("%Y-%m-%d"),
    "City": m_test["City"],
    "y_true": tt,
    "gru_pred": tp
})

val_df.to_csv(f"{PREDICTIONS_PATH}/gru_val_preds.csv", index=False)
test_df.to_csv(f"{PREDICTIONS_PATH}/gru_test_preds.csv", index=False)

print("✅ GRU ready for hybrid")
