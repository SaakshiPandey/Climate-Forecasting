"""
LightGBM Visualization Script
Saves plots to: data/processed/plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= CONFIG =================
PREDICTIONS_PATH = "data/processed"
PLOTS_PATH = os.path.join(PREDICTIONS_PATH, "plots")

VAL_FILE = os.path.join(PREDICTIONS_PATH, "lgbm_val_preds.csv")
TEST_FILE = os.path.join(PREDICTIONS_PATH, "lgbm_test_preds.csv")

os.makedirs(PLOTS_PATH, exist_ok=True)

# ================= STYLE =================
sns.set(style="whitegrid")

ACTUAL_COLOR = "#1f77b4"   # Blue
PRED_COLOR = "#ff7f0e"     # Orange (different from GRU red)
ERROR_COLOR = "#2ca02c"    # Green

plt.rcParams["figure.figsize"] = (12, 6)

# ================= LOAD DATA =================
print("Loading LightGBM prediction files...")

val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

val_df["Date"] = pd.to_datetime(val_df["Date"])
test_df["Date"] = pd.to_datetime(test_df["Date"])

# ================= 1. TIME SERIES =================
def plot_timeseries(df, filename, title):
    df = df.sort_values("Date")

    plt.figure()
    plt.plot(df["Date"], df["y_true"], label="Actual", color=ACTUAL_COLOR, linewidth=2)
    plt.plot(df["Date"], df["lgbm_pred"], label="Predicted", color=PRED_COLOR, linewidth=2)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(PLOTS_PATH, filename), dpi=300)
    plt.close()

    print(f"Saved: {filename}")


# ================= 2. SCATTER =================
def plot_scatter(df, filename, title):
    plt.figure()

    plt.scatter(df["y_true"], df["lgbm_pred"], alpha=0.5, color=PRED_COLOR)

    min_val = min(df["y_true"].min(), df["lgbm_pred"].min())
    max_val = max(df["y_true"].max(), df["lgbm_pred"].max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color="black")

    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.title(title)

    plt.savefig(os.path.join(PLOTS_PATH, filename), dpi=300)
    plt.close()

    print(f"Saved: {filename}")


# ================= 3. ERROR DISTRIBUTION =================
def plot_error_distribution(df, filename, title):
    errors = df["y_true"] - df["lgbm_pred"]

    plt.figure()
    sns.histplot(errors, bins=50, kde=True, color=ERROR_COLOR)

    plt.xlabel("Error (°C)")
    plt.title(title)

    plt.savefig(os.path.join(PLOTS_PATH, filename), dpi=300)
    plt.close()

    print(f"Saved: {filename}")


# ================= 4. ERROR OVER TIME =================
def plot_error_time(df, filename, title):
    df = df.sort_values("Date")
    df["error"] = df["y_true"] - df["lgbm_pred"]

    plt.figure()
    plt.plot(df["Date"], df["error"], color=ERROR_COLOR)

    plt.axhline(0, linestyle='--', color="black")

    plt.xlabel("Date")
    plt.ylabel("Error (°C)")
    plt.title(title)

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(PLOTS_PATH, filename), dpi=300)
    plt.close()

    print(f"Saved: {filename}")


# ================= RUN =================
print("\nGenerating LightGBM plots...")

plot_timeseries(
    test_df,
    "lgbm_actual_vs_pred_timeseries.png",
    "LightGBM: Actual vs Predicted Temperature"
)

plot_scatter(
    test_df,
    "lgbm_scatter_plot.png",
    "LightGBM: Actual vs Predicted Scatter"
)

plot_error_distribution(
    test_df,
    "lgbm_error_distribution.png",
    "LightGBM: Error Distribution"
)

plot_error_time(
    test_df,
    "lgbm_error_over_time.png",
    "LightGBM: Error Over Time"
)

print("\n✅ LightGBM plots saved successfully!")