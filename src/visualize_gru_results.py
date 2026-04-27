"""
GRU Visualization Script (Saved Plots Version)
Saves all plots to: data/processed/plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= CONFIG =================
PREDICTIONS_PATH = "data/processed"
PLOTS_PATH = os.path.join(PREDICTIONS_PATH, "plots")

VAL_FILE = os.path.join(PREDICTIONS_PATH, "gru_val_preds_final.csv")
TEST_FILE = os.path.join(PREDICTIONS_PATH, "gru_test_preds_final.csv")

os.makedirs(PLOTS_PATH, exist_ok=True)

# ================= STYLE =================
sns.set(style="whitegrid")

ACTUAL_COLOR = "#1f77b4"     # Blue
PRED_COLOR = "#d62728"       # Red
ERROR_COLOR = "#9467bd"      # Purple

plt.rcParams["figure.figsize"] = (12, 6)

# ================= LOAD DATA =================
print("Loading data...")

val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

val_df["Date"] = pd.to_datetime(val_df["Date"])
test_df["Date"] = pd.to_datetime(test_df["Date"])

# ================= 1. TIME SERIES =================
def plot_timeseries(df, filename, title):
    df = df.sort_values("Date")

    plt.figure()
    plt.plot(df["Date"], df["y_true"], label="Actual", color=ACTUAL_COLOR, linewidth=2)
    plt.plot(df["Date"], df["gru_pred"], label="Predicted", color=PRED_COLOR, linewidth=2)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_PATH, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


# ================= 2. SCATTER =================
def plot_scatter(df, filename, title):
    plt.figure()

    plt.scatter(df["y_true"], df["gru_pred"], alpha=0.5, color=PRED_COLOR)

    # Perfect line
    min_val = min(df["y_true"].min(), df["gru_pred"].min())
    max_val = max(df["y_true"].max(), df["gru_pred"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color="black")

    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.title(title)

    save_path = os.path.join(PLOTS_PATH, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


# ================= 3. ERROR DISTRIBUTION =================
def plot_error_distribution(df, filename, title):
    errors = df["y_true"] - df["gru_pred"]

    plt.figure()
    sns.histplot(errors, bins=50, kde=True, color=ERROR_COLOR)

    plt.xlabel("Error (°C)")
    plt.title(title)

    save_path = os.path.join(PLOTS_PATH, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


# ================= 4. ERROR OVER TIME =================
def plot_error_time(df, filename, title):
    df = df.sort_values("Date")
    df["error"] = df["y_true"] - df["gru_pred"]

    plt.figure()
    plt.plot(df["Date"], df["error"], color=ERROR_COLOR)

    plt.axhline(0, linestyle='--', color="black")

    plt.xlabel("Date")
    plt.ylabel("Error (°C)")
    plt.title(title)

    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_PATH, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


# ================= RUN =================
print("\nGenerating GRU plots...")

# Match your naming style
plot_timeseries(
    test_df,
    "gru_actual_vs_pred_timeseries.png",
    "GRU: Actual vs Predicted Temperature"
)

plot_scatter(
    test_df,
    "gru_scatter_plot.png",
    "GRU: Actual vs Predicted Scatter"
)

plot_error_distribution(
    test_df,
    "gru_error_distribution.png",
    "GRU: Error Distribution"
)

plot_error_time(
    test_df,
    "gru_error_over_time.png",
    "GRU: Error Over Time"
)

print("\n✅ All GRU plots saved in /plots folder!")