import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


TARGET_TOLERANCE = 1e-3
PREDICTIONS_PATH = "data/processed"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_and_align_split(lgbm_path, gru_path, split_name):
    lgbm_df = pd.read_csv(lgbm_path)
    gru_df = pd.read_csv(gru_path)

    required_lgbm_cols = {"Date", "City", "y_true", "lgbm_pred"}
    required_gru_cols = {"Date", "City", "y_true", "gru_pred"}

    missing_lgbm = required_lgbm_cols - set(lgbm_df.columns)
    missing_gru = required_gru_cols - set(gru_df.columns)

    if missing_lgbm:
        raise ValueError(f"{split_name}: LightGBM predictions are missing columns: {sorted(missing_lgbm)}")
    if missing_gru:
        raise ValueError(f"{split_name}: GRU predictions are missing columns: {sorted(missing_gru)}")

    merged = lgbm_df.merge(
        gru_df,
        on=["Date", "City"],
        how="inner",
        suffixes=("_lgbm", "_gru"),
    )

    if merged.empty:
        raise ValueError(
            f"{split_name}: no overlapping rows were found between LightGBM and GRU predictions."
        )

    merged["target_gap"] = (merged["y_true_lgbm"] - merged["y_true_gru"]).abs()
    max_gap = merged["target_gap"].max()
    if max_gap > TARGET_TOLERANCE:
        raise ValueError(
            f"{split_name}: prediction files are misaligned because matched rows have different targets. "
            f"Max target gap: {max_gap:.6f}"
        )
    if max_gap > 0:
        print(
            f"{split_name.title()}: matched target values differ slightly due to floating-point precision "
            f"(max gap {max_gap:.6f})."
        )

    merged = merged.rename(columns={"y_true_lgbm": "y_true"})
    return merged[["Date", "City", "y_true", "lgbm_pred", "gru_pred"]].copy()


def fit_best_weighted_average(val_df):
    y_true = val_df["y_true"].to_numpy()
    lgbm_pred = val_df["lgbm_pred"].to_numpy()
    gru_pred = val_df["gru_pred"].to_numpy()

    diff = lgbm_pred - gru_pred
    denom = np.dot(diff, diff)

    if denom == 0:
        alpha = 0.5
    else:
        alpha = np.dot(y_true - gru_pred, diff) / denom

    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended_pred = alpha * lgbm_pred + (1 - alpha) * gru_pred

    return {
        "name": "weighted_average",
        "alpha": alpha,
        "val_pred": blended_pred,
        "val_rmse": rmse(y_true, blended_pred),
        "val_mae": mean_absolute_error(y_true, blended_pred),
    }


def fit_linear_stacker(val_df):
    X = np.column_stack(
        [
            np.ones(len(val_df)),
            val_df["lgbm_pred"].to_numpy(),
            val_df["gru_pred"].to_numpy(),
        ]
    )
    y = val_df["y_true"].to_numpy()

    coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    intercept, w_lgbm, w_gru = coefficients
    val_pred = X @ coefficients

    return {
        "name": "linear_stacking",
        "intercept": float(intercept),
        "w_lgbm": float(w_lgbm),
        "w_gru": float(w_gru),
        "val_pred": val_pred,
        "val_rmse": rmse(y, val_pred),
        "val_mae": mean_absolute_error(y, val_pred),
    }


def evaluate_candidate_on_test(candidate, test_df):
    if candidate["name"] == "weighted_average":
        test_pred = (
            candidate["alpha"] * test_df["lgbm_pred"].to_numpy()
            + (1 - candidate["alpha"]) * test_df["gru_pred"].to_numpy()
        )
    elif candidate["name"] == "linear_stacking":
        test_pred = (
            candidate["intercept"]
            + candidate["w_lgbm"] * test_df["lgbm_pred"].to_numpy()
            + candidate["w_gru"] * test_df["gru_pred"].to_numpy()
        )
    elif candidate["name"] == "lightgbm_only":
        test_pred = test_df["lgbm_pred"].to_numpy()
    elif candidate["name"] == "gru_only":
        test_pred = test_df["gru_pred"].to_numpy()
    else:
        raise ValueError(f"Unknown candidate: {candidate['name']}")

    y_true = test_df["y_true"].to_numpy()
    return {
        **candidate,
        "test_pred": test_pred,
        "test_rmse": rmse(y_true, test_pred),
        "test_mae": mean_absolute_error(y_true, test_pred),
    }


def save_predictions(result, test_df):
    os.makedirs(PREDICTIONS_PATH, exist_ok=True)

    output = test_df[["Date", "City", "y_true"]].copy()
    output["hybrid_pred"] = result["test_pred"]
    output["method"] = result["name"]

    output.to_csv(f"{PREDICTIONS_PATH}/hybrid_test_preds.csv", index=False)


val_df = load_and_align_split(
    "data/processed/lgbm_val_preds.csv",
    "data/processed/gru_val_preds.csv",
    "validation",
)
test_df = load_and_align_split(
    "data/processed/lgbm_test_preds.csv",
    "data/processed/gru_test_preds.csv",
    "test",
)

print("Aligned validation rows:", len(val_df))
print("Aligned test rows:", len(test_df))

baseline_candidates = [
    {
        "name": "lightgbm_only",
        "val_pred": val_df["lgbm_pred"].to_numpy(),
        "val_rmse": rmse(val_df["y_true"], val_df["lgbm_pred"]),
        "val_mae": mean_absolute_error(val_df["y_true"], val_df["lgbm_pred"]),
    },
    {
        "name": "gru_only",
        "val_pred": val_df["gru_pred"].to_numpy(),
        "val_rmse": rmse(val_df["y_true"], val_df["gru_pred"]),
        "val_mae": mean_absolute_error(val_df["y_true"], val_df["gru_pred"]),
    },
]

hybrid_candidates = [
    fit_best_weighted_average(val_df),
    fit_linear_stacker(val_df),
]

all_candidates = baseline_candidates + hybrid_candidates

print("\nValidation Results")
for candidate in all_candidates:
    if candidate["name"] == "weighted_average":
        print(
            f"weighted_average | alpha={candidate['alpha']:.4f} | "
            f"RMSE={candidate['val_rmse']:.4f} | MAE={candidate['val_mae']:.4f}"
        )
    elif candidate["name"] == "linear_stacking":
        print(
            f"linear_stacking | intercept={candidate['intercept']:.4f} | "
            f"w_lgbm={candidate['w_lgbm']:.4f} | w_gru={candidate['w_gru']:.4f} | "
            f"RMSE={candidate['val_rmse']:.4f} | MAE={candidate['val_mae']:.4f}"
        )
    else:
        print(
            f"{candidate['name']} | RMSE={candidate['val_rmse']:.4f} | "
            f"MAE={candidate['val_mae']:.4f}"
        )

best_candidate = min(all_candidates, key=lambda item: item["val_rmse"])
best_result = evaluate_candidate_on_test(best_candidate, test_df)

print("\nSelected Model:", best_result["name"])
print("Best Validation RMSE:", best_result["val_rmse"])
print("Best Validation MAE:", best_result["val_mae"])
print("\nTest RMSE:", best_result["test_rmse"])
print("Test MAE:", best_result["test_mae"])

if best_result["name"] == "weighted_average":
    print(f"Chosen alpha: {best_result['alpha']:.4f}")
elif best_result["name"] == "linear_stacking":
    print(
        f"Chosen coefficients: intercept={best_result['intercept']:.4f}, "
        f"w_lgbm={best_result['w_lgbm']:.4f}, w_gru={best_result['w_gru']:.4f}"
    )

save_predictions(best_result, test_df)
print("\nHybrid predictions saved to data/processed/hybrid_test_preds.csv")
