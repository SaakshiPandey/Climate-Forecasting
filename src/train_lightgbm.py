import pandas as pd
import lightgbm as lgb
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

DATA_PATH = "data/processed/split_data"
MODEL_PATH = "models"

def train_lightgbm():

    # Load split data
    train = pd.read_csv(f"{DATA_PATH}/train.csv")
    val = pd.read_csv(f"{DATA_PATH}/val.csv")
    test = pd.read_csv(f"{DATA_PATH}/test.csv")

    TARGET = "Target_T2M"
    DROP_COLS = ["Date", "City", TARGET]

    X_train = train.drop(columns=DROP_COLS)
    y_train = train[TARGET]

    X_val = val.drop(columns=DROP_COLS)
    y_val = val[TARGET]

    X_test = test.drop(columns=DROP_COLS)
    y_test = test[TARGET]

    # -----------------------
    # Define model
    # -----------------------

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=20,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42
    )

    # -----------------------
    # Train model
    # -----------------------

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(30)]
    )

    # -----------------------
    # Evaluate
    # -----------------------

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    val_mae = mean_absolute_error(y_val, val_preds)
    test_mae = mean_absolute_error(y_test, test_preds)

    train_preds = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    print("\nTrain RMSE:", train_rmse)

    print("\nValidation RMSE:", val_rmse)
    print("Validation MAE:", val_mae)

    print("\nTest RMSE:", test_rmse)
    print("Test MAE:", test_mae)

    # -----------------------
    # Save model
    # -----------------------

    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(model, f"{MODEL_PATH}/lightgbm_model.pkl")

    print("\nModel saved successfully.")


if __name__ == "__main__":
    train_lightgbm()