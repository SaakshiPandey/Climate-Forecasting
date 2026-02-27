import pandas as pd
import numpy as np
import os

INPUT_PATH = "data/processed/climate_with_target.csv"
OUTPUT_PATH = "data/processed/climate_feature_engineered.csv"


def create_features():

    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort properly
    df = df.sort_values(["City", "Date"])

    # -----------------------
    # 1️⃣ Lag Features
    # -----------------------

    lag_features = ["T2M", "RH2M", "PRECTOTCORR"]

    for feature in lag_features:
        df[f"{feature}_lag1"] = df.groupby("City")[feature].shift(1)
        df[f"{feature}_lag3"] = df.groupby("City")[feature].shift(3)
        df[f"{feature}_lag7"] = df.groupby("City")[feature].shift(7)

    # -----------------------
    # 2️⃣ Rolling Features
    # -----------------------

    df["T2M_roll7_mean"] = (
        df.groupby("City")["T2M"]
        .rolling(window=7)
        .mean()
        .reset_index(0, drop=True)
    )

    df["T2M_roll30_mean"] = (
        df.groupby("City")["T2M"]
        .rolling(window=30)
        .mean()
        .reset_index(0, drop=True)
    )

    df["T2M_roll7_std"] = (
        df.groupby("City")["T2M"]
        .rolling(window=7)
        .std()
        .reset_index(0, drop=True)
    )

    # -----------------------
    # 3️⃣ Time-Based Features
    # -----------------------

    df["Month"] = df["Date"].dt.month
    df["Day_of_Year"] = df["Date"].dt.dayofyear
    df["Week_of_Year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Year"] = df["Date"].dt.year

    # -----------------------
    # 4️⃣ Encode City
    # -----------------------

    df["City_Code"] = df["City"].astype("category").cat.codes

    # -----------------------
    # Remove rows with NaN (from lags & rolling)
    # -----------------------

    df = df.dropna()

    df.to_csv(OUTPUT_PATH, index=False)

    print("Feature engineering completed successfully.")


if __name__ == "__main__":
    create_features()