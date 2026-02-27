import pandas as pd
import os

INPUT_PATH = "data/processed/climate_feature_engineered.csv"
OUTPUT_PATH = "data/processed/split_data"


def split_data():

    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # Define split boundaries
    train_end = "2020-12-31"
    val_end = "2022-12-31"

    train_df = df[df["Date"] <= train_end]
    val_df = df[(df["Date"] > train_end) & (df["Date"] <= val_end)]
    test_df = df[df["Date"] > val_end]

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    train_df.to_csv(f"{OUTPUT_PATH}/train.csv", index=False)
    val_df.to_csv(f"{OUTPUT_PATH}/val.csv", index=False)
    test_df.to_csv(f"{OUTPUT_PATH}/test.csv", index=False)

    print("Data split completed.")
    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)


if __name__ == "__main__":
    split_data()