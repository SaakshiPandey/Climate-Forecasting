import pandas as pd
import os

DATA_PATH = "data/processed/climate_multicity.csv"
OUTPUT_PATH = "data/processed/climate_with_target.csv"


def create_target():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort properly
    df = df.sort_values(["City", "Date"])

    # Create target column per city
    df["Target_T2M"] = df.groupby("City")["T2M"].shift(-1)

    # Remove last day of each city (no next-day target available)
    df = df.dropna(subset=["Target_T2M"])

    df.to_csv(OUTPUT_PATH, index=False)

    print("Target column created successfully.")


if __name__ == "__main__":
    create_target()