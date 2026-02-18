import os
import pandas as pd

CLEAN_PATH = "data/processed/clean"
FINAL_PATH = "data/processed"

def merge_all():
    dfs = []

    for file in os.listdir(CLEAN_PATH):
        df = pd.read_csv(os.path.join(CLEAN_PATH, file))
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    final_df = final_df.sort_values(["City", "Date"])

    final_df.to_csv(os.path.join(FINAL_PATH, "climate_multicity.csv"), index=False)

    print("Final merged dataset saved.")

if __name__ == "__main__":
    merge_all()
