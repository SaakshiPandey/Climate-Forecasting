import os
import pandas as pd

RAW_PATH = "data/raw"
CLEAN_PATH = "data/processed/clean"


def clean_city_file(filename):
    file_path = os.path.join(RAW_PATH, filename)

    # Find where actual data starts (after -END HEADER-)
    with open(file_path, "r") as f:
        lines = f.readlines()

    header_index = None
    for i, line in enumerate(lines):
        if line.startswith("YEAR"):
            header_index = i
            break

    if header_index is None:
        raise ValueError(f"Header not found in {filename}")

    df = pd.read_csv(file_path, skiprows=header_index)

    # Create Date from YEAR + DOY
    df["Date"] = pd.to_datetime(
        df["YEAR"].astype(int).astype(str) + "-" +
        df["DOY"].astype(int).astype(str),
        format="%Y-%j"
    )

    # Drop YEAR and DOY after creating Date
    df = df.drop(columns=["YEAR", "DOY"])

    return df


def clean_all():
    os.makedirs(CLEAN_PATH, exist_ok=True)

    for file in os.listdir(RAW_PATH):
        if file.endswith(".csv"):
            city_name = file.replace(".csv", "")

            df = clean_city_file(file)
            df["City"] = city_name

            output_path = os.path.join(CLEAN_PATH, file)
            df.to_csv(output_path, index=False)

            print(f"{city_name} cleaned successfully.")


if __name__ == "__main__":
    clean_all()
