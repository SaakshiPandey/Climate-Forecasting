import os
import requests
from config import CITIES, START_DATE, END_DATE, PARAMETERS, BASE_URL

RAW_DATA_PATH = "data/raw"

def download_city_data(city, lat, lon):
    url = (
        f"{BASE_URL}?parameters={PARAMETERS}"
        f"&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={START_DATE}"
        f"&end={END_DATE}"
        f"&format=CSV"
    )

    response = requests.get(url)

    file_path = os.path.join(RAW_DATA_PATH, f"{city}.csv")

    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"{city} data downloaded.")

def download_all():
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    for city, coords in CITIES.items():
        download_city_data(city, coords["lat"], coords["lon"])

if __name__ == "__main__":
    download_all()
