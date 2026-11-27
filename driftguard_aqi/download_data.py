"""Fetch the real dataset. Needs a free Kaggle account + API token (kaggle.json), else follow the manual steps.

Dataset: 'Air Quality Data in India (2015 - 2020)' by Rohan Rao  ->  city_day.csv
(CPCB station data aggregated per city per day; includes AQI, PM2.5, PM10, NO2, SO2, CO, O3)
"""
import subprocess, sys, os
os.makedirs("data/raw", exist_ok=True)
try:
    subprocess.check_call(["kaggle", "datasets", "download", "-d", "rohanrao/air-quality-data-in-india",
                           "-f", "city_day.csv", "-p", "data/raw", "--unzip"])
    print("Saved data/raw/city_day.csv")
except Exception as e:
    print(f"Automatic download failed ({e}).\nManual: open https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india , "
          "download city_day.csv and place it at data/raw/city_day.csv")
    sys.exit(1)
