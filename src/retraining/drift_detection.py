# retraining/drift_detection.py
import numpy as np
import pandas as pd
import logging
from scipy.stats import ks_2samp

logging.basicConfig(filename='logs/drift_detection.log', level=logging.INFO)

# Load past and current data
data_past = pd.read_csv("data/past_data.csv")
data_current = pd.read_csv("data/current_data.csv")

# Compute drift for each feature
drift_results = {}
for column in data_past.columns:
    stat, p_value = ks_2samp(data_past[column], data_current[column])
    drift_results[column] = p_value < 0.05  # True if drift detected

# Log detected drift
logging.info(f"Drift results: {drift_results}")
if any(drift_results.values()):
    logging.warning("Data drift detected! Triggering retraining.")
    with open("logs/drift_trigger.txt", "w") as f:
        f.write("Retrain")

print("Drift detection completed.")