# automated_retraining.py
import time
import requests
import logging

logging.basicConfig(filename='logs/retraining.log', level=logging.INFO)

THRESHOLD = 0.75  # Retrain model if accuracy drops below this
CHECK_INTERVAL = 600  # Check every 10 minutes

while True:
    response = requests.get("http://localhost:8080/metrics")
    metrics = response.json()
    accuracy = metrics.get("model_accuracy", 1.0)

    if accuracy < THRESHOLD:
        logging.warning(f"Model accuracy dropped below {THRESHOLD}, triggering retraining...")
        retrain_response = requests.post("http://localhost:8080/retrain")
        logging.info(f"Retraining response: {retrain_response.status_code}")
    
    time.sleep(CHECK_INTERVAL)

print("Automated model retraining script running.")
