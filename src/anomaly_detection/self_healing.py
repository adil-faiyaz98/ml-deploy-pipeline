# anomaly_detection/self_healing.py
import logging
import requests
import time

logging.basicConfig(filename='logs/anomaly_detection.log', level=logging.INFO)

THRESHOLD = 0.85  # Accuracy threshold for anomalies
CHECK_INTERVAL = 600  # Check every 10 minutes

while True:
    response = requests.get("http://localhost:8080/metrics")
    metrics = response.json()
    accuracy = metrics.get("model_accuracy", 1.0)

    if accuracy < THRESHOLD:
        logging.warning(f"Anomaly detected! Model accuracy dropped below {THRESHOLD}. Triggering self-healing.")
        heal_response = requests.post("http://localhost:8080/self_heal")
        logging.info(f"Self-healing response: {heal_response.status_code}")
    
    time.sleep(CHECK_INTERVAL)

print("Self-healing infrastructure monitoring started.")