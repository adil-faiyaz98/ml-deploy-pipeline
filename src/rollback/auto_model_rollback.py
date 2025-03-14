# rollback/auto_model_rollback.py
import logging
import requests
import json

logging.basicConfig(filename='logs/auto_model_rollback.log', level=logging.INFO)

THRESHOLD = 0.80  # Rollback model if accuracy drops below this
ENDPOINT = "http://localhost:8080/metrics"

response = requests.get(ENDPOINT)
metrics = response.json()
accuracy = metrics.get("model_accuracy", 1.0)

if accuracy < THRESHOLD:
    logging.warning(f"Model accuracy dropped below {THRESHOLD}, initiating rollback...")
    rollback_response = requests.post("http://localhost:8080/rollback")
    logging.info(f"Rollback response: {rollback_response.status_code}")
    print("Automatic rollback executed due to accuracy drop.")
