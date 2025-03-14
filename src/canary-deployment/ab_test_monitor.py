
# ab_test_monitor.py
import requests
import logging

logging.basicConfig(filename='logs/ab_test.log', level=logging.INFO)

ENDPOINT_V1 = "http://ml-model-v1/predict"
ENDPOINT_V2 = "http://ml-model-v2/predict"

def get_predictions(endpoint):
    response = requests.post(endpoint, json={"input": [2.3, 4.2, 5.7]})
    return response.json()

logging.info(f"V1 prediction: {get_predictions(ENDPOINT_V1)}")
logging.info(f"V2 prediction: {get_predictions(ENDPOINT_V2)}")