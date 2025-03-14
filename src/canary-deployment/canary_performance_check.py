
# canary_performance_check.py
import requests
import logging

logging.basicConfig(filename='logs/canary.log', level=logging.INFO)

ENDPOINT = "http://ml-model-canary/predict"

response = requests.post(ENDPOINT, json={"input": [3.1, 5.2, 7.4]})
logging.info(f"Canary model response: {response.json()}")
