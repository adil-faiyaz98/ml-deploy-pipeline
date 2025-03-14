# collect_latency.py
import time
import requests
import logging

logging.basicConfig(filename='logs/latency.log', level=logging.INFO)

ENDPOINT = "http://localhost:8080/predict"

while True:
    start_time = time.time()
    response = requests.post(ENDPOINT, json={"input": [1.2, 3.4, 5.6]})
    latency = time.time() - start_time
    logging.info(f"Inference latency: {latency:.4f} seconds")
    time.sleep(10)  # Monitor every 10 seconds