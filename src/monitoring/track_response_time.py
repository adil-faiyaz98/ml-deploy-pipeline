# track_response_time.py
import requests
import logging
import time

logging.basicConfig(filename='logs/response_time.log', level=logging.INFO)

ENDPOINT = "http://localhost:8080/predict"

for _ in range(10):
    start_time = time.time()
    response = requests.post(ENDPOINT, json={"input": [2.5, 4.1, 6.3]})
    response_time = time.time() - start_time
    logging.info(f"Response time: {response_time:.4f} seconds")
    time.sleep(5)