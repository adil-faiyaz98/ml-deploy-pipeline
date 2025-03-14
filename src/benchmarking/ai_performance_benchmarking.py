
# benchmarking/ai_performance_benchmarking.py
import logging
import time
import requests

logging.basicConfig(filename='logs/ai_performance_benchmarking.log', level=logging.INFO)

ENDPOINT = "http://localhost:8080/predict"

start_time = time.time()
response = requests.post(ENDPOINT, json={"input": [1.2, 3.4, 5.6]})
inference_time = time.time() - start_time

logging.info(f"Model inference time: {inference_time:.4f} seconds")
print(f"AI Performance Benchmarking: Inference time = {inference_time:.4f} seconds")