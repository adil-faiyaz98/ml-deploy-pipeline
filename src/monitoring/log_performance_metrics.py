# log_performance_metrics.py
import logging
import json

logging.basicConfig(filename='logs/performance_metrics.log', level=logging.INFO)

with open('logs/latency.log', 'r') as f:
    latency_data = f.readlines()

with open('logs/response_time.log', 'r') as f:
    response_data = f.readlines()

metrics = {
    "latency": latency_data[-1] if latency_data else "N/A",
    "response_time": response_data[-1] if response_data else "N/A"
}

logging.info(json.dumps(metrics))