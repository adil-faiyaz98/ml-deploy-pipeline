# autoscaling/ai_workload_autoscaler.py
import logging
import requests

logging.basicConfig(filename='logs/ai_workload_autoscaler.log', level=logging.INFO)

THRESHOLD_HIGH = 80  # Scale up if CPU usage > 80%
THRESHOLD_LOW = 30   # Scale down if CPU usage < 30%
ENDPOINT = "http://localhost:9090/api/v1/query?query=node_cpu_usage"

response = requests.get(ENDPOINT)
cpu_usage = float(response.json().get("data", {}).get("result", [{}])[0].get("value", [0, 0])[1])

if cpu_usage > THRESHOLD_HIGH:
    logging.info("High workload detected. Scaling up resources.")
    requests.post("http://localhost:8080/scale_up")
elif cpu_usage < THRESHOLD_LOW:
    logging.info("Low workload detected. Scaling down resources.")
    requests.post("http://localhost:8080/scale_down")

print("AI-driven workload autoscaling executed.")
