# anomaly_detection/root_cause_analysis.py
import logging
import requests
import json

logging.basicConfig(filename='logs/root_cause_analysis.log', level=logging.INFO)

ENDPOINTS = {
    "Model Metrics": "http://localhost:8080/metrics",
    "System Health": "http://localhost:8080/system_health",
    "Network Logs": "http://localhost:8080/network_logs"
}

root_cause_report = {}
for category, url in ENDPOINTS.items():
    try:
        response = requests.get(url)
        root_cause_report[category] = response.json()
    except Exception as e:
        root_cause_report[category] = f"Error fetching data: {str(e)}"
        logging.error(f"{category} check failed: {str(e)}")

with open("logs/root_cause_analysis.json", "w") as f:
    json.dump(root_cause_report, f, indent=4)

logging.info("Root cause analysis completed and saved.")
print("Root cause analysis completed.")