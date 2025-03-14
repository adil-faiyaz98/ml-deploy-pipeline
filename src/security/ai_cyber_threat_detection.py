# security/ai_cyber_threat_detection.py
import logging
import requests
import json

logging.basicConfig(filename='logs/ai_cyber_threat_detection.log', level=logging.INFO)

ENDPOINTS = {
    "Network Logs": "http://localhost:8080/network_logs",
    "System Health": "http://localhost:8080/system_health",
    "Authentication Logs": "http://localhost:8080/auth_logs"
}

cyber_threat_report = {}
for category, url in ENDPOINTS.items():
    try:
        response = requests.get(url)
        cyber_threat_report[category] = response.json()
    except Exception as e:
        cyber_threat_report[category] = f"Error fetching data: {str(e)}"
        logging.error(f"{category} check failed: {str(e)}")

with open("logs/cyber_threat_report.json", "w") as f:
    json.dump(cyber_threat_report, f, indent=4)

logging.info("AI cyber threat detection completed and saved.")
print("AI cyber threat detection completed.")
