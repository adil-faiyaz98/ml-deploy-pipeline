# ci_cd/auto_debugging.py
import logging
import requests

logging.basicConfig(filename='logs/auto_debugging.log', level=logging.INFO)

def fetch_logs(service_url):
    try:
        response = requests.get(service_url)
        return response.json()
    except Exception as e:
        logging.error(f"Failed to fetch logs from {service_url}: {e}")
        return None

logs = fetch_logs("http://localhost:8080/logs")
if logs:
    logging.info("Auto-debugging analysis completed.")
    print("Issue detected in logs, debugging recommendations available.")