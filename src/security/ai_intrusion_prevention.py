# security/ai_intrusion_prevention.py
import logging
import requests
import json

logging.basicConfig(filename='logs/ai_intrusion_prevention.log', level=logging.INFO)

# Simulated API for AI-driven intrusion prevention
def block_intrusion(ip_address):
    response = requests.post("http://localhost:8080/block_ip", json={"ip": ip_address})
    logging.info(f"Blocked IP {ip_address}, response: {response.status_code}")

# Fetch suspicious IPs from cyber threat report
with open("logs/cyber_threat_report.json", "r") as f:
    threats = json.load(f)
    suspicious_ips = threats.get("Network Logs", {}).get("suspicious_ips", [])

for ip in suspicious_ips:
    block_intrusion(ip)

print("AI-powered intrusion prevention executed.")
