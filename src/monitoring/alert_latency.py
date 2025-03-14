# alert_latency.py
import logging
import smtplib
import time

logging.basicConfig(filename='logs/latency.log', level=logging.INFO)
THRESHOLD = 1.0  # Alert if latency exceeds 1 second

while True:
    with open('logs/latency.log', 'r') as f:
        lines = f.readlines()
        if lines:
            latest_latency = float(lines[-1].strip().split()[-2])
            if latest_latency > THRESHOLD:
                logging.warning("High latency detected! Sending alert.")
                with smtplib.SMTP('smtp.example.com') as server:
                    server.sendmail("alert@example.com", "admin@example.com", f"High latency: {latest_latency}s")
    time.sleep(60)
