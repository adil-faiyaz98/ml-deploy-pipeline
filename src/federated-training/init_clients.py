# federated_learning/init_clients.py
import json
import logging

logging.basicConfig(filename='logs/federated_learning.log', level=logging.INFO)

clients = {"client_1": {}, "client_2": {}, "client_3": {}}
with open("federated_learning/clients.json", "w") as f:
    json.dump(clients, f)

logging.info("Initialized federated learning clients.")
print("Federated learning clients initialized.")