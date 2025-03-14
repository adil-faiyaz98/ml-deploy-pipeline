# federated_learning/aggregate_updates.py
import json
import logging
import numpy as np

logging.basicConfig(filename='logs/federated_aggregation.log', level=logging.INFO)

# Load client updates
with open("federated_learning/client_updates.json", "r") as f:
    clients = json.load(f)

# Aggregate model updates
aggregated_weights = np.mean([np.array(clients[client]["weights"]) for client in clients], axis=0)

with open("federated_learning/global_model.json", "w") as f:
    json.dump({"weights": list(aggregated_weights)}, f)

logging.info("Aggregated federated model updates.")
print("Federated model aggregation completed.")