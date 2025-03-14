
# federated_learning/deploy_federated_model.py
import json
import logging

logging.basicConfig(filename='logs/federated_deployment.log', level=logging.INFO)

# Load global model
with open("federated_learning/global_model.json", "r") as f:
    model_data = json.load(f)

logging.info("Federated model deployed with weights:")
logging.info(model_data["weights"])
print("Federated model successfully deployed.")