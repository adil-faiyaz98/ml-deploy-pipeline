# Azure ML Deployment Script (deploy_azure_ml.sh)

#!/bin/bash
set -e

RESOURCE_GROUP="your-resource-group"
WORKSPACE_NAME="your-ml-workspace"
MODEL_NAME="ml_model"
MODEL_VERSION=$(date +%Y%m%d%H%M%S)

az ml model register --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --name $MODEL_NAME --path models/model.pkl

az ml model deploy --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --name $MODEL_NAME-deployment --model $MODEL_NAME:$MODEL_VERSION --compute-type aks --overwrite

echo "Azure ML Deployment Completed!"