# GCP Vertex AI Deployment Script (deploy_vertex_ai.sh)

#!/bin/bash
set -e

PROJECT_ID="your-gcp-project"
MODEL_NAME="ml_model"
MODEL_VERSION=$(date +%Y%m%d%H%M%S)
BUCKET_NAME="your-gcs-bucket"
REGION="us-central1"

gsutil cp models/model.pkl gs://$BUCKET_NAME/$MODEL_NAME/$MODEL_VERSION/

gcloud ai models upload \
    --project=$PROJECT_ID \
    --region=$REGION \
    --display-name=$MODEL_NAME-$MODEL_VERSION \
    --artifact-uri=gs://$BUCKET_NAME/$MODEL_NAME/$MODEL_VERSION/

gcloud ai endpoints create \
    --project=$PROJECT_ID \
    --region=$REGION \
    --display-name=$MODEL_NAME-endpoint

echo "GCP Vertex AI Deployment Completed!"