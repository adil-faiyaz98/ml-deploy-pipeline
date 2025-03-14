# Model Monitoring Script (model_monitoring.sh)

#!/bin/bash
PROMETHEUS_URL="http://localhost:9090"
CLOUDWATCH_NAMESPACE="MLModelMonitoring"
MODEL_NAME="ml_model"
ENDPOINT_URL="http://localhost:8080/predict"

# Check Model Latency
START_TIME=$(date +%s%3N)
curl -X POST -H "Content-Type: application/json" -d '{"input": [1.2, 3.4, 5.6]}' $ENDPOINT_URL > response.json
END_TIME=$(date +%s%3N)
LATENCY=$((END_TIME - START_TIME))

# Extract Model Accuracy
ACCURACY=$(jq .accuracy response.json)

# Push Metrics to Prometheus
curl -X POST --data "model_latency_seconds $LATENCY" $PROMETHEUS_URL/metrics/job/model_monitoring
curl -X POST --data "model_accuracy $ACCURACY" $PROMETHEUS_URL/metrics/job/model_monitoring

# Push Metrics to CloudWatch
aws cloudwatch put-metric-data --namespace $CLOUDWATCH_NAMESPACE --metric-name ModelLatency --value $LATENCY
aws cloudwatch put-metric-data --namespace $CLOUDWATCH_NAMESPACE --metric-name ModelAccuracy --value $ACCURACY

echo "Model Monitoring Completed!"
