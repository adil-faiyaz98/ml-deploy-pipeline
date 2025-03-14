# Alerting Script (alert_system.sh)

#!/bin/bash
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/your/slack/webhook"
EMAIL_RECIPIENT="your-email@example.com"
PAGERDUTY_SERVICE_KEY="your-pagerduty-key"

# Check Model Performance
MODEL_ACCURACY=$(curl -s http://localhost:9090/api/v1/query?query=model_accuracy | jq .data.result[0].value[1])
LATENCY=$(curl -s http://localhost:9090/api/v1/query?query=model_latency_seconds | jq .data.result[0].value[1])
THRESHOLD=0.85

if (( $(echo "$MODEL_ACCURACY < $THRESHOLD" | bc -l) )); then
    echo "Model accuracy dropped below threshold! Sending alerts..."

    curl -X POST -H 'Content-type: application/json' --data "{
      \"text\": \"*Model Alert:* Accuracy dropped below $THRESHOLD (Current: $MODEL_ACCURACY). Immediate action required!\"
    }" $SLACK_WEBHOOK_URL

    echo "🚨 Model Alert: Accuracy dropped below $THRESHOLD (Current: $MODEL_ACCURACY). Check logs immediately!" | mail -s "[ALERT] Model Performance Issue" $EMAIL_RECIPIENT

    curl -X POST -H "Content-Type: application/json" -d "{
      \"routing_key\": \"$PAGERDUTY_SERVICE_KEY\",
      \"event_action\": \"trigger\",
      \"payload\": {
        \"summary\": \"Model accuracy dropped below threshold\",
        \"severity\": \"critical\",
        \"source\": \"Model Monitoring System\"
      }
    }" "https://events.pagerduty.com/v2/enqueue"

    echo "Alerts Sent!"
else
    echo "Model is performing within acceptable limits. No alerts triggered."
fi