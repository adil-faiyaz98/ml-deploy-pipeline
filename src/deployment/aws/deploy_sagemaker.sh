# AWS SageMaker Deployment Script (deploy_sagemaker.sh)

#!/bin/bash
set -e

MODEL_NAME="ml_model"
MODEL_VERSION=$(date +%Y%m%d%H%M%S)
S3_BUCKET="your-sagemaker-bucket"
ROLE_ARN="arn:aws:iam::your-account-id:role/SageMakerExecutionRole"
REGION="us-east-1"

aws s3 cp models/model.pkl s3://$S3_BUCKET/$MODEL_NAME/$MODEL_VERSION/model.pkl

aws sagemaker create-model \
    --model-name $MODEL_NAME-$MODEL_VERSION \
    --primary-container Image="382416733822.dkr.ecr.$REGION.amazonaws.com/pytorch-inference:1.5.1-cpu",ModelDataUrl="s3://$S3_BUCKET/$MODEL_NAME/$MODEL_VERSION/model.pkl" \
    --execution-role-arn $ROLE_ARN \
    --region $REGION

aws sagemaker create-endpoint-config \
    --endpoint-config-name $MODEL_NAME-config-$MODEL_VERSION \
    --production-variants VariantName=AllTraffic,ModelName=$MODEL_NAME-$MODEL_VERSION,InitialInstanceCount=1,InstanceType=ml.m5.large \
    --region $REGION

aws sagemaker create-endpoint \
    --endpoint-name $MODEL_NAME-endpoint \
    --endpoint-config-name $MODEL_NAME-config-$MODEL_VERSION \
    --region $REGION

echo "AWS SageMaker Deployment Completed!"