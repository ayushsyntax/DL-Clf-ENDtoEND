#!/bin/bash
set -e

# Load credentials
# Load credentials from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Ensure AWS_DEFAULT_REGION is set
export AWS_DEFAULT_REGION=${AWS_REGION:-us-east-1}

source venv_deploy/bin/activate

echo "Creating clusters and repos..."
aws ecs create-cluster --cluster-name brain-tumor-cluster 2>/dev/null || echo "Cluster exists"
aws ecr create-repository --repository-name brain-tumor-inference 2>/dev/null || echo "Repo exists"

echo "Registering Task Definition..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
cat infra/task_definition.json | sed \
  -e "s/\${AWS_ACCOUNT_ID}/$AWS_ACCOUNT_ID/g" \
  -e "s/\${AWS_REGION}/us-east-1/g" \
  -e "s/\${AWS_S3_BUCKET}/brain-tumor-mri-registry-prod/g" \
  -e "s/\${ECR_IMAGE}/$AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com\/brain-tumor-inference:latest/g" \
  > infra/tmp_task_def.json

aws ecs register-task-definition --cli-input-json file://infra/tmp_task_def.json >/dev/null
rm infra/tmp_task_def.json

echo "Setting subnets..."
SUBNET_IDS="subnet-07ff7ba793509d75c,subnet-0facb4c7383df453f,subnet-077861eb6080c1e00"

echo "Creating Service..."
aws ecs create-service \
  --cluster brain-tumor-cluster \
  --service-name brain-tumor-service \
  --task-definition brain-tumor-inference-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],assignPublicIp=ENABLED}" 2>/dev/null || \
  echo "Service might already exist."

echo "Provisioning complete!"
