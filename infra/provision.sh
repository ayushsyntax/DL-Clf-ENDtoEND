#!/bin/bash
set -e

# infra/provision.sh
# Run this script once manually to set up AWS resources
# Usage: ./infra/provision.sh

echo "Fetching AWS Account ID..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_S3_BUCKET=${AWS_S3_BUCKET:-brain-tumor-mri-registry-prod}
ECR_REPOSITORY=${ECR_REPOSITORY:-brain-tumor-inference}
ECS_CLUSTER=${ECS_CLUSTER:-brain-tumor-cluster}
ECS_SERVICE=${ECS_SERVICE:-brain-tumor-service}
TASK_FAMILY=${ECS_TASK_FAMILY:-brain-tumor-inference-task}

echo "AWS_ACCOUNT_ID: $AWS_ACCOUNT_ID"
echo "AWS_REGION: $AWS_REGION"
echo "AWS_S3_BUCKET: $AWS_S3_BUCKET"
echo "ECR_REPOSITORY: $ECR_REPOSITORY"
echo "ECS_CLUSTER: $ECS_CLUSTER"
echo "ECS_SERVICE: $ECS_SERVICE"

# 1. Create S3 bucket
echo "Creating S3 bucket..."
if ! aws s3api head-bucket --bucket "$AWS_S3_BUCKET" 2>/dev/null; then
  aws s3 mb s3://"$AWS_S3_BUCKET" --region "$AWS_REGION" || true
else
  echo "Bucket $AWS_S3_BUCKET already exists."
fi

# 2. Create ECR repository
echo "Creating ECR repository..."
aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region "$AWS_REGION" >/dev/null 2>&1 || \
aws ecr create-repository --repository-name "$ECR_REPOSITORY" --region "$AWS_REGION"

# 3. Create ECS cluster
echo "Creating ECS cluster..."
aws ecs describe-clusters --clusters "$ECS_CLUSTER" --region "$AWS_REGION" >/dev/null 2>&1 || \
aws ecs create-cluster --cluster-name "$ECS_CLUSTER" --region "$AWS_REGION"

# 4. Register Task Definition
echo "Registering Task Definition..."
# We generate a temporary task definition JSON with the replaced variables
cat infra/task_definition.json | sed \
  -e "s/\${AWS_ACCOUNT_ID}/$AWS_ACCOUNT_ID/g" \
  -e "s/\${AWS_REGION}/$AWS_REGION/g" \
  -e "s/\${AWS_S3_BUCKET}/$AWS_S3_BUCKET/g" \
  -e "s/\${ECR_IMAGE}/$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com\/$ECR_REPOSITORY:latest/g" \
  > infra/tmp_task_def.json

aws ecs register-task-definition --cli-input-json file://infra/tmp_task_def.json --region "$AWS_REGION"
rm infra/tmp_task_def.json

# 5. Create ECS service
echo "Creating ECS service..."
# Get default VPC and subnets
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region "$AWS_REGION")
SUBNET_IDS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID --query 'Subnets[*].SubnetId' --output text --region "$AWS_REGION" | sed 's/\t/,/g')
SUBNET_ARRAY=$(echo '"'"${SUBNET_IDS//,/'","'}"'"')

aws ecs describe-services --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" --region "$AWS_REGION" >/dev/null 2>&1 || \
aws ecs create-service \
  --cluster "$ECS_CLUSTER" \
  --service-name "$ECS_SERVICE" \
  --task-definition "$TASK_FAMILY" \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_IDS}],assignPublicIp=ENABLED}" \
  --region "$AWS_REGION"

echo "Provisioning complete! You can now run the GitHub Actions workflow."
