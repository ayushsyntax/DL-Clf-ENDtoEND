#!/bin/bash
# Load credentials from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Ensure AWS_DEFAULT_REGION is set
export AWS_DEFAULT_REGION=${AWS_REGION:-us-east-1}

source venv_deploy/bin/activate

echo "Checking Service Events..."
aws ecs describe-services --cluster brain-tumor-cluster --services brain-tumor-service --query 'services[0].events[0:5].message' --output table
echo "Checking Deployment status..."
aws ecs describe-services --cluster brain-tumor-cluster --services brain-tumor-service --query 'services[0].deployments[*].[status, taskDefinition, rolloutState]' --output table
