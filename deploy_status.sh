#!/bin/bash
# Load credentials from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Ensure AWS_DEFAULT_REGION is set
export AWS_DEFAULT_REGION=${AWS_REGION:-us-east-1}

source venv_deploy/bin/activate

echo "Checking ECS Tasks..."
TASK_ARNS=$(aws ecs list-tasks --cluster brain-tumor-cluster --service-name brain-tumor-service --query 'taskArns' --output text)

if [ "$TASK_ARNS" == "None" ] || [ -z "$TASK_ARNS" ]; then
    echo "No tasks found."
    exit 0
fi

echo "Task ARNs: $TASK_ARNS"
aws ecs describe-tasks --cluster brain-tumor-cluster --tasks $TASK_ARNS --query 'tasks[].[taskArn, lastStatus]' --output table
