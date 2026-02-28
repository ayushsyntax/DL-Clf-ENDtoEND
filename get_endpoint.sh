#!/bin/bash
# Load credentials from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Ensure AWS_DEFAULT_REGION is set
export AWS_DEFAULT_REGION=${AWS_REGION:-us-east-1}

source venv_deploy/bin/activate

echo "Waiting for task to transition to RUNNING..."
for i in {1..30}; do
    TASK_ARN=$(aws ecs list-tasks --cluster brain-tumor-cluster --service-name brain-tumor-service --query 'taskArns[0]' --output text)
    if [ "$TASK_ARN" != "None" ] && [ -n "$TASK_ARN" ]; then
        STATUS=$(aws ecs describe-tasks --cluster brain-tumor-cluster --tasks "$TASK_ARN" --query 'tasks[0].lastStatus' --output text)
        echo "Attempt $i: Task ARN: $TASK_ARN, Status: $STATUS"
        if [ "$STATUS" == "RUNNING" ]; then
            break
        fi
    else
        echo "Attempt $i: No tasks found yet..."
    fi
    sleep 10
done

if [ "$STATUS" != "RUNNING" ]; then
    echo "Wait timed out or task failed. Check ECS events."
    exit 1
fi

ENI_ID=$(aws ecs describe-tasks --cluster brain-tumor-cluster --tasks "$TASK_ARN" --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
echo "ENI: $ENI_ID"

PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI_ID" --query 'NetworkInterfaces[0].Association.PublicIp' --output text)

echo ""
echo "----------------------------------------------------"
echo "Live endpoint: http://$PUBLIC_IP:8000"
echo "----------------------------------------------------"
echo "Check health: curl http://$PUBLIC_IP:8000/health"
