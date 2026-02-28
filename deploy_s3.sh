#!/bin/bash
set -e

# Load credentials from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Hardcoded for safety in this session
# Ensure AWS_DEFAULT_REGION and S3 bucket are set
export AWS_DEFAULT_REGION=${AWS_REGION:-us-east-1}
export AWS_S3_BUCKET=${AWS_S3_BUCKET:-brain-tumor-mri-registry-prod}

python3 -m venv venv_deploy
source venv_deploy/bin/activate
pip install boto3 python-dotenv

python3 -c "import boto3, os; s3 = boto3.client('s3'); s3.create_bucket(Bucket=os.environ['AWS_S3_BUCKET'])" 2>/dev/null || echo "Bucket might already exist"

python3 upload_model.py
