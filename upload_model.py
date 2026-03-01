import os
import boto3
from dotenv import load_dotenv

def upload_to_s3(local_file, bucket, s3_file):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )

    try:
        print(f"Uploading {local_file} to s3://{bucket}/{s3_file}...")
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful!")
        return True
    except Exception as e:
        print(f"Error uploading file: {e}")
        return False

if __name__ == "__main__":
    load_dotenv()

    bucket_name = os.getenv('AWS_S3_BUCKET')
    local_model_path = 'artifacts/best_model.keras'
    s3_model_path = 'models/best_model.keras'

    if os.path.exists(local_model_path):
        upload_to_s3(local_model_path, bucket_name, s3_model_path)
    else:
        print(f"Error: {local_model_path} not found.")
