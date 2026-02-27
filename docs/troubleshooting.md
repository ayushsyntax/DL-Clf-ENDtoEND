# Troubleshooting

- GPU not detected: Check CUDA/cuDNN versions (TF 2.16 requires CUDA 12.3).
- Kaggle download fails: Ensure API keys in .env.
- S3 upload error: Verify IAM permissions for put_object.
- OOM in training: Reduce batch_size in config.
- ECS deployment: Ensure AWS CLI configured, Fargate supports CPU.
