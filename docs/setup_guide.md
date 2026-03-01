# Getting Started

Complete environment setup for local development, training, and deployment verification.

## Prerequisites

- WSL2 (Ubuntu) or native Linux
- Python 3.12+
- Docker Desktop with WSL2 integration enabled
- AWS CLI configured with valid credentials
- Kaggle API credentials

## Environment Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv_deploy
   source venv_deploy/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   The full `requirements.txt` includes training dependencies (TensorFlow, KerasTuner, MLflow, scikit-learn). For inference-only environments, the Dockerfile installs a minimal subset.

3. **Configure secrets**:
   ```bash
   cp .env.example .env
   ```
   Fill in the following fields:
   - `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` for S3 and ECR access
   - `KAGGLE_USERNAME` and `KAGGLE_KEY` for dataset download
   - `API_URL` for the Streamlit client to target (defaults to `http://localhost:8000`)

## GPU Verification

If training locally with GPU support:
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs detected: {len(gpus)}")
```

The training pipeline automatically enables incremental VRAM allocation via `configure_gpu_memory()` and sets mixed-precision (float16) via `set_precision_policy()` in `src/training_pipeline/build_model.py`.

## Running the Training Pipeline

```bash
python -m src.training_pipeline.train
```

This executes the full lifecycle:
1. GPU configuration and precision policy setup
2. Dataset loading via `DataIngestion` and `DataValidator`
3. Bayesian hyperparameter search (10 trials)
4. Fine-tuning with the best configuration
5. Evaluation (threshold optimization, confusion matrix, classification report)
6. MLflow archiving of the final model and metrics

Artifacts are written to `artifacts/` (model, plots, metrics, tuner trials).

## Running the Streamlit Client

```bash
streamlit run apps/streamlit_app.py
```

The client reads `API_URL` from `.env` and connects to the live FastAPI endpoint. It displays a health status indicator and allows MRI image upload for classification.

## Local Docker Build

Before pushing to ECR, verify the inference image locally:
```bash
docker build -t brain-tumor-inference -f infra/Dockerfile .
docker run -p 8000:8000 brain-tumor-inference
```

Test with:
```bash
curl http://localhost:8000/health
```

## AWS Infrastructure Provisioning

Run once to create S3 bucket, ECR repository, ECS cluster, and service:
```bash
bash infra/provision.sh
```

After provisioning, push the model to S3:
```bash
python upload_model.py
```

Subsequent deployments are handled automatically by the GitHub Actions CI/CD pipeline on push to `main`.
