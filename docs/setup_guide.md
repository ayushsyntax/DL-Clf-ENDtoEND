# 🚀 Getting Started

This guide covers setting up your local environment in WSL2 for training, and how to verify everything is ready for the cloud.

## 🛠️ Local Environment (WSL2)
We recommend using **WSL2 (Ubuntu)** for development as it mirrors the production Linux environment perfectly.

1. **Virtual Environments**: Create a clean environment. We used `venv_deploy` for our final push.
   ```bash
   python -m venv venv_deploy
   source venv_deploy/bin/activate
   ```

2. **Core Dependencies**: Install the baseline requirements.
   ```bash
   pip install -r requirements.txt
   ```
   > [!NOTE]
   > For training, ensure you have the full `tensorflow` package. For inference deployment, the `Dockerfile` uses a lean `tensorflow-cpu` if specialized GPU hardware isn't available in the cloud.

3. **Secret Configuration**: Fill in your `.env`.
   ```bash
   cp .env.example .env
   # Open .env and add your Kaggle, AWS, and API configuration.
   ```

## 🧠 GPU Verification
If you're training locally, don't waste time on a CPU. Check your GPU connectivity immediately:
```python
import tensorflow as tf
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
```

## 🏃 Running the Pipeline
- **Full Model Lifecycle**: From data ingestion to model checkpointing.
  ```bash
  python -m src.training_pipeline.train
  ```
- **Local Streamlit UI**: Perfect for manual testing.
  ```bash
  streamlit run apps/streamlit_app.py
  ```

## 🐋 Local Docker Deployment
Before pushing to ECR/ECS, build and run locally to catch dependency issues early.
```bash
docker build -t brain-tumor-mri-clf -f infra/Dockerfile .
docker run -p 8000:8000 brain-tumor-mri-clf
```
Test with `curl http://localhost:8000/health`.
