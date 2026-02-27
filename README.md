# Brain Tumor MRI Classifier

Production-grade binary classifier for brain tumor MRI scans.

## Project Overview
- **Goal:** Classify MRI scans into "Tumor" (Glioma, Meningioma, Pituitary) or "No Tumor".
- **Backbone:** EfficientNetV2-S (ImageNet pretrained).
- **Stack:** TensorFlow 2.16+, Keras 3.x, MLflow, FastAPI, AWS ECS.

## Architecture
See `docs/architecture_diagram.mmd` for details.

### Key Features
- **Binary Mapping:** Automatically maps 4 Kaggle classes to 2 binary classes.
- **Data Pipeline:** High-performance `tf.data` pipeline with native augmentation.
- **Training:** AdamW + Cosine Decay + Warmup, Mixed Precision, Class Weights.
- **Tracking:** Local MLflow tracking for hyperparameters and metrics.
- **Inference:** FastAPI service designed for CPU execution on AWS ECS Fargate.
- **Security:** API Key authentication for the `/predict` endpoint.
- **Demo:** Local Streamlit app for interactive testing.

## Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file based on `.env.example`:
```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
AWS_ACCESS_KEY_ID=your_id
AWS_SECRET_ACCESS_KEY=your_secret
API_KEY=your_auth_key
```

### 3. Pipeline Execution
1. **Download Data:** `python scripts/download_dataset.py`
2. **Run Training:** `python scripts/run_training.py`
3. **Upload to S3:** `python scripts/upload_model_to_s3.py`
4. **Run API:** `python src/api/app.py`
5. **Run Demo:** `python scripts/run_streamlit.py`

## Deployment
Automated via GitHub Actions (`.github/workflows/ci_cd.yml`).
- Builds `Dockerfile.inference`
- Pushes to Amazon ECR
- Updates ECS Fargate Service

## Evaluation
Metrics (Accuracy, Precision, Recall, F1, AUC) and Confusion Matrix are saved to `artifacts/evaluation_results.json`.
Note: Explainability tools (Grad-CAM, SHAP) and monitoring instrumentators (Prometheus) have been removed from this version.

## Final Test Metrics (Threshold-Tuned)
| Metric | Value |
|--------|-------|
| Test AUC | [from eval] |
| Best Thresh | [X.XXX] |
| Precision | XX.X% |
| Recall | XX.X% |
| F1 | XX.X% |



## Model Card
- Dataset: 7200 JPEG (imbalanced tumor/no-tumor)
- AUC: XX.X% test
- Thresh: Prior recall (med FN costly)
- Limits: Axial MRI binary class only
