# Project Walkthrough

> A complete system-level walkthrough of the **Brain Tumor MRI Classifier** — from architecture philosophy to deployment mechanics.
> Designed for presenting the project in interviews, demos, and technical discussions.

---

## Table of Contents

1. [30-Second Elevator Pitch](#30-second-elevator-pitch)
2. [System Architecture](#system-architecture)
3. [Pipeline Breakdown](#pipeline-breakdown)
4. [Data Flow Diagram](#data-flow-diagram)
5. [Technology Decisions](#technology-decisions)
6. [MLOps Maturity](#mlops-maturity)
7. [Performance Summary](#performance-summary)
8. [Deployment Architecture](#deployment-architecture)
9. [Project Timeline & Challenges](#project-timeline--challenges)
10. [Demo Script](#demo-script)

---

## 30-Second Elevator Pitch

> *"I built a production-grade deep learning system that classifies Brain MRI scans as Tumor or No Tumor with 98.19% accuracy. The system uses EfficientNetV2-S with Bayesian hyperparameter optimization, tracks every experiment in MLflow with dataset fingerprinting, and deploys automatically to AWS ECS Fargate through a 10-step GitHub Actions pipeline. I handle the entire lifecycle — from Kaggle data ingestion and DVC versioning through to a FastAPI inference endpoint consumed by a Streamlit client."*

---

## System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TRAINING PHASE                             │
│                                                                     │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐    ┌────────────┐  │
│  │  Kaggle   │───▶│   Data     │───▶│ Bayesian │───▶│  Fine-Tune │  │
│  │  Dataset  │    │  Pipeline  │    │  Tuner   │    │  + Eval    │  │
│  └──────────┘    └────────────┘    └──────────┘    └─────┬──────┘  │
│                                                          │         │
│                                    ┌────────────┐        │         │
│                                    │   MLflow   │◀───────┘         │
│                                    │  Tracking  │                  │
│                                    └────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                                       │
                              best_model.keras
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT PHASE                            │
│                                                                     │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐    ┌────────────┐  │
│  │   S3     │───▶│  GitHub    │───▶│  Docker  │───▶│   ECS      │  │
│  │ Registry │    │  Actions   │    │  + ECR   │    │  Fargate   │  │
│  └──────────┘    └────────────┘    └──────────┘    └─────┬──────┘  │
│                                                          │         │
│                                                   FastAPI /predict │
│                                                          │         │
└─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                              ┌────────────────┐
                              │   Streamlit    │
                              │   Client UI    │
                              └────────────────┘
```

### Repository Structure Overview

```
brain-tumor-mri-classifier/
│
├── src/                         # All application code
│   ├── common/                  # Shared: config, logging, hashing
│   ├── data_pipeline/           # Ingestion → Validation → Preprocessing
│   ├── training_pipeline/       # Model → Tuner → Train → Evaluate → MLflow
│   ├── inference_pipeline/      # Model loading + prediction
│   └── api/                     # FastAPI endpoints + middleware
│
├── configs/                     # YAML model configuration
├── infra/                       # Dockerfile, ECS task def, provision script
├── tests/                       # pytest: model + API integration tests
├── apps/                        # Streamlit client
├── artifacts/                   # Trained model, plots, eval JSONs
├── docs/                        # This documentation
└── .github/workflows/           # CI/CD pipeline
```

---

## Pipeline Breakdown

### Stage 1: Data Ingestion

| What | How | Code |
|:---|:---|:---|
| Download from Kaggle | `KaggleApi.dataset_download_files()` | `data_ingestion.py` |
| Credentials | `.env` → `pydantic_settings` → `os.environ` | `config.py` |
| Storage | `data/raw/Training/` + `data/raw/Testing/` | Auto-extracted ZIP |
| Versioning | DVC tracks `data.dvc` hash | `dvc add data/` |

### Stage 2: Data Validation

| What | How | Code |
|:---|:---|:---|
| Scan directories | Walk `Training/` & `Testing/` for `.jpg` files | `data_validation.py` |
| Label mapping | 4 classes → binary: `{glioma:1, meningioma:1, pituitary:1, notumor:0}` | `DataValidator.binary_map` |
| Quality checks | Deduplicate by filepath, raise on empty result | `validate_and_map()` |
| Output | `pd.DataFrame` with `filepath`, `original_label`, `label` | Passed to preprocessing |

### Stage 3: Preprocessing

| What | How | Code |
|:---|:---|:---|
| Split strategy | Kaggle Training → 80% train / 20% val; Kaggle Testing → test | `preprocessing.py` |
| Image pipeline | `tf.data.Dataset` with parallel decode + resize | `create_tensorflow_dataset()` |
| Normalization | `efficientnet_v2.preprocess_input()` → [-1, 1] | Matches backbone training |
| Augmentation | Horizontal flip, brightness ±10%, contrast 0.9–1.1 | Train split only |
| Performance | `AUTOTUNE` prefetch + parallel map | CPU/GPU overlap |

### Stage 4: Hyperparameter Search

| What | How | Code |
|:---|:---|:---|
| Algorithm | Bayesian Optimization (Gaussian Process surrogate) | `keras_tuner.BayesianOptimization` |
| Objective | Maximize `val_auc` | `Objective("val_auc", "max")` |
| Budget | 10 trials × 5 epochs (early stopping patience=2) | `tuner.py` |
| Search space | LR, dropout, dense units, L2, unfreeze depth | 5 hyperparameters |
| Best result | Trial 09: val_auc=0.9998, accuracy=99.11% | Stored in `artifacts/tuner/` |

### Stage 5: Fine-Tuning

| What | How | Code |
|:---|:---|:---|
| Model | EfficientNetV2-S + Dense(256) + Dropout(0.4) + Sigmoid | `build_model.py` |
| LR schedule | Best tuner LR / 10 = 9.77e-06 | Prevents catastrophic forgetting |
| Regularization | L2=0.00224, Dropout=0.4, label_smoothing=0.1 | Triple regularization |
| Callbacks | EarlyStopping(5) + ModelCheckpoint(save_best) | Monitor `val_auc` |
| Class weights | Inverse frequency: upweight minority class | `calculate_imbalance_weights()` |
| Precision | Mixed float16 (output layer float32) | ~2× GPU throughput |

### Stage 6: Evaluation

| What | How | Code |
|:---|:---|:---|
| Test set | 1,600 images (Kaggle Testing folder) | Never seen during training |
| Threshold | F1-optimal via precision-recall curve | `_optimal_threshold()` |
| Metrics | AUC, accuracy, precision, recall, F1, confusion matrix | `evaluate.py` |
| Artifacts | `confusion_matrix.json`, `report.json`, `metrics.json`, `cm.png` | Saved to `artifacts/eval/` |

### Stage 7: Experiment Tracking

| What | How | Code |
|:---|:---|:---|
| Platform | MLflow (local file URI) | `mlflow_tracking.py` |
| Trial logging | Each of 10 trials → own MLflow run with HPs + val_auc | `log_trial_run()` |
| Final run | HPs + metrics + model + artifacts + dataset hash | `_archive()` in `train.py` |
| Dataset lineage | SHA-256 of sorted file paths | `generate_data_state_hash()` |

### Stage 8: Deployment

| What | How | Code |
|:---|:---|:---|
| Model registry | `upload_model.py` → S3 bucket | `boto3.upload_file()` |
| Container | `python:3.12-slim` + src + configs + model | `infra/Dockerfile` |
| CI/CD | 10-step GitHub Actions workflow | `deploy.yml` |
| Runtime | AWS ECS Fargate (2 vCPU, 4GB) | `task_definition.json` |
| Endpoint | FastAPI on Uvicorn, port 8000 | `/health`, `/predict` |

---

## Data Flow Diagram

```
                    ┌─────────────────────────────────┐
                    │      Kaggle Dataset (7,023)      │
                    │  glioma │ meningioma │ pituitary  │
                    │              notumor              │
                    └────────────────┬────────────────┘
                                     │
                            DataValidator
                     4-class → binary mapping
                                     │
                    ┌────────────────┴────────────────┐
                    │          DataFrame               │
                    │  filepath | original_label | label│
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                  │
              ┌─────┴─────┐                    ┌──────┴──────┐
              │ Training/  │                    │  Testing/   │
              │   ~5,600   │                    │   ~1,600    │
              └─────┬──────┘                    └──────┬──────┘
                    │                                  │
           ┌───────┴───────┐                    Held-out Test
           │               │                    (no augment, no shuffle)
     ┌─────┴─────┐  ┌─────┴─────┐
     │  80% Train │  │  20% Val  │
     │   ~4,480   │  │  ~1,120   │
     └─────┬─────┘  └─────┬─────┘
           │               │
     augment + shuffle    raw only

           │               │
     ┌─────┴───────────────┴──────┐
     │     tf.data Pipelines       │
     │  decode → resize → norm →   │
     │  batch(16) → prefetch       │
     └────────────────────────────┘
```

---

## Technology Decisions

### Why Each Tool Was Chosen

| Component | Chosen | Alternatives Considered | Decision Rationale |
|:---|:---|:---|:---|
| **Backbone** | EfficientNetV2-S | ResNet50, VGG16, ViT | Best accuracy/parameter ratio; fast inference on CPU |
| **HP Search** | Bayesian (KerasTuner) | Grid, Random, Optuna | Sample-efficient; 10 trials vs 1000+ for grid |
| **Tracking** | MLflow | W&B, TensorBoard | Free, self-hosted, model registry built-in |
| **Data Versioning** | DVC | Git LFS, S3 manual | Git-native workflow, handles large binary files |
| **API Framework** | FastAPI | Flask, Django | Async, auto-docs (OpenAPI), type validation |
| **Container** | Docker | VM, conda-pack | Reproducible, standard for cloud-native |
| **Compute** | ECS Fargate | EC2, Lambda, SageMaker | Serverless containers, no instance management |
| **CI/CD** | GitHub Actions | Jenkins, CircleCI | Integrated with GitHub, free for public repos |
| **Config** | pydantic_settings | argparse, configparser | Typed validation, .env + env var auto-loading |
| **Logging** | structlog | stdlib logging, loguru | Structured JSON output, CloudWatch-compatible |
| **Client** | Streamlit | React, Gradio | Rapid prototyping, Python-native, zero frontend code |

### Architecture Principles

1. **Separation of Concerns** — Each pipeline (`data_pipeline/`, `training_pipeline/`, `inference_pipeline/`, `api/`) is independently importable and testable
2. **Configuration as Code** — All settings externalized in `.env` + YAML. No magic numbers
3. **Fail Fast** — Pydantic validates config at startup; `DataValidator` raises on empty datasets
4. **Training-Serving Parity** — Same `preprocess_input()` + `IMAGE_SIZE` used in both pipelines
5. **Defense in Depth** — Model baked into Docker image + S3 fallback for cold starts

---

## MLOps Maturity

### Current Implementation

| MLOps Practice | Status | Implementation |
|:---|:---:|:---|
| Version-controlled code | ✅ | Git + GitHub |
| Version-controlled data | ✅ | DVC with MD5 hashing |
| Experiment tracking | ✅ | MLflow (params + metrics + model + artifacts) |
| Dataset fingerprinting | ✅ | SHA-256 linked to each MLflow run |
| Automated training | ✅ | `python -m src.training_pipeline.train` (single command) |
| Hyperparameter optimization | ✅ | Bayesian search (10 trials) |
| Model registry | ✅ | MLflow + S3 |
| Automated deployment | ✅ | GitHub Actions → ECR → ECS |
| Zero-downtime deploy | ✅ | ECS rolling update + `--force-new-deployment` |
| Health checks | ✅ | `/health` endpoint probed by ECS |
| Structured logging | ✅ | structlog → JSON in production → CloudWatch |
| Container-based serving | ✅ | Docker + Fargate |
| Reproducible training | ✅ | Global seeds (42) + DVC + externalized config |
| Unit tests | ✅ | pytest: model architecture + API contracts |

### Future Improvements

| Practice | Priority | Approach |
|:---|:---:|:---|
| Model drift monitoring | 🔴 High | CloudWatch metrics on prediction confidence distribution |
| A/B testing | 🟡 Medium | S3 model versioning + traffic splitting in ALB |
| Grad-CAM explainability | 🟡 Medium | `/explain` endpoint with heatmap overlay |
| Multi-class expansion | 🟢 Low | Already have 4-class labels; change output layer |
| GPU inference | 🟢 Low | SageMaker real-time endpoints for high throughput |

---

## Performance Summary

### Model Metrics (Hold-out Test Set: 1,600 images)

| Metric | Value |
|:---|:---|
| **Test Accuracy** | 98.19% |
| **AUC** | 0.9998 |
| **Precision (Tumor)** | 0.9843 |
| **Recall (Tumor)** | 0.9917 |
| **F1 (Macro)** | 0.9756 |
| **False Negative Rate** | 0.83% (10 / 1,200) |
| **False Positive Rate** | 4.75% (19 / 400) |

### Confusion Matrix

```
                 Predicted
              No Tumor  Tumor
Actual  No Tumor  381      19
        Tumor      10    1,190
```

### Optimal Hyperparameters (Trial 09)

| Parameter | Value |
|:---|:---|
| Learning Rate | 9.77e-05 |
| Unfrozen Layers | 30 |
| Dense Units | 256 |
| Dropout | 0.4 |
| L2 Regularization | 0.00224 |

---

## Deployment Architecture

### AWS Infrastructure

```
┌──────────────────────────────────────────────────────────┐
│                     AWS Cloud                            │
│                                                          │
│  ┌─────────┐     ┌──────────┐     ┌────────────────┐   │
│  │   S3    │     │   ECR    │     │  ECS Fargate   │   │
│  │ Bucket  │     │ Registry │     │    Cluster      │   │
│  │         │     │          │     │                 │   │
│  │ models/ │     │ brain-   │     │ ┌─────────────┐│   │
│  │ best_   │     │ tumor-   │────▶│ │  Container  ││   │
│  │ model.  │     │ inference│     │ │             ││   │
│  │ keras   │     │          │     │ │ FastAPI     ││   │
│  └────┬────┘     └──────────┘     │ │ :8000       ││   │
│       │                           │ └──────┬──────┘│   │
│       │                           └────────┼──────┘│   │
│       │                                    │        │   │
│       │  ┌──────────────┐                  │        │   │
│       │  │  CloudWatch  │◀── JSON logs ────┘        │   │
│       │  │    Logs      │                           │   │
│       │  └──────────────┘                           │   │
│       │                                              │   │
│       └───── S3 fallback on cold start ──────────────┘   │
│                                                          │
└──────────────────────────────────────────────────────────┘
         │
    Public IP :8000
         │
         ▼
┌──────────────────┐
│  Streamlit       │  (Local machine)
│  Client          │
│  apps/streamlit_ │
│  app.py          │
└──────────────────┘
```

### CI/CD Pipeline Steps

```
1. Checkout ──▶ 2. AWS Auth ──▶ 3. ECR Login ──▶ 4. S3 Model Download
                                                          │
                                                          ▼
10. Output IP ◀── 9. Wait Stable ◀── 8. Update Service ◀── 7. Register Task
                                                          │
                                                          ▲
                                             6. Push ECR ◀── 5. Docker Build
```

### Infrastructure Provisioning (`infra/provision.sh`)

Run **once** to create all AWS resources:

| Step | AWS Resource | Command |
|:---|:---|:---|
| 1 | S3 Bucket | `aws s3 mb s3://brain-tumor-mri-registry-prod` |
| 2 | ECR Repository | `aws ecr create-repository --repository-name brain-tumor-inference` |
| 3 | ECS Cluster | `aws ecs create-cluster --cluster-name brain-tumor-cluster` |
| 4 | Task Definition | `aws ecs register-task-definition --cli-input-json file://task_definition.json` |
| 5 | ECS Service | `aws ecs create-service --launch-type FARGATE --desired-count 1` |

---

## Project Timeline & Challenges

### Development Journey

| Phase | Key Activities | Challenges Solved |
|:---|:---|:---|
| **1. Data** | Kaggle ingestion, DVC setup, validation | Handling 4→2 class mapping cleanly |
| **2. Model** | EfficientNetV2 transfer learning, mixed precision | Keras layers in `tf.function` causing memory leaks |
| **3. Tuning** | Bayesian search design, 10-trial budget | Training hanging during class weight computation |
| **4. Evaluation** | Threshold optimization, artifact generation | Precision-recall curve edge cases |
| **5. MLflow** | Tracking service, dataset fingerprinting | Serializing Keras 3 models in MLflow |
| **6. API** | FastAPI + middleware + error handling | S3 cold start latency on Fargate |
| **7. Infrastructure** | Docker, ECS, provision script | Task definition JSON formatting for `register-task-definition` |
| **8. CI/CD** | GitHub Actions 10-step workflow | `ecs update-service` and `wait services-stable` CLI syntax |

### Technical Challenges & Solutions

**Challenge 1: Training Pipeline Hang**
- **Problem:** Class weight calculation iterated over `tf.data` pipeline, consuming all memory
- **Solution:** Compute weights from metadata DataFrame, not from the data pipeline

**Challenge 2: Keras Layers in `tf.function`**
- **Problem:** Instantiating layers inside graph-traced functions creates new objects per call
- **Solution:** Move all layer creation outside `tf.function` scope

**Challenge 3: ECS Cold Start Model Loading**
- **Problem:** First container startup had no model file
- **Solution:** Dual strategy — bake model into Docker image + S3 fallback in `InferencePipeline`

**Challenge 4: Training-Serving Skew**
- **Problem:** Different preprocessing in training vs inference causes accuracy degradation
- **Solution:** Both pipelines use the same `preprocess_input()` function and `IMAGE_SIZE` from config

---

## Demo Script

Use this script when presenting the project live:

### 1. Repository Overview (30 seconds)

> "Let me start by showing the project structure. The repository is organized into four pipelines..."
>
> Show: `README.md` header, repository tree, architecture diagram

### 2. Configuration System (1 minute)

> "All configuration is centralized in a single Pydantic settings class with a YAML overlay..."
>
> Show: `src/common/config.py`, `configs/model_config.yaml`

### 3. Data Pipeline (1 minute)

> "Data flows from Kaggle through validation and a tf.data pipeline..."
>
> Show: `data_validation.py` (binary_map), `preprocessing.py` (create_tensorflow_dataset)

### 4. Model & Training (2 minutes)

> "The model uses EfficientNetV2-S with a custom head. The Bayesian tuner searches 5 hyperparameters..."
>
> Show: `build_model.py` (model architecture), `tuner.py` (Bayesian search), `train.py` (orchestrator)

### 5. Evaluation Results (1 minute)

> "We achieve 98.19% accuracy with a 0.83% false negative rate..."
>
> Show: Confusion matrix plot, training curves, classification report

### 6. MLflow Tracking (1 minute)

> "Every experiment is tracked with dataset fingerprints for full lineage..."
>
> Show: MLflow UI (if available), `mlflow_tracking.py`

### 7. Deployment (2 minutes)

> "The model deploys to AWS ECS Fargate through a GitHub Actions pipeline..."
>
> Show: `Dockerfile`, `deploy.yml`, `task_definition.json`

### 8. Live Demo (1 minute)

> "Let me demonstrate the inference endpoint..."
>
> Show: Streamlit app → upload MRI → prediction result

### 9. Q&A Preparation

Common follow-up questions after a demo:
- "Why not multi-class?" → See answer Q2 in interview_questions.md
- "How do you handle drift?" → See answer Q34
- "What would you change?" → See answer Q36
- "How would you scale?" → See answer Q37

---

## Key Files Quick Reference

| File | Purpose | Lines |
|:---|:---|:---:|
| `src/common/config.py` | Pydantic settings + YAML overlay | 67 |
| `src/common/logging.py` | structlog (JSON/console auto-switch) | 35 |
| `src/data_pipeline/data_ingestion.py` | Kaggle API download | 42 |
| `src/data_pipeline/data_validation.py` | 4-class → binary mapping | 74 |
| `src/data_pipeline/preprocessing.py` | tf.data pipeline + augmentation | 88 |
| `src/training_pipeline/build_model.py` | EfficientNetV2-S architecture | 87 |
| `src/training_pipeline/tuner.py` | Bayesian hyperparameter search | 64 |
| `src/training_pipeline/train.py` | Master orchestrator | 128 |
| `src/training_pipeline/evaluate.py` | Threshold optimization + metrics | 97 |
| `src/training_pipeline/mlflow_tracking.py` | MLflow tracking service | 55 |
| `src/inference_pipeline/infer.py` | Model loading + prediction | 101 |
| `src/api/app.py` | FastAPI endpoints | 64 |
| `src/api/middleware.py` | Request/response logging | 12 |
| `apps/streamlit_app.py` | Test client UI | 71 |
| `infra/Dockerfile` | Production container | 10 |
| `.github/workflows/deploy.yml` | CI/CD pipeline | 107 |
| `configs/model_config.yaml` | Model configuration | 17 |
| `tests/test_model.py` | Architecture unit tests | 34 |
| `tests/test_api.py` | API integration tests | 29 |
