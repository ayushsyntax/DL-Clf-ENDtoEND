# Brain Tumor MRI Classifier — Full Flow & Tech Stack

An end-to-end Machine Learning pipeline and live REST API for classifying Brain Tumor MRIs.

Here's the complete picture of the project from raw data to live inference.

***

## Full End-to-End Flow

```
Kaggle Dataset
     ↓
  Training (local, Python/Keras)
     ↓
artifacts/best_model.keras
     ↓
python upload_model.py → AWS S3
     ↓
git push to main
     ↓
GitHub Actions (deploy.yml)
  ├── Pull model from S3
  ├── Build Docker image
  ├── Push image to AWS ECR
  ├── Register new ECS Task Definition
  ├── Update ECS Service (Fargate)
  └── Print: Live endpoint: http://<public-ip>:8000
             ↓
     FastAPI running on ECS Fargate
      ├── GET  /health  → 200 OK
      └── POST /predict → { label, confidence }
             ↓
     streamlit run streamlit_app.py (local)
      └── Upload MRI image → hit live ECS API → show result
```

***

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| **Model** | Keras / TensorFlow | CNN trained on brain MRI dataset |
| **Dataset** | Kaggle (via API key) | `KAGGLE_USERNAME` + `KAGGLE_KEY` |
| **Model Storage** | AWS S3 | Model too large for git; versioned in S3 |
| **API** | FastAPI + Uvicorn | Async, fast, auto Swagger docs at `/docs` |
| **Containerization** | Docker (`python:3.12-slim`) | Reproducible, portable runtime |
| **Container Registry** | AWS ECR | Stores Docker image, pulled by ECS |
| **Compute** | AWS ECS Fargate | Serverless containers — no EC2 to manage |
| **CI/CD** | GitHub Actions | Push to `main` → full deploy automatically |
| **Local Testing UI** | Streamlit | Hits live ECS API, shows label + confidence |
| **Config** | `.env` + `pydantic-settings` | Clean secret management |
| **Logging** | `structlog` | Structured JSON logs on every request |

***

## Data Flow on a `/predict` Request

```
User uploads MRI image (Streamlit)
        ↓
POST /predict  (multipart/form-data)
        ↓
FastAPI receives image bytes
        ↓
Preprocessed → resized → normalized
        ↓
model.predict() → softmax probabilities
        ↓
{ "label": "No Tumor", "confidence": 0.91 }
        ↓
Streamlit renders result + progress bar
```

***

## Key Design Decisions

- **Model in S3, not git** — `.keras` files are hundreds of MBs; S3 is the right store [linkedin](https://www.linkedin.com/posts/rajkumar-mistry-1b8862178_aws-fastapi-python-activity-7370061922685280257-mESr)
- **Fargate over EC2** — zero server management, pay per task second [linkedin](https://www.linkedin.com/posts/rajkumar-mistry-1b8862178_aws-fastapi-python-activity-7370061922685280257-mESr)
- **No auth** — public inference API, anyone can hit `/predict` directly
- **One deploy file** — entire CI/CD lives in `.github/workflows/deploy.yml`, no scripts folder
- **Streamlit local only** — UI is a dev/demo tool; the production artifact is the API itself

***

## Reproduction Steps

1. **Clone repo**
   ```bash
   git clone https://github.com/ayushsyntax/DL-Clf-ENDtoEND.git
   cd DL-Clf-ENDtoEND
   ```

2. **Configure Environment**
   Copy `.env.example` -> `.env` and fill in your AWS credentials.
   (This keeps the Kaggle data ingestion environment intact).
   ```bash
   cp .env.example .env
   ```

3. **Provide Artifact**
   Place your trained model locally at `artifacts/best_model.keras`.

4. **Upload to S3**
   Run the upload script to push your model to your configured S3 bucket.
   ```bash
   python upload_model.py
   ```

5. **Deploy via GitHub Actions**
   Push to the `main` branch. GitHub Actions will deploy to AWS ECS Fargate automatically.

6. **Test Live Endpoint**
   Copy the printed Live endpoint URL from GitHub Actions logs into `.env` as `API_URL`.
   Run the local Streamlit UI to test the live API:
   ```bash
   streamlit run streamlit_app.py
   ```
