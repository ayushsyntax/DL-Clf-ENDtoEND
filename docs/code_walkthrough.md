# Code Walkthrough

> Line-by-line explanation of every module in the Brain Tumor MRI Classifier project.
> Use this document alongside the source code during interviews and code reviews.

---

## Table of Contents

1. [Configuration Layer — `src/common/`](#1-configuration-layer)
2. [Data Pipeline — `src/data_pipeline/`](#2-data-pipeline)
3. [Training Pipeline — `src/training_pipeline/`](#3-training-pipeline)
4. [Inference Pipeline — `src/inference_pipeline/`](#4-inference-pipeline)
5. [API Layer — `src/api/`](#5-api-layer)
6. [Supporting Files](#6-supporting-files)

---

## 1. Configuration Layer

### `src/common/config.py` — Centralized Settings

This is the **single source of truth** for every configurable value in the project.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application-wide configuration loaded from .env and model_config.yaml."""

    # ── Path Resolution ──
    # Compute the project root by navigating up from this file's location
    # config.py → common/ → src/ → project root
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    EVAL_ARTIFACTS_DIR: Path = ARTIFACTS_DIR / "eval"

    # ── Secrets (loaded from .env) ──
    KAGGLE_USERNAME: str = ""
    KAGGLE_KEY: str = ""
    AWS_S3_BUCKET: str = os.getenv("AWS_S3_BUCKET", "brain-tumor-mri-registry")

    # ── Model Configuration ──
    MODEL_PATH: str = "artifacts/best_model.keras"
    IMAGE_SIZE: int = 224           # EfficientNetV2 expects 224×224
    CONFIDENCE_THRESHOLD: float = 0.5

    # ── Training Hyperparameters ──
    BATCH_SIZE: int = 16
    TUNER_MAX_TRIALS: int = 10      # Bayesian search budget
    FINE_TUNE_EPOCHS: int = 20      # Max epochs for final training
    LABEL_SMOOTHING: float = 0.1    # Softens hard labels → better calibration

    # ── MLflow ──
    MLFLOW_TRACKING_URI: str = f"file://{(BASE_DIR / 'mlruns').as_posix()}"

    # Pydantic auto-loads from .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def __init__(self, **data):
        super().__init__(**data)
        self._load_yaml_config()  # Second layer: YAML overlay

    def _load_yaml_config(self) -> None:
        """Overlay configs/model_config.yaml onto pydantic defaults."""
        yaml_path = self.BASE_DIR / "configs" / "model_config.yaml"
        if not yaml_path.exists():
            return
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f).get("model", {})
        # YAML values override pydantic defaults (not env vars)
        self.IMAGE_SIZE = cfg.get("input_size", [self.IMAGE_SIZE])[0]
        self.CONFIDENCE_THRESHOLD = float(
            cfg.get("confidence_threshold", self.CONFIDENCE_THRESHOLD)
        )
```

**Key design points:**

| Decision | Why |
|:---|:---|
| `pydantic_settings` | Type validation, `.env` auto-loading, fail-fast on missing config |
| YAML overlay | Model-specific config is version-controlled separately from secrets |
| `Path` objects | OS-agnostic path handling (works on Windows WSL + Linux ECS) |
| Singleton (`settings = Settings()`) | One global instance, imported everywhere |

---

### `src/common/logging.py` — Structured Logging

```python
import structlog

def setup_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,   # Thread-local context
            structlog.processors.add_log_level,         # INFO, ERROR, etc.
            structlog.processors.StackInfoRenderer(),   # Stack traces on errors
            structlog.processors.format_exc_info,       # Exception formatting
            structlog.processors.TimeStamper(fmt="iso"),# ISO-8601 timestamps
            # ↓ Auto-detect environment
            (structlog.processors.JSONRenderer()         # Production: JSON
             if not sys.stderr.isatty()
             else structlog.dev.ConsoleRenderer()),      # Dev: Colored output
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,  # Performance: don't re-configure per call
    )
```

**How it works:**

- `sys.stderr.isatty()` — Returns `True` in terminals (VSCode, bash), `False` in Docker/ECS
- In Docker → **JSON output** → CloudWatch can parse and index every key-value field
- In development → **colored console** → human-readable during debugging
- Every call passes structured data: `logger.info("Download", dataset="brain-tumor", size=7023)`

---

### `src/common/utils.py` — Hashing & DVC

```python
class HashingUtils:
    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """SHA-256 of a file, read in 4KB chunks to handle large models."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

class DVCUtils:
    @staticmethod
    def push_artifacts():
        """Run 'dvc push' if .dvc directory exists."""
        if not Path(".dvc").exists():
            return
        subprocess.run(["dvc", "push"], check=True, capture_output=True)
```

**Why chunked reading?** — Model files can be 100MB+. Reading the entire file into memory for hashing would spike RAM. The 4KB chunk approach keeps memory usage constant.

---

## 2. Data Pipeline

### `src/data_pipeline/data_ingestion.py` — Kaggle Download

```python
class DataIngestion:
    """Downloads and extracts the brain tumor MRI dataset from Kaggle."""

    def __init__(self):
        # Kaggle API requires credentials in env vars
        self.api = KaggleApi()
        self.api.authenticate()

    def download_dataset(self) -> None:
        raw_dir = Path(settings.RAW_DATA_DIR)
        raw_dir.mkdir(parents=True, exist_ok=True)  # Create data/raw/ if missing
        self.api.dataset_download_files(
            settings.KAGGLE_DATASET,  # "masoudnickparvar/brain-tumor-mri-dataset"
            path=raw_dir,
            unzip=True               # Auto-extract the ZIP
        )
```

**Flow:** `.env` → `Settings` → `os.environ["KAGGLE_USERNAME"]` → `KaggleApi.authenticate()` → Download + unzip to `data/raw/`

---

### `src/data_pipeline/data_validation.py` — Label Mapping

This is where the **4-class → binary** transformation happens:

```python
class DataValidator:
    def __init__(self, raw_data_path: Path):
        self.classes = ["glioma", "meningioma", "pituitary", "notumor"]
        # ↓ Domain-specific mapping: all tumor types → 1, no tumor → 0
        self.binary_map = {"glioma": 1, "meningioma": 1, "pituitary": 1, "notumor": 0}

    def validate_and_map(self) -> pd.DataFrame:
        records = []
        for subset in ["Training", "Testing"]:
            subset_path = self.raw_data_path / subset
            for label in self.classes:
                label_path = subset_path / label
                for img_path in label_path.glob("*.jpg"):
                    records.append({
                        "filepath": str(img_path.absolute()),
                        "original_label": label,          # Preserved for debugging
                        "label": self.binary_map[label]   # Binary target
                    })

        df = pd.DataFrame(records)

        # Data quality: remove duplicate file paths
        duplicates = df.duplicated(subset=["filepath"]).sum()
        if duplicates > 0:
            df = df.drop_duplicates(subset=["filepath"])

        return df
```

**Key decisions:**
- **Preserves `original_label`** — Useful for error analysis ("Are all false negatives pituitary tumors?")
- **Deduplication** — Defensive check against filesystem issues
- **Raises `ValueError` on empty result** — Fail fast if data directory is misconfigured

---

### `src/data_pipeline/preprocessing.py` — tf.data Pipeline

```python
def load_and_preprocess_image(image_path, label):
    """Decode JPEG, resize, apply EfficientNetV2 normalization."""
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, [224, 224])
    # ↓ CRITICAL: This applies the EXACT normalization EfficientNetV2 expects
    # Scales pixel values to [-1, 1] range
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return image, label


def apply_training_augmentations(image, label):
    """Conservative augmentations for medical imaging."""
    image = tf.image.random_flip_left_right(image)   # Brain symmetry makes this safe
    image = tf.image.random_brightness(image, 0.1)    # ±10% — preserves MRI intensity
    image = tf.image.random_contrast(image, 0.9, 1.1) # Subtle contrast variation
    return image, label


def create_tensorflow_dataset(file_paths, labels, should_augment=False, should_shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if should_shuffle:
        ds = ds.shuffle(buffer_size=len(labels))  # Full shuffle

    ds = ds.map(load_and_preprocess_image,
                num_parallel_calls=tf.data.AUTOTUNE)  # ← Parallel CPU decoding

    if should_augment:
        ds = ds.map(apply_training_augmentations,
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(16).prefetch(tf.data.AUTOTUNE)  # ← Overlap CPU/GPU
```

**Performance pipeline:**

```
[File paths] → shuffle → parallel decode/resize → parallel augment → batch(16) → prefetch
                                  ↑ CPU                    ↑ CPU              ↑ GPU reads next
```

### Dataset Splitting

```python
def get_experimental_datasets():
    df = DataValidator(settings.RAW_DATA_DIR).validate_and_map()

    # Kaggle's built-in split
    train_meta = df[df["filepath"].str.contains("Training")]
    test_meta = df[df["filepath"].str.contains("Testing")]

    # Carve 20% validation from training
    val_meta = train_meta.sample(frac=0.2, random_state=42)  # Deterministic
    train_meta = train_meta.drop(val_meta.index)              # Remove val from train

    # Build tf.data pipelines with appropriate flags
    train_ds = create_tensorflow_dataset(..., should_augment=True,  should_shuffle=True)
    val_ds   = create_tensorflow_dataset(..., should_augment=False, should_shuffle=False)
    test_ds  = create_tensorflow_dataset(..., should_augment=False, should_shuffle=False)

    return train_ds, val_ds, test_ds, df
```

**Why return `df`?** — The metadata DataFrame is needed downstream for class weight calculation and dataset fingerprinting in MLflow.

---

## 3. Training Pipeline

### `src/training_pipeline/build_model.py` — Model Architecture

```python
def configure_gpu_memory():
    """Prevent TensorFlow from allocating all GPU memory at once."""
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def set_precision_policy():
    """Enable mixed float16 for ~2× throughput on supported GPUs."""
    keras.mixed_precision.set_global_policy("mixed_float16")


def create_efficientnet_base(unfreeze_layers: int):
    """Load pretrained backbone and selectively unfreeze top layers."""
    base = keras.applications.EfficientNetV2S(
        include_top=False,        # Remove ImageNet classification head
        weights="imagenet",       # Load pretrained weights
        input_shape=(224, 224, 3)
    )
    base.trainable = unfreeze_layers > 0
    if unfreeze_layers > 0:
        # Freeze all EXCEPT the last N layers
        for layer in base.layers[:len(base.layers) - unfreeze_layers]:
            layer.trainable = False
    return base


def build_model(hp: keras_tuner.HyperParameters):
    """Construct model — called by KerasTuner for each trial."""

    # ── Hyperparameter Definitions (searched by Bayesian Optimization) ──
    lr       = hp.Float("learning_rate",  1e-5, 1e-2, sampling="log")
    dropout  = hp.Float("dropout_rate",   0.2,  0.7,  step=0.1)
    units    = hp.Int("dense_units",      128,  512,  step=128)
    l2       = hp.Float("l2_reg",         1e-5, 1e-2, sampling="log")
    unfreeze = hp.Int("unfreeze_layers",  0,    20,   step=5)

    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        create_efficientnet_base(unfreeze),         # Pretrained backbone
        keras.layers.GlobalAveragePooling2D(),       # (batch, H, W, C) → (batch, C)
        keras.layers.Dense(units, activation="relu",
                          kernel_regularizer=keras.regularizers.l2(l2)),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="sigmoid",
                          dtype="float32")  # ← MUST be float32 for precision
    ])

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )
    return model
```

**Architecture diagram:**

```
Input (224×224×3)
    ↓
EfficientNetV2-S (pretrained, partial unfreeze)
    ↓
GlobalAveragePooling2D (spatial collapse)
    ↓
Dense(256, relu, L2=0.00224)
    ↓
Dropout(0.4)
    ↓
Dense(1, sigmoid, float32)   →   P(tumor)
```

**Why `GlobalAveragePooling2D` over `Flatten`?**
- Flatten on EfficientNetV2-S output would produce ~100K+ parameters → overfitting risk
- GAP reduces each feature map to a single value → dramatic parameter reduction
- Also provides spatial invariance (tumor location doesn't matter)

---

### `src/training_pipeline/tuner.py` — Bayesian Hyperparameter Search

```python
def run_hyperparameter_search(train_ds, val_ds, class_weights):
    tuner = keras_tuner.BayesianOptimization(
        build_model,
        objective=keras_tuner.Objective("val_auc", direction="max"),
        max_trials=10,
        executions_per_trial=1,     # 1 run per config (not avg over multiple)
        directory="artifacts/tuner",
        project_name="mri_brain_tuner",
        overwrite=False             # Resume if previous search exists
    )

    tuner.search(
        train_ds,
        validation_data=val_ds,
        class_weight=class_weights,  # Pass imbalance weights
        epochs=5,                    # Short search — just need relative ranking
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=2, mode="max"
            )
        ]
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hp, tuner
```

**How Bayesian Optimization works internally:**
1. Trial 1–3: Explore randomly to build an initial surrogate model
2. Trial 4+: Use the surrogate to predict which HP region is most promising
3. Select the next trial configuration to maximize **Expected Improvement**
4. After all trials, return the best-performing HPs

---

### `src/training_pipeline/train.py` — Master Orchestrator

```python
class TrainingPipeline:
    def run_full_training_lifecycle(self):
        # ── Stage 1: Hardware Setup ──
        configure_gpu_memory()
        set_precision_policy()

        # ── Stage 2: Data Loading ──
        train_ds, val_ds, test_ds, meta_df = get_experimental_datasets()

        # ── Stage 3: Class Weights ──
        class_weights = calculate_imbalance_weights({
            0: len(meta_df[meta_df["label"] == 0]),  # ~1,600 no-tumor
            1: len(meta_df[meta_df["label"] == 1])   # ~5,400 tumor
        })
        # Result: {0: ~1.7, 1: ~0.6} — upweights minority class

        # ── Stage 4: Bayesian Search ──
        best_hp, tuner = run_hyperparameter_search(train_ds, val_ds, class_weights)

        # ── Stage 5: Log All Trials to MLflow ──
        for trial_id, trial in tuner.oracle.trials.items():
            if trial.status == "COMPLETED":
                self.tracking.log_trial_run(trial_id, trial.hyperparameters.values, trial.score)

        # ── Stage 6: Rebuild Model with Best HPs ──
        final_model = build_model(best_hp)
        final_model.optimizer.learning_rate = best_hp.get("learning_rate") / 10.0
        # ↑ Reduce LR by 10× for fine-tuning (prevent catastrophic forgetting)

        # ── Stage 7: Fine-Tune ──
        callbacks = [
            EarlyStopping(patience=5, monitor="val_auc", restore_best_weights=True),
            ModelCheckpoint("artifacts/best_model.keras", monitor="val_auc", save_best_only=True)
        ]
        history = final_model.fit(
            train_ds, validation_data=val_ds,
            epochs=20, class_weight=class_weights,
            callbacks=callbacks
        )

        # ── Stage 8: Evaluate ──
        metrics = ModelEvaluator(final_model).run_evaluation_suite(test_ds)

        # ── Stage 9: Archive to MLflow ──
        self._archive(final_model, best_hp.values, metrics, meta_df)
```

**Training Timeline:**

```
[GPU Memory] → [Precision] → [Data] → [Weights] → [10 Tuner Trials]
     ↓                                                    ↓
[Log Trials to MLflow]  ←──  best HPs  ←──  [Select Best]
     ↓
[Rebuild Model] → [LR/10] → [Fine-tune 20 epochs] → [Evaluate] → [MLflow Archive]
```

---

### `src/training_pipeline/evaluate.py` — Evaluation Suite

```python
class ModelEvaluator:
    def run_evaluation_suite(self, test_ds):
        # Collect all predictions on test set
        y_true, y_scores = [], []
        for images, labels in test_ds:
            y_scores.extend(self.model.predict(images, verbose=0).flatten())
            y_true.extend(labels.numpy())

        # Find optimal classification threshold
        best_thresh, best_recall, best_f1 = self._optimal_threshold(y_true, y_scores)
        y_pred = (y_scores >= best_thresh).astype(int)

        # Compute comprehensive metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics = {
            "auc": roc_auc_score(y_true, y_scores),
            "accuracy": report["accuracy"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "best_threshold": best_thresh,
        }

        # Save everything
        self._save_artifacts(confusion_matrix, report, metrics)
        self._plot_confusion_matrix(y_true, y_pred)
        return metrics

    def _optimal_threshold(self, y_true, y_scores):
        """Maximize F1 score across all possible thresholds."""
        prec, rec, thresh = precision_recall_curve(y_true, y_scores)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)  # Harmonic mean
        idx = int(numpy.argmax(f1))
        return float(thresh[idx]), float(rec[idx]), float(f1[idx])
```

**Why precision-recall curve, not ROC?**
- ROC can be misleading with class imbalance (the TN count inflates the true negative rate)
- Precision-recall focuses **only on the positive class** (tumors), which is what matters clinically

---

### `src/training_pipeline/mlflow_tracking.py` — Experiment Tracking

```python
class TrackingService:
    def __init__(self):
        mlflow.set_tracking_uri("file:///project/mlruns")  # Local storage
        mlflow.set_experiment("brain-tumor-classification")

    def generate_data_state_hash(self, df):
        """SHA-256 of all sorted file paths → dataset fingerprint."""
        combined = "".join(df["filepath"].sort_values().tolist())
        return hashlib.sha256(combined.encode()).hexdigest()

    def log_trial_run(self, trial_id, hyperparameters, val_auc):
        """Each Bayesian trial gets its own MLflow run."""
        with self.begin_tracking_session(f"trial_{trial_id}"):
            mlflow.log_params(hyperparameters)
            mlflow.log_metric("val_auc", val_auc)

    def log_experimental_metadata(self, df, params):
        """Link model to exact dataset version."""
        mlflow.log_param("dataset_hash", self.generate_data_state_hash(df))
        mlflow.log_params(params)

    def log_trained_model(self, model):
        """Serialize Keras model to MLflow's model registry."""
        mlflow.keras.log_model(model, "brain_tumor_detector_model")
```

**What gets logged per experiment:**

```
MLflow Run: "optimized_trial"
├── Parameters
│   ├── dataset_hash: "a1b2c3d4..."
│   ├── learning_rate: 9.77e-05
│   ├── dropout_rate: 0.4
│   └── ... (all HPs)
├── Metrics
│   ├── auc: 0.9998
│   ├── accuracy: 0.9819
│   ├── precision: 0.9843
│   └── recall: 0.9917
├── Artifacts
│   ├── brain_tumor_detector_model/ (serialized Keras)
│   ├── confusion_matrix.json
│   └── report.json
```

---

## 4. Inference Pipeline

### `src/inference_pipeline/infer.py` — Production Inference

```python
class InferencePipeline:
    def __init__(self, override_model_path=None):
        self.model_path = override_model_path or "artifacts/best_model.keras"
        self.model = self._load_model()

    def _load_model(self):
        """Load from disk, fall back to S3 download."""
        if not Path(self.model_path).exists():
            self._download_from_s3()
        return keras.models.load_model(self.model_path)

    def _download_from_s3(self):
        """Cold start handler: pull model from S3 registry."""
        s3_client = boto3.client("s3")
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            settings.AWS_S3_BUCKET,     # "brain-tumor-mri-registry"
            settings.MODEL_S3_KEY,      # "models/best_model.keras"
            self.model_path
        )

    def preprocess_image(self, raw_image_bytes):
        """Convert upload bytes → model-ready tensor."""
        decoded = tf.io.decode_image(raw_image_bytes, channels=3)
        resized = tf.image.resize(decoded, [224, 224])
        preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(
            tf.cast(resized, tf.float32)
        )
        return tf.expand_dims(preprocessed, axis=0)  # Add batch dimension: (1, 224, 224, 3)

    def predict(self, raw_image_bytes):
        """Binary classification: Tumor vs No Tumor."""
        input_tensor = self.preprocess_image(raw_image_bytes)
        raw_prediction = self.model.predict(input_tensor, verbose=0)[0][0]
        confidence_score = float(raw_prediction)
        is_tumor = confidence_score > settings.CONFIDENCE_THRESHOLD  # Default: 0.5

        return {
            "label": "Tumor" if is_tumor else "No Tumor",
            "probability": round(confidence_score, 4),
            "class_idx": 1 if is_tumor else 0
        }
```

**Inference flow:**

```
Raw bytes → decode_image → resize(224,224) → preprocess_input → expand_dims
    ↓
model.predict → sigmoid output → threshold → {"label", "probability", "class_idx"}
```

**Training-serving parity checklist:**

| Step | Training | Inference | Match? |
|:---|:---|:---|:---:|
| Decode | `decode_jpeg` | `decode_image` | ✅ (same output) |
| Resize | `[224, 224]` | `[224, 224]` | ✅ |
| Normalize | `preprocess_input` | `preprocess_input` | ✅ |
| Batch | via `tf.data.batch()` | `expand_dims(0)` | ✅ |

---

## 5. API Layer

### `src/api/app.py` — FastAPI Application

```python
app = FastAPI(
    title="Brain Tumor MRI Classifier API",
    description="Binary inference: Tumor vs No Tumor from MRI scans.",
    version="1.0.0"
)

# ↓ Register structured logging middleware
app.add_middleware(BaseHTTPMiddleware, dispatch=logging_middleware)

# ↓ Global reference, initialized once at startup
inference_engine: InferencePipeline = None


@app.on_event("startup")
async def startup_initialization():
    """Load model once — amortize the ~5s load time across all requests."""
    global inference_engine
    inference_engine = InferencePipeline()


@app.get("/health")
async def health_check():
    """ECS/ALB health probe. Returns model load status."""
    if inference_engine is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False}


@app.post("/predict")
async def predict(uploaded_image: UploadFile = File(...)):
    """
    Main inference endpoint.
    Accepts: multipart/form-data with 'uploaded_image' field
    Returns: {"label": "Tumor", "probability": 0.9876, "class_idx": 1}
    """
    if inference_engine is None:
        raise HTTPException(503, "Model not loaded")

    try:
        result = inference_engine.predict(await uploaded_image.read())
        logger.info("Prediction complete", **result)
        return result
    except ValueError:
        raise HTTPException(422, "Invalid or corrupt image.")
    except Exception as e:
        logger.error("Inference failed", error=str(e))
        raise HTTPException(500, "Internal inference error.")
```

**Request lifecycle:**

```
Client POST /predict
    ↓
[Middleware: log method + path]
    ↓
[File validation: UploadFile required]
    ↓
[Read bytes: await uploaded_image.read()]
    ↓
[InferencePipeline.predict()]
    ↓
[Middleware: log status code]
    ↓
JSON Response
```

---

### `src/api/middleware.py` — Request/Response Logging

```python
async def logging_middleware(request: Request, call_next):
    """Log every HTTP request and response for observability."""
    logger.info("Request", method=request.method, path=request.url.path)
    response = await call_next(request)
    logger.info("Response", status_code=response.status_code)
    return response
```

**Production log output (JSON):**

```json
{"method": "POST", "path": "/predict", "level": "info", "timestamp": "2026-03-01T15:00:00Z"}
{"status_code": 200, "level": "info", "timestamp": "2026-03-01T15:00:01Z"}
```

---

## 6. Supporting Files

### `apps/streamlit_app.py` — Test Client

```python
def main():
    st.set_page_config(page_title="Brain Tumor MRI Classifier")

    # ── Health Check ──
    health = requests.get(f"{API_URL}/health", timeout=5)
    if health.json().get("status") == "healthy":
        st.success("🟢 API Online")

    # ── File Upload ──
    uploaded = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

    if uploaded and st.button("Predict"):
        files = {"uploaded_image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        payload = resp.json()

        st.markdown(f"### Result: **{payload['label']}**")
        st.progress(float(payload['probability']))
```

**Purpose:** Local manual verification tool. Connects to the deployed FastAPI endpoint (local or ECS) and provides a visual interface for ad-hoc testing.

---

### `infra/Dockerfile` — Production Container

```dockerfile
FROM python:3.12-slim           # Minimal base (~120MB)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  # No pip cache → smaller image
COPY src/ src/                  # Only inference code
COPY configs/ configs/          # Model configuration
COPY artifacts/best_model.keras artifacts/best_model.keras  # Baked-in model
ENV PYTHONPATH=/app             # Enable absolute imports
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**What's NOT in the image:** `data/`, `mlruns/`, `tests/`, `apps/`, `.git/`, `.env` — smaller image, less attack surface.

---

### `.github/workflows/deploy.yml` — CI/CD Pipeline

```yaml
name: Deploy to ECS
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      # 1. Checkout repository
      - uses: actions/checkout@v4

      # 2. Configure AWS credentials from GitHub Secrets
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      # 3. Authenticate with ECR
      - uses: aws-actions/amazon-ecr-login@v2

      # 4. Pull model from S3 (not in Git)
      - run: aws s3 cp s3://${{ secrets.AWS_S3_BUCKET }}/models/best_model.keras artifacts/

      # 5. Build Docker image with commit-SHA tag
      - run: docker build -f infra/Dockerfile -t $REGISTRY/$REPO:${{ github.sha }} .

      # 6. Push to ECR
      - run: docker push $REGISTRY/$REPO:${{ github.sha }}

      # 7. Register new task definition with updated image URI
      - run: |
          aws ecs describe-task-definition --task-definition brain-tumor-inference-task \
              --query taskDefinition > task-def.json
          jq '.containerDefinitions[0].image = "NEW_IMAGE"' task-def.json > new.json
          aws ecs register-task-definition --cli-input-json file://new.json

      # 8. Rolling update on ECS (zero downtime)
      - run: aws ecs update-service --force-new-deployment

      # 9. Wait for healthy tasks
      - run: aws ecs wait services-stable

      # 10. Output public IP
      - run: echo "http://$PUBLIC_IP:8000"
```

---

### `tests/test_model.py` — Model Architecture Tests

```python
def _default_model():
    """Build with minimal HPs for testing."""
    hp = keras_tuner.HyperParameters()
    hp.values["learning_rate"] = 1e-4
    hp.values["unfreeze_layers"] = 0  # Frozen backbone = fast build
    return build_model(hp)


def test_model_output_shape():
    """Output must be (batch, 1) for binary sigmoid."""
    assert _default_model().output_shape == (None, 1)

def test_model_optimizer_type():
    """Must use AdamW, not Adam."""
    assert isinstance(_default_model().optimizer, tf.keras.optimizers.AdamW)

def test_backbone_frozen_by_default():
    """With unfreeze_layers=0, entire backbone is non-trainable."""
    backbone = _default_model().layers[0]
    assert backbone.trainable is False
```

**Testing philosophy:** Test **contracts** (shape, type, trainability) not values. These tests remain valid regardless of HP changes or retraining.

---

### `tests/test_api.py` — API Integration Tests

```python
client = TestClient(app)

def test_health_check_reachable():
    assert client.get("/health").status_code == 200

def test_health_check_schema():
    body = client.get("/health").json()
    assert "status" in body
    assert "model_loaded" in body

def test_predict_without_file_returns_422():
    """Missing file upload → FastAPI's built-in validation error."""
    assert client.post("/predict").status_code == 422
```

**Why `TestClient` not `requests`?** — FastAPI's `TestClient` runs the app in-process (no server needed), making tests fast and deterministic.

---

## Module Dependency Graph

```
src/common/config.py  ←────────────────────────────────────┐
src/common/logging.py ←──────────────────────────────┐     │
src/common/utils.py   ←───────────────────────┐      │     │
                                               │      │     │
src/data_pipeline/data_ingestion.py  ──────────┼──────┼─────┤
src/data_pipeline/data_validation.py ──────────┼──────┼─────┤
src/data_pipeline/preprocessing.py   ──────────┼──────┼─────┤
                                               │      │     │
src/training_pipeline/build_model.py ──────────┼──────┼─────┤
src/training_pipeline/tuner.py       ──────────┼──────┼─────┤
src/training_pipeline/evaluate.py    ──────────┼──────┼─────┤
src/training_pipeline/mlflow_tracking.py ──────┼──────┼─────┤
src/training_pipeline/train.py       ──────────┘      │     │
                                                      │     │
src/inference_pipeline/infer.py ──────────────────────┼─────┤
                                                      │     │
src/api/middleware.py ────────────────────────────────┘     │
src/api/app.py ────────────────────────────────────────────┘
```

Every module depends on `config.py` and `logging.py`, creating a clean **hub-and-spoke** architecture with no circular dependencies.
