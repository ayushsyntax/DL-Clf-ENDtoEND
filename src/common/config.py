"""
Central configuration for the brain tumor classifier.
Manages environment variables and hardware defaults.
Contains specific values safe for GTX 1650 4GB VRAM.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    """Stores application constants and training parameters.

    Attributes:
        BASE_DIR: Project root path.
    """

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    EVAL_ARTIFACTS_DIR: Path = ARTIFACTS_DIR / "eval"

    KAGGLE_DATASET: str = "masoudnickparvar/brain-tumor-mri-dataset"
    KAGGLE_USERNAME: str = ""
    KAGGLE_KEY: str = ""

    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 16
    CHANNELS: int = 3

    TUNER_MAX_TRIALS: int = 10
    TUNER_EPOCHS: int = 5
    TUNER_PATIENCE: int = 2

    WARMUP_EPOCHS: int = 5
    FINE_TUNE_EPOCHS: int = 20
    EARLY_STOPPING_PATIENCE: int = 5

    INITIAL_LEARNING_RATE: float = 1e-4
    DROPOUT_RATE: float = 0.5
    LABEL_SMOOTHING: float = 0.1

    AWS_S3_BUCKET: str = os.getenv("AWS_S3_BUCKET", "brain-tumor-mri-registry")
    MODEL_S3_PATH: str = "models/best_model.keras"
    API_KEY: str = os.getenv("API_KEY", "dev-key-123")

    MLFLOW_TRACKING_URI: str = f"file://{(BASE_DIR / 'mlruns').as_posix()}"
    MLFLOW_EXPERIMENT_NAME: str = "brain-tumor-classification"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
