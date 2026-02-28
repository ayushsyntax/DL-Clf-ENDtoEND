"""
Central configuration for the brain tumor classifier.
Manages environment variables and hardware defaults.
Contains specific values safe for GTX 1650 4GB VRAM.
"""

import os
import yaml
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    """Stores application constants, loaded from .env and model_config.yaml."""

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    EVAL_ARTIFACTS_DIR: Path = ARTIFACTS_DIR / "eval"

    # From .env
    KAGGLE_USERNAME: str = ""
    KAGGLE_KEY: str = ""
    AWS_S3_BUCKET: str = os.getenv("AWS_S3_BUCKET", "brain-tumor-mri-registry")
    API_URL: str = os.getenv("API_URL", "http://localhost:8000")

    # From YAML
    MODEL_PATH: str = "artifacts/best_model.keras"
    MODEL_S3_KEY: str = "models/best_model.keras"
    IMAGE_SIZE: int = 224
    CLASSES: list = ["glioma", "meningioma", "notumor", "pituitary"]
    CONFIDENCE_THRESHOLD: float = 0.5

    # Training logic unchanged
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

    MLFLOW_TRACKING_URI: str = f"file://{(BASE_DIR / 'mlruns').as_posix()}"
    MLFLOW_EXPERIMENT_NAME: str = "brain-tumor-classification"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    def __init__(self, **data):
        super().__init__(**data)
        self._load_yaml_config()

    def _load_yaml_config(self):
        yaml_path = self.BASE_DIR / "configs" / "model_config.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if 'model' in config_data:
                    model_cfg = config_data['model']
                    self.MODEL_PATH = model_cfg.get('path', self.MODEL_PATH)
                    self.MODEL_S3_KEY = model_cfg.get('s3_key', self.MODEL_S3_KEY)
                    input_size = model_cfg.get('input_size', [self.IMAGE_SIZE, self.IMAGE_SIZE])
                    self.IMAGE_SIZE = input_size[0]
                    self.CLASSES = model_cfg.get('classes', self.CLASSES)
                    self.CONFIDENCE_THRESHOLD = float(model_cfg.get('confidence_threshold', self.CONFIDENCE_THRESHOLD))


settings = Settings()
