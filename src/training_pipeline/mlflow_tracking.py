"""MLflow tracking service: logs params, metrics, model, and artifacts per trial."""

import hashlib
from pathlib import Path

import mlflow
import pandas
import structlog

from src.common.config import settings

logger = structlog.get_logger()


class TrackingService:
    """Wraps MLflow to track hyperparameter trials and final experiment results."""

    def __init__(self) -> None:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    def generate_data_state_hash(self, df: pandas.DataFrame) -> str:
        """SHA-256 fingerprint of sorted filepaths — detects dataset drift between runs."""
        combined = "".join(df["filepath"].sort_values().tolist())
        return hashlib.sha256(combined.encode()).hexdigest()

    def begin_tracking_session(self, name: str) -> mlflow.ActiveRun:
        """Open a named MLflow run context."""
        return mlflow.start_run(run_name=name)

    def log_trial_run(self, trial_id: str, hyperparameters: dict, val_auc: float) -> None:
        """Log a single tuner trial with its params and validation AUC."""
        with self.begin_tracking_session(f"trial_{trial_id}") as run:
            mlflow.log_params(hyperparameters)
            mlflow.log_metric("val_auc", val_auc)
            logger.info("Trial logged", run_id=run.info.run_id, trial_id=trial_id)

    def log_experimental_metadata(self, df: pandas.DataFrame, params: dict) -> None:
        """Log dataset hash + hyperparameters to the active run."""
        mlflow.log_param("dataset_hash", self.generate_data_state_hash(df))
        mlflow.log_params(params)

    def log_performance_metrics(self, metrics: dict) -> None:
        """Push evaluation metric dict to the active run dashboard."""
        mlflow.log_metrics(metrics)

    def log_trained_model(self, model) -> None:
        """Serialize and archive model weights and graph to MLflow."""
        mlflow.keras.log_model(model, "brain_tumor_detector_model")

    def upload_persistent_artifact(self, file_path: Path) -> None:
        """Attach a local file as a persistent artifact to the active run."""
        mlflow.log_artifact(str(file_path))
        logger.info("Artifact uploaded", file=file_path.name)
