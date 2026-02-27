"""
MLflow tracking service for brain MRI classification.
Archives data hash, trial parameters, and metric scores.
Handles model serialization and artifact uploads.
"""

import hashlib
from pathlib import Path
import mlflow
import pandas
import structlog
from src.common.config import settings

logger = structlog.get_logger()


class TrackingService:
    """Manage the connection to the MLflow tracking server.

    Attributes:
        BASE_DIR: Path reference.
    """

    def __init__(self) -> None:
        """Initialize server connection and target experiment."""
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    def generate_data_state_hash(self, dataframe: pandas.DataFrame) -> str:
        """Create a fingerprint for dataset versioning.

        Args:
            dataframe (DataFrame): Dataset metadata.

        Returns:
            str: SHA-256 state hash.
        """
        sorted_filepaths = dataframe["filepath"].sort_values().tolist()
        combined_string = "".join(sorted_filepaths)
        data_hash = hashlib.sha256(combined_string.encode()).hexdigest()

        return data_hash

    def begin_tracking_session(self, trial_name: str) -> mlflow.ActiveRun:
        """Enter a new experiment recording block.

        Args:
            trial_name (str): Label for the active run.

        Returns:
            ActiveRun: MLflow context object.
        """
        return mlflow.start_run(run_name=trial_name)

    def log_experimental_metadata(
        self,
        df: pandas.DataFrame,
        hyperparameters: dict
    ) -> None:
        """Trace data version and trial configurations.

        Args:
            df (DataFrame): Dataframe for hashing.
            hyperparameters (dict): Hyperparameters for run.
        """
        current_data_hash = self.generate_data_state_hash(df)

        mlflow.log_param("dataset_hash", current_data_hash)
        mlflow.log_params(hyperparameters)

        logger.info("Metadata cataloged in MLflow", hash=current_data_hash)

    def log_performance_metrics(self, metrics: dict) -> None:
        """Push score mappings to the dashboard.

        Args:
            metrics (dict): Performance result map.
        """
        mlflow.log_metrics(metrics)

    def log_trained_model(self, model_instance: any) -> None:
        """Archive weights and graph to server.

        Args:
            model_instance: The trained model.
        """
        mlflow.keras.log_model(model_instance, "brain_tumor_detector_model")

    def upload_persistent_artifact(self, file_path: Path) -> None:
        """Archive static result files to experiment history.

        Args:
            file_path (Path): System path to artifact.
        """
        mlflow.log_artifact(str(file_path))
        logger.info("Artifact upload complete", filename=file_path.name)
