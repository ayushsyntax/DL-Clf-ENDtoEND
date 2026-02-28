"""
Master training orchestrator for binary MRI classification.
Orchestrates data preparation, hyperparameter search, and training.
Implements warmup and fine-tuning cycles.
"""

import pandas
import tensorflow
import structlog
import numpy
import random

tensorflow.random.set_seed(42)
numpy.random.seed(42)
random.seed(42)

from src.common.config import settings
from src.data_pipeline.preprocessing import get_experimental_datasets
from src.training_pipeline.build_model import configure_gpu_memory
from src.training_pipeline.build_model import set_precision_policy
from src.training_pipeline.build_model import build_model
from src.training_pipeline.tuner import run_hyperparameter_search
from src.training_pipeline.evaluate import ModelEvaluator
from src.training_pipeline.mlflow_tracking import TrackingService

logger = structlog.get_logger()


def calculate_imbalance_weights(
    class_distribution: dict[int, int]
) -> dict[int, float]:
    """Computes inverse-frequency class weights from known class counts.

    Args:
        class_distribution: Dict mapping class index to sample count.

    Returns:
        Dict mapping class index to its scalar weight.
    """
    total_samples = sum(class_distribution.values())
    num_classes = len(class_distribution)
    weights = {
        cls: total_samples / (num_classes * count)
        for cls, count in class_distribution.items()
    }
    logger.info("Class weights computed", weights=weights)
    return weights


class TrainingPipeline:
    """End-to-end lifecycle manager for MRI classifier experiments.

    Attributes:
        tracking_service (TrackingService): Logging interface.
    """

    def __init__(self) -> None:
        """Initialize pipeline tracking and logging."""
        self.tracking_service = TrackingService()
        self.logger = structlog.get_logger()

    def run_full_training_lifecycle(self) -> None:
        """Execute all experimental stages sequentially."""
        self.logger.info("Step 1: Configuring GPU")
        configure_gpu_memory()
        set_precision_policy()

        self.logger.info("Step 2: Loading datasets")
        train_ds, val_ds, test_ds, metadata_df = get_experimental_datasets()

        self.logger.info("Step 3: Computing class weights")
        class_0_count = len(metadata_df[metadata_df["label"] == 0])
        class_1_count = len(metadata_df[metadata_df["label"] == 1])
        class_weights = calculate_imbalance_weights({0: class_0_count, 1: class_1_count})

        self.logger.info("Step 4: Running tuner search")
        best_hyperparameters, tuner = run_hyperparameter_search(train_ds, val_ds, class_weights)

        self.logger.info("Step 4.5: Logging tuner trials")
        for trial_id, trial in tuner.oracle.trials.items():
            if trial.status == "COMPLETED":
                self.tracking_service.log_trial_run(
                    trial_id=trial_id,
                    hyperparameters=trial.hyperparameters.values,
                    val_auc=trial.score
                )

        self.logger.info("Step 5: Building final model with best hyperparameters")
        original_unfreeze_layers = best_hyperparameters.values.get("unfreeze_layers", 0)
        best_hyperparameters.values["unfreeze_layers"] = 0
        best_hyperparameters.values["unfreeze_layers"] = original_unfreeze_layers
        final_model = build_model(best_hyperparameters)

        self.logger.info("Step 6: Commencing fine-tuning cycle")
        scaled_learning_rate = best_hyperparameters.get("learning_rate") / 10.0
        final_model.optimizer.learning_rate = scaled_learning_rate

        model_checkpoint_path = str(settings.ARTIFACTS_DIR / "best_model.keras")
        training_callbacks = [
            tensorflow.keras.callbacks.EarlyStopping(
                patience=settings.EARLY_STOPPING_PATIENCE,
                monitor="val_auc",
                mode="max",
                restore_best_weights=True
            ),
            tensorflow.keras.callbacks.ModelCheckpoint(
                filepath=model_checkpoint_path,
                monitor="val_auc",
                mode="max",
                save_best_only=True
            )
        ]

        fine_tune_history = final_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=settings.FINE_TUNE_EPOCHS,
            class_weight=class_weights,
            callbacks=training_callbacks,
            verbose=1
        )

        self._save_training_curves(fine_tune_history)

        self.logger.info("Step 7: Commencing evaluation stage")
        evaluator = ModelEvaluator(final_model)
        performance_metrics = evaluator.run_evaluation_suite(test_ds)

        self.logger.info("Step 8: Archiving experiment data to MLflow")
        self._archive_trial_record(
            final_model,
            best_hyperparameters.values,
            performance_metrics,
            metadata_df
        )

        self.logger.info("Step 9: Best model saved to artifacts directory", path=model_checkpoint_path)

    def _save_training_curves(
        self,
        fine_tune_history: tensorflow.keras.callbacks.History
    ) -> None:
        """Plot and save training metrics to disk.

        Args:
            fine_tune_history (History): Results from tuning phase.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_directory = settings.ARTIFACTS_DIR / "plots"
        plot_directory.mkdir(parents=True, exist_ok=True)

        figure, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(fine_tune_history.history["loss"], label="train")
        axes[0].plot(fine_tune_history.history["val_loss"], label="val")
        axes[0].set_title("Fine-tune Loss")
        axes[0].legend()

        axes[1].plot(fine_tune_history.history["auc"], label="train")
        axes[1].plot(fine_tune_history.history["val_auc"], label="val")
        axes[1].set_title("Fine-tune AUC")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(plot_directory / "training_curves.png")
        plt.close()

    def _archive_trial_record(
        self,
        model: tensorflow.keras.Model,
        parameters: dict,
        metrics: dict,
        dataframe: pandas.DataFrame
    ) -> None:
        """Transfer trial data to MLflow registry.

        Args:
            model (Model): Trained instance.
            parameters (dict): Search parameters.
            metrics (dict): Test scores.
            dataframe (DataFrame): Dataset metadata.
        """
        with self.tracking_service.begin_tracking_session("optimized_trial") as active_run:
            self.tracking_service.log_experimental_metadata(dataframe, parameters)
            self.tracking_service.log_performance_metrics(metrics)
            self.tracking_service.log_trained_model(model)

            confusion_matrix_path = settings.EVAL_ARTIFACTS_DIR / "confusion_matrix.json"
            report_path = settings.EVAL_ARTIFACTS_DIR / "report.json"

            self.tracking_service.upload_persistent_artifact(confusion_matrix_path)
            self.tracking_service.upload_persistent_artifact(report_path)

            self.logger.info("Record archived", run_id=active_run.info.run_id)
