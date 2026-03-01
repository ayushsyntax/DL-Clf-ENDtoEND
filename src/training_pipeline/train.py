"""Master orchestrator: GPU setup → data → tuner search → fine-tuning → evaluation → MLflow."""

import random

import numpy
import pandas
import structlog
import tensorflow

tensorflow.random.set_seed(42)
numpy.random.seed(42)
random.seed(42)

from src.common.config import settings
from src.data_pipeline.preprocessing import get_experimental_datasets
from src.training_pipeline.build_model import build_model, configure_gpu_memory, set_precision_policy
from src.training_pipeline.evaluate import ModelEvaluator
from src.training_pipeline.mlflow_tracking import TrackingService
from src.training_pipeline.tuner import run_hyperparameter_search

logger = structlog.get_logger()


def calculate_imbalance_weights(distribution: dict[int, int]) -> dict[int, float]:
    """Inverse-frequency class weights to counteract label imbalance."""
    total = sum(distribution.values())
    n = len(distribution)
    weights = {cls: total / (n * count) for cls, count in distribution.items()}
    logger.info("Class weights computed", weights=weights)
    return weights


class TrainingPipeline:
    """End-to-end training lifecycle: tuner search → fine-tuning → eval → MLflow archive."""

    def __init__(self) -> None:
        self.tracking = TrackingService()
        self.logger = structlog.get_logger()

    def run_full_training_lifecycle(self) -> None:
        """Execute all stages sequentially with structured logging at each step."""
        configure_gpu_memory()
        set_precision_policy()

        train_ds, val_ds, test_ds, meta_df = get_experimental_datasets()

        class_weights = calculate_imbalance_weights({
            0: len(meta_df[meta_df["label"] == 0]),
            1: len(meta_df[meta_df["label"] == 1])
        })

        best_hp, tuner = run_hyperparameter_search(train_ds, val_ds, class_weights)

        for trial_id, trial in tuner.oracle.trials.items():
            if trial.status == "COMPLETED":
                self.tracking.log_trial_run(trial_id, trial.hyperparameters.values, trial.score)

        final_model = build_model(best_hp)

        final_model.optimizer.learning_rate = best_hp.get("learning_rate") / 10.0

        checkpoint_path = str(settings.ARTIFACTS_DIR / "best_model.keras")
        callbacks = [
            tensorflow.keras.callbacks.EarlyStopping(
                patience=settings.EARLY_STOPPING_PATIENCE,
                monitor="val_auc", mode="max", restore_best_weights=True
            ),
            tensorflow.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, monitor="val_auc", mode="max", save_best_only=True
            )
        ]

        history = final_model.fit(
            train_ds, validation_data=val_ds,
            epochs=settings.FINE_TUNE_EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks, verbose=1
        )

        self._save_training_curves(history)

        metrics = ModelEvaluator(final_model).run_evaluation_suite(test_ds)

        self._archive(final_model, best_hp.values, metrics, meta_df)
        self.logger.info("Training complete", checkpoint=checkpoint_path)

    def _save_training_curves(self, history: tensorflow.keras.callbacks.History) -> None:
        """Plot loss and AUC curves from fine-tuning and save to artifacts/plots/."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = settings.ARTIFACTS_DIR / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(history.history["loss"], label="train")
        axes[0].plot(history.history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(history.history["auc"], label="train")
        axes[1].plot(history.history["val_auc"], label="val")
        axes[1].set_title("AUC")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "training_curves.png")
        plt.close()

    def _archive(
        self,
        model: tensorflow.keras.Model,
        params: dict,
        metrics: dict,
        df: pandas.DataFrame
    ) -> None:
        """Push final model, params, metrics, and eval artifacts to MLflow."""
        with self.tracking.begin_tracking_session("optimized_trial") as run:
            self.tracking.log_experimental_metadata(df, params)
            self.tracking.log_performance_metrics(metrics)
            self.tracking.log_trained_model(model)
            self.tracking.upload_persistent_artifact(settings.EVAL_ARTIFACTS_DIR / "confusion_matrix.json")
            self.tracking.upload_persistent_artifact(settings.EVAL_ARTIFACTS_DIR / "report.json")
            self.logger.info("Experiment archived", run_id=run.info.run_id)


if __name__ == "__main__":
    TrainingPipeline().run_full_training_lifecycle()
