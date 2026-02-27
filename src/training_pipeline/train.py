"""
Master training orchestrator for binary MRI classification.
Orchestrates data preparation, hyperparameter search, and training.
Implements warmup and fine-tuning cycles.
"""

import pandas
import tensorflow
import structlog
import numpy as np, random
tensorflow.random.set_seed(42); np.random.seed(42); random.seed(42)
from src.common.config import settings
from src.data_pipeline.preprocessing import get_experimental_datasets
from src.training_pipeline.build_model import configure_gpu_memory
from src.training_pipeline.build_model import set_precision_policy
from src.training_pipeline.build_model import build_model
from src.training_pipeline.tuner import run_hyperparameter_search
from src.training_pipeline.evaluate import ModelEvaluator
from src.training_pipeline.mlflow_tracking import TrackingService

logger = structlog.get_logger()


def count_labels_in_dataset(dataset: tensorflow.data.Dataset) -> tuple[int, int]:
    """Count class frequencies in a prepared dataset.

    Args:
        dataset (Dataset): Target dataset for counting.

    Returns:
        tuple[int, int]: Counts for classes 0 and 1.
    """
    class_0_sum = 0
    class_1_sum = 0

    for _, batch_of_labels in dataset.unbatch():
        integer_label = int(batch_of_labels.numpy())
        if integer_label == 0:
            class_0_sum += 1
        elif integer_label == 1:
            class_1_sum += 1

    return class_0_sum, class_1_sum


def calculate_imbalance_weights(dataset: tensorflow.data.Dataset) -> dict[int, float]:
    """Compute balanced weights from label sums.

    Args:
        dataset (Dataset): Dataset for weight analysis.

    Returns:
        dict[int, float]: Weighted class map.
    """
    count_0, count_1 = count_labels_in_dataset(dataset)
    total_samples = count_0 + count_1

    weight_0 = total_samples / (2 * count_0)
    weight_1 = total_samples / (2 * count_1)

    balanced_mapping = {0: float(weight_0), 1: float(weight_1)}
    logger.info("Class weights computed", weights=balanced_mapping)

    return balanced_mapping


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
        class_weights = calculate_imbalance_weights(train_ds)

        self.logger.info("Step 4: Running tuner search")
        best_hyperparameters = run_hyperparameter_search(train_ds, val_ds, class_weights)

        self.logger.info("Step 5: Building final model with best hyperparameters")
        best_hyperparameters.values["unfreeze_layers"] = 0
        final_model = build_model(best_hyperparameters)

        self.logger.info("Step 6: Commencing warmup training cycle")
        warmup_history = final_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=settings.WARMUP_EPOCHS,
            class_weight=class_weights,
            verbose=1
        )

        self.logger.info("Step 7: Commencing fine-tuning cycle")
        final_model = build_model(best_hyperparameters)
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

        import os
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        for ax, hist, title in zip(axs.flat,
          [warmup_history.history['loss'], warmup_history.history['val_loss'],
           warmup_history.history['auc'], warmup_history.history['val_auc'],
           fine_tune_history.history['loss'], fine_tune_history.history['val_loss'],
           fine_tune_history.history['auc'], fine_tune_history.history['val_auc']],
          ['Warmup Loss','Warmup AUC','Fine-tune Loss','Fine-tune AUC']):
            pass  # see below
        axs[0,0].plot(warmup_history.history['loss'], label='train')
        axs[0,0].plot(warmup_history.history['val_loss'], label='val')
        axs[0,0].set_title('Warmup Loss'); axs[0,0].legend()
        axs[0,1].plot(warmup_history.history['auc'], label='train')
        axs[0,1].plot(warmup_history.history['val_auc'], label='val')
        axs[0,1].set_title('Warmup AUC'); axs[0,1].legend()
        axs[1,0].plot(fine_tune_history.history['loss'], label='train')
        axs[1,0].plot(fine_tune_history.history['val_loss'], label='val')
        axs[1,0].set_title('Fine-tune Loss'); axs[1,0].legend()
        axs[1,1].plot(fine_tune_history.history['auc'], label='train')
        axs[1,1].plot(fine_tune_history.history['val_auc'], label='val')
        axs[1,1].set_title('Fine-tune AUC'); axs[1,1].legend()
        plt.tight_layout()
        os.makedirs('artifacts/plots', exist_ok=True)
        plt.savefig('artifacts/plots/training_curves.png'); plt.close()

        self.logger.info("Step 8: Commencing evaluation stage")
        evaluator = ModelEvaluator(final_model)
        performance_metrics = evaluator.run_evaluation_suite(test_ds)

        self.logger.info("Step 9: Archiving experiment data to MLflow")
        self._archive_trial_record(
            final_model,
            best_hyperparameters.values,
            performance_metrics,
            metadata_df
        )

        self.logger.info("Step 10: Best model saved to artifacts directory", path=model_checkpoint_path)

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
