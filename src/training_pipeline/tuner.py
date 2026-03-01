"""Bayesian hyperparameter search over EfficientNetV2-S architecture space."""

import random

import keras_tuner
import numpy
import structlog
import tensorflow

tensorflow.random.set_seed(42)
numpy.random.seed(42)
random.seed(42)

from src.common.config import settings
from src.training_pipeline.build_model import build_model

logger = structlog.get_logger()


def run_hyperparameter_search(
    train_ds: tensorflow.data.Dataset,
    val_ds: tensorflow.data.Dataset,
    class_weights: dict[int, float]
) -> tuple[keras_tuner.HyperParameters, keras_tuner.Tuner]:
    """
    Run Bayesian Optimization over model hyperparameters.

    Args:
        train_ds: Training dataset.
        val_ds: Validation dataset used as tuner objective.
        class_weights: Inverse-frequency weights to handle label imbalance.

    Returns:
        Best HyperParameters object and the fitted Tuner instance.
    """
    logger.info("Starting Bayesian hyperparameter search")

    tuner = keras_tuner.BayesianOptimization(
        build_model,
        objective=keras_tuner.Objective("val_auc", direction="max"),
        max_trials=settings.TUNER_MAX_TRIALS,
        executions_per_trial=1,
        directory=str(settings.ARTIFACTS_DIR / "tuner"),
        project_name="mri_brain_tuner",
        overwrite=False
    )

    tuner.search(
        train_ds,
        validation_data=val_ds,
        class_weight=class_weights,
        epochs=settings.TUNER_EPOCHS,
        callbacks=[
            tensorflow.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=settings.TUNER_PATIENCE, mode="max"
            )
        ],
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info("Search complete", best=best_hp.values)
    return best_hp, tuner
