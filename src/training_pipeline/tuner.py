"""
Hyperparameter optimization via Bayesian search.
Uses Keras Tuner to find optimal model settings.
Implements early stopping for trial efficiency.
"""

import keras_tuner
import tensorflow
import structlog
import numpy as np, random
tensorflow.random.set_seed(42); np.random.seed(42); random.seed(42)
from src.training_pipeline.build_model import build_model
from src.common.config import settings

logger = structlog.get_logger()


def run_hyperparameter_search(
    training_dataset: tensorflow.data.Dataset,
    validation_dataset: tensorflow.data.Dataset,
    class_weights: dict[int, float]
) -> keras_tuner.HyperParameters:
    """Run a search to discover optimal architecture.

    Args:
        training_dataset (Dataset): Data to iterate on.
        validation_dataset (Dataset): Data to monitor on.
        class_weights (dict): Imbalance correction mapping.

    Returns:
        HyperParameters: Discovered best configuration.
    """
    logger.info("Initializing Bayesian Optimization loop")

    objective_metric = keras_tuner.Objective("val_auc", direction="max")
    tuner_directory = str(settings.ARTIFACTS_DIR / "tuner")

    tuner = keras_tuner.BayesianOptimization(
        build_model,
        objective=objective_metric,
        max_trials=settings.TUNER_MAX_TRIALS,
        executions_per_trial=1,
        directory=tuner_directory,
        project_name="mri_brain_tuner",
        overwrite=False
    )

    stop_early_callback = tensorflow.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=settings.TUNER_PATIENCE,
        mode="max"
    )

    tuner.search(
        training_dataset,
        validation_data=validation_dataset,
        class_weight=class_weights,
        epochs=settings.TUNER_EPOCHS,
        callbacks=[stop_early_callback],
        verbose=1
    )

    search_results = tuner.get_best_hyperparameters(num_trials=1)
    best_discovered_hyperparameters = search_results[0]

    logger.info("Best search parameters found",
                best_values=best_discovered_hyperparameters.values)

    return best_discovered_hyperparameters
