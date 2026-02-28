"""
Model architecture for brain tumor classification.
Uses EfficientNetV2-S as a feature extraction base.
Implements hardware optimizations for memory and precision.
"""

import keras
import keras_tuner
import tensorflow
import structlog
from src.common.config import settings

logger = structlog.get_logger()


def configure_gpu_memory() -> None:
    """Enable incremental memory allocation for GPUs.

    Prevents TensorFlow from taking all VRAM immediately.
    """
    physical_gpus = tensorflow.config.list_physical_devices("GPU")
    for gpu_device in physical_gpus:
        try:
            tensorflow.config.experimental.set_memory_growth(gpu_device, True)
        except RuntimeError:
            pass

    if physical_gpus:
        logger.info("GPU hardware found and memory growth enabled")


def set_precision_policy() -> None:
    """Set global precision policy based on hardware.

    Uses mixed float16 on GPU to double training speed.
    """
    keras.mixed_precision.set_global_policy("mixed_float16")
    logger.info("Precision policy set to mixed_float16")


def create_efficientnet_base(unfreeze_layers: int) -> keras.Model:
    """Create a pretrained feature extractor with partial unfreezing.

    Args:
        unfreeze_layers (int): Number of top layers to remain trainable.

    Returns:
        Model: Configured feature extractor base.
    """
    image_shape = (settings.IMAGE_SIZE, settings.IMAGE_SIZE, settings.CHANNELS)

    base_model = keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=image_shape
    )

    base_model.trainable = True

    if unfreeze_layers == 0:
        base_model.trainable = False
        return base_model

    total_layer_count = len(base_model.layers)
    freeze_until_index = total_layer_count - unfreeze_layers

    for layer in base_model.layers[:freeze_until_index]:
        layer.trainable = False

    return base_model


def build_model(hyperparameters: keras_tuner.HyperParameters) -> keras.Model:
    """Construct and compile the binary classification model.

    Args:
        hyperparameters (HyperParameters): Tunable trial values.

    Returns:
        Model: Compiled Keras model instance.
    """
    configure_gpu_memory()
    set_precision_policy()

    image_shape = (settings.IMAGE_SIZE, settings.IMAGE_SIZE, settings.CHANNELS)

    learning_rate = hyperparameters.Float(
        "learning_rate",
        min_value=1e-5, max_value=1e-2, sampling="log"
    )
    dropout_rate = hyperparameters.Float(
        "dropout_rate",
        min_value=0.2, max_value=0.7, step=0.1
    )
    dense_units = hyperparameters.Int(
        "dense_units",
        min_value=128, max_value=512, step=128
    )
    l2_regularization = hyperparameters.Float(
        "l2_reg",
        min_value=1e-5, max_value=1e-2, sampling="log"
    )
    unfreeze_layers = hyperparameters.Int(
        "unfreeze_layers",
        min_value=0, max_value=20, step=5
    )

    base_model = create_efficientnet_base(unfreeze_layers)

    model = keras.Sequential([
        keras.layers.Input(shape=image_shape),
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2_regularization)
        ),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation="sigmoid", dtype="float32")
    ])

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=settings.LABEL_SMOOTHING),
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    return model
