"""EfficientNetV2-S binary classifier with hardware-aware GPU and precision setup."""

import keras
import keras_tuner
import tensorflow
import structlog
from src.common.config import settings

logger = structlog.get_logger()


def configure_gpu_memory() -> None:
    """Enable incremental VRAM allocation to prevent OOM on limited hardware."""
    for gpu in tensorflow.config.list_physical_devices("GPU"):
        try:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if tensorflow.config.list_physical_devices("GPU"):
        logger.info("GPU memory growth enabled")


def set_precision_policy() -> None:
    """Use mixed float16 on GPU to halve memory usage and double throughput."""
    keras.mixed_precision.set_global_policy("mixed_float16")
    logger.info("Precision policy set to mixed_float16")


def create_efficientnet_base(unfreeze_layers: int) -> keras.Model:
    """
    Load pretrained EfficientNetV2-S and freeze all but the top N layers.

    Args:
        unfreeze_layers: Number of top layers to keep trainable (0 = fully frozen).
    """
    image_shape = (settings.IMAGE_SIZE, settings.IMAGE_SIZE, settings.CHANNELS)
    base = keras.applications.EfficientNetV2S(
        include_top=False, weights="imagenet", input_shape=image_shape
    )
    base.trainable = unfreeze_layers > 0
    if unfreeze_layers > 0:
        for layer in base.layers[:len(base.layers) - unfreeze_layers]:
            layer.trainable = False
    return base


def build_model(hyperparameters: keras_tuner.HyperParameters) -> keras.Model:
    """
    Construct and compile the binary MRI classification model.

    Args:
        hyperparameters: Keras Tuner trial values for LR, dropout, dense units, etc.

    Returns:
        Compiled Keras model with sigmoid output and AUC primary metric.
    """
    configure_gpu_memory()
    set_precision_policy()

    image_shape = (settings.IMAGE_SIZE, settings.IMAGE_SIZE, settings.CHANNELS)
    lr = hyperparameters.Float("learning_rate", 1e-5, 1e-2, sampling="log")
    dropout = hyperparameters.Float("dropout_rate", 0.2, 0.7, step=0.1)
    units = hyperparameters.Int("dense_units", 128, 512, step=128)
    l2 = hyperparameters.Float("l2_reg", 1e-5, 1e-2, sampling="log")
    unfreeze = hyperparameters.Int("unfreeze_layers", 0, 20, step=5)

    model = keras.Sequential([
        keras.layers.Input(shape=image_shape),
        create_efficientnet_base(unfreeze),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units, activation="relu", kernel_regularizer=keras.regularizers.l2(l)),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="sigmoid", dtype="float32")
    ])

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=settings.LABEL_SMOOTHING),
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )
    return model
