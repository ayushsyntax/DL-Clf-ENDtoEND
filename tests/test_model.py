"""Unit tests for EfficientNetV2-S model architecture and compilation."""

import keras_tuner
import tensorflow as tf

from src.training_pipeline.build_model import build_model


def _default_model():
    """Build model with minimal default hyperparameters for testing."""
    hp = keras_tuner.HyperParameters()
    hp.values["learning_rate"] = 1e-4
    hp.values["dropout_rate"] = 0.3
    hp.values["dense_units"] = 128
    hp.values["l2_reg"] = 1e-4
    hp.values["unfreeze_layers"] = 0
    return build_model(hp)


def test_model_output_shape():
    """Model head must output a single sigmoid probability per sample."""
    assert _default_model().output_shape == (None, 1)


def test_model_optimizer_type():
    """Model must be compiled with AdamW as per training spec."""
    assert isinstance(_default_model().optimizer, tf.keras.optimizers.AdamW)


def test_backbone_frozen_by_default():
    """EfficientNetV2-S backbone must be fully frozen when unfreeze_layers=0."""
    backbone = _default_model().layers[0]
    assert backbone.trainable is False
