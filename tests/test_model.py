import tensorflow as tf

from src.training_pipeline.build_model import build_efficientnet_v2_s


def test_model_architecture_output():
    """
    Validates that the model head produces a single binary probability.
    """
    model_instance = build_efficientnet_v2_s()
    expected_shape = (None, 1)

    assert model_instance.output_shape == expected_shape


def test_model_optimizer_type():
    """
    Ensures the model is compiled with the AdamW optimizer as per spec.
    """
    model_instance = build_efficientnet_v2_s()
    assert isinstance(model_instance.optimizer, tf.keras.optimizers.AdamW)


def test_backbone_frozen_state():
    """
    Verifies the EfficientNet backbone is non-trainable for transfer learning.
    """
    model_instance = build_efficientnet_v2_s()
    backbone_layer = model_instance.get_layer('efficientnetv2-s')

    assert backbone_layer.trainable is False
