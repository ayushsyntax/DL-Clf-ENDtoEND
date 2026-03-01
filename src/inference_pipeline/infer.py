from pathlib import Path

import boto3
import keras
import tensorflow as tf

from src.common.config import settings
from src.common.logging import logger


class InferencePipeline:
    """
    Handles model loading and prediction for production inference.

    This class encapsulates the logic for obtaining the model (locally or from S3)
    and performing preprocessing on raw image input.
    """

    def __init__(self, override_model_path: str = None):
        """
        Initializes the inference engine.

        Args:
            override_model_path (str, optional): Explicit path to a .keras file.
        """
        self.model_path = override_model_path or str(settings.BASE_DIR / settings.MODEL_PATH)
        self.model = self._load_model()

    def _load_model(self) -> keras.Model:
        """
        Loads the Keras model from the local filesystem with an S3 fallback.

        Returns:
            keras.Model: The loaded model instance.
        """
        if not Path(self.model_path).exists():
            logger.info("Local model footprint missing, downloading from S3")
            self._download_from_s3()

        logger.info("Instantiating Keras model", path=self.model_path)
        return keras.models.load_model(self.model_path)

    def _download_from_s3(self):
        """
        Transfers the trained model from the central S3 registry to local storage.

        Raises:
            Exception: If AWS credentials or network connectivity fail.
        """
        try:
            s3_client = boto3.client("s3")
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(
                settings.AWS_S3_BUCKET,
                settings.MODEL_S3_KEY,
                self.model_path
            )
            logger.info("Artifact successfully synchronized from S3")
        except Exception as download_error:
            logger.error("S3 artifact synchronization failed", error=str(download_error))
            raise

    def preprocess_image(self, raw_image_bytes: bytes) -> tf.Tensor:
        """
        Converts raw image bytes into a preprocessed tensor batch.
        Uses EfficientNetV2 preprocessing to match training pipeline exactly.

        Args:
            raw_image_bytes (bytes): Binary content of the uploaded image.

        Returns:
            tf.Tensor: A tensor of shape (1, 224, 224, 3) ready for prediction.
        """
        decoded_image = tf.io.decode_image(raw_image_bytes, channels=3)
        resized_image = tf.image.resize(decoded_image, [settings.IMAGE_SIZE, settings.IMAGE_SIZE])
        preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(
            tf.cast(resized_image, tf.float32)
        )
        return tf.expand_dims(preprocessed, axis=0)

    def predict(self, raw_image_bytes: bytes) -> dict:
        """
        Performs binary classification of an MRI scan.

        Args:
            raw_image_bytes (bytes): Binary content of the image.

        Returns:
            dict: Classification results including label, probability, class_idx.
        """
        input_tensor = self.preprocess_image(raw_image_bytes)
        raw_prediction = self.model.predict(input_tensor, verbose=0)[0][0]
        confidence_score = float(raw_prediction)
        is_tumor = confidence_score > settings.CONFIDENCE_THRESHOLD

        return {
            "label": "Tumor" if is_tumor else "No Tumor",
            "probability": round(confidence_score, 4),
            "class_idx": 1 if is_tumor else 0
        }
