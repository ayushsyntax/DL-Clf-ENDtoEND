from pathlib import Path

import boto3

from src.common.config import settings
from src.common.logging import logger, setup_logging

setup_logging()


def upload_trained_model():
    """
    Transfers the best local model artifact to the central AWS S3 registry.

    This script is intended for use in CI/CD or post-training workflows to
    ensure the model registry is updated with the latest performance candidate.
    """
    local_model_path = settings.ARTIFACTS_DIR / "best_model.keras"

    if not local_model_path.exists():
        logger.error(
            "Model artifact footprint not found, skipping upload",
            path=str(local_model_path)
        )
        return

    logger.info(
        "Initiating model artifact upload to S3",
        bucket=settings.AWS_S3_BUCKET,
        key=settings.MODEL_S3_PATH
    )

    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(
            str(local_model_path),
            settings.AWS_S3_BUCKET,
            settings.MODEL_S3_PATH
        )
        logger.info("Artifact upload sequence finalized successfully")

    except Exception as s3_error:
        logger.error("AWS S3 interaction failed", error=str(s3_error))


if __name__ == "__main__":
    upload_trained_model()
