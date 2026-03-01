import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from src.common.config import settings
from src.common.logging import logger, setup_logging

os.environ["KAGGLE_USERNAME"] = settings.KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = settings.KAGGLE_KEY

setup_logging()


class DataIngestion:
    """Downloads and extracts the brain tumor MRI dataset from Kaggle."""

    def __init__(self):
        """Authenticate with Kaggle API using credentials from settings."""
        self.api = KaggleApi()
        self.api.authenticate()

    def download_dataset(self) -> None:
        """Download and unzip dataset into RAW_DATA_DIR."""
        raw_dir = Path(settings.RAW_DATA_DIR)
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading dataset", dataset=settings.KAGGLE_DATASET)
        try:
            self.api.dataset_download_files(
                settings.KAGGLE_DATASET,
                path=raw_dir,
                unzip=True
            )
            logger.info("Dataset ready", path=str(raw_dir))
        except Exception as e:
            logger.error("Download failed", error=str(e))
            raise


if __name__ == "__main__":
    DataIngestion().download_dataset()
