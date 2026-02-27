import os
from src.common.config import settings

# Set credentials BEFORE kaggle import to bypass kaggle.json requirement
os.environ['KAGGLE_USERNAME'] = settings.KAGGLE_USERNAME
os.environ['KAGGLE_KEY'] = settings.KAGGLE_KEY

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

from src.common.logging import logger, setup_logging

setup_logging()


class DataIngestion:
    """
    Handles dataset downloading and extraction using Kaggle API.
    """

    def __init__(self):
        """
        Initializes the DataIngestion class by authenticating with Kaggle.
        """
        self.api = KaggleApi()
        self.api.authenticate()

    def download_dataset(self):
        """
        Downloads the brain tumor MRI dataset from Kaggle.
        """
        raw_dir = Path(settings.RAW_DATA_DIR)
        raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading dataset from Kaggle", dataset=settings.KAGGLE_DATASET)

        try:
            self.api.dataset_download_files(
                settings.KAGGLE_DATASET,
                path=raw_dir,
                unzip=True
            )
            logger.info("Dataset downloaded and extracted successfully", path=str(raw_dir))
        except Exception as e:
            logger.error("Failed to download dataset", error=str(e))
            raise


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.download_dataset()
