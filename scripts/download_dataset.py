from src.common.logging import logger, setup_logging
from src.data_pipeline.data_ingestion import DataIngestion

setup_logging()


def execute_download_pipeline():
    """
    Triggers the end-to-end dataset ingestion process.

    This script serves as a simple entry point to hydrate the local data
    directory using the Kaggle API.
    """
    logger.info("Initializing automated data ingestion")

    try:
        data_engine = DataIngestion()
        data_engine.download_dataset()
        logger.info("Data ingestion completed successfully")

    except Exception as pipeline_error:
        logger.error("Data ingestion pipeline crashed", error=str(pipeline_error))


if __name__ == "__main__":
    execute_download_pipeline()
