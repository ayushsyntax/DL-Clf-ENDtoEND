"""
Execution entry point for brain tumor classification training.
Orchestrates hardware log silencing and pipeline initialization.
Acts as the main script for background training runs.
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import structlog
from src.training_pipeline.train import TrainingPipeline

logger = structlog.get_logger()


def launch_training_procedure() -> None:
    """Initialize and run the full training experiment lifecycle.

    Executes the top-down sequence from data preparation to archival.
    Handles top-level execution crashes gracefully via direct logging.
    """
    logger.info("Starting background training lifecycle context")

    try:
        pipeline_service = TrainingPipeline()
        pipeline_service.run_full_training_lifecycle()

        logger.info("Training cycle successfully terminated")
        sys.exit(0)

    except Exception as hardware_failure_or_software_bug:
        logger.error(
            "Fatal training failure identified and recorded",
            error_message=str(hardware_failure_or_software_bug)
        )
        sys.exit(1)


if __name__ == "__main__":
    launch_training_procedure()
