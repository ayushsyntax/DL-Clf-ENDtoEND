import hashlib
import subprocess
from pathlib import Path

from src.common.logging import logger


class HashingUtils:
    """
    Provides utility methods for cryptographic hashing and data integrity.
    """

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """
        Computes the SHA-256 hash of a specific file.

        Args:
            file_path (Path): Absolute path to the file.

        Returns:
            str: Hexadecimal string of the SHA-256 hash.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


class DVCUtils:
    """
    Encapsulates DVC command interactions for data and model versioning.
    """

    @staticmethod
    def push_artifacts():
        """
        Triggers a 'dvc push' command if a DVC repository is detected.

        This assumes that the DVC remote (e.g., S3) is already configured.
        """
        if not Path(".dvc").exists():
            logger.warning("DVC directory (.dvc) not found. Skipping push.")
            return

        try:
            logger.info("Executing DVC push for artifact versioning")
            subprocess.run(["dvc", "push"], check=True, capture_output=True)
            logger.info("DVC artifacts pushed successfully")
        except subprocess.CalledProcessError as e:
            logger.error("DVC push failed", error=str(e.stderr.decode()))
        except Exception as e:
            logger.error("Error during DVC interaction", error=str(e))
