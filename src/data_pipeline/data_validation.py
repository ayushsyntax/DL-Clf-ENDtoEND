from pathlib import Path

import pandas as pd

from src.common.config import settings
from src.common.logging import logger


class DataValidator:
    """
    Scans raw Kaggle folders, applies binary label mapping, and returns a clean DataFrame.

    4 raw classes (glioma, meningioma, pituitary, notumor) → binary (0=notumor, 1=tumor).
    """

    def __init__(self, raw_data_path: Path):
        self.raw_data_path = raw_data_path
        self.classes = ["glioma", "meningioma", "pituitary", "notumor"]
        self.binary_map = {"glioma": 1, "meningioma": 1, "pituitary": 1, "notumor": 0}

    def validate_and_map(self) -> pd.DataFrame:
        """
        Walk Training/Testing directories, map labels, deduplicate, and return DataFrame.

        Returns:
            pd.DataFrame: Columns — filepath, original_label, label (binary).

        Raises:
            ValueError: If no .jpg images are found.
        """
        records = []

        for subset in ["Training", "Testing"]:
            subset_path = self.raw_data_path / subset
            if not subset_path.exists():
                logger.warning("Subset missing, skipping", path=str(subset_path))
                continue
            for label in self.classes:
                label_path = subset_path / label
                if not label_path.exists():
                    continue
                for img_path in label_path.glob("*.jpg"):
                    records.append({
                        "filepath": str(img_path.absolute()),
                        "original_label": label,
                        "label": self.binary_map[label]
                    })

        df = pd.DataFrame(records)

        if df.empty:
            raise ValueError(f"No .jpg images found in {self.raw_data_path}")

        duplicates = df.duplicated(subset=["filepath"]).sum()
        if duplicates > 0:
            logger.warning("Duplicates removed", count=duplicates)
            df = df.drop_duplicates(subset=["filepath"])

        logger.info(
            "Validation complete",
            total=len(df),
            distribution=df["label"].value_counts().to_dict()
        )
        return df


if __name__ == "__main__":
    validator = DataValidator(Path(settings.RAW_DATA_DIR))
    try:
        df = validator.validate_and_map()
        logger.info("Success", rows=len(df))
    except Exception as e:
        logger.error("Validation failed", error=str(e))
