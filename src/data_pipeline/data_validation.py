from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.common.config import settings
from src.common.logging import logger


class DataValidator:
    """
    Validates raw data, performs binary mapping, and checks for corruption.

    Attributes:
        raw_data_path (Path): Path to the raw Kaggle dataset.
        classes (List[str]): Expected original class names in the dataset.
        binary_map (Dict[str, int]): Mapping from original labels to binary classes.
    """

    def __init__(self, raw_data_path: Path):
        """
        Initializes the DataValidator with the source directory.

        Args:
            raw_data_path (Path): Path to the raw dataset directory.
        """
        self.raw_data_path = raw_data_path
        self.classes = ["glioma", "meningioma", "pituitary", "notumor"]
        self.binary_map = {
            "glioma": 1,
            "meningioma": 1,
            "pituitary": 1,
            "notumor": 0
        }

    def validate_and_map(self) -> pd.DataFrame:
        """
        Scans data directory, maps to binary classes, and returns a validated DataFrame.

        The method iterates through Training and Testing directories, checks for
        image files, and consolidates them into a single pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing 'filepath', 'original_label', and 'label'.

        Raises:
            ValueError: If no valid JPEG images are found in the directory.
        """
        data_records = []

        for subset in ["Training", "Testing"]:
            subset_path = self.raw_data_path / subset
            if not subset_path.exists():
                logger.warning("Subset path does not exist. Skipping.", path=str(subset_path))
                continue

            for label in self.classes:
                label_path = subset_path / label
                if not label_path.exists():
                    continue

                for img_path in label_path.glob("*.jpg"):
                    data_records.append({
                        "filepath": str(img_path.absolute()),
                        "original_label": label,
                        "label": self.binary_map[label]
                    })

        df = pd.DataFrame(data_records)

        if df.empty:
            raise ValueError(f"No valid .jpg images found in {self.raw_data_path}")

        logger.info("Data validation and binary mapping complete",
                    total_samples=len(df),
                    class_distribution=df['label'].value_counts().to_dict())

        duplicates = df.duplicated(subset=['filepath']).sum()
        if duplicates > 0:
            logger.warning("Duplicates detected in dataset", count=duplicates)
            df = df.drop_duplicates(subset=['filepath'])

        return df


if __name__ == "__main__":
    validator = DataValidator(Path(settings.RAW_DATA_DIR))
    try:
        final_df = validator.validate_and_map()
        logger.info("Validation successful", first_rows=final_df.head().to_dict())
    except Exception as validation_error:
        logger.error("Validation failed", error=str(validation_error))
