"""
Preprocessing module for brain MRI scans.
Handles directory crawling, data validation, and dataset creation.
Uses EfficientNetV2 standards for input preparation.
"""

import pandas
import tensorflow
import structlog
from src.common.config import settings
from src.data_pipeline.data_validation import DataValidator


def load_and_preprocess_image(
    image_path: str,
    label: int
) -> tuple[tensorflow.Tensor, tensorflow.Tensor]:
    """Load and scale an image for model input.

    Args:
        image_path (str): Path to JPEG file.
        label (int): Target classification label.

    Returns:
        tuple[Tensor, Tensor]: Processed image and label.
    """
    raw_file_bytes = tensorflow.io.read_file(image_path)
    decoded_image = tensorflow.image.decode_jpeg(raw_file_bytes, channels=3)
    resized_image = tensorflow.image.resize(
        decoded_image,
        [settings.IMAGE_SIZE, settings.IMAGE_SIZE]
    )
    preprocessed_image = tensorflow.keras.applications.efficientnet_v2.preprocess_input(
        resized_image
    )

    return preprocessed_image, label


def apply_training_augmentations(
    image: tensorflow.Tensor,
    label: tensorflow.Tensor
) -> tuple[tensorflow.Tensor, tensorflow.Tensor]:
    """Applies random augmentations to a single training image.

    Args:
        image: Float32 image tensor of shape (224, 224, 3).
        label: Binary integer label tensor.

    Returns:
        Tuple of augmented image tensor and unchanged label.
    """
    image = tensorflow.image.random_flip_left_right(image)
    image = tensorflow.image.random_brightness(image, max_delta=0.1)
    image = tensorflow.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def create_tensorflow_dataset(
    file_paths: list[str],
    labels: list[int],
    should_augment: bool = False,
    should_shuffle: bool = False
) -> tensorflow.data.Dataset:
    """Build a high-performance data pipeline.

    Args:
        file_paths (list[str]): List of file strings.
        labels (list[int]): List of binary integers.
        should_augment (bool): Whether to apply random augmentations.
        should_shuffle (bool): Whether to shuffle the data.

    Returns:
        Dataset: Batched and prefetched data.
    """
    dataset = tensorflow.data.Dataset.from_tensor_slices((file_paths, labels))

    if should_shuffle:
        total_samples = len(labels)
        dataset = dataset.shuffle(buffer_size=total_samples)

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tensorflow.data.AUTOTUNE
    )
    # dataset = dataset.cache() # Removed due to OOM issue in WSL

    if should_augment:
        dataset = dataset.map(
            apply_training_augmentations,
            num_parallel_calls=tensorflow.data.AUTOTUNE
        )

    dataset = dataset.batch(settings.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tensorflow.data.AUTOTUNE)

    return dataset


def get_experimental_datasets() -> tuple[tensorflow.data.Dataset, tensorflow.data.Dataset, tensorflow.data.Dataset, pandas.DataFrame]:
    """Create train, validation, and test datasets.

    Returns:
        tuple[Dataset, Dataset, Dataset, DataFrame]: Final grouped datasets and metadata.
    """
    logger = structlog.get_logger()
    logger.info("Initializing dataset distribution", raw_path=str(settings.RAW_DATA_DIR))

    validator = DataValidator(settings.RAW_DATA_DIR)
    full_dataframe = validator.validate_and_map()

    training_metadata = full_dataframe[full_dataframe["filepath"].str.contains("Training")]
    testing_metadata = full_dataframe[full_dataframe["filepath"].str.contains("Testing")]

    validation_metadata = training_metadata.sample(frac=0.2, random_state=42)
    final_training_metadata = training_metadata.drop(validation_metadata.index)

    train_dataset = create_tensorflow_dataset(
        final_training_metadata["filepath"].tolist(),
        final_training_metadata["label"].tolist(),
        should_augment=True,
        should_shuffle=True
    )

    validation_dataset = create_tensorflow_dataset(
        validation_metadata["filepath"].tolist(),
        validation_metadata["label"].tolist(),
        should_augment=False,
        should_shuffle=False
    )

    testing_dataset = create_tensorflow_dataset(
        testing_metadata["filepath"].tolist(),
        testing_metadata["label"].tolist(),
        should_augment=False,
        should_shuffle=False
    )

    logger.info("Dataset splits successfully created")

    return train_dataset, validation_dataset, testing_dataset, full_dataframe
