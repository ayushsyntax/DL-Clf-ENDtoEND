import pandas
import structlog
import tensorflow
from src.common.config import settings
from src.data_pipeline.data_validation import DataValidator


def load_and_preprocess_image(
    image_path: str,
    label: int
) -> tuple[tensorflow.Tensor, tensorflow.Tensor]:
    """Decode JPEG, resize, and apply EfficientNetV2 scaling."""
    raw = tensorflow.io.read_file(image_path)
    image = tensorflow.image.decode_jpeg(raw, channels=3)
    image = tensorflow.image.resize(image, [settings.IMAGE_SIZE, settings.IMAGE_SIZE])
    image = tensorflow.keras.applications.efficientnet_v2.preprocess_input(image)
    return image, label


def apply_training_augmentations(
    image: tensorflow.Tensor,
    label: tensorflow.Tensor
) -> tuple[tensorflow.Tensor, tensorflow.Tensor]:
    """Random flip, brightness, and contrast — training only."""
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
    """Build batched, prefetched tf.data pipeline. Cache disabled to avoid WSL OOM."""
    ds = tensorflow.data.Dataset.from_tensor_slices((file_paths, labels))

    if should_shuffle:
        ds = ds.shuffle(buffer_size=len(labels))

    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tensorflow.data.AUTOTUNE)

    if should_augment:
        ds = ds.map(apply_training_augmentations, num_parallel_calls=tensorflow.data.AUTOTUNE)

    return ds.batch(settings.BATCH_SIZE).prefetch(tensorflow.data.AUTOTUNE)


def get_experimental_datasets() -> tuple[
    tensorflow.data.Dataset,
    tensorflow.data.Dataset,
    tensorflow.data.Dataset,
    pandas.DataFrame
]:
    """
    Build train / val / test datasets from raw directory.

    Val split: 20% of Training data (stratified by filepath, random_state=42).

    Returns:
        train_ds, val_ds, test_ds, full_metadata_df
    """
    logger = structlog.get_logger()
    logger.info("Building datasets", raw_path=str(settings.RAW_DATA_DIR))

    df = DataValidator(settings.RAW_DATA_DIR).validate_and_map()

    train_meta = df[df["filepath"].str.contains("Training")]
    test_meta = df[df["filepath"].str.contains("Testing")]
    val_meta = train_meta.sample(frac=0.2, random_state=42)
    train_meta = train_meta.drop(val_meta.index)

    train_ds = create_tensorflow_dataset(
        train_meta["filepath"].tolist(), train_meta["label"].tolist(),
        should_augment=True, should_shuffle=True
    )
    val_ds = create_tensorflow_dataset(
        val_meta["filepath"].tolist(), val_meta["label"].tolist()
    )
    test_ds = create_tensorflow_dataset(
        test_meta["filepath"].tolist(), test_meta["label"].tolist()
    )

    logger.info("Splits ready", train=len(train_meta), val=len(val_meta), test=len(test_meta))
    return train_ds, val_ds, test_ds, df
