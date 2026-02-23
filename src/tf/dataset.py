"""TF data pipeline: progressive loading of TFRecord shards for model training."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


# ---------------------------------------------------------------------------
# TFRecord feature spec (must match export_tfrecords_job.py)
# ---------------------------------------------------------------------------

_FEATURE_SPEC = {
    "image_raw": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "pathology": tf.io.FixedLenFeature([], tf.string),
    "modality": tf.io.FixedLenFeature([], tf.string),
}


# ---------------------------------------------------------------------------
# Parsing & decoding
# ---------------------------------------------------------------------------

def _parse_example(serialized: tf.Tensor) -> dict[str, tf.Tensor]:
    """Parse a single TFRecord example into a dict of tensors."""
    return tf.io.parse_single_example(serialized, _FEATURE_SPEC)


def _decode_and_normalize(
    parsed: dict[str, tf.Tensor],
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
) -> tuple[tf.Tensor, tf.Tensor]:
    """Decode PNG bytes → float32 image [0,1] and one-hot label."""
    image = tf.io.decode_png(parsed["image_raw"], channels=1)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(parsed["label"], tf.int32)
    label_onehot = tf.one_hot(label, depth=num_classes)

    return image, label_onehot


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Light data augmentation for training set."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset(
    tfrecord_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer: int = 1000,
    num_classes: int = 15,
    image_size: tuple[int, int] = (224, 224),
    augment: bool = False,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset that streams TFRecord shards progressively.

    Parameters
    ----------
    tfrecord_dir : str
        Directory containing shard-XXXX.tfrecord files (e.g. "data/processed/tfrecords/v1/train").
    batch_size : int
        Number of examples per batch.
    shuffle : bool
        Whether to shuffle the dataset (recommended for training).
    shuffle_buffer : int
        Buffer size for shuffling.
    num_classes : int
        Number of distinct classes for one-hot encoding.
    image_size : tuple[int, int]
        Target (height, width) for images.
    augment : bool
        Whether to apply data augmentation (only for training).

    Returns
    -------
    tf.data.Dataset
        Yields (image, label_onehot) batches ready for model.fit().
    """
    pattern = str(Path(tfrecord_dir) / "*.tfrecord")
    files_ds = tf.data.Dataset.list_files(pattern, shuffle=shuffle)

    ds = files_ds.interleave(
        lambda path: tf.data.TFRecordDataset(path),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,
    )

    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda parsed: _decode_and_normalize(parsed, num_classes, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_datasets(
    tfrecords_root: str,
    batch_size: int = 32,
    num_classes: int = 15,
    image_size: tuple[int, int] = (224, 224),
    shuffle_buffer: int = 1000,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build train, val, and test datasets from a TFRecords export directory.

    Parameters
    ----------
    tfrecords_root : str
        Root directory containing train/, val/, test/ subdirectories
        (e.g. "data/processed/tfrecords/v1").
    batch_size : int
        Number of examples per batch.
    num_classes : int
        Number of distinct classes for one-hot encoding.
    image_size : tuple[int, int]
        Target (height, width) for images.
    shuffle_buffer : int
        Buffer size for training shuffle.

    Returns
    -------
    tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]
        (train_ds, val_ds, test_ds) ready for model.fit() / model.evaluate().
    """
    train_ds = build_dataset(
        tfrecord_dir=str(Path(tfrecords_root) / "train"),
        batch_size=batch_size,
        shuffle=True,
        shuffle_buffer=shuffle_buffer,
        num_classes=num_classes,
        image_size=image_size,
        augment=True,
    )
    val_ds = build_dataset(
        tfrecord_dir=str(Path(tfrecords_root) / "val"),
        batch_size=batch_size,
        shuffle=False,
        num_classes=num_classes,
        image_size=image_size,
        augment=False,
    )
    test_ds = build_dataset(
        tfrecord_dir=str(Path(tfrecords_root) / "test"),
        batch_size=batch_size,
        shuffle=False,
        num_classes=num_classes,
        image_size=image_size,
        augment=False,
    )
    return train_ds, val_ds, test_ds
