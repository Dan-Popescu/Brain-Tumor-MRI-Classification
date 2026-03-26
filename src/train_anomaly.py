"""CNN Autoencoder anomaly detector trained on normal brain scans only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from spark_jobs.config_utils import (
    as_positive_float,
    as_positive_int,
    as_positive_int_or_none,
    load_config,
)
from training.dataset_artifacts import (
    get_label_count,
    load_dataset_summary,
    load_split_files,
    lookup_label_idx,
    resolve_dataset_artifact_paths,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN autoencoder for anomaly detection on normal MRI scans."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/train_anomaly.yaml",
        help="YAML/JSON config path",
    )
    return parser.parse_args()


def _parse_image_size(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or int(value[0]) <= 0
        or int(value[1]) <= 0
    ):
        raise ValueError("`image_size` must be [height, width] with positive ints.")
    return int(value[0]), int(value[1])
def _resolve_settings(config: dict[str, Any]) -> dict[str, Any]:
    image_height, image_width = _parse_image_size(
        config.get("image_size", [224, 224])
    )
    model_output_dir = Path(
        str(config.get("model_output_dir", "models/anomaly_autoencoder"))
    )

    return {
        "input_tfrecord_path": str(
            config.get(
                "input_tfrecord_path", "data/processed/training_tfrecord/current"
            )
        ),
        "seed": as_positive_int(config.get("seed", 42), "seed"),
        "image_height": image_height,
        "image_width": image_width,
        "batch_size": as_positive_int(config.get("batch_size", 32), "batch_size"),
        "epochs": as_positive_int(config.get("epochs", 50), "epochs"),
        "learning_rate": as_positive_float(
            config.get("learning_rate"), "learning_rate", 1e-3
        ),
        "shuffle_buffer": as_positive_int(
            config.get("shuffle_buffer", 2048), "shuffle_buffer"
        ),
        "latent_dim": as_positive_int(
            config.get("latent_dim", 128), "latent_dim"
        ),
        "normal_label_name": str(config.get("normal_label_name", "_NORMAL")),
        "normal_label_idx": config.get("normal_label_idx"),
        "steps_per_epoch": as_positive_int_or_none(
            config.get("steps_per_epoch"), "steps_per_epoch"
        ),
        "early_stopping_patience": as_positive_int_or_none(
            config.get("early_stopping_patience"), "early_stopping_patience"
        ),
        "model_output_dir": str(model_output_dir),
        "best_model_path": str(
            model_output_dir
            / str(config.get("best_model_filename", "best_autoencoder.keras"))
        ),
        "final_model_path": str(
            model_output_dir
            / str(config.get("final_model_filename", "final_autoencoder.keras"))
        ),
        "threshold_path": str(
            model_output_dir
            / str(config.get("threshold_filename", "anomaly_threshold.json"))
        ),
        "run_summary_path": str(
            model_output_dir
            / str(
                config.get("run_summary_filename", "anomaly_train_run_summary.json")
            )
        ),
    }


def load_settings(
    config_path: str | None = "conf/train_anomaly.yaml",
) -> dict[str, Any]:
    return _resolve_settings(load_config(config_path))


# ── TFRecord parsing ────────────────────────────────────────────────────────

_FEATURE_SPEC = {
    "image_bytes": tf.io.FixedLenFeature([], tf.string),
    "label_idx": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_tfrecord(
    serialized_record: tf.Tensor, image_height: int, image_width: int
) -> tuple[tf.Tensor, tf.Tensor]:
    """Parse a TFRecord and return (image, label_idx)."""
    parsed = tf.io.parse_single_example(serialized_record, _FEATURE_SPEC)
    image = tf.image.decode_image(
        parsed["image_bytes"], channels=1, expand_animations=False
    )
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape((image_height, image_width, 1))
    label = tf.cast(parsed["label_idx"], tf.int32)
    return image, label


def _parse_tfrecord_autoencoder(
    serialized_record: tf.Tensor, image_height: int, image_width: int
) -> tuple[tf.Tensor, tf.Tensor]:
    """Parse for autoencoder: input=image, target=image."""
    image, _ = _parse_tfrecord(serialized_record, image_height, image_width)
    return image, image


# ── File collection ──────────────────────────────────────────────────────────
def _load_dataset_inputs(
    input_tfrecord_path: str,
) -> tuple[Path, dict[str, list[str]], dict[str, Any]]:
    root, shard_manifest_path, summary_path = resolve_dataset_artifact_paths(
        input_tfrecord_path
    )
    if not root.exists():
        raise FileNotFoundError(f"TFRecord root not found: {root}")

    split_files = load_split_files(root, shard_manifest_path)
    dataset_summary = load_dataset_summary(summary_path)
    return root, split_files, dataset_summary


# ── Normal-only filtering ────────────────────────────────────────────────────


def _detect_normal_label_idx(
    normal_label_idx: int | None,
    normal_label_name: str,
    dataset_summary: dict[str, Any],
) -> int:
    """Return the label_idx corresponding to normal scans."""
    if normal_label_idx is not None:
        return int(normal_label_idx)

    summary_label_idx = lookup_label_idx(dataset_summary, normal_label_name)
    if summary_label_idx is not None:
        print(
            f"[anomaly] resolved normal_label_idx={summary_label_idx} "
            f"from dataset summary for '{normal_label_name}'"
        )
        return summary_label_idx

    raise ValueError(
        "Could not resolve `normal_label_idx` from dataset_summary.json. "
        "Provide `normal_label_idx` explicitly or ensure the summary contains "
        f"the pathology '{normal_label_name}'."
    )


def _build_normal_only_dataset(
    files: list[str],
    normal_label_idx: int,
    image_height: int,
    image_width: int,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
    shuffle: bool,
    repeat: bool,
    known_normal_count: int,
) -> tuple[tf.data.Dataset, int]:
    """Build a dataset containing only normal scans (autoencoder target=input).

    Returns (dataset, num_normal_examples).
    """
    if not files:
        raise ValueError("No files provided for normal-only dataset.")

    cycle_length = min(16, max(1, len(files)))
    normal_count = int(known_normal_count)

    # Rebuild dataset pipeline with filter
    file_ds2 = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        file_ds2 = file_ds2.shuffle(
            buffer_size=max(1, len(files)),
            reshuffle_each_iteration=True,
            seed=seed,
        )
    raw_ds2 = file_ds2.interleave(
        lambda path: tf.data.TFRecordDataset(path),
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,
    )

    def _filter_and_parse(record: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        parsed = tf.io.parse_single_example(record, _FEATURE_SPEC)
        label = tf.cast(parsed["label_idx"], tf.int32)
        image = tf.image.decode_image(
            parsed["image_bytes"], channels=1, expand_animations=False
        )
        image = tf.cast(image, tf.float32) / 255.0
        image.set_shape((image_height, image_width, 1))
        return image, image, label

    parsed_ds = raw_ds2.map(_filter_and_parse, num_parallel_calls=tf.data.AUTOTUNE)

    # Filter to keep only normal
    normal_ds = parsed_ds.filter(
        lambda img, target, label: tf.equal(label, normal_label_idx)
    )
    # Drop the label column – autoencoder only needs (image, image)
    normal_ds = normal_ds.map(
        lambda img, target, label: (img, target),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        normal_ds = normal_ds.shuffle(
            buffer_size=shuffle_buffer,
            reshuffle_each_iteration=True,
            seed=seed,
        )
    if repeat:
        normal_ds = normal_ds.repeat()

    normal_ds = normal_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return normal_ds, int(normal_count)


# ── Model ────────────────────────────────────────────────────────────────────


def _build_autoencoder(
    image_height: int,
    image_width: int,
    latent_dim: int,
    learning_rate: float,
) -> tf.keras.Model:
    """CNN autoencoder: encoder compresses, decoder reconstructs."""

    # ── Encoder ──
    encoder_input = tf.keras.layers.Input(
        shape=(image_height, image_width, 1), name="encoder_input"
    )
    x = tf.keras.layers.Conv2D(
        32, kernel_size=3, strides=2, padding="same", activation="relu"
    )(encoder_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        64, kernel_size=3, strides=2, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        128, kernel_size=3, strides=2, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        256, kernel_size=3, strides=2, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    shape_before_flatten = x.shape[1:]  # (h, w, c) after encoding
    x = tf.keras.layers.Flatten()(x)
    bottleneck = tf.keras.layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)

    # ── Decoder ──
    h_enc = int(shape_before_flatten[0])
    w_enc = int(shape_before_flatten[1])
    c_enc = int(shape_before_flatten[2])

    x = tf.keras.layers.Dense(h_enc * w_enc * c_enc, activation="relu")(bottleneck)
    x = tf.keras.layers.Reshape((h_enc, w_enc, c_enc))(x)

    x = tf.keras.layers.Conv2DTranspose(
        128, kernel_size=3, strides=2, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(
        64, kernel_size=3, strides=2, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(
        32, kernel_size=3, strides=2, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(
        1, kernel_size=3, strides=2, padding="same", activation="sigmoid"
    )(x)

    # Crop/resize to exact input dimensions (handles rounding from strides)
    decoder_output = tf.keras.layers.Resizing(
        image_height, image_width, name="decoder_output"
    )(x)

    autoencoder = tf.keras.Model(
        inputs=encoder_input, outputs=decoder_output, name="cnn_autoencoder"
    )
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return autoencoder


# ── Threshold calibration ────────────────────────────────────────────────────


def _calibrate_threshold(
    model: tf.keras.Model,
    normal_ds_no_repeat: tf.data.Dataset,
    num_normal: int,
) -> dict[str, float]:
    """Compute per-pixel MSE on all normal images and set threshold so 0% are flagged.

    Strategy:
      1. For each normal image compute the per-pixel reconstruction error map.
      2. For each image take the *maximum* pixel error (worst-case pixel).
      3. The global threshold = max over all normal images of that worst-case pixel.
         This guarantees that **no** normal image contains any pixel above threshold.
      4. We also store the 99th-percentile and mean of the max-pixel distribution
         for optional softer thresholds in the app.
    """
    print(f"[anomaly] calibrating threshold on {num_normal} normal images...")

    max_pixel_errors: list[float] = []
    mean_pixel_errors: list[float] = []

    for batch_images, _ in normal_ds_no_repeat:
        reconstructed = model.predict(batch_images, verbose=0)
        # Per-pixel squared error: (batch, H, W, 1)
        pixel_errors = np.square(batch_images.numpy() - reconstructed)

        for i in range(pixel_errors.shape[0]):
            error_map = pixel_errors[i]  # (H, W, 1)
            max_pixel_errors.append(float(np.max(error_map)))
            mean_pixel_errors.append(float(np.mean(error_map)))

    max_pixel_errors_arr = np.array(max_pixel_errors)
    mean_pixel_errors_arr = np.array(mean_pixel_errors)

    # Threshold: the absolute maximum pixel error across all normal images
    # ensures zero false positives on the normal set
    threshold_strict = float(np.max(max_pixel_errors_arr))
    threshold_p99 = float(np.percentile(max_pixel_errors_arr, 99))
    threshold_p95 = float(np.percentile(max_pixel_errors_arr, 95))

    result = {
        "threshold_strict": threshold_strict,
        "threshold_p99": threshold_p99,
        "threshold_p95": threshold_p95,
        "normal_images_evaluated": num_normal,
        "max_pixel_error_mean": float(np.mean(max_pixel_errors_arr)),
        "max_pixel_error_std": float(np.std(max_pixel_errors_arr)),
        "mean_pixel_error_mean": float(np.mean(mean_pixel_errors_arr)),
        "mean_pixel_error_std": float(np.std(mean_pixel_errors_arr)),
    }

    print(f"[anomaly] threshold_strict (0% FP on normals): {threshold_strict:.6f}")
    print(f"[anomaly] threshold_p99:                       {threshold_p99:.6f}")
    print(f"[anomaly] threshold_p95:                       {threshold_p95:.6f}")

    return result


# ── Main training loop ───────────────────────────────────────────────────────


def run_anomaly_training(settings: dict[str, Any]) -> dict[str, Any]:
    resolved = _resolve_settings(settings)
    tf.keras.utils.set_random_seed(resolved["seed"])

    tfrecord_root, split_files, dataset_summary = _load_dataset_inputs(
        resolved["input_tfrecord_path"]
    )

    # Detect which label_idx corresponds to normal
    normal_label_idx = _detect_normal_label_idx(
        resolved.get("normal_label_idx"),
        resolved["normal_label_name"],
        dataset_summary,
    )
    resolved["normal_label_idx"] = normal_label_idx

    train_normal_count = get_label_count(
        dataset_summary,
        split="train",
        label_idx=normal_label_idx,
    )
    val_normal_count = get_label_count(
        dataset_summary,
        split="val",
        label_idx=normal_label_idx,
    )
    if train_normal_count is None:
        raise ValueError(
            "Missing normal label counts for train split in dataset_summary.json."
        )
    if split_files["val"] and val_normal_count is None:
        raise ValueError(
            "Missing normal label counts for val split in dataset_summary.json."
        )
    if val_normal_count is None:
        val_normal_count = 0

    # ── Training dataset (normal only, train split) ──
    train_ds, num_train_normal = _build_normal_only_dataset(
        files=split_files["train"],
        normal_label_idx=normal_label_idx,
        image_height=resolved["image_height"],
        image_width=resolved["image_width"],
        batch_size=resolved["batch_size"],
        shuffle_buffer=resolved["shuffle_buffer"],
        seed=resolved["seed"],
        shuffle=True,
        repeat=True,
        known_normal_count=train_normal_count,
    )
    print(f"[anomaly] normal training examples: {num_train_normal}")
    if num_train_normal == 0:
        raise ValueError("No normal training examples found.")

    # ── Validation dataset (normal only, val split) ──
    val_ds: tf.data.Dataset | None = None
    num_val_normal = 0
    if split_files["val"]:
        val_ds, num_val_normal = _build_normal_only_dataset(
            files=split_files["val"],
            normal_label_idx=normal_label_idx,
            image_height=resolved["image_height"],
            image_width=resolved["image_width"],
            batch_size=resolved["batch_size"],
            shuffle_buffer=resolved["shuffle_buffer"],
            seed=resolved["seed"],
            shuffle=False,
            repeat=False,
            known_normal_count=val_normal_count,
        )
        print(f"[anomaly] normal validation examples: {num_val_normal}")

    # ── Build model ──
    autoencoder = _build_autoencoder(
        image_height=resolved["image_height"],
        image_width=resolved["image_width"],
        latent_dim=resolved["latent_dim"],
        learning_rate=resolved["learning_rate"],
    )
    autoencoder.summary()

    output_dir = Path(resolved["model_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=resolved["best_model_path"],
            monitor="val_loss" if val_ds is not None else "loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]
    if resolved["early_stopping_patience"] is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if val_ds is not None else "loss",
                mode="min",
                patience=resolved["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            )
        )

    fit_kwargs: dict[str, Any] = {
        "epochs": resolved["epochs"],
        "callbacks": callbacks,
    }

    if resolved["steps_per_epoch"] is not None:
        fit_kwargs["steps_per_epoch"] = resolved["steps_per_epoch"]
    else:
        fit_kwargs["steps_per_epoch"] = max(
            1, math.ceil(num_train_normal / resolved["batch_size"])
        )

    if val_ds is not None:
        fit_kwargs["validation_data"] = val_ds
        fit_kwargs["validation_steps"] = max(
            1, math.ceil(num_val_normal / resolved["batch_size"])
        )

    history = autoencoder.fit(train_ds, **fit_kwargs)
    autoencoder.save(resolved["final_model_path"])

    # ── Calibrate threshold on ALL normal images (train + val) ──
    # Build a non-repeating, non-shuffled dataset of all normal images
    all_normal_files = split_files["train"] + split_files.get("val", [])
    calibration_ds, num_calibration = _build_normal_only_dataset(
        files=all_normal_files,
        normal_label_idx=normal_label_idx,
        image_height=resolved["image_height"],
        image_width=resolved["image_width"],
        batch_size=resolved["batch_size"],
        shuffle_buffer=1,
        seed=resolved["seed"],
        shuffle=False,
        repeat=False,
        known_normal_count=int(train_normal_count) + int(val_normal_count),
    )

    # Load best model for threshold calibration
    best_model = tf.keras.models.load_model(resolved["best_model_path"])
    threshold_info = _calibrate_threshold(
        best_model, calibration_ds, num_calibration
    )

    # Save threshold
    threshold_payload = {
        **threshold_info,
        "normal_label_idx": normal_label_idx,
        "normal_label_name": resolved["normal_label_name"],
        "image_height": resolved["image_height"],
        "image_width": resolved["image_width"],
    }
    Path(resolved["threshold_path"]).write_text(
        json.dumps(threshold_payload, indent=2), encoding="utf-8"
    )
    print(f"[anomaly] threshold saved to {resolved['threshold_path']}")

    # ── Summary ──
    summary = {
        "settings": resolved,
        "dataset_summary_used": True,
        "normal_label_idx": normal_label_idx,
        "num_train_normal": num_train_normal,
        "num_val_normal": num_val_normal,
        "num_calibration_normal": num_calibration,
        "threshold": threshold_info,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "best_model_path": resolved["best_model_path"],
        "final_model_path": resolved["final_model_path"],
        "threshold_path": resolved["threshold_path"],
    }
    Path(resolved["run_summary_path"]).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = _parse_args()
    settings = load_settings(args.config)
    result = run_anomaly_training(settings)

    print("[anomaly] settings:")
    print(json.dumps(result["settings"], indent=2))
    print(f"[anomaly] normal train examples: {result['num_train_normal']}")
    print(f"[anomaly] normal val examples: {result['num_val_normal']}")
    print(f"[anomaly] calibration examples: {result['num_calibration_normal']}")
    print(f"[anomaly] strict threshold: {result['threshold']['threshold_strict']:.6f}")
    print(f"[anomaly] best model: {result['best_model_path']}")
    print(f"[anomaly] threshold file: {result['threshold_path']}")


if __name__ == "__main__":
    main()
