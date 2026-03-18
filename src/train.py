"""TensorFlow training entrypoint from TFRecord shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import tensorflow as tf
from spark_jobs.config_utils import (
    as_positive_float,
    as_positive_int,
    as_positive_int_or_none,
    load_config,
    resolve_local_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline TensorFlow model from TFRecord shards."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/train.yaml",
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


def _to_abs_local_path(path: str) -> Path:
    local = Path(resolve_local_path(path))
    if local.is_absolute():
        return local
    return (Path.cwd() / local).resolve()


def _resolve_settings(config: dict[str, Any]) -> dict[str, Any]:
    image_height, image_width = _parse_image_size(config.get("image_size", [224, 224]))
    model_output_dir = Path(str(config.get("model_output_dir", "models/baseline_cnn")))

    return {
        "input_tfrecord_path": str(
            config.get("input_tfrecord_path", "data/processed/training_tfrecord/current")
        ),
        "seed": as_positive_int(config.get("seed", 42), "seed"),
        "image_height": image_height,
        "image_width": image_width,
        "batch_size": as_positive_int(config.get("batch_size", 32), "batch_size"),
        "epochs": as_positive_int(config.get("epochs", 10), "epochs"),
        "learning_rate": as_positive_float(
            config.get("learning_rate"), "learning_rate", 1e-3
        ),
        "shuffle_buffer": as_positive_int(
            config.get("shuffle_buffer", 2048), "shuffle_buffer"
        ),
        "num_classes": as_positive_int_or_none(config.get("num_classes"), "num_classes"),
        "steps_per_epoch": as_positive_int_or_none(
            config.get("steps_per_epoch"), "steps_per_epoch"
        ),
        "validation_steps": as_positive_int_or_none(
            config.get("validation_steps"), "validation_steps"
        ),
        "early_stopping_patience": as_positive_int_or_none(
            config.get("early_stopping_patience"), "early_stopping_patience"
        ),
        "model_output_dir": str(model_output_dir),
        "best_model_path": str(
            model_output_dir / str(config.get("best_model_filename", "best.keras"))
        ),
        "final_model_path": str(
            model_output_dir / str(config.get("final_model_filename", "final.keras"))
        ),
        "run_summary_path": str(
            model_output_dir
            / str(config.get("run_summary_filename", "train_run_summary.json"))
        ),
    }


def load_settings(config_path: str | None = "conf/train.yaml") -> dict[str, Any]:
    return _resolve_settings(load_config(config_path))


def _collect_split_files(input_tfrecord_path: str) -> tuple[Path, dict[str, list[str]]]:
    tfrecord_root = _to_abs_local_path(input_tfrecord_path)
    if not tfrecord_root.exists():
        raise FileNotFoundError(f"TFRecord root path not found: {tfrecord_root}")

    split_files = {
        "train": sorted(
            str(path) for path in tfrecord_root.glob("split=train/shard_id=*/*.tfrecord")
        ),
        "val": sorted(
            str(path) for path in tfrecord_root.glob("split=val/shard_id=*/*.tfrecord")
        ),
        "test": sorted(
            str(path) for path in tfrecord_root.glob("split=test/shard_id=*/*.tfrecord")
        ),
    }
    if not split_files["train"]:
        raise ValueError(
            "No training TFRecord files found under "
            f"{tfrecord_root} with pattern split=train/shard_id=*/*.tfrecord"
        )
    return tfrecord_root, split_files


def _extract_label_idx(serialized_example: bytes) -> int:
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    label_feature = example.features.feature.get("label_idx")
    if label_feature is None or not label_feature.int64_list.value:
        raise ValueError("Missing `label_idx` in TFRecord example.")
    return int(label_feature.int64_list.value[0])


def _scan_stats(split_files: dict[str, list[str]]) -> tuple[dict[str, int], set[int]]:
    rows_by_split = {"train": 0, "val": 0, "test": 0}
    labels: set[int] = set()

    for split, files in split_files.items():
        if not files:
            continue
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
        for raw_record in dataset:
            serialized = bytes(raw_record.numpy())
            rows_by_split[split] += 1
            labels.add(_extract_label_idx(serialized))

    return rows_by_split, labels


_FEATURE_SPEC = {
    "image_bytes": tf.io.FixedLenFeature([], tf.string),
    "label_idx": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_tfrecord(
    serialized_record: tf.Tensor, image_height: int, image_width: int
) -> tuple[tf.Tensor, tf.Tensor]:
    parsed = tf.io.parse_single_example(serialized_record, _FEATURE_SPEC)
    image = tf.image.decode_image(
        parsed["image_bytes"], channels=1, expand_animations=False
    )

    # Verify shape
    shape = tf.shape(image)
    tf.debugging.assert_equal(
        shape[0], image_height,  message="Image height mismatch in TFRecord. Image Height does not match target image height defined in train.yaml"
    )
    tf.debugging.assert_equal(
        shape[1], image_width, message="Image width mismatch in TFRecord. Image width does not match target image width defined in train.yaml"
    )
    tf.debugging.assert_equal(
        shape[2], 1, message="Image channel mismatch in TFRecord. Image in tfrecord is not grayscale. Expected grayscale. Verify pipeline."
    )

    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape((image_height, image_width, 1))
    label = tf.cast(parsed["label_idx"], tf.int32)
    return image, label


def _build_dataset(
    *,
    files: list[str],
    seed: int,
    image_height: int,
    image_width: int,
    batch_size: int,
    shuffle_buffer: int,
    shuffle_files: bool,
    shuffle_examples: bool,
    repeat: bool,
) -> tf.data.Dataset:
    file_ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle_files:
        file_ds = file_ds.shuffle(
            buffer_size=max(1, len(files)),
            reshuffle_each_iteration=True,
            seed=seed,
        )

    cycle_length = min(16, max(1, len(files)))
    dataset = file_ds.interleave(
        lambda path: tf.data.TFRecordDataset(path),
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle_examples,
    )
    if shuffle_examples:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer,
            reshuffle_each_iteration=True,
            seed=seed,
        )
    dataset = dataset.map(
        lambda record: _parse_tfrecord(record, image_height, image_width),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if repeat:
        dataset = dataset.repeat()
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _build_model(
    *, image_height: int, image_width: int, num_classes: int, learning_rate: float
) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            # block 1
            tf.keras.layers.Input(shape=(image_height, image_width, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # block 2
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # block 3
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # block 4, denser layers
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def run_training(settings: dict[str, Any]) -> dict[str, Any]:
    resolved = _resolve_settings(settings)
    tf.keras.utils.set_random_seed(resolved["seed"])
    tfrecord_root, split_files = _collect_split_files(resolved["input_tfrecord_path"])

    rows_by_split, labels = _scan_stats(split_files)
    rows_train = int(rows_by_split["train"])
    rows_val = int(rows_by_split["val"])
    rows_test = int(rows_by_split["test"])
    if rows_train == 0:
        raise ValueError("No train records found in TFRecord dataset.")

    if resolved["num_classes"] is None:
        if not labels:
            raise ValueError("Unable to infer `num_classes`: no labels found.")
        num_classes = max(labels) + 1
    else:
        num_classes = int(resolved["num_classes"])

    train_ds = _build_dataset(
        files=split_files["train"],
        seed=resolved["seed"],
        image_height=resolved["image_height"],
        image_width=resolved["image_width"],
        batch_size=resolved["batch_size"],
        shuffle_buffer=resolved["shuffle_buffer"],
        shuffle_files=True,
        shuffle_examples=True,
        repeat=True,
    )

    val_ds: tf.data.Dataset | None = None
    if split_files["val"]:
        val_ds = _build_dataset(
            files=split_files["val"],
            seed=resolved["seed"],
            image_height=resolved["image_height"],
            image_width=resolved["image_width"],
            batch_size=resolved["batch_size"],
            shuffle_buffer=resolved["shuffle_buffer"],
            shuffle_files=False,
            shuffle_examples=False,
            repeat=False,
        )

    model = _build_model(
        image_height=resolved["image_height"],
        image_width=resolved["image_width"],
        num_classes=num_classes,
        learning_rate=resolved["learning_rate"],
    )

    output_dir = Path(resolved["model_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    monitor_metric = "val_accuracy" if val_ds is not None else "accuracy"
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=resolved["best_model_path"],
            monitor=monitor_metric,
            mode="max",
            save_best_only=True,
            verbose=1,
        )
    ]
    if resolved["early_stopping_patience"] is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                mode="max",
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
            1, math.ceil(rows_train / resolved["batch_size"])
        )

    if val_ds is not None:
        fit_kwargs["validation_data"] = val_ds
        if resolved["validation_steps"] is not None:
            fit_kwargs["validation_steps"] = resolved["validation_steps"]
        else:
            fit_kwargs["validation_steps"] = max(
                1, math.ceil(rows_val / resolved["batch_size"])
            )

    history = model.fit(train_ds, **fit_kwargs)
    model.save(resolved["final_model_path"])

    test_metrics: dict[str, float] | None = None
    if split_files["test"]:
        test_ds = _build_dataset(
            files=split_files["test"],
            seed=resolved["seed"],
            image_height=resolved["image_height"],
            image_width=resolved["image_width"],
            batch_size=resolved["batch_size"],
            shuffle_buffer=resolved["shuffle_buffer"],
            shuffle_files=False,
            shuffle_examples=False,
            repeat=False,
        )
        test_metrics = {
            key: float(value)
            for key, value in model.evaluate(test_ds, verbose=1, return_dict=True).items()
        }

    summary = {
        "settings": resolved,
        "input_tfrecord_path_resolved": str(tfrecord_root),
        "rows_total": rows_train + rows_val + rows_test,
        "rows_train": rows_train,
        "rows_val": rows_val,
        "rows_test": rows_test,
        "num_classes_used": num_classes,
        "train_files_count": len(split_files["train"]),
        "val_files_count": len(split_files["val"]),
        "test_files_count": len(split_files["test"]),
        "history": history.history,
        "test_metrics": test_metrics,
        "best_model_path": resolved["best_model_path"],
        "final_model_path": resolved["final_model_path"],
    }
    Path(resolved["run_summary_path"]).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = _parse_args()
    settings = load_settings(args.config)
    result = run_training(settings)

    print("[train] settings:")
    print(json.dumps(result["settings"], indent=2))
    print(f"[train] input tfrecord path: {result['input_tfrecord_path_resolved']}")
    print(f"[train] rows total: {result['rows_total']}")
    print(f"[train] rows train: {result['rows_train']}")
    print(f"[train] rows val: {result['rows_val']}")
    print(f"[train] rows test: {result['rows_test']}")
    print(f"[train] num_classes used: {result['num_classes_used']}")
    print(f"[train] train files: {result['train_files_count']}")
    print(f"[train] val files: {result['val_files_count']}")
    print(f"[train] test files: {result['test_files_count']}")
    print(f"[train] best model: {result['best_model_path']}")
    print(f"[train] final model: {result['final_model_path']}")


if __name__ == "__main__":
    main()
