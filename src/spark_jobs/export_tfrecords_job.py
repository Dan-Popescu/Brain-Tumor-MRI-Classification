"""Spark export job: write TFRecord shards from split manifest for progressive TF loading."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T


# ---------------------------------------------------------------------------
# Config helpers (same pattern as other jobs)
# ---------------------------------------------------------------------------

def _load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(raw)
        return parsed or {}
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        raise ValueError(f"Invalid YAML config: {config_path}") from exc

    try:
        parsed = json.loads(raw)
        return parsed or {}
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Config parsing failed. Install PyYAML or provide JSON config."
        ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export split manifest to TFRecord shards for progressive TF loading."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/spark_export_tfrecords.yaml",
        help="YAML/JSON config path",
    )
    return parser.parse_args()


def _as_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `{field_name}` value: {value}") from exc
    if parsed <= 0:
        raise ValueError(f"`{field_name}` must be > 0, got {parsed}")
    return parsed


def _resolve_local_path(path: str) -> str:
    """Convert file:// URIs to local paths for filesystem access."""
    if path.startswith("file:"):
        parsed = urlparse(path)
        if parsed.scheme == "file":
            if parsed.netloc:
                return unquote(f"//{parsed.netloc}{parsed.path}")
            return unquote(parsed.path or path[len("file:"):])
    return path


def _parse_shards(raw_shards: Any) -> dict[str, int]:
    """Parse shard counts per split from config."""
    defaults = {"train": 8, "val": 2, "test": 2}
    if raw_shards is None:
        return defaults
    if isinstance(raw_shards, dict):
        result = {}
        for split_name in ("train", "val", "test"):
            val = raw_shards.get(split_name, defaults.get(split_name, 2))
            parsed = _as_positive_int(val, f"shards.{split_name}")
            result[split_name] = parsed if parsed is not None else defaults[split_name]
        return result
    raise ValueError("Invalid `shards` config. Expected mapping {train: N, val: N, test: N}.")


def _find_latest_split_id(splits_path: str) -> str:
    """Find the most recent split_id by directory name (lexicographic sort)."""
    local_path = _resolve_local_path(splits_path)
    splits_dir = Path(local_path)
    if not splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {splits_path}")

    candidates = sorted(
        [d.name for d in splits_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No split directories found in: {splits_path}")

    return candidates[0]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _resolve_settings(config: dict[str, Any]) -> dict[str, Any]:
    input_splits_path = str(config.get("input_splits_path", "data/processed/splits"))

    split_id = config.get("split_id")
    if split_id is not None:
        split_id = str(split_id).strip() or None
    if split_id is None:
        split_id = _find_latest_split_id(input_splits_path)

    input_manifest_path = str(Path(input_splits_path) / split_id / "manifest.parquet")

    return {
        "input_splits_path": input_splits_path,
        "input_manifest_path": input_manifest_path,
        "split_id": split_id,
        "output_tfrecords_path": str(config.get("output_tfrecords_path", "data/processed/tfrecords")),
        "export_version": str(config.get("export_version", "v1")),
        "app_name": str(config.get("app_name", "mri-export-tfrecords")),
        "master": config.get("master"),
        "partitions": _as_positive_int(config.get("partitions"), "partitions"),
        "shuffle_partitions": _as_positive_int(
            config.get("shuffle_partitions"), "shuffle_partitions"
        ),
        "shards": _parse_shards(config.get("shards")),
    }


def load_settings(config_path: str | None = "conf/spark_export_tfrecords.yaml") -> dict[str, Any]:
    """Load and normalize settings from config file."""
    return _resolve_settings(_load_config(config_path))


# ---------------------------------------------------------------------------
# TFRecord helpers
# ---------------------------------------------------------------------------

def _make_example(
    image_bytes: bytes,
    label: int,
    image_id: str,
    pathology: str,
    modality: str,
) -> Any:
    """Build a tf.train.Example from image data and metadata.

    TensorFlow is imported lazily so this module can be loaded by Spark
    without requiring TF at the driver level.
    """
    import tensorflow as tf

    def _bytes_feature(value: bytes) -> tf.train.Feature:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value: int) -> tf.train.Feature:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    feature = {
        "image_raw": _bytes_feature(image_bytes),
        "label": _int64_feature(label),
        "image_id": _bytes_feature(image_id.encode("utf-8")),
        "pathology": _bytes_feature(pathology.encode("utf-8")),
        "modality": _bytes_feature(modality.encode("utf-8")),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _write_tfrecord_partition(
    partition_idx: int,
    rows_iter: Any,
    output_dir: str,
) -> list[tuple[int, int]]:
    """Write one TFRecord shard from a Spark partition.

    Executed on Spark workers — imports TF lazily.
    Returns [(partition_idx, rows_written)] for stats collection.
    """
    import tensorflow as tf  # noqa: F811 — lazy import on executor

    shard_path = os.path.join(output_dir, f"shard-{partition_idx:04d}.tfrecord")
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    with tf.io.TFRecordWriter(shard_path) as writer:
        for row in rows_iter:
            processed_path = row["processed_path"]
            local_path = _resolve_local_path(processed_path)
            image_bytes = Path(local_path).read_bytes()

            example = _make_example(
                image_bytes=image_bytes,
                label=int(row["label_idx"]),
                image_id=str(row["image_id"]),
                pathology=str(row["pathology"]),
                modality=str(row["modality"]),
            )
            writer.write(example.SerializeToString())
            count += 1

    return [(partition_idx, count)]


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------

def _export_split(
    split_df: DataFrame,
    split_name: str,
    settings: dict[str, Any],
) -> int:
    """Repartition a split DataFrame and write TFRecord shards.

    Returns the total number of rows written for this split.
    """
    output_dir = str(
        Path(settings["output_tfrecords_path"])
        / settings["export_version"]
        / split_name
    )
    num_shards = settings["shards"].get(split_name, 2)

    repartitioned = split_df.repartition(num_shards)

    counts_rdd = repartitioned.rdd.mapPartitionsWithIndex(
        lambda idx, it: _write_tfrecord_partition(idx, it, output_dir)
    )
    shard_counts = counts_rdd.collect()

    total = sum(count for _, count in shard_counts)
    for shard_idx, count in sorted(shard_counts):
        print(f"  [export] {split_name}/shard-{shard_idx:04d}.tfrecord: {count} examples")

    return total


def _write_export_metadata(settings: dict[str, Any], split_counts: dict[str, int]) -> str:
    """Persist export settings as metadata.json under tfrecords/<version>/."""
    version_root = Path(settings["output_tfrecords_path"]) / settings["export_version"]
    version_root.mkdir(parents=True, exist_ok=True)

    metadata_path = version_root / "metadata.json"
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "export_version": settings["export_version"],
        "split_id": settings["split_id"],
        "input_manifest_path": settings["input_manifest_path"],
        "shards": settings["shards"],
        "split_counts": split_counts,
        "total_examples": sum(split_counts.values()),
        "tfrecord_features": {
            "image_raw": "bytes (PNG)",
            "label": "int64",
            "image_id": "bytes (utf-8)",
            "pathology": "bytes (utf-8)",
            "modality": "bytes (utf-8)",
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return str(metadata_path)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _registry_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("export_version", T.StringType(), nullable=False),
            T.StructField("generated_at_utc", T.StringType(), nullable=False),
            T.StructField("split_id", T.StringType(), nullable=False),
            T.StructField("rows_train", T.LongType(), nullable=False),
            T.StructField("rows_val", T.LongType(), nullable=False),
            T.StructField("rows_test", T.LongType(), nullable=False),
            T.StructField("rows_total", T.LongType(), nullable=False),
            T.StructField("shards_train", T.IntegerType(), nullable=False),
            T.StructField("shards_val", T.IntegerType(), nullable=False),
            T.StructField("shards_test", T.IntegerType(), nullable=False),
            T.StructField("input_manifest_path", T.StringType(), nullable=False),
            T.StructField("output_tfrecords_path", T.StringType(), nullable=False),
        ]
    )


def _upsert_export_registry(
    spark: SparkSession,
    settings: dict[str, Any],
    split_counts: dict[str, int],
) -> str:
    registry_path = str(
        Path(settings["output_tfrecords_path"]) / "export_versions.parquet"
    )
    local_registry_path = _resolve_local_path(registry_path)

    rows_total = sum(split_counts.values())
    new_row = (
        settings["export_version"],
        datetime.now(timezone.utc).isoformat(),
        settings["split_id"],
        int(split_counts.get("train", 0)),
        int(split_counts.get("val", 0)),
        int(split_counts.get("test", 0)),
        int(rows_total),
        settings["shards"]["train"],
        settings["shards"]["val"],
        settings["shards"]["test"],
        settings["input_manifest_path"],
        settings["output_tfrecords_path"],
    )
    new_df = spark.createDataFrame([new_row], schema=_registry_schema())

    if Path(local_registry_path).exists():
        existing_df = spark.read.parquet(registry_path)
        merged_df = existing_df.filter(
            F.col("export_version") != settings["export_version"]
        ).unionByName(new_df)
    else:
        merged_df = new_df

    merged_df.write.mode("overwrite").parquet(registry_path)
    return registry_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_export(spark: SparkSession, settings: dict[str, Any]) -> dict[str, Any]:
    """Run TFRecord export pipeline."""
    resolved = _resolve_settings(settings)

    if resolved["shuffle_partitions"]:
        spark.conf.set(
            "spark.sql.shuffle.partitions",
            resolved["shuffle_partitions"],
        )

    print(f"[export_job] Reading split manifest: {resolved['input_manifest_path']}")
    manifest_df = spark.read.parquet(resolved["input_manifest_path"])

    required_columns = {"image_id", "processed_path", "label_idx", "pathology", "modality", "split"}
    missing = sorted(required_columns - set(manifest_df.columns))
    if missing:
        raise ValueError(
            "Input manifest is missing required columns: " + ", ".join(missing)
        )

    split_counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        split_df = manifest_df.filter(F.col("split") == split_name)
        print(f"[export_job] Exporting split '{split_name}'...")
        count = _export_split(split_df, split_name, resolved)
        split_counts[split_name] = count
        print(f"[export_job] {split_name}: {count} examples written")

    metadata_path = _write_export_metadata(resolved, split_counts)
    registry_path = _upsert_export_registry(spark, resolved, split_counts)

    return {
        "settings": resolved,
        "split_counts": split_counts,
        "rows_total": sum(split_counts.values()),
        "output_metadata_path": metadata_path,
        "output_registry_path": registry_path,
    }


def main() -> None:
    args = _parse_args()
    settings = load_settings(args.config)

    spark_builder = SparkSession.builder.appName(settings["app_name"])
    if settings["master"]:
        spark_builder = spark_builder.master(settings["master"])
    spark = spark_builder.getOrCreate()

    result = run_export(spark, settings)

    print("[export_job] settings:")
    print(json.dumps(result["settings"], indent=2))
    print(f"[export_job] rows total: {result['rows_total']}")
    for split_name, count in result["split_counts"].items():
        print(f"[export_job] rows {split_name}: {count}")
    print(f"[export_job] output metadata: {result['output_metadata_path']}")
    print(f"[export_job] output registry: {result['output_registry_path']}")

    spark.stop()


if __name__ == "__main__":
    main()
