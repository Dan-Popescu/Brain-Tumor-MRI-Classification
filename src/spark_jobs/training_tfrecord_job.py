"""Create TFRecord training shards from a split manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

from spark_jobs.config_utils import (
    as_int,
    as_positive_int,
    as_positive_int_or_none,
    load_config,
    parse_partition_columns,
    to_abs_local_path,
)
from spark_jobs.spark_df_utils import (
    build_spark_session,
    configure_shuffle_partitions,
    repartition_dataframe,
    verify_partition_columns,
)
from training.dataset_artifacts import (
    DATASET_SUMMARY_FILENAME,
    SHARD_MANIFEST_DIRNAME,
)

# from project_paths import PROJECT_ROOT
# project_root = str(PROJECT_ROOT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export split manifest to sharded TFRecord files for training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/spark_training_tfrecord.yaml",
        help="YAML/JSON config path",
    )
    return parser.parse_args()


def _resolve_settings(config: dict[str, Any]) -> dict[str, Any]:
    seed = as_int(config.get("seed"), "seed", 42)
    n_shards = as_positive_int(config.get("n_shards", 16), "n_shards")

    export_id = config.get("export_id")
    if export_id is not None:
        export_id = str(export_id).strip() or None
    if export_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        export_id = f"training_tfrecord_seed{seed}_n{n_shards}_{timestamp}"

    output_root_path = str(
        config.get("output_root_path", "data/processed/training_tfrecord")
    )
    output_path = config.get("output_path")
    if output_path:
        output_path = str(output_path)
    else:
        output_path = str(Path(output_root_path) / export_id)

    return {
        "input_manifest_path": str(
            config.get(
                "input_manifest_path",
                "data/processed/splits/current/training_manifest.parquet",
            )
        ),
        "output_root_path": output_root_path,
        "output_path": output_path,
        "current_output_path": str(
            config.get("current_output_path", Path(output_root_path) / "current")
        ),
        "training_tfrecord_registry_path": str(
            config.get(
                "training_tfrecord_registry_path",
                "data/processed/training_tfrecord_versions.parquet",
            )
        ),
        "app_name": str(config.get("app_name", "mri-training-tfrecord")),
        "master": config.get("master"),
        "partitions": as_positive_int_or_none(config.get("partitions"), "partitions"),
        "shuffle_partitions": as_positive_int_or_none(
            config.get("shuffle_partitions"), "shuffle_partitions"
        ),
        "partition_by": parse_partition_columns(
            config.get("partition_by", ["split", "shard_id"])
        ),
        "seed": seed,
        "n_shards": n_shards,
        "export_id": export_id,
        "compression": "none",
        "output_format": "tfrecord",
    }


def load_settings(
    config_path: str | None = "conf/spark_training_tfrecord.yaml",
) -> dict[str, Any]:
    """Load and normalize settings from config file."""
    return _resolve_settings(load_config(config_path))


def _build_tfrecord_df(manifest_df: DataFrame, settings: dict[str, Any]) -> DataFrame:
    required_columns = {"image_id", "processed_bytes", "label_idx", "split", "split_id"}
    missing = sorted(required_columns - set(manifest_df.columns))
    if missing:
        raise ValueError(
            "Input split manifest is missing required columns: " + ", ".join(missing)
        )

    base_df = manifest_df.select(
        F.col("image_id"),
        F.col("processed_bytes"),
        F.col("label_idx"),
        F.col("split"),
        F.col("split_id"),
    )
    seed_str = str(settings["seed"])
    n_shards = settings["n_shards"]

    shard_id = F.pmod(
        F.xxhash64(F.concat_ws("::", F.col("image_id"), F.lit(seed_str))),
        F.lit(n_shards),
    ).cast("int")

    return (
        base_df.withColumn("shard_id", shard_id)
        .withColumn("export_id", F.lit(settings["export_id"]))
        .withColumn("shard_seed", F.lit(settings["seed"]).cast("int"))
        .withColumn("n_shards", F.lit(n_shards).cast("int"))
    )
def _prepare_output_dir(local_output_path: Path) -> None:
    resolved = local_output_path.resolve()
    if resolved == resolved.parent:
        raise ValueError("Refusing to clear filesystem root as output directory.")
    if local_output_path.is_symlink() or local_output_path.is_file():
        local_output_path.unlink()
    elif local_output_path.exists():
        shutil.rmtree(local_output_path)
    local_output_path.mkdir(parents=True, exist_ok=True)


def _refresh_current_alias(local_output_path: Path, local_current_path: Path) -> None:
    local_current_path.parent.mkdir(parents=True, exist_ok=True)
    if local_current_path.is_symlink() or local_current_path.is_file():
        local_current_path.unlink()
    elif local_current_path.exists():
        shutil.rmtree(local_current_path)
    local_current_path.symlink_to(local_output_path.resolve(), target_is_directory=True)


def _write_partition_tfrecords(
    partition_index: int,
    rows: Iterable[Any],
    output_local_root: str,
) -> Iterator[tuple[str, int, str, int]]:
    import tensorflow as tf

    output_root = Path(output_local_root)
    writers: dict[tuple[str, int], Any] = {}
    counts: dict[tuple[str, int], int] = {}

    def _bytes_feature(value: bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value: int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

    try:
        for row in rows:
            split = str(row["split"])
            shard_id = int(row["shard_id"])
            key = (split, shard_id)

            writer = writers.get(key)
            if writer is None:
                shard_dir = output_root / f"split={split}" / f"shard_id={shard_id}"
                shard_dir.mkdir(parents=True, exist_ok=True)
                file_path = shard_dir / f"part-{partition_index:05d}.tfrecord"
                writer = tf.io.TFRecordWriter(str(file_path))
                writers[key] = writer
                counts[key] = 0

            image_bytes_value = row["processed_bytes"]
            if image_bytes_value is None:
                raise ValueError("Missing required `processed_bytes` for a row.")
            image_bytes = bytes(image_bytes_value)

            image_id = str(row["image_id"])
            split_id = "" if row["split_id"] is None else str(row["split_id"])
            label_idx = int(row["label_idx"])

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_bytes": _bytes_feature(image_bytes),
                        "label_idx": _int64_feature(label_idx),
                        "image_id": _bytes_feature(image_id.encode("utf-8")),
                        "split": _bytes_feature(split.encode("utf-8")),
                        "split_id": _bytes_feature(split_id.encode("utf-8")),
                        "shard_id": _int64_feature(shard_id),
                    }
                )
            )
            writer.write(example.SerializeToString())
            counts[key] += 1
    finally:
        for writer in writers.values():
            writer.close()

    for (split, shard_id), rows_written in counts.items():
        if rows_written > 0:
            relative_path = str(
                Path(f"split={split}") / f"shard_id={shard_id}" / f"part-{partition_index:05d}.tfrecord"
            )
            yield (split, shard_id, relative_path, rows_written)


def _shard_manifest_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("export_id", T.StringType(), nullable=False),
            T.StructField("split", T.StringType(), nullable=False),
            T.StructField("shard_id", T.IntegerType(), nullable=False),
            T.StructField("relative_path", T.StringType(), nullable=False),
            T.StructField("rows_written", T.LongType(), nullable=False),
        ]
    )


def _registry_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("export_id", T.StringType(), nullable=False),
            T.StructField("generated_at_utc", T.StringType(), nullable=False),
            T.StructField("seed", T.IntegerType(), nullable=False),
            T.StructField("n_shards", T.IntegerType(), nullable=False),
            T.StructField("rows_total", T.LongType(), nullable=False),
            T.StructField("rows_train", T.LongType(), nullable=False),
            T.StructField("rows_val", T.LongType(), nullable=False),
            T.StructField("rows_test", T.LongType(), nullable=False),
            T.StructField("distinct_labels", T.IntegerType(), nullable=False),
            T.StructField("input_manifest_path", T.StringType(), nullable=False),
            T.StructField("output_path", T.StringType(), nullable=False),
            T.StructField("current_output_path", T.StringType(), nullable=False),
            T.StructField("output_format", T.StringType(), nullable=False),
            T.StructField("compression", T.StringType(), nullable=False),
            T.StructField("partition_by", T.StringType(), nullable=False),
            T.StructField("partitions", T.IntegerType(), nullable=True),
            T.StructField("shuffle_partitions", T.IntegerType(), nullable=True),
        ]
    )


def _upsert_training_tfrecord_registry(
    spark: SparkSession,
    settings: dict[str, Any],
    rows_by_split: dict[str, int],
    distinct_labels: int,
) -> str:
    registry_path = settings["training_tfrecord_registry_path"]
    rows_total = int(
        rows_by_split.get("train", 0)
        + rows_by_split.get("val", 0)
        + rows_by_split.get("test", 0)
    )

    new_row = (
        settings["export_id"],
        datetime.now(timezone.utc).isoformat(),
        settings["seed"],
        settings["n_shards"],
        rows_total,
        int(rows_by_split.get("train", 0)),
        int(rows_by_split.get("val", 0)),
        int(rows_by_split.get("test", 0)),
        int(distinct_labels),
        settings["input_manifest_path"],
        settings["output_path"],
        settings["current_output_path"],
        settings["output_format"],
        settings["compression"],
        ",".join(settings["partition_by"]),
        settings["partitions"],
        settings["shuffle_partitions"],
    )

    conf_key = "spark.sql.sources.partitionOverwriteMode"
    previous_overwrite_mode = spark.conf.get(conf_key, "static") or "static"
    spark.conf.set(conf_key, "dynamic")
    try:
        (
            spark.createDataFrame([new_row], schema=_registry_schema())
            .write.mode("overwrite")
            .partitionBy("export_id")
            .parquet(registry_path)
        )
    finally:
        spark.conf.set(conf_key, previous_overwrite_mode)
    return registry_path


def _collect_label_rows(manifest_df: DataFrame) -> list[dict[str, Any]]:
    rows = (
        manifest_df.select("label_idx", "pathology")
        .where(F.col("label_idx").isNotNull())
        .distinct()
        .orderBy("label_idx")
        .collect()
    )
    return [
        {
            "label_idx": int(row["label_idx"]),
            "pathology": None if row["pathology"] is None else str(row["pathology"]),
        }
        for row in rows
    ]


def _collect_label_counts_by_split(manifest_df: DataFrame) -> dict[str, dict[str, int]]:
    rows = (
        manifest_df.groupBy("split", "label_idx")
        .count()
        .where(F.col("label_idx").isNotNull())
        .collect()
    )
    result: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for row in rows:
        split = str(row["split"])
        if split not in result:
            result[split] = {}
        result[split][str(int(row["label_idx"]))] = int(row["count"])
    return result


def _collect_image_spec(manifest_df: DataFrame) -> dict[str, int] | None:
    required_columns = {"new_height", "new_width", "channels"}
    if not required_columns.issubset(manifest_df.columns):
        return None

    specs = (
        manifest_df.select("new_height", "new_width", "channels")
        .distinct()
        .collect()
    )
    if len(specs) != 1:
        return None

    row = specs[0]
    return {
        "image_height": int(row["new_height"]),
        "image_width": int(row["new_width"]),
        "channels": int(row["channels"]),
    }


def _write_dataset_summary(
    *,
    local_output_path: Path,
    settings: dict[str, Any],
    rows_by_split: dict[str, int],
    file_counts_by_split: dict[str, int],
    distinct_labels: int,
    label_rows: list[dict[str, Any]],
    label_counts_by_split: dict[str, dict[str, int]],
    image_spec: dict[str, int] | None,
) -> str:
    summary_payload: dict[str, Any] = {
        "export_id": settings["export_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": settings["seed"],
        "n_shards": settings["n_shards"],
        "rows_total": int(
            rows_by_split.get("train", 0)
            + rows_by_split.get("val", 0)
            + rows_by_split.get("test", 0)
        ),
        "rows_by_split": {
            "train": int(rows_by_split.get("train", 0)),
            "val": int(rows_by_split.get("val", 0)),
            "test": int(rows_by_split.get("test", 0)),
        },
        "files_total": int(
            file_counts_by_split.get("train", 0)
            + file_counts_by_split.get("val", 0)
            + file_counts_by_split.get("test", 0)
        ),
        "files_by_split": {
            "train": int(file_counts_by_split.get("train", 0)),
            "val": int(file_counts_by_split.get("val", 0)),
            "test": int(file_counts_by_split.get("test", 0)),
        },
        "distinct_labels": int(distinct_labels),
        "labels": label_rows,
        "label_counts_by_split": label_counts_by_split,
        "input_manifest_path": settings["input_manifest_path"],
        "output_path": settings["output_path"],
        "current_output_path": settings["current_output_path"],
        "shard_manifest_path": str(local_output_path / SHARD_MANIFEST_DIRNAME),
        "dataset_summary_path": str(local_output_path / DATASET_SUMMARY_FILENAME),
        "output_format": settings["output_format"],
        "compression": settings["compression"],
    }
    if image_spec is not None:
        summary_payload.update(image_spec)

    summary_path = local_output_path / DATASET_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return str(summary_path)


def run_training_tfrecord(spark: SparkSession, settings: dict[str, Any]) -> dict[str, Any]:
    """Run TFRecord export pipeline from normalized or partial settings."""
    resolved_settings = _resolve_settings(settings)

    configure_shuffle_partitions(
        spark,
        resolved_settings["shuffle_partitions"],
    )

    manifest_df = spark.read.parquet(resolved_settings["input_manifest_path"])
    tfrecord_df = _build_tfrecord_df(manifest_df, resolved_settings)

    input_counts_rows = manifest_df.groupBy("split").count().collect()  # 3 rows
    rows_by_split_input = {row["split"]: int(row["count"]) for row in input_counts_rows}
    rows_input_total = int(sum(rows_by_split_input.values()))
    distinct_labels = int(manifest_df.select("label_idx").distinct().count())
    label_rows = _collect_label_rows(manifest_df)
    label_counts_by_split = _collect_label_counts_by_split(manifest_df)
    image_spec = _collect_image_spec(manifest_df)

    partition_by = resolved_settings["partition_by"]
    verify_partition_columns(tfrecord_df, partition_by)
    tfrecord_df = repartition_dataframe(
        tfrecord_df,
        resolved_settings["partitions"],
        partition_by,
    )

    local_output_path = to_abs_local_path(resolved_settings["output_path"])
    _prepare_output_dir(local_output_path)

    # execute once for each partition
    write_counts_rdd = tfrecord_df.rdd.mapPartitionsWithIndex(
        lambda partition_index, rows: _write_partition_tfrecords(
            partition_index=partition_index,
            rows=rows,
            output_local_root=str(local_output_path),
        )
    )

    write_counts_df = spark.createDataFrame(
        write_counts_rdd,
        "split string, shard_id int, relative_path string, rows_written long"
    ).withColumn("export_id", F.lit(resolved_settings["export_id"]))

    input_counts_df = (
        manifest_df.groupBy("split")
        .count()
        .withColumnRenamed("count", "input_count")
    )

    output_counts_df = (
        write_counts_df.groupBy("split")
        .agg(F.sum("rows_written").alias("output_count"))
    )

    mismatch_df = (
        input_counts_df.join(output_counts_df, on="split", how="full")
        .na.fill(0, ["input_count", "output_count"])
        .where(F.col("input_count") != F.col("output_count"))
    )

    if mismatch_df.count() > 0:
        raise ValueError("TFRecord write validation failed")

    # After a successful mismatch check, output counts are identical to input counts.
    rows_by_split_written = dict(rows_by_split_input)
    rows_written_total = int(rows_input_total)
    file_counts_by_split_rows = (
        write_counts_df.groupBy("split").count().collect()
    )
    file_counts_by_split = {
        row["split"]: int(row["count"]) for row in file_counts_by_split_rows
    }

    shard_manifest_output_path = local_output_path / SHARD_MANIFEST_DIRNAME
    (
        write_counts_df.select(
            "export_id",
            "split",
            "shard_id",
            "relative_path",
            "rows_written",
        )
        .coalesce(1)
        .write.mode("overwrite")
        .parquet(str(shard_manifest_output_path))
    )

    local_current_path = to_abs_local_path(resolved_settings["current_output_path"])
    if local_current_path.resolve() == local_output_path.resolve():
        raise ValueError("`current_output_path` must be different from `output_path`.")
    dataset_summary_path = _write_dataset_summary(
        local_output_path=local_output_path,
        settings=resolved_settings,
        rows_by_split=rows_by_split_written,
        file_counts_by_split=file_counts_by_split,
        distinct_labels=distinct_labels,
        label_rows=label_rows,
        label_counts_by_split=label_counts_by_split,
        image_spec=image_spec,
    )
    _refresh_current_alias(local_output_path, local_current_path)

    registry_path = _upsert_training_tfrecord_registry(
        spark=spark,
        settings=resolved_settings,
        rows_by_split=rows_by_split_written,
        distinct_labels=distinct_labels,
    )

    return {
        "settings": resolved_settings,
        "rows_total": int(rows_written_total),
        "rows_train": int(rows_by_split_written.get("train", 0)),
        "rows_val": int(rows_by_split_written.get("val", 0)),
        "rows_test": int(rows_by_split_written.get("test", 0)),
        "distinct_labels": int(distinct_labels),
        "output_path": resolved_settings["output_path"],
        "output_current_path": resolved_settings["current_output_path"],
        "shard_manifest_path": str(shard_manifest_output_path),
        "dataset_summary_path": dataset_summary_path,
        "output_registry_path": registry_path,
    }


def main() -> None:
    args = _parse_args()
    settings = load_settings(args.config)

    spark = build_spark_session(settings["app_name"], settings["master"])

    result = run_training_tfrecord(spark, settings)

    print("[training_tfrecord_job] settings:")
    print(json.dumps(result["settings"], indent=2))
    print(f"[training_tfrecord_job] rows total: {result['rows_total']}")
    print(f"[training_tfrecord_job] rows train: {result['rows_train']}")
    print(f"[training_tfrecord_job] rows val: {result['rows_val']}")
    print(f"[training_tfrecord_job] rows test: {result['rows_test']}")
    print(f"[training_tfrecord_job] distinct labels: {result['distinct_labels']}")
    print(f"[training_tfrecord_job] output path: {result['output_path']}")
    print(f"[training_tfrecord_job] output current path: {result['output_current_path']}")
    print(f"[training_tfrecord_job] shard manifest: {result['shard_manifest_path']}")
    print(f"[training_tfrecord_job] dataset summary: {result['dataset_summary_path']}")
    print(f"[training_tfrecord_job] output registry: {result['output_registry_path']}")

    spark.stop()


if __name__ == "__main__":
    main()
