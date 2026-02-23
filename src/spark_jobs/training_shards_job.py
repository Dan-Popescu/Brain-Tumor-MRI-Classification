"Create training-ready parquet shards from a split manifest, with deterministic shard assignment."

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

from config_utils import (
    as_int,
    as_positive_int,
    as_positive_int_or_none,
    load_config,
    parse_partition_columns,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export split manifest to sharded parquet files for training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/spark_training_shards.yaml",
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
        export_id = f"training_shards_seed{seed}_n{n_shards}_{timestamp}"

    output_root_path = str(
        config.get("output_root_path", "data/processed/training_shards")
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
        "training_shards_registry_path": str(
            config.get(
                "training_shards_registry_path",
                "data/processed/training_shards_versions.parquet",
            )
        ),
        "app_name": str(config.get("app_name", "mri-training-shards")),
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
    }


def load_settings(
    config_path: str | None = "conf/spark_training_shards.yaml",
) -> dict[str, Any]:
    """Load and normalize settings from config file."""
    return _resolve_settings(load_config(config_path))


def _build_shards_df(manifest_df: DataFrame, settings: dict[str, Any]) -> DataFrame:
    required_columns = {"image_id", "processed_path", "label_idx", "split", "split_id"}
    missing = sorted(required_columns - set(manifest_df.columns))
    if missing:
        raise ValueError(
            "Input split manifest is missing required columns: " + ", ".join(missing)
        )

    base_df = manifest_df.select(
        "image_id", "processed_path", "label_idx", "split", "split_id"
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
            T.StructField("partition_by", T.StringType(), nullable=False),
            T.StructField("partitions", T.IntegerType(), nullable=True),
            T.StructField("shuffle_partitions", T.IntegerType(), nullable=True),
        ]
    )


def _upsert_training_shards_registry(
    spark: SparkSession,
    settings: dict[str, Any],
    rows_by_split: dict[str, int],
    distinct_labels: int,
) -> str:
    registry_path = settings["training_shards_registry_path"]

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
        ",".join(settings["partition_by"]),
        settings["partitions"],
        settings["shuffle_partitions"],
    )
    conf_key = "spark.sql.sources.partitionOverwriteMode"
    previous_overwrite_mode = spark.conf.get(conf_key, "static")
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


def run_training_shards(spark: SparkSession, settings: dict[str, Any]) -> dict[str, Any]:
    """Run training shards export pipeline from normalized or partial settings."""
    resolved_settings = _resolve_settings(settings)

    if resolved_settings["shuffle_partitions"]:
        spark.conf.set(
            "spark.sql.shuffle.partitions",
            resolved_settings["shuffle_partitions"],
        )

    manifest_df = spark.read.parquet(resolved_settings["input_manifest_path"])
    shards_df = _build_shards_df(manifest_df, resolved_settings)

    partition_by = resolved_settings["partition_by"]
    if partition_by:
        unknown = [col for col in partition_by if col not in shards_df.columns]
        if unknown:
            raise ValueError(
                "Unknown partition columns: "
                + ", ".join(unknown)
                + ". Available columns: "
                + ", ".join(shards_df.columns)
            )

    if resolved_settings["partitions"]:
        if partition_by:
            shards_df = shards_df.repartition(
                resolved_settings["partitions"], *partition_by
            )
        else:
            shards_df = shards_df.repartition(resolved_settings["partitions"])
    elif partition_by:
        shards_df = shards_df.repartition(*partition_by)

    writer = shards_df.write.mode("overwrite")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.parquet(resolved_settings["output_path"])

    rows_by_split_rows = shards_df.groupBy("split").count().collect()
    rows_by_split = {row["split"]: int(row["count"]) for row in rows_by_split_rows}
    distinct_labels = shards_df.select("label_idx").distinct().count()

    registry_path = _upsert_training_shards_registry(
        spark=spark,
        settings=resolved_settings,
        rows_by_split=rows_by_split,
        distinct_labels=distinct_labels,
    )

    return {
        "settings": resolved_settings,
        "rows_total": int(
            rows_by_split.get("train", 0)
            + rows_by_split.get("val", 0)
            + rows_by_split.get("test", 0)
        ),
        "rows_train": int(rows_by_split.get("train", 0)),
        "rows_val": int(rows_by_split.get("val", 0)),
        "rows_test": int(rows_by_split.get("test", 0)),
        "distinct_labels": int(distinct_labels),
        "output_path": resolved_settings["output_path"],
        "output_registry_path": registry_path,
    }


def main() -> None:
    args = _parse_args()
    settings = load_settings(args.config)

    spark_builder = SparkSession.builder.appName(settings["app_name"])
    if settings["master"]:
        spark_builder = spark_builder.master(settings["master"])
    spark = spark_builder.getOrCreate()

    result = run_training_shards(spark, settings)

    print("[training_shards_job] settings:")
    print(json.dumps(result["settings"], indent=2))
    print(f"[training_shards_job] rows total: {result['rows_total']}")
    print(f"[training_shards_job] rows train: {result['rows_train']}")
    print(f"[training_shards_job] rows val: {result['rows_val']}")
    print(f"[training_shards_job] rows test: {result['rows_test']}")
    print(f"[training_shards_job] distinct labels: {result['distinct_labels']}")
    print(f"[training_shards_job] output path: {result['output_path']}")
    print(f"[training_shards_job] output registry: {result['output_registry_path']}")

    spark.stop()


if __name__ == "__main__":
    main()
