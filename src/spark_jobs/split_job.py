"""Spark split job: deterministic train/val/test assignment from silver manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

from spark_jobs.config_utils import (
    as_int,
    as_positive_int_or_none,
    load_config,
    parse_partition_columns,
)
from spark_jobs.spark_df_utils import (
    build_spark_session,
    configure_shuffle_partitions,
    repartition_dataframe,
    verify_partition_columns,
    write_partitioned_parquet,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic train/val/test splits from silver manifest."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/spark_split.yaml",
        help="YAML/JSON config path",
    )
    return parser.parse_args()


def _as_ratio(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `{field_name}` value: {value}") from exc
    if parsed <= 0.0 or parsed >= 1.0:
        raise ValueError(f"`{field_name}` must be in (0, 1), got {parsed}")
    return parsed


def _parse_split_ratios(raw_ratios: Any) -> tuple[float, float, float]:
    if isinstance(raw_ratios, (list, tuple)) and len(raw_ratios) == 3:
        train_ratio = _as_ratio(raw_ratios[0], "split_ratios[0]")
        val_ratio = _as_ratio(raw_ratios[1], "split_ratios[1]")
        test_ratio = _as_ratio(raw_ratios[2], "split_ratios[2]")
    else:
        raise ValueError("Invalid `split_ratios`. Expected [train, val, test].")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            f"`split_ratios` must sum to 1.0, got {ratio_sum:.6f}"
        )

    return train_ratio, val_ratio, test_ratio


def _ratio_tag(train_ratio: float, val_ratio: float, test_ratio: float) -> str:
    return f"{int(round(train_ratio * 100))}_{int(round(val_ratio * 100))}_{int(round(test_ratio * 100))}"


def _resolve_settings(config: dict[str, Any]) -> dict[str, Any]:
    train_ratio, val_ratio, test_ratio = _parse_split_ratios(
        config.get("split_ratios", [0.7, 0.15, 0.15])
    )
    seed = as_int(config.get("seed"), "seed", 42)
    split_id = config.get("split_id")
    if split_id is not None:
        split_id = str(split_id).strip() or None
    if split_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        split_id = f"split_seed{seed}_r{_ratio_tag(train_ratio, val_ratio, test_ratio)}_{timestamp}"

    output_splits_path = str(config.get("output_splits_path", "data/processed/splits"))
    output_manifest_path = config.get("output_manifest_path")
    if output_manifest_path:
        output_manifest_path = str(output_manifest_path)
    else:
        output_manifest_path = str(
            Path(output_splits_path) / split_id / "training_manifest.parquet"
        )
    current_output_manifest_path = str(
        config.get(
            "current_output_manifest_path",
            Path(output_splits_path) / "current" / "training_manifest.parquet",
        )
    )

    return {
        "input_manifest_path": str(
            config.get("input_manifest_path", "data/processed/manifest_silver.parquet")
        ),
        "output_splits_path": output_splits_path,
        "output_manifest_path": output_manifest_path,
        "current_output_manifest_path": current_output_manifest_path,
        "split_versions_registry_path": str(
            config.get(
                "split_versions_registry_path",
                "data/processed/split_versions.parquet",
            )
        ),
        "app_name": str(config.get("app_name", "mri-split")),
        "master": config.get("master"),
        "partitions": as_positive_int_or_none(config.get("partitions"), "partitions"),
        "shuffle_partitions": as_positive_int_or_none(
            config.get("shuffle_partitions"), "shuffle_partitions"
        ),
        "partition_by": parse_partition_columns(config.get("partition_by", ["split"])),
        "split_id": split_id,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }


def load_settings(config_path: str | None = "conf/spark_split.yaml") -> dict[str, Any]:
    """Load and normalize settings from config file."""
    return _resolve_settings(load_config(config_path))


def _build_split_df(manifest_df: DataFrame, settings: dict[str, Any]) -> DataFrame:
    if "image_id" not in manifest_df.columns:
        raise ValueError("Input manifest must contain `image_id` column.")

    train_ratio = settings["train_ratio"]
    val_ratio = settings["val_ratio"]
    train_val_threshold = train_ratio + val_ratio
    seed_str = str(settings["seed"])
    split_ratio_text = f"{train_ratio:.4f}/{val_ratio:.4f}/{settings['test_ratio']:.4f}"

    # Deterministic pseudo-random value in [0, 1): depends only on image_id + seed.
    random_key = (
        F.pmod(
            F.xxhash64(F.concat_ws("::", F.col("image_id"), F.lit(seed_str))),
            F.lit(1_000_000),
        )
        / F.lit(1_000_000.0)
    )


    return (
        manifest_df.withColumn("__split_key", random_key)
        .withColumn(
            "split",
            F.when(F.col("__split_key") < F.lit(train_ratio), F.lit("train"))
            .when(F.col("__split_key") < F.lit(train_val_threshold), F.lit("val"))
            .otherwise(F.lit("test")),
        )
        .withColumn("split_id", F.lit(settings["split_id"]))
        .withColumn("split_seed", F.lit(settings["seed"]).cast("int"))
        .withColumn("split_ratios", F.lit(split_ratio_text))
        .drop("__split_key")
    )


def _registry_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("split_id", T.StringType(), nullable=False),
            T.StructField("generated_at_utc", T.StringType(), nullable=False),
            T.StructField("seed", T.IntegerType(), nullable=False),
            T.StructField("train_ratio", T.DoubleType(), nullable=False),
            T.StructField("val_ratio", T.DoubleType(), nullable=False),
            T.StructField("test_ratio", T.DoubleType(), nullable=False),
            T.StructField("rows_total", T.LongType(), nullable=False),
            T.StructField("rows_train", T.LongType(), nullable=False),
            T.StructField("rows_val", T.LongType(), nullable=False),
            T.StructField("rows_test", T.LongType(), nullable=False),
            T.StructField("input_manifest_path", T.StringType(), nullable=False),
            T.StructField("output_manifest_path", T.StringType(), nullable=False),
            T.StructField("output_splits_path", T.StringType(), nullable=False),
            T.StructField("partition_by", T.StringType(), nullable=False),
            T.StructField("partitions", T.IntegerType(), nullable=True),
            T.StructField("shuffle_partitions", T.IntegerType(), nullable=True),
        ]
    )


def _upsert_split_registry(
    spark: SparkSession,
    settings: dict[str, Any],
    split_counts: dict[str, int],
) -> str:
    registry_path = settings["split_versions_registry_path"]

    rows_total = int(
        split_counts.get("train", 0) + split_counts.get("val", 0) + split_counts.get("test", 0)
    )
    new_row = (
        settings["split_id"],
        datetime.now(timezone.utc).isoformat(),
        settings["seed"],
        float(settings["train_ratio"]),
        float(settings["val_ratio"]),
        float(settings["test_ratio"]),
        rows_total,
        int(split_counts.get("train", 0)),
        int(split_counts.get("val", 0)),
        int(split_counts.get("test", 0)),
        settings["input_manifest_path"],
        settings["output_manifest_path"],
        settings["output_splits_path"],
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
            .partitionBy("split_id")
            .parquet(registry_path)
        )
    finally:
        spark.conf.set(conf_key, previous_overwrite_mode)
    return registry_path


def run_split(spark: SparkSession, settings: dict[str, Any]) -> dict[str, Any]:
    """Run split pipeline from normalized or partial settings."""
    resolved_settings = _resolve_settings(settings)

    configure_shuffle_partitions(
        spark,
        resolved_settings["shuffle_partitions"],
    )

    manifest_df = spark.read.parquet(resolved_settings["input_manifest_path"])
    split_df = _build_split_df(manifest_df, resolved_settings)

    partition_by = resolved_settings["partition_by"]
    verify_partition_columns(split_df, partition_by)
    split_df = repartition_dataframe(
        split_df,
        resolved_settings["partitions"],
        partition_by,
    )

    write_partitioned_parquet(
        split_df,
        resolved_settings["output_manifest_path"],
        partition_by,
    )
    write_partitioned_parquet(
        split_df,
        resolved_settings["current_output_manifest_path"],
        partition_by,
    )

    split_counts_rows = split_df.groupBy("split").count().collect()
    split_counts = {row["split"]: int(row["count"]) for row in split_counts_rows}
    registry_path = _upsert_split_registry(spark, resolved_settings, split_counts)

    return {
        "settings": resolved_settings,
        "rows_total": int(
            split_counts.get("train", 0)
            + split_counts.get("val", 0)
            + split_counts.get("test", 0)
        ),
        "rows_train": int(split_counts.get("train", 0)),
        "rows_val": int(split_counts.get("val", 0)),
        "rows_test": int(split_counts.get("test", 0)),
        "output_manifest_path": resolved_settings["output_manifest_path"],
        "output_current_manifest_path": resolved_settings["current_output_manifest_path"],
        "output_registry_path": registry_path,
    }


def main() -> None:
    args = _parse_args()
    settings = load_settings(args.config)

    spark = build_spark_session(settings["app_name"], settings["master"])

    result = run_split(spark, settings)

    print("[split_job] settings:")
    print(json.dumps(result["settings"], indent=2))
    print(f"[split_job] rows total: {result['rows_total']}")
    print(f"[split_job] rows train: {result['rows_train']}")
    print(f"[split_job] rows val: {result['rows_val']}")
    print(f"[split_job] rows test: {result['rows_test']}")
    print(f"[split_job] output manifest: {result['output_manifest_path']}")
    print(f"[split_job] output current manifest: {result['output_current_manifest_path']}")
    print(f"[split_job] output registry: {result['output_registry_path']}")

    spark.stop()


if __name__ == "__main__":
    main()
