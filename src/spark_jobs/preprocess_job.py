"""Spark preprocessing job: manifest-only (no image transform)."""

from __future__ import annotations

import argparse
import json
from typing import Any, cast

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

from spark_jobs.config_utils import (
    as_bool,
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
    write_partitioned_parquet,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Spark manifest parquet from raw MRI image files."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/spark_preprocess.yaml",
        help="YAML/JSON config path",
    )
    return parser.parse_args()


def _resolve_settings(config: dict[str, Any]) -> dict[str, Any]:
    partition_by = parse_partition_columns(config.get("partition_by", ["pathology", "modality"]))

    return {
        "input_path": config.get("input_path", "data/raw"),
        "output_manifest_path": config.get(
            "output_manifest_path", "data/processed/manifest_bronze.parquet"
        ),
        "app_name": config.get("app_name", "mri-manifest-preprocess"),
        "master": config.get("master"),
        "partitions": as_positive_int_or_none(config.get("partitions"), "partitions"),
        "shuffle_partitions": as_positive_int_or_none(
            config.get("shuffle_partitions"), "shuffle_partitions"
        ),
        "partition_by": partition_by,
        "sort_within_partitions": as_bool(
            config.get("sort_within_partitions"), "sort_within_partitions", True
        ),
    }


def load_settings(config_path: str | None = "conf/spark_preprocess.yaml") -> dict[str, Any]:
    """Load and normalize settings from config file."""
    return _resolve_settings(load_config(config_path))

def _resolve_binary_input_paths(input_path: str) -> list[str]:
    """Resolve top-level class directories as explicit binaryFile inputs.

    Passing explicit class directories allows Spark to read folders like `_NORMAL ...`
    while still keeping file listing and decoding distributed.
    """
    local_root = to_abs_local_path(input_path)
    if not local_root.exists():
        raise FileNotFoundError(f"Input path not found: {local_root}")
    if not local_root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {local_root}")

    class_dirs = sorted(
        path
        for path in local_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if class_dirs:
        return [str(path) for path in class_dirs]
    return [str(local_root)]


def _build_base_df(spark: SparkSession, input_path: str) -> DataFrame:
    input_paths = _resolve_binary_input_paths(input_path)
    df = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(input_paths)
    )

    # Keep only actual image files and ignore Windows metadata side-files.
    image_pattern = r"(?i)\.(jpg|jpeg|png)$"

    return (
        df.filter(F.col("path").rlike(image_pattern))
        # Keep data under directories like `_NORMAL ...` while excluding hidden side-files.
        .withColumn("__filename", F.regexp_extract(F.col("path"), r"([^/\\]+)$", 1))
        .filter(~F.col("__filename").rlike(r"^\."))
        .drop("__filename")
    )


def _build_label_mapping(df: DataFrame) -> DataFrame:
    """Build deterministic pathology -> label_idx mapping without window execution."""
    label_schema = T.StructType(
        [
            T.StructField("pathology", T.StringType(), nullable=True),
            T.StructField("label_idx", T.IntegerType(), nullable=False),
        ]
    )

    labels = [
        row["pathology"]
        for row in df.select("pathology")
        .where(F.col("pathology").isNotNull() & (F.col("pathology") != ""))
        .distinct()
        .orderBy("pathology")
        .collect()
    ]
    label_rows = [(pathology, idx) for idx, pathology in enumerate(labels)]
    spark = cast(SparkSession, df.sparkSession)
    return spark.createDataFrame(label_rows, schema=label_schema)


def _enrich_manifest(df: DataFrame) -> DataFrame:
    parent_dir_pattern = r"[/\\]([^/\\]+)[/\\][^/\\]+$"
    modality_pattern = r"(T1C\+|T1|T2)$"
    pathology_pattern = r"^(.*)\s+(?:T1C\+|T1|T2)$"

    enriched = (
        df.withColumn("raw_path", F.col("path"))
        .withColumn("parent_dir", F.regexp_extract(F.col("raw_path"), parent_dir_pattern, 1))
        .withColumn("modality", F.regexp_extract(F.col("parent_dir"), modality_pattern, 1))
        .withColumn("pathology", F.regexp_extract(F.col("parent_dir"), pathology_pattern, 1))
        .withColumn(
            "pathology",
            F.when(F.col("pathology") == "", F.col("parent_dir")).otherwise(F.col("pathology")),
        )
        .withColumn("image_id", F.sha2(F.col("raw_path"), 256))
        .withColumn("file_size", F.col("length").cast("long"))
    )

    label_mapping = _build_label_mapping(enriched)

    return (
        enriched.join(F.broadcast(label_mapping), on="pathology", how="left")
        .select(
            "image_id",
            "raw_path",
            "pathology",
            "label_idx",
            "modality",
            "file_size",
        )
    )


def run_preprocess(spark: SparkSession, settings: dict[str, Any]) -> dict[str, Any]:
    """Run the preprocess pipeline from normalized or partial settings."""
    resolved_settings = _resolve_settings(settings)
    configure_shuffle_partitions(
        spark,
        resolved_settings["shuffle_partitions"],
    )

    input_df = _build_base_df(
        spark,
        resolved_settings["input_path"],
    )
    manifest_df = _enrich_manifest(input_df)

    partition_by = resolved_settings["partition_by"]
    verify_partition_columns(manifest_df, partition_by)
    manifest_df = repartition_dataframe(
        manifest_df,
        resolved_settings["partitions"],
        partition_by,
    )

    if resolved_settings["sort_within_partitions"]:
        sort_columns = [col for col in ["pathology", "modality", "raw_path"] if col in manifest_df.columns]
        if sort_columns:
            manifest_df = manifest_df.sortWithinPartitions(*sort_columns)

    output_path = resolved_settings["output_manifest_path"]
    write_partitioned_parquet(manifest_df, output_path, partition_by)

    total_rows = manifest_df.count()
    return {
        "settings": resolved_settings,
        "rows_written": total_rows,
        "output_path": output_path,
    }


def main() -> None:
    args = _parse_args()
    settings = load_settings(args.config)

    spark = build_spark_session(settings["app_name"], settings["master"])

    result = run_preprocess(spark, settings)

    print("[preprocess_job] settings:")
    print(json.dumps(result["settings"], indent=2))
    print(f"[preprocess_job] rows written: {result['rows_written']}")
    print(f"[preprocess_job] output: {result['output_path']}")

    spark.stop()


if __name__ == "__main__":
    main()
