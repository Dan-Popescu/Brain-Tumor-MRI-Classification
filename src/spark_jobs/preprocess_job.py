#!/usr/bin/env python3
"""Spark preprocessing job: manifest-only (no image transform)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T


def _load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    # Try YAML first when available, then JSON as fallback.
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
        description="Build a Spark manifest parquet from raw MRI image files."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/spark_preprocess.yaml",
        help="YAML/JSON config path",
    )
    return parser.parse_args()


def _parse_partition_columns(raw_columns: Any) -> list[str]:
    if raw_columns is None:
        return []

    if isinstance(raw_columns, str):
        return [col.strip() for col in raw_columns.split(",") if col.strip()]

    if isinstance(raw_columns, list):
        return [str(col).strip() for col in raw_columns if str(col).strip()]

    raise ValueError(
        "Invalid `partition_by` value in config. Use a comma-separated string or list."
    )


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


def _as_bool(value: Any, field_name: str, default: bool) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False

    raise ValueError(f"Invalid `{field_name}` value: {value}")


def _resolve_settings(config: dict[str, Any]) -> dict[str, Any]:
    partition_by = _parse_partition_columns(config.get("partition_by", ["pathology", "modality"]))

    return {
        "input_path": config.get("input_path", "data/raw"),
        "output_manifest_path": config.get(
            "output_manifest_path", "data/processed/manifest_bronze.parquet"
        ),
        "app_name": config.get("app_name", "mri-manifest-preprocess"),
        "master": config.get("master"),
        "partitions": _as_positive_int(config.get("partitions"), "partitions"),
        "shuffle_partitions": _as_positive_int(
            config.get("shuffle_partitions"), "shuffle_partitions"
        ),
        "partition_by": partition_by,
        "sort_within_partitions": _as_bool(
            config.get("sort_within_partitions"), "sort_within_partitions", True
        ),
    }


def load_settings(config_path: str | None = "conf/spark_preprocess.yaml") -> dict[str, Any]:
    """Load and normalize settings from config file."""
    return _resolve_settings(_load_config(config_path))


def _build_base_df(spark: SparkSession, input_path: str) -> DataFrame:
    df = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(input_path)
    )

    # Keep only actual image files and ignore Windows metadata sidecar files.
    image_pattern = r"(?i)\.(jpg|jpeg|webp)$"

    return (
        df.filter(~F.col("path").endswith(":Zone.Identifier"))
        .filter(~F.col("path").endswith(".gitkeep"))
        .filter(F.col("path").rlike(image_pattern))
    )


def _build_label_mapping(df: DataFrame) -> DataFrame:
    """Build deterministic pathology -> label_idx mapping without window execution."""
    label_schema = T.StructType(
        [
            T.StructField("pathology", T.StringType(), nullable=True),
            T.StructField("label_idx", T.IntegerType(), nullable=False),
        ]
    )

    # Pathology cardinality is expected to be low; collecting labels keeps mapping deterministic
    # and removes the global Window step that forces a single partition.
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
    parent_dir_pattern = r"/([^/]+)/[^/]+$"
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
        .withColumn("is_valid", F.lit(True))
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
            "is_valid",
        )
    )


def run_preprocess(spark: SparkSession, settings: dict[str, Any]) -> dict[str, Any]:
    """Run the preprocess pipeline from normalized or partial settings."""
    resolved_settings = _resolve_settings(settings)
    if resolved_settings["shuffle_partitions"]:
        spark.conf.set(
            "spark.sql.shuffle.partitions",
            resolved_settings["shuffle_partitions"],
        )

    input_df = _build_base_df(spark, resolved_settings["input_path"])
    manifest_df = _enrich_manifest(input_df)

    partition_by = resolved_settings["partition_by"]
    if partition_by:
        missing_partition_cols = [col for col in partition_by if col not in manifest_df.columns]
        if missing_partition_cols:
            raise ValueError(
                "Unknown partition columns: "
                + ", ".join(missing_partition_cols)
                + ". Available columns: "
                + ", ".join(manifest_df.columns)
            )

    if resolved_settings["partitions"]:
        if partition_by:
            manifest_df = manifest_df.repartition(resolved_settings["partitions"], *partition_by)
        else:
            manifest_df = manifest_df.repartition(resolved_settings["partitions"])
    elif partition_by:
        manifest_df = manifest_df.repartition(*partition_by)

    if resolved_settings["sort_within_partitions"]:
        sort_columns = [col for col in ["pathology", "modality", "raw_path"] if col in manifest_df.columns]
        if sort_columns:
            manifest_df = manifest_df.sortWithinPartitions(*sort_columns)

    output_path = resolved_settings["output_manifest_path"]
    writer = manifest_df.write.mode("overwrite")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.parquet(output_path)

    total_rows = manifest_df.count()
    return {
        "settings": resolved_settings,
        "rows_written": total_rows,
        "output_path": output_path,
    }


def main() -> None:
    args = _parse_args()
    settings = load_settings(args.config)

    spark_builder = SparkSession.builder.appName(settings["app_name"])
    if settings["master"]:
        spark_builder = spark_builder.master(settings["master"])
    spark = spark_builder.getOrCreate()

    result = run_preprocess(spark, settings)

    print("[preprocess_job] settings:")
    print(json.dumps(result["settings"], indent=2))
    print(f"[preprocess_job] rows written: {result['rows_written']}")
    print(f"[preprocess_job] output: {result['output_path']}")

    spark.stop()


if __name__ == "__main__":
    main()
