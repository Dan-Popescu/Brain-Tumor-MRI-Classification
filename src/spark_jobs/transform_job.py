#!/usr/bin/env python3
"""Spark transform job: Pillow grayscale + resize image pipeline."""

from __future__ import annotations

import argparse
import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from PIL import Image
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from spark_jobs.config_utils import (
    as_bool,
    as_int,
    as_positive_int,
    as_positive_int_or_none,
    load_config,
    parse_partition_columns,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform MRI images from bronze manifest into silver dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/spark_transform.yaml",
        help="YAML/JSON config path",
    )
    return parser.parse_args()


def _parse_target_size(raw_size: Any) -> tuple[int, int]:
    if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
        width = as_positive_int(raw_size[0], "target_size[0]")
        height = as_positive_int(raw_size[1], "target_size[1]")
        return width, height

    raise ValueError("Invalid `target_size`. Expected [width, height].")


def _parse_resize_strategy(value: Any) -> str:
    strategy = str(value or "keep_aspect_pad").strip().lower()
    allowed = {"stretch", "keep_aspect_pad", "center_crop"}
    if strategy not in allowed:
        raise ValueError(
            f"Invalid `resize_strategy`: {strategy}. Allowed: {sorted(allowed)}"
        )
    return strategy


def _resolve_settings(config: dict[str, Any]) -> dict[str, Any]:
    target_width, target_height = _parse_target_size(config.get("target_size", [224, 224]))
    debug_export_n = as_positive_int_or_none(config.get("debug_export_n"), "debug_export_n")
    if debug_export_n is None:
        debug_export_n = 100

    return {
        "input_manifest_path": config.get(
            "input_manifest_path", "data/processed/manifest_bronze.parquet"
        ),
        "output_images_path": config.get(
            "output_images_path", "data/processed/images_silver"
        ),
        "output_manifest_path": config.get(
            "output_manifest_path", "data/processed/manifest_silver.parquet"
        ),
        "versions_registry_path": config.get(
            "versions_registry_path", "data/processed/transform_versions.parquet"
        ),
        "app_name": config.get("app_name", "mri-transform"),
        "master": config.get("master"),
        "partitions": as_positive_int_or_none(config.get("partitions"), "partitions"),
        "shuffle_partitions": as_positive_int_or_none(
            config.get("shuffle_partitions"), "shuffle_partitions"
        ),
        "partition_by": parse_partition_columns(
            config.get("partition_by", ["pathology", "modality"])
        ),
        "transform_version": str(config.get("transform_version", "v1")),
        "resize_strategy": _parse_resize_strategy(config.get("resize_strategy")),
        "target_width": target_width,
        "target_height": target_height,
        "debug_export_enabled": as_bool(
            config.get("debug_export_enabled"),
            "debug_export_enabled",
            False,
        ),
        "debug_export_path": str(
            config.get("debug_export_path", "data/debug/transform_preview")
        ),
        "debug_export_n": debug_export_n,
        "debug_export_per_class": as_positive_int_or_none(
            config.get("debug_export_per_class"), "debug_export_per_class"
        ),
        "debug_export_seed": as_int(config.get("debug_export_seed"), "debug_export_seed", 42),
    }


def load_settings(config_path: str | None = "conf/spark_transform.yaml") -> dict[str, Any]:
    """Load and normalize settings from config file."""
    return _resolve_settings(load_config(config_path))


def _safe_partition_value(value: Any) -> str:
    return str(value).replace("/", "_")


def _resolve_local_path(path: str) -> str:
    if path.startswith("file:"):
        parsed = urlparse(path)
        if parsed.scheme == "file":
            if parsed.netloc:
                return unquote(f"//{parsed.netloc}{parsed.path}")
            return unquote(parsed.path or path[len("file:") :])
    return path


def _resample_filter() -> Any:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return resampling.LANCZOS
    return getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 3))


def _resize_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
    strategy: str,
    resample_filter: Any,
) -> Image.Image:
    if strategy == "stretch":
        return image.resize((target_width, target_height), resample_filter)

    src_w, src_h = image.size
    src_ratio = src_w / src_h
    dst_ratio = target_width / target_height

    if strategy == "center_crop":
        if src_ratio > dst_ratio:
            crop_w = int(src_h * dst_ratio)
            left = (src_w - crop_w) // 2
            image = image.crop((left, 0, left + crop_w, src_h))
        else:
            crop_h = int(src_w / dst_ratio)
            top = (src_h - crop_h) // 2
            image = image.crop((0, top, src_w, top + crop_h))
        return image.resize((target_width, target_height), resample_filter)

    resized = image.copy()
    resized.thumbnail((target_width, target_height), resample_filter)
    canvas = Image.new("L", (target_width, target_height), color=0)
    paste_x = (target_width - resized.width) // 2
    paste_y = (target_height - resized.height) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def _write_version_metadata(settings: dict[str, Any]) -> str:
    """Persist transform settings for traceability under images_silver/<version>/."""
    version_root = Path(settings["output_images_path"]) / settings["transform_version"]
    version_root.mkdir(parents=True, exist_ok=True)

    metadata_path = version_root / "metadata.json"
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "transform_version": settings["transform_version"],
        "target_size": [settings["target_width"], settings["target_height"]],
        "resize_strategy": settings["resize_strategy"],
        "image_mode": "L",
        "image_format": "PNG",
        "transform_backend": "pillow_map_partitions",
        "input_manifest_path": settings["input_manifest_path"],
        "output_manifest_path": settings["output_manifest_path"],
        "partition_by": settings["partition_by"],
        "partitions": settings["partitions"],
        "shuffle_partitions": settings["shuffle_partitions"],
        "debug_export_enabled": settings["debug_export_enabled"],
        "debug_export_path": settings["debug_export_path"],
        "debug_export_n": settings["debug_export_n"],
        "debug_export_per_class": settings["debug_export_per_class"],
        "debug_export_seed": settings["debug_export_seed"],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return str(metadata_path)


def _registry_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("transform_version", T.StringType(), nullable=False),
            T.StructField("generated_at_utc", T.StringType(), nullable=False),
            T.StructField("target_width", T.IntegerType(), nullable=False),
            T.StructField("target_height", T.IntegerType(), nullable=False),
            T.StructField("resize_strategy", T.StringType(), nullable=False),
            T.StructField("image_mode", T.StringType(), nullable=False),
            T.StructField("image_format", T.StringType(), nullable=False),
            T.StructField("partition_by", T.StringType(), nullable=False),
            T.StructField("partitions", T.IntegerType(), nullable=True),
            T.StructField("shuffle_partitions", T.IntegerType(), nullable=True),
            T.StructField("rows_written", T.LongType(), nullable=False),
            T.StructField("input_manifest_path", T.StringType(), nullable=False),
            T.StructField("output_manifest_path", T.StringType(), nullable=False),
            T.StructField("output_images_path", T.StringType(), nullable=False),
        ]
    )


def _upsert_transform_registry(
    spark: SparkSession,
    settings: dict[str, Any],
    rows_written: int,
) -> str:
    registry_path = str(settings["versions_registry_path"])
    new_row = (
        settings["transform_version"],
        datetime.now(timezone.utc).isoformat(),
        settings["target_width"],
        settings["target_height"],
        settings["resize_strategy"],
        "L",
        "PNG",
        ",".join(settings["partition_by"]),
        settings["partitions"],
        settings["shuffle_partitions"],
        int(rows_written),
        settings["input_manifest_path"],
        settings["output_manifest_path"],
        settings["output_images_path"],
    )

    conf_key = "spark.sql.sources.partitionOverwriteMode"
    previous_overwrite_mode = spark.conf.get(conf_key, "static")
    spark.conf.set(conf_key, "dynamic")
    try:
        (
            spark.createDataFrame([new_row], schema=_registry_schema())
            .write.mode("overwrite")
            .partitionBy("transform_version")
            .parquet(registry_path)
        )
    finally:
        spark.conf.set(conf_key, previous_overwrite_mode)
    return registry_path


def _transform_partition(
    rows: Iterable[Row],
    settings: dict[str, Any],
) -> Iterable[tuple[Any, ...]]:
    transform_version = settings["transform_version"]
    target_width = int(settings["target_width"])
    target_height = int(settings["target_height"])
    resize_strategy = str(settings["resize_strategy"])
    resample_filter = _resample_filter()

    for row in rows:
        image_id = row["image_id"]
        raw_path = str(row["raw_path"])
        pathology = row["pathology"]
        label_idx = row["label_idx"]
        modality = row["modality"]
        file_size = row["file_size"]
        is_valid = row["is_valid"]

        local_raw_path = _resolve_local_path(raw_path)
        with Image.open(local_raw_path) as image:
            gray = image.convert("L")
            orig_w, orig_h = gray.size
            transformed = _resize_image(
                image=gray,
                target_width=target_width,
                target_height=target_height,
                strategy=resize_strategy,
                resample_filter=resample_filter,
            )
            buffer = io.BytesIO()
            transformed.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()

        yield (
            image_id,
            raw_path,
            pathology,
            label_idx,
            modality,
            file_size,
            is_valid,
            png_bytes,
            len(png_bytes),
            int(orig_w),
            int(orig_h),
            target_width,
            target_height,
            1,
            transform_version,
        )


def _output_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("image_id", T.StringType(), nullable=False),
            T.StructField("raw_path", T.StringType(), nullable=False),
            T.StructField("pathology", T.StringType(), nullable=True),
            T.StructField("label_idx", T.IntegerType(), nullable=True),
            T.StructField("modality", T.StringType(), nullable=True),
            T.StructField("file_size", T.LongType(), nullable=True),
            T.StructField("is_valid", T.BooleanType(), nullable=True),
            T.StructField("processed_bytes", T.BinaryType(), nullable=False),
            T.StructField("processed_file_size", T.LongType(), nullable=False),
            T.StructField("orig_width", T.IntegerType(), nullable=False),
            T.StructField("orig_height", T.IntegerType(), nullable=False),
            T.StructField("new_width", T.IntegerType(), nullable=False),
            T.StructField("new_height", T.IntegerType(), nullable=False),
            T.StructField("channels", T.IntegerType(), nullable=False),
            T.StructField("transform_version", T.StringType(), nullable=False),
        ]
    )


def _build_transform_df(
    spark: SparkSession,
    manifest_df: DataFrame,
    settings: dict[str, Any],
) -> DataFrame:
    required_columns = {
        "image_id",
        "raw_path",
        "pathology",
        "label_idx",
        "modality",
        "file_size",
        "is_valid",
    }
    missing = sorted(required_columns - set(manifest_df.columns))
    if missing:
        raise ValueError(
            "Input manifest is missing required columns: " + ", ".join(missing)
        )

    base_df = manifest_df.select(
        "image_id",
        "raw_path",
        "pathology",
        "label_idx",
        "modality",
        "file_size",
        "is_valid",
    )

    transformed_rdd = base_df.rdd.mapPartitions(
        lambda rows: _transform_partition(rows, settings)
    )
    return spark.createDataFrame(transformed_rdd, schema=_output_schema())


def _resolve_debug_sample_df(
    transformed_df: DataFrame,
    settings: dict[str, Any],
) -> DataFrame:
    debug_export_per_class = settings["debug_export_per_class"]
    seed = settings["debug_export_seed"]

    if debug_export_per_class:
        sample_window = Window.partitionBy("pathology").orderBy(F.rand(seed))
        return (
            transformed_df.withColumn(
                "__sample_row_number", F.row_number().over(sample_window)
            )
            .where(F.col("__sample_row_number") <= F.lit(debug_export_per_class))
            .drop("__sample_row_number")
        )

    return transformed_df.orderBy(F.rand(seed)).limit(int(settings["debug_export_n"]))


def _export_debug_preview(transformed_df: DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
    if not settings["debug_export_enabled"]:
        return {
            "debug_rows_exported": 0,
            "debug_export_path": None,
        }

    debug_version_root = (
        Path(settings["debug_export_path"]) / settings["transform_version"]
    ).resolve()
    if debug_version_root.is_symlink() or debug_version_root.is_file():
        debug_version_root.unlink()
    elif debug_version_root.exists():
        shutil.rmtree(debug_version_root)
    debug_version_root.mkdir(parents=True, exist_ok=True)

    sample_df = _resolve_debug_sample_df(
        transformed_df.select(
            "image_id",
            "pathology",
            "modality",
            "processed_bytes",
        ),
        settings,
    )

    exported_rows = 0
    for row in sample_df.collect():
        image_bytes = row["processed_bytes"]
        if image_bytes is None:
            raise ValueError("Debug export requires non-null `processed_bytes`.")

        output_file = (
            debug_version_root
            / f"pathology={_safe_partition_value(row['pathology'])}"
            / f"modality={_safe_partition_value(row['modality'])}"
            / f"{row['image_id']}.png"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(bytes(image_bytes))
        exported_rows += 1

    return {
        "debug_rows_exported": exported_rows,
        "debug_export_path": str(debug_version_root),
    }


def run_transform(spark: SparkSession, settings: dict[str, Any]) -> dict[str, Any]:
    """Run transform pipeline from normalized or partial settings."""
    resolved_settings = _resolve_settings(settings)
    metadata_path = _write_version_metadata(resolved_settings)

    if resolved_settings["shuffle_partitions"]:
        spark.conf.set(
            "spark.sql.shuffle.partitions",
            resolved_settings["shuffle_partitions"],
        )

    manifest_df = spark.read.parquet(resolved_settings["input_manifest_path"])
    input_rows = manifest_df.count()
    transformed_df = _build_transform_df(spark, manifest_df, resolved_settings)

    partition_by = resolved_settings["partition_by"]
    if partition_by:
        unknown = [col for col in partition_by if col not in transformed_df.columns]
        if unknown:
            raise ValueError(
                "Unknown partition columns: "
                + ", ".join(unknown)
                + ". Available columns: "
                + ", ".join(transformed_df.columns)
            )

    if resolved_settings["partitions"]:
        if partition_by:
            transformed_df = transformed_df.repartition(
                resolved_settings["partitions"], *partition_by
            )
        else:
            transformed_df = transformed_df.repartition(resolved_settings["partitions"])
    elif partition_by:
        transformed_df = transformed_df.repartition(*partition_by)

    writer = transformed_df.write.mode("overwrite")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.parquet(resolved_settings["output_manifest_path"])

    debug_info = {
        "debug_rows_exported": 0,
        "debug_export_path": None,
    }
    if resolved_settings["debug_export_enabled"]:
        debug_info = _export_debug_preview(
            spark.read.parquet(resolved_settings["output_manifest_path"]),
            resolved_settings,
        )
    registry_path = _upsert_transform_registry(spark, resolved_settings, input_rows)

    return {
        "settings": resolved_settings,
        "rows_written": input_rows,
        "output_manifest_path": resolved_settings["output_manifest_path"],
        "output_metadata_path": metadata_path,
        "output_registry_path": registry_path,
        "debug_rows_exported": debug_info["debug_rows_exported"],
        "debug_export_path": debug_info["debug_export_path"],
    }


def main() -> None:
    args = _parse_args()
    settings = load_settings(args.config)

    spark_builder = SparkSession.builder.appName(settings["app_name"])
    if settings["master"]:
        spark_builder = spark_builder.master(settings["master"])
    spark = spark_builder.getOrCreate()

    result = run_transform(spark, settings)

    print("[transform_job] settings:")
    print(json.dumps(result["settings"], indent=2))
    print(f"[transform_job] rows written: {result['rows_written']}")
    print(f"[transform_job] output manifest: {result['output_manifest_path']}")
    print(f"[transform_job] output metadata: {result['output_metadata_path']}")
    print(f"[transform_job] output registry: {result['output_registry_path']}")
    print(f"[transform_job] debug rows exported: {result['debug_rows_exported']}")
    print(f"[transform_job] debug export path: {result['debug_export_path']}")

    spark.stop()


if __name__ == "__main__":
    main()
