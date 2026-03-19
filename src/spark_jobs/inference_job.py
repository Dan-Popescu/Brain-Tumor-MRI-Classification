from __future__ import annotations

import argparse
import json
from typing import Any, Iterable

from pyspark.sql import Row, SparkSession

from spark_jobs.config_utils import (
    as_bool,
    as_positive_int_or_none,
    load_config,
    parse_partition_columns,
)
from inference.schemas import prediction_schema
from inference.model_registry import (
    load_autoencoder_model,
    load_autoencoder_threshold,
    load_classifier_model,
)
from inference.classification import decode_processed_bytes, predict_classifier
from inference.anomaly_autoencoder import predict_anomaly
from inference.artifacts import save_autoencoder_artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spark inference job.")
    parser.add_argument("--config", type=str, default="conf/spark_inference.yaml")
    return parser.parse_args()


def _resolve_settings(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_manifest_path": config["input_manifest_path"],
        "output_predictions_path": config["output_predictions_path"],
        "output_artifacts_path": config.get("output_artifacts_path"),
        "app_name": config.get("app_name", "mri-inference"),
        "master": config.get("master"),
        "top_k": int(config.get("top_k", 5)),
        "enabled_outputs": list(config.get("enabled_outputs", ["classification"])),
        "models": dict(config.get("models", {})),
        "threshold_mode": str(config.get("threshold_mode", "strict")),
        "write_visual_artifacts": as_bool(
            config.get("write_visual_artifacts"),
            "write_visual_artifacts",
            False,
        ),
        "partitions": as_positive_int_or_none(config.get("partitions"), "partitions"),
        "partition_by": parse_partition_columns(config.get("partition_by", [])),
    }


def _validate_input_manifest_columns(columns: list[str]) -> None:
    required_columns = {"image_id", "raw_path", "processed_bytes"}
    missing = sorted(required_columns - set(columns))
    if missing:
        raise ValueError(
            "Input manifest is missing required columns: " + ", ".join(missing)
        )


def _predict_partition(
    rows: Iterable[Row],
    settings: dict[str, Any],
) -> Iterable[tuple[Any, ...]]:
    classifier = None
    autoencoder = None
    threshold_info = None

    if "classification" in settings["enabled_outputs"]:
        classifier = load_classifier_model(settings["models"]["classifier_model_path"])

    if "anomaly_autoencoder" in settings["enabled_outputs"]:
        autoencoder = load_autoencoder_model(
            settings["models"]["autoencoder_model_path"]
        )
        threshold_info = load_autoencoder_threshold(
            settings["models"]["autoencoder_threshold_path"]
        )

    image_height = int(threshold_info["image_height"]) if threshold_info else 224
    image_width = int(threshold_info["image_width"]) if threshold_info else 224

    for row in rows:
        image_array = decode_processed_bytes(
            image_bytes=bytes(row["processed_bytes"]),
            image_height=image_height,
            image_width=image_width,
        )

        result = {
            "request_id": row["request_id"] if "request_id" in settings["input_columns"] else None,
            "image_id": row["image_id"],
            "raw_path": row["raw_path"],
            "pathology": row["pathology"] if "pathology" in settings["input_columns"] else None,
            "label_idx": row["label_idx"] if "label_idx" in settings["input_columns"] else None,
            "transform_version": row["transform_version"] if "transform_version" in settings["input_columns"] else None,
            "classifier_pred_idx": None,
            "classifier_pred_label": None,
            "classifier_confidence": None,
            "classifier_topk_json": None,
            "anomaly_max_error": None,
            "anomaly_threshold": None,
            "anomaly_is_detected": None,
            "anomalous_pixel_count": None,
            "anomaly_ratio": None,
            "reconstruction_path": None,
            "error_map_path": None,
            "anomaly_overlay_path": None,
            "gradcam_path": None,
            "gradcam_overlay_path": None,
        }

        if classifier is not None:
            result.update(
                predict_classifier(classifier, image_array, top_k=settings["top_k"])
            )

        if autoencoder is not None and threshold_info is not None:
            anomaly = predict_anomaly(
                autoencoder,
                image_array,
                threshold_info,
                threshold_mode=settings["threshold_mode"],
            )
            result["anomaly_max_error"] = anomaly["anomaly_max_error"]
            result["anomaly_threshold"] = anomaly["anomaly_threshold"]
            result["anomaly_is_detected"] = anomaly["anomaly_is_detected"]
            result["anomalous_pixel_count"] = anomaly["anomalous_pixel_count"]
            result["anomaly_ratio"] = anomaly["anomaly_ratio"]

            if (
                settings["write_visual_artifacts"]
                and settings["output_artifacts_path"]
            ):
                paths = save_autoencoder_artifacts(
                    output_artifacts_dir=settings["output_artifacts_path"],
                    image_id=row["image_id"],
                    autoencoder_reconstruction=anomaly["autoencoder_reconstruction"],
                    error_map=anomaly["error_map"],
                )
                result["reconstruction_path"] = paths["reconstruction_path"]
                result["error_map_path"] = paths["error_map_path"]

        yield tuple(result[field.name] for field in prediction_schema().fields)


def run_inference(spark: SparkSession, settings: dict[str, Any]) -> dict[str, Any]:
    manifest_df = spark.read.parquet(settings["input_manifest_path"])
    _validate_input_manifest_columns(manifest_df.columns)

    runtime_settings = dict(settings)
    runtime_settings["input_columns"] = set(manifest_df.columns)

    predictions_rdd = manifest_df.rdd.mapPartitions(
        lambda rows: _predict_partition(rows, runtime_settings)
    )
    predictions_df = spark.createDataFrame(predictions_rdd, schema=prediction_schema())

    partition_by = settings["partition_by"]
    if partition_by:
        unknown = [col for col in partition_by if col not in predictions_df.columns]
        if unknown:
            raise ValueError(
                "Unknown partition columns: "
                + ", ".join(unknown)
                + ". Available columns: "
                + ", ".join(predictions_df.columns)
            )

    if settings["partitions"] is not None:
        if partition_by:
            predictions_df = predictions_df.repartition(
                settings["partitions"],
                *partition_by,
            )
        else:
            predictions_df = predictions_df.repartition(settings["partitions"])
    elif partition_by:
        predictions_df = predictions_df.repartition(*partition_by)

    writer = predictions_df.write.mode("overwrite")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.parquet(settings["output_predictions_path"])

    rows_written = spark.read.parquet(settings["output_predictions_path"]).count()

    return {
        "rows_written": rows_written,
        "output_predictions_path": settings["output_predictions_path"],
    }


def main() -> None:
    args = _parse_args()
    settings = _resolve_settings(load_config(args.config))

    builder = SparkSession.builder.appName(settings["app_name"])
    if settings["master"]:
        builder = builder.master(settings["master"])
    spark = builder.getOrCreate()

    result = run_inference(spark, settings)
    print(json.dumps(result, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
