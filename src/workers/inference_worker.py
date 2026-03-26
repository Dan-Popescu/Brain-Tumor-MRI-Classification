from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import SparkSession, types as T

from spark_jobs.config_utils import load_config
from spark_jobs.inference_job import run_inference, _resolve_settings
from spark_jobs.transform_job import load_settings as load_transform_settings
from spark_jobs.transform_job import run_transform


POLL_INTERVAL_SECONDS = 2.0
IGNORED_REQUEST_DIR_NAMES = {"templates"}

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference worker watching request folders.")
    parser.add_argument(
        "config",
        nargs="?",
        default="conf/spark_inference.yaml",
        help="Path to inference config file.",
    )
    return parser.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ensure_requests_root(requests_root: Path) -> Path:
    requests_root.mkdir(parents=True, exist_ok=True)
    return requests_root


def _list_pending_requests(requests_root: Path) -> list[Path]:
    request_dirs = []
    for path in requests_root.iterdir():
        if not path.is_dir():
            continue
        if path.name in IGNORED_REQUEST_DIR_NAMES:
            continue
        status = _read_json(path / "status.json").get("status")
        if status == "pending":
            request_dirs.append(path)
    return sorted(request_dirs)


def _build_request_paths(request_dir: Path) -> dict[str, Path]:
    return {
        "request_dir": request_dir,
        "raw_dir": request_dir / "raw",
        "metadata_path": request_dir / "metadata.json",
        "status_path": request_dir / "status.json",
        "manifest_bronze_path": request_dir / "manifest_bronze.parquet",
        "manifest_silver_path": request_dir / "manifest_silver.parquet",
        "predictions_path": request_dir / "predictions.parquet",
        "artifacts_dir": request_dir / "artifacts",
        "result_path": request_dir / "result.json",
        "error_path": request_dir / "error.json",
    }


def _update_status(
    request_dir: Path,
    *,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    paths = _build_request_paths(request_dir)
    current = _read_json(paths["status_path"])
    current.update(
        {
            "request_id": request_dir.name,
            "status": status,
            "updated_at": _utc_now_iso(),
        }
    )
    if extra:
        current.update(extra)
    _write_json(paths["status_path"], current)


def _load_request_metadata(request_dir: Path) -> dict[str, Any]:
    paths = _build_request_paths(request_dir)
    return _read_json(paths["metadata_path"])


def _request_has_raw_input(request_dir: Path) -> bool:
    raw_dir = _build_request_paths(request_dir)["raw_dir"]
    return raw_dir.is_dir() and any(path.is_file() for path in raw_dir.rglob("*"))


def _find_single_raw_image(request_dir: Path) -> Path:
    raw_dir = _build_request_paths(request_dir)["raw_dir"]
    files = [path for path in raw_dir.rglob("*") if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No raw image found in {raw_dir}")
    if len(files) > 1:
        raise ValueError(
            f"Expected exactly one raw image in {raw_dir}, found {len(files)}"
        )
    return files[0]


def _single_image_bronze_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("request_id", T.StringType(), nullable=False),
            T.StructField("image_id", T.StringType(), nullable=False),
            T.StructField("raw_path", T.StringType(), nullable=False),
            T.StructField("pathology", T.StringType(), nullable=True),
            T.StructField("label_idx", T.IntegerType(), nullable=True),
            T.StructField("modality", T.StringType(), nullable=True),
            T.StructField("file_size", T.LongType(), nullable=True),
        ]
    )


def _write_single_image_bronze_manifest(
    spark: SparkSession,
    request_dir: Path,
    request_id: str,
) -> str:
    paths = _build_request_paths(request_dir)
    raw_image_path = _find_single_raw_image(request_dir)

    raw_path_str = str(raw_image_path.resolve())
    image_id = hashlib.sha256(raw_path_str.encode("utf-8")).hexdigest()
    file_size = int(raw_image_path.stat().st_size)

    row = (
        request_id,
        image_id,
        raw_path_str,
        None,
        None,
        None,
        file_size,
    )

    (
        spark.createDataFrame([row], schema=_single_image_bronze_schema())
        .write.mode("overwrite")
        .parquet(str(paths["manifest_bronze_path"]))
    )
    return str(paths["manifest_bronze_path"])


def _run_request_transform_only(
    spark: SparkSession,
    request_dir: Path,
    metadata: dict[str, Any],
    base_settings: dict[str, Any],
) -> None:
    paths = _build_request_paths(request_dir)

    _update_status(request_dir, status="building_bronze")
    _write_single_image_bronze_manifest(
        spark=spark,
        request_dir=request_dir,
        request_id=request_dir.name,
    )

    _update_status(request_dir, status="building_silver")
    transform_settings = load_transform_settings("conf/spark_transform.yaml")
    transform_settings["input_manifest_path"] = str(paths["manifest_bronze_path"])
    transform_settings["output_manifest_path"] = str(paths["manifest_silver_path"])
    transform_settings["output_images_path"] = str(request_dir / "images_silver")
    transform_settings["versions_registry_path"] = str(
        request_dir / "transform_versions.parquet"
    )
    transform_settings["master"] = base_settings.get("master")
    transform_settings["partitions"] = int(metadata.get("transform_partitions", 1))
    transform_settings["shuffle_partitions"] = int(
        metadata.get("transform_shuffle_partitions", 1)
    )
    transform_settings["partition_by"] = []
    transform_settings["debug_export_enabled"] = False

    run_transform(spark, transform_settings)


def _resolve_request_input_manifest(
    request_dir: Path,
    metadata: dict[str, Any],
    base_settings: dict[str, Any],
) -> str:
    paths = _build_request_paths(request_dir)

    manifest_override = metadata.get("input_manifest_path")
    if manifest_override:
        return str(Path(str(manifest_override)).resolve())

    if paths["manifest_silver_path"].exists():
        return str(paths["manifest_silver_path"])

    if base_settings.get("input_manifest_path"):
        return str(Path(str(base_settings["input_manifest_path"])).resolve())

    raise FileNotFoundError(
        f"No input manifest found for request {request_dir.name}. "
        "Expected `metadata.json` to contain `input_manifest_path` "
        f"or local file {paths['manifest_silver_path']}."
    )


def _prepare_request_settings(
    base_settings: dict[str, Any],
    request_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    paths = _build_request_paths(request_dir)

    settings = dict(base_settings)
    settings["input_manifest_path"] = _resolve_request_input_manifest(
        request_dir,
        metadata,
        base_settings,
    )
    settings["output_predictions_path"] = str(paths["predictions_path"])
    settings["output_artifacts_path"] = str(paths["artifacts_dir"])

    for key in [
        "top_k",
        "threshold_mode",
        "custom_threshold",
        "write_visual_artifacts",
        "master",
        "partitions",
        "partition_by",
        "enabled_outputs",
    ]:
        if key in metadata:
            settings[key] = metadata[key]

    return settings


def _prediction_preview(predictions_path: Path) -> dict[str, Any]:
    if not predictions_path.exists():
        return {"rows": 0, "first_prediction": None}

    df = pd.read_parquet(predictions_path)
    if df.empty:
        return {"rows": 0, "first_prediction": None}

    return {
        "rows": int(len(df)),
        "first_prediction": df.iloc[0].to_dict(),
    }


def _build_result_payload(
    request_dir: Path,
    inference_result: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    paths = _build_request_paths(request_dir)
    preview = _prediction_preview(paths["predictions_path"])
    return {
        "request_id": request_dir.name,
        "status": "done",
        "finished_at": _utc_now_iso(),
        "input_manifest_path": metadata.get(
            "input_manifest_path",
            str(paths["manifest_silver_path"]),
        ),
        "predictions_path": str(paths["predictions_path"]),
        "artifacts_dir": str(paths["artifacts_dir"]),
        "run_summary": inference_result,
        "prediction_preview": preview,
    }


def _process_request(
    spark: SparkSession,
    *,
    request_dir: Path,
    base_settings: dict[str, Any],
) -> dict[str, Any]:
    metadata = _load_request_metadata(request_dir)
    paths = _build_request_paths(request_dir)

    _update_status(request_dir, status="resolving_input")
    if (
        not metadata.get("input_manifest_path")
        and not paths["manifest_silver_path"].exists()
        and _request_has_raw_input(request_dir)
    ):
        _run_request_transform_only(
            spark,
            request_dir,
            metadata,
            base_settings,
        )

    request_settings = _prepare_request_settings(base_settings, request_dir, metadata)
    _update_status(request_dir, status="running_inference")
    result = run_inference(spark, request_settings)
    _update_status(request_dir, status="writing_results")
    result_payload = _build_result_payload(request_dir, result, metadata)
    _write_json(_build_request_paths(request_dir)["result_path"], result_payload)
    return result_payload


def run_worker(config_path: str = "conf/spark_inference.yaml") -> None:
    raw_config = load_config(config_path)
    settings = _resolve_settings(raw_config)

    requests_root = Path("data/inference/requests")
    _ensure_requests_root(requests_root)

    builder = SparkSession.builder.appName(f"{settings['app_name']}-worker")
    if settings["master"]:
        builder = builder.master(settings["master"])
    spark = builder.getOrCreate()

    print(f"[worker] watching {requests_root}")
    print(f"[worker] base config: {config_path}")

    try:
        while True:
            pending_requests = _list_pending_requests(requests_root)
            if not pending_requests:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            request_dir = pending_requests[0]

            try:
                _update_status(
                    request_dir,
                    status="queued",
                    extra={"started_at": _utc_now_iso()},
                )

                result_payload = _process_request(
                    spark,
                    request_dir=request_dir,
                    base_settings=settings,
                )

                _update_status(
                    request_dir,
                    status="done",
                    extra={
                        "finished_at": _utc_now_iso(),
                        "result_path": str(_build_request_paths(request_dir)["result_path"]),
                        "predictions_path": result_payload["predictions_path"],
                    },
                )
                print(f"[worker] done: {request_dir.name}")

            except Exception as exc:
                error_payload = {
                    "request_id": request_dir.name,
                    "failed_at": _utc_now_iso(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                _write_json(_build_request_paths(request_dir)["error_path"], error_payload)
                _update_status(
                    request_dir,
                    status="failed",
                    extra={"finished_at": _utc_now_iso()},
                )
                print(f"[worker] failed: {request_dir.name} -> {exc}")

    finally:
        spark.stop()


if __name__ == "__main__":
    args = _parse_args()
    run_worker(args.config)
