"""Helpers to consume Spark-produced TFRecord dataset artifacts for training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spark_jobs.config_utils import to_abs_local_path

SHARD_MANIFEST_DIRNAME = "shard_manifest.parquet"
DATASET_SUMMARY_FILENAME = "dataset_summary.json"
def resolve_dataset_artifact_paths(input_tfrecord_path: str) -> tuple[Path, Path, Path]:
    root = to_abs_local_path(input_tfrecord_path)
    return (
        root,
        root / SHARD_MANIFEST_DIRNAME,
        root / DATASET_SUMMARY_FILENAME,
    )


def load_dataset_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Required dataset summary not found: {summary_path}"
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_split_files(root: Path, shard_manifest_path: Path) -> dict[str, list[str]]:
    if not shard_manifest_path.exists():
        raise FileNotFoundError(
            f"Required shard manifest not found: {shard_manifest_path}"
        )

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "pyarrow is required to read shard_manifest.parquet in the new training pipeline."
        )

    table = pq.read_table(
        str(shard_manifest_path),
        columns=["split", "relative_path"],
    )
    split_files = {"train": [], "val": [], "test": []}
    for row in table.to_pylist():
        split = str(row["split"])
        if split not in split_files:
            continue
        relative_path = str(row["relative_path"])
        split_files[split].append(str((root / relative_path).resolve()))

    for split in split_files:
        split_files[split].sort()
    return split_files


def lookup_label_idx(summary: dict[str, Any], pathology_name: str) -> int | None:
    normalized = pathology_name.strip()
    for row in summary.get("labels", []):
        if str(row.get("pathology", "")).strip() == normalized:
            value = row.get("label_idx")
            if value is None:
                return None
            return int(value)
    return None


def get_label_count(
    summary: dict[str, Any],
    *,
    split: str,
    label_idx: int,
) -> int | None:
    split_counts = summary.get("label_counts_by_split", {}).get(split)
    if not isinstance(split_counts, dict):
        return None

    count_value = split_counts.get(str(label_idx))
    if count_value is None:
        return 0
    return int(count_value)
