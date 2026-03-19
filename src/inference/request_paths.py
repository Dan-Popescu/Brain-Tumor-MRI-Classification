from __future__ import annotations

from pathlib import Path


def build_request_paths(request_root: str, request_id: str) -> dict[str, str]:
    root = Path(request_root) / request_id
    return {
        "request_root": str(root),
        "raw_dir": str(root / "raw"),
        "bronze_manifest_path": str(root / "manifest_bronze.parquet"),
        "silver_manifest_path": str(root / "manifest_silver.parquet"),
        "predictions_path": str(root / "predictions.parquet"),
        "artifacts_dir": str(root / "artifacts"),
        "status_path": str(root / "status.json"),
    }
