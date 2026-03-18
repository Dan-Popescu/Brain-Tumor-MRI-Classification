"""Shared config and parsing helpers for local Spark and Tensorflow scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def load_config(config_path: str | None) -> dict[str, Any]:
    """Load YAML/JSON config from disk. Returns empty dict for missing path arg."""
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

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


def as_positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `{field_name}` value: {value}") from exc
    if parsed <= 0:
        raise ValueError(f"`{field_name}` must be > 0, got {parsed}")
    return parsed


def as_positive_int_or_none(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return as_positive_int(value, field_name)


def as_int(value: Any, field_name: str, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `{field_name}` value: {value}") from exc


def as_positive_float(value: Any, field_name: str, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `{field_name}` value: {value}") from exc
    if parsed <= 0:
        raise ValueError(f"`{field_name}` must be > 0, got {parsed}")
    return parsed


def as_bool(value: Any, field_name: str, default: bool) -> bool:
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


def parse_partition_columns(raw_columns: Any) -> list[str]:
    if raw_columns is None:
        return []
    if isinstance(raw_columns, str):
        return [col.strip() for col in raw_columns.split(",") if col.strip()]
    if isinstance(raw_columns, list):
        return [str(col).strip() for col in raw_columns if str(col).strip()]
    raise ValueError(
        "Invalid `partition_by` value in config. Use a comma-separated string or list."
    )


def resolve_local_path(path: str) -> str:
    """Convert file URI paths from Spark (file:/...) to local filesystem paths."""
    if path.startswith("file:"):
        parsed = urlparse(path)
        if parsed.scheme == "file":
            if parsed.netloc:
                return unquote(f"//{parsed.netloc}{parsed.path}")
            return unquote(parsed.path or path[len("file:") :])
    return path
