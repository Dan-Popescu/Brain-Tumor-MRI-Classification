from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tensorflow as tf


def resolve_model_paths(settings: dict[str, Any]) -> dict[str, str | None]:
    models = settings.get("models", {})
    return {
        "classifier_model_path": models.get("classifier_model_path"),
        "autoencoder_model_path": models.get("autoencoder_model_path"),
        "autoencoder_threshold_path": models.get("autoencoder_threshold_path"),
    }


def load_classifier_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def load_autoencoder_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def load_autoencoder_threshold(threshold_path: str) -> dict[str, Any]:
    return json.loads(Path(threshold_path).read_text(encoding="utf-8"))
