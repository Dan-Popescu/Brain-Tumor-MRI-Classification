from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _save_grayscale(array_2d: np.ndarray, path: str) -> str:
    arr = np.clip(array_2d, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)
    return path


def _overlay_heatmap(
    original_gray: np.ndarray,
    error_map: np.ndarray,
    binary_mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    if error_map.max() > 0:
        norm_error = error_map / error_map.max()
    else:
        norm_error = error_map

    rgb = np.stack([original_gray] * 3, axis=-1)
    overlay = rgb.copy()
    overlay[binary_mask, 0] = np.clip(
        overlay[binary_mask, 0] * (1 - alpha) + alpha * norm_error[binary_mask] * 255,
        0,
        255,
    ).astype(np.uint8)
    overlay[binary_mask, 1] = (overlay[binary_mask, 1] * (1 - alpha)).astype(np.uint8)
    overlay[binary_mask, 2] = (overlay[binary_mask, 2] * (1 - alpha)).astype(np.uint8)
    return overlay


def save_autoencoder_artifacts(
    *,
    output_artifacts_dir: str,
    image_id: str,
    image_array: np.ndarray,
    autoencoder_reconstruction: np.ndarray,
    error_map: np.ndarray,
    binary_mask: np.ndarray,
) -> dict[str, str | None]:
    
    artifacts_root = Path(output_artifacts_dir)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    reconstruction_image_path = str(
        artifacts_root / f"{image_id}_reconstruction.png"
    )
    error_map_image_path = str(
        artifacts_root / f"{image_id}_error_map.png"
    )
    anomaly_overlay_image_path = str(
        artifacts_root / f"{image_id}_anomaly_overlay.png"
    )

    original_gray_2d = image_array[0, :, :, 0] * 255.0
    reconstruction_image_2d = autoencoder_reconstruction[0, :, :, 0] * 255.0

    if error_map.max() > 0:
        normalized_error_map_image = (error_map / error_map.max()) * 255.0
    else:
        normalized_error_map_image = error_map * 255.0

    _save_grayscale(reconstruction_image_2d, reconstruction_image_path)
    _save_grayscale(normalized_error_map_image, error_map_image_path)

    anomaly_overlay_path: str | None = None
    if np.any(binary_mask):
        overlay = _overlay_heatmap(
            original_gray_2d.astype(np.uint8),
            error_map,
            binary_mask,
        )
        Image.fromarray(overlay).save(anomaly_overlay_image_path)
        anomaly_overlay_path = anomaly_overlay_image_path

    return {
        "reconstruction_path": reconstruction_image_path,
        "error_map_path": error_map_image_path,
        "anomaly_overlay_path": anomaly_overlay_path,
    }
