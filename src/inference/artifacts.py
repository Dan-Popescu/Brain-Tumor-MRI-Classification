from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _save_grayscale(array_2d: np.ndarray, path: str) -> str:
    arr = np.clip(array_2d, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)
    return path


def save_autoencoder_artifacts(
    *,
    output_artifacts_dir: str,
    image_id: str,
    autoencoder_reconstruction: np.ndarray,
    error_map: np.ndarray,
) -> dict[str, str]:
    
    artifacts_root = Path(output_artifacts_dir)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    reconstruction_image_path = str(
        artifacts_root / f"{image_id}_reconstruction.png"
    )
    error_map_image_path = str(
        artifacts_root / f"{image_id}_error_map.png"
    )

    reconstruction_image_2d = autoencoder_reconstruction[0, :, :, 0] * 255.0

    if error_map.max() > 0:
        normalized_error_map_image = (error_map / error_map.max()) * 255.0
    else:
        normalized_error_map_image = error_map * 255.0

    _save_grayscale(reconstruction_image_2d, reconstruction_image_path)
    _save_grayscale(normalized_error_map_image, error_map_image_path)

    return {
        "reconstruction_path": reconstruction_image_path,
        "error_map_path": error_map_image_path,
    }