from __future__ import annotations 

from typing import Any 

import numpy as np 

def predict_anomaly(
        model: Any,
        image_array: np.ndarray,
        threshold_info: dict[str, Any],
        threshold_mode: str = "strict",
) -> dict[str, Any]:
    
    if threshold_mode == "p99":
        threshold = float(threshold_info["threshold_p99"])
    elif threshold_mode == "p95":
        threshold = float(threshold_info["threshold_p95"])
    else:
        threshold = float(threshold_info["threshold_strict"])

    autoencoder_reconstruction = model.predict(image_array, verbose=0)

    error_map = np.square(image_array[0, :, :, 0] - autoencoder_reconstruction[0, :, :, 0])
    binary_mask = error_map > threshold 

    anomalous_pixel_count = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)

    return {
        "anomaly_max_error": float(np.max(error_map)),
        "anomaly_threshold": threshold,
        "anomaly_is_detected": anomalous_pixel_count > 0,
        "anomalous_pixel_count": anomalous_pixel_count,
        "anomaly_ratio": anomalous_pixel_count / total_pixels if total_pixels else 0.0,
        "autoencoder_reconstruction": autoencoder_reconstruction,
        "reconstructed_array": autoencoder_reconstruction,
        "error_map": error_map,
        "binary_mask": binary_mask,
    }
