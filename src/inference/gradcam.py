from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


DEFAULT_GRADCAM_LAYER_NAME = "final_conv_features"


def _find_last_4d_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        output = getattr(layer, "output", None)
        shape = getattr(output, "shape", None)
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("No 4D feature layer found for Grad-CAM.")


def _resolve_gradcam_layer_name(
    model: tf.keras.Model,
    preferred_layer_name: str = DEFAULT_GRADCAM_LAYER_NAME,
) -> str:
    try:
        model.get_layer(preferred_layer_name)
        return preferred_layer_name
    except ValueError:
        return _find_last_4d_layer(model)


def compute_gradcam(
    model: tf.keras.Model,
    image_array: np.ndarray,
    pred_index: int,
    layer_name: str = DEFAULT_GRADCAM_LAYER_NAME,
) -> np.ndarray:
    target_layer_name = _resolve_gradcam_layer_name(model, layer_name)
    with tf.GradientTape() as tape:
        x = tf.convert_to_tensor(image_array)
        conv_outputs = None

        for layer in model.layers:
            x = layer(x, training=False)
            if layer.name == target_layer_name:
                conv_outputs = x

        predictions = x
        if conv_outputs is None:
            raise ValueError(f"Grad-CAM target layer not reached: {target_layer_name}")

        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        raise ValueError(
            f"Unable to compute Grad-CAM gradients for layer `{target_layer_name}`."
        )
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if float(max_val) > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def save_gradcam_artifacts(
    *,
    output_artifacts_dir: str,
    image_id: str,
    image_array: np.ndarray,
    heatmap: np.ndarray,
) -> dict[str, str]:
    artifacts_root = Path(output_artifacts_dir)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    original = (image_array[0, :, :, 0] * 255.0).clip(0, 255).astype(np.uint8)
    original_img = Image.fromarray(original, mode="L").convert("RGB")

    heatmap_img = Image.fromarray((heatmap * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    heatmap_img = heatmap_img.resize(original_img.size, Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img, dtype=np.uint8)

    heatmap_rgb = np.zeros((heatmap_arr.shape[0], heatmap_arr.shape[1], 3), dtype=np.uint8)
    heatmap_rgb[:, :, 0] = heatmap_arr

    overlay = (
        0.65 * np.array(original_img, dtype=np.float32)
        + 0.35 * heatmap_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    gradcam_path = str(artifacts_root / f"{image_id}_gradcam.png")
    gradcam_overlay_path = str(artifacts_root / f"{image_id}_gradcam_overlay.png")

    heatmap_img.save(gradcam_path)
    Image.fromarray(overlay).save(gradcam_overlay_path)

    return {
        "gradcam_path": gradcam_path,
        "gradcam_overlay_path": gradcam_overlay_path,
    }
