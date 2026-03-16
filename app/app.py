"""Streamlit app: Brain Tumor Classification + Anomaly Detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ── Paths (defaults – adjust if models are elsewhere) ────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASSIFIER_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_cnn" / "best.keras"
AUTOENCODER_MODEL_PATH = (
    PROJECT_ROOT / "models" / "anomaly_autoencoder" / "best_autoencoder.keras"
)
THRESHOLD_PATH = (
    PROJECT_ROOT / "models" / "anomaly_autoencoder" / "anomaly_threshold.json"
)

# Label names in alphabetical order (same order as preprocess_job label_idx)
LABEL_NAMES: list[str] = [
    "Astrocitoma",
    "Carcinoma",
    "Ependimoma",
    "Ganglioglioma",
    "Germinoma",
    "Glioblastoma",
    "Granuloma",
    "Meduloblastoma",
    "Meningioma",
    "Neurocitoma",
    "Oligodendroglioma",
    "Papiloma",
    "Schwannoma",
    "Tuberculoma",
    "_NORMAL",
]


# ── Cached model loading ─────────────────────────────────────────────────────


@st.cache_resource
def load_classifier() -> tf.keras.Model | None:
    path = CLASSIFIER_MODEL_PATH
    if not path.exists():
        return None
    return tf.keras.models.load_model(str(path))


@st.cache_resource
def load_autoencoder() -> tf.keras.Model | None:
    path = AUTOENCODER_MODEL_PATH
    if not path.exists():
        return None
    return tf.keras.models.load_model(str(path))


@st.cache_data
def load_threshold() -> dict | None:
    path = THRESHOLD_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ── Image preprocessing ──────────────────────────────────────────────────────


def preprocess_image(
    pil_image: Image.Image, target_height: int, target_width: int
) -> np.ndarray:
    """Convert uploaded image to model-ready array (1, H, W, 1) float32 in [0,1]."""
    gray = pil_image.convert("L")

    # Resize with aspect ratio preservation + padding (same as transform_job)
    orig_w, orig_h = gray.size
    scale = min(target_width / orig_w, target_height / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = gray.resize((new_w, new_h), Image.LANCZOS)

    # Center-pad to target size
    padded = Image.new("L", (target_width, target_height), color=0)
    paste_x = (target_width - new_w) // 2
    paste_y = (target_height - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))

    arr = np.array(padded, dtype=np.float32) / 255.0
    return arr.reshape(1, target_height, target_width, 1)


# ── Anomaly heatmap ──────────────────────────────────────────────────────────


def compute_anomaly_map(
    model: tf.keras.Model,
    image_array: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (error_map, binary_mask, max_error).

    error_map : (H, W) per-pixel MSE
    binary_mask : (H, W) bool – True where error > threshold
    max_error : scalar – maximum pixel error
    """
    reconstructed = model.predict(image_array, verbose=0)
    error_map = np.square(image_array[0, :, :, 0] - reconstructed[0, :, :, 0])
    binary_mask = error_map > threshold
    max_error = float(np.max(error_map))
    return error_map, binary_mask, max_error


def overlay_heatmap(
    original_gray: np.ndarray,
    error_map: np.ndarray,
    binary_mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Create an RGB overlay: original in gray, anomalous pixels highlighted red."""
    # Normalize error map for visualization
    if error_map.max() > 0:
        norm_error = error_map / error_map.max()
    else:
        norm_error = error_map

    # Build RGB from grayscale original
    h, w = original_gray.shape
    rgb = np.stack([original_gray] * 3, axis=-1)  # (H, W, 3)

    # Red overlay where mask is True
    overlay = rgb.copy()
    overlay[binary_mask, 0] = np.clip(
        overlay[binary_mask, 0] * (1 - alpha) + alpha * norm_error[binary_mask] * 255,
        0,
        255,
    ).astype(np.uint8)
    overlay[binary_mask, 1] = (overlay[binary_mask, 1] * (1 - alpha)).astype(np.uint8)
    overlay[binary_mask, 2] = (overlay[binary_mask, 2] * (1 - alpha)).astype(np.uint8)

    return overlay


# ── Streamlit UI ─────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Brain Tumor MRI Analysis",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Brain Tumor MRI Analysis")
    st.markdown(
        "Upload a brain MRI scan to **classify the tumor type** and "
        "**detect anomalous regions** via reconstruction-based anomaly detection."
    )

    # Sidebar – model status
    st.sidebar.header("Model Status")
    classifier = load_classifier()
    autoencoder = load_autoencoder()
    threshold_info = load_threshold()

    if classifier is not None:
        st.sidebar.success("✅ Classifier loaded")
    else:
        st.sidebar.error(f"❌ Classifier not found at `{CLASSIFIER_MODEL_PATH}`")

    if autoencoder is not None and threshold_info is not None:
        st.sidebar.success("✅ Anomaly autoencoder loaded")
        st.sidebar.info(
            f"Strict threshold: {threshold_info['threshold_strict']:.6f}\n\n"
            f"Normal images used for calibration: {threshold_info['normal_images_evaluated']}"
        )
    else:
        st.sidebar.warning("⚠️ Anomaly model or threshold not found")

    # Sidebar – threshold selector
    threshold_mode = "strict"
    if threshold_info is not None:
        threshold_mode = st.sidebar.selectbox(
            "Anomaly Threshold",
            ["strict", "p99", "p95", "custom"],
            help=(
                "**strict**: guarantees 0% false positives on normal scans. "
                "**p99/p95**: slightly more sensitive. "
                "**custom**: set your own value."
            ),
        )
    custom_threshold: float | None = None
    if threshold_mode == "custom" and threshold_info is not None:
        custom_threshold = st.sidebar.slider(
            "Custom threshold",
            min_value=0.0,
            max_value=float(threshold_info["threshold_strict"]) * 2,
            value=float(threshold_info["threshold_strict"]),
            step=0.0001,
            format="%.6f",
        )

    # ── File uploader ──
    uploaded_file = st.file_uploader(
        "Upload a brain MRI image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPEG, PNG",
    )

    if uploaded_file is None:
        st.info("👆 Upload an MRI image to get started.")
        return

    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Uploaded MRI", use_container_width=False, width=300)

    # Determine image dimensions from threshold info or default
    img_h = threshold_info["image_height"] if threshold_info else 224
    img_w = threshold_info["image_width"] if threshold_info else 224

    image_array = preprocess_image(pil_image, img_h, img_w)

    col1, col2 = st.columns(2)

    # ── Classification ──
    with col1:
        st.subheader("🔬 Tumor Classification")
        if classifier is not None:
            preds = classifier.predict(image_array, verbose=0)
            pred_idx = int(np.argmax(preds[0]))
            confidence = float(preds[0][pred_idx])

            if pred_idx < len(LABEL_NAMES):
                pred_label = LABEL_NAMES[pred_idx]
            else:
                pred_label = f"Unknown (idx={pred_idx})"

            st.metric("Predicted Class", pred_label)
            st.metric("Confidence", f"{confidence:.2%}")

            # Show top-5 predictions
            top5_indices = np.argsort(preds[0])[::-1][:5]
            st.markdown("**Top 5 Predictions:**")
            for rank, idx in enumerate(top5_indices, 1):
                name = LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else f"idx={idx}"
                prob = float(preds[0][idx])
                bar_width = int(prob * 100)
                st.markdown(
                    f"{rank}. **{name}** — {prob:.2%} "
                    f"`{'█' * max(1, bar_width // 5)}{'░' * (20 - max(1, bar_width // 5))}`"
                )
        else:
            st.warning("Classifier model not available.")

    # ── Anomaly Detection ──
    with col2:
        st.subheader("🔥 Anomaly Detection")
        if autoencoder is not None and threshold_info is not None:
            # Select threshold
            if threshold_mode == "strict":
                active_threshold = threshold_info["threshold_strict"]
            elif threshold_mode == "p99":
                active_threshold = threshold_info["threshold_p99"]
            elif threshold_mode == "p95":
                active_threshold = threshold_info["threshold_p95"]
            else:
                active_threshold = custom_threshold or threshold_info["threshold_strict"]

            error_map, binary_mask, max_error = compute_anomaly_map(
                autoencoder, image_array, active_threshold
            )

            anomalous_pixels = int(np.sum(binary_mask))
            total_pixels = binary_mask.size
            anomaly_ratio = anomalous_pixels / total_pixels

            is_anomalous = anomalous_pixels > 0

            if is_anomalous:
                st.error(
                    f"⚠️ Anomaly detected! {anomalous_pixels:,} pixels "
                    f"({anomaly_ratio:.2%}) exceed threshold."
                )
            else:
                st.success("✅ No anomalous pixels detected (appears normal).")

            st.metric("Max Pixel Error", f"{max_error:.6f}")
            st.metric("Active Threshold", f"{active_threshold:.6f}")
            st.metric("Anomalous Pixels", f"{anomalous_pixels:,} / {total_pixels:,}")

            # Visualizations
            gray_2d = (image_array[0, :, :, 0] * 255).astype(np.uint8)

            # Heatmap overlay
            if is_anomalous:
                overlay = overlay_heatmap(gray_2d, error_map, binary_mask)
                st.image(
                    overlay,
                    caption="Anomaly Overlay (red = anomalous pixels)",
                    use_container_width=True,
                )

            # Raw error map
            if error_map.max() > 0:
                error_display = (error_map / error_map.max() * 255).astype(np.uint8)
            else:
                error_display = (error_map * 255).astype(np.uint8)
            st.image(
                error_display,
                caption="Reconstruction Error Map",
                use_container_width=True,
            )

            # Reconstruction
            reconstructed = autoencoder.predict(image_array, verbose=0)
            recon_display = (reconstructed[0, :, :, 0] * 255).astype(np.uint8)
            st.image(
                recon_display,
                caption="Autoencoder Reconstruction",
                use_container_width=True,
            )
        else:
            st.warning("Anomaly detection model not available.")


if __name__ == "__main__":
    main()