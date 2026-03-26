from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUESTS_ROOT = PROJECT_ROOT / "data" / "inference" / "requests"
CLASSIFIER_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_cnn" / "best.keras"
AUTOENCODER_MODEL_PATH = (
    PROJECT_ROOT / "models" / "anomaly_autoencoder" / "best_autoencoder.keras"
)
THRESHOLD_PATH = (
    PROJECT_ROOT / "models" / "anomaly_autoencoder" / "anomaly_threshold.json"
)
def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _request_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]


def _request_dir(request_id: str) -> Path:
    return REQUESTS_ROOT / request_id


def _find_request_dir(request_id: str) -> Path | None:
    direct = _request_dir(request_id)
    if direct.exists():
        return direct
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_threshold_info() -> dict[str, Any] | None:
    if not THRESHOLD_PATH.exists():
        return None
    return _read_json(THRESHOLD_PATH)


def _read_predictions_preview(request_dir: Path) -> dict[str, Any] | None:
    predictions_path = request_dir / "predictions.parquet"
    if not predictions_path.exists():
        return None

    df = pd.read_parquet(predictions_path)
    if df.empty:
        return None

    return df.iloc[0].to_dict()


def _parse_top5(first_prediction: dict[str, Any]) -> list[dict[str, Any]]:
    raw = first_prediction.get("classifier_topk_json")
    if not raw:
        return []
    return json.loads(raw)


def _artifact_path(request_dir: Path, first_prediction: dict[str, Any], key: str) -> Path | None:
    raw_value = first_prediction.get(key)
    if raw_value:
        candidate = Path(str(raw_value))
        if candidate.exists():
            return candidate

    image_id = first_prediction.get("image_id")
    if not image_id:
        return None

    suffix_by_key = {
        "reconstruction_path": f"{image_id}_reconstruction.png",
        "error_map_path": f"{image_id}_error_map.png",
        "anomaly_overlay_path": f"{image_id}_anomaly_overlay.png",
        "gradcam_path": f"{image_id}_gradcam.png",
        "gradcam_overlay_path": f"{image_id}_gradcam_overlay.png",
    }
    suffix = suffix_by_key.get(key)
    if suffix is None:
        return None

    fallback = request_dir / "artifacts" / suffix
    if fallback.exists():
        return fallback
    return None


def _request_raw_image(request_dir: Path) -> Path | None:
    raw_dir = request_dir / "raw"
    if not raw_dir.exists():
        return None
    files = [path for path in raw_dir.rglob("*") if path.is_file()]
    if not files:
        return None
    return files[0]


def _resolve_total_pixels(
    first_prediction: dict[str, Any],
    threshold_info: dict[str, Any] | None,
) -> int | None:
    height = first_prediction.get("new_height") or first_prediction.get("image_height")
    width = first_prediction.get("new_width") or first_prediction.get("image_width")
    if height is not None and width is not None:
        return int(height) * int(width)

    if threshold_info is not None:
        image_height = threshold_info.get("image_height")
        image_width = threshold_info.get("image_width")
        if image_height is not None and image_width is not None:
            return int(image_height) * int(image_width)

    anomalous_pixels = first_prediction.get("anomalous_pixel_count")
    anomaly_ratio = first_prediction.get("anomaly_ratio")
    if anomalous_pixels is not None and anomaly_ratio:
        ratio = float(anomaly_ratio)
        if ratio > 0:
            return int(round(int(anomalous_pixels) / ratio))

    return None


def _render_processing_indicator(status: dict[str, Any], current_state: str) -> None:
    started_at = status.get("started_at")
    status_label = "Pending" if current_state in {"pending", "queued"} else "Processing"
    st.write(f"Status: `{status_label}`")
    if started_at:
        st.caption(f"Started at {started_at}")


def _submit_request(
    uploaded_file,
    threshold_mode: str,
    custom_threshold: float | None = None,
) -> str:
    request_id = _request_id()
    request_dir = _request_dir(request_id)
    raw_dir = request_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    filename = uploaded_file.name or "uploaded_image.png"
    raw_path = raw_dir / filename
    raw_path.write_bytes(uploaded_file.getvalue())

    metadata = {
        "request_id": request_id,
        "created_at": _utc_now_iso(),
        "source": "streamlit_app_new",
        "filename": filename,
        "write_visual_artifacts": True,
        "partitions": 1,
        "partition_by": [],
        "enabled_outputs": [
            "classification", 
            "anomaly_autoencoder", 
            # "gradcam"
        ],
        "threshold_mode": threshold_mode,
    }
    if threshold_mode == "custom" and custom_threshold is not None:
        metadata["custom_threshold"] = float(custom_threshold)

    _write_json(
        request_dir / "metadata.json",
        metadata,
    )
    _write_json(
        request_dir / "status.json",
        {
            "request_id": request_id,
            "status": "pending",
            "updated_at": _utc_now_iso(),
        },
    )

    return request_id


def _clear_active_request() -> None:
    st.session_state.pop("request_id", None)


def _show_request_status(
    request_id: str,
    *,
    debug_mode: bool,
    refresh_interval_seconds: float,
    threshold_info: dict[str, Any] | None,
) -> None:
    request_dir = _find_request_dir(request_id)
    if request_dir is None:
        st.warning("Request not found.")
        return

    status = _read_json(request_dir / "status.json")
    result = _read_json(request_dir / "result.json")
    error = _read_json(request_dir / "error.json")
    first_prediction = _read_predictions_preview(request_dir)
    raw_image_path = _request_raw_image(request_dir)

    current_state = status.get("status", "unknown")

    if current_state in {
        "pending",
        "queued",
        "processing",
        "resolving_input",
        "building_bronze",
        "building_silver",
        "running_inference",
        "writing_results",
    }:
        _render_processing_indicator(status, current_state)
        time.sleep(refresh_interval_seconds)
        st.rerun()
        return

    st.write(f"Status: `{current_state}`")

    if error:
        st.subheader("Error")
        st.error(error.get("error_message", "An error occurred."))
        if debug_mode:
            st.json(status)
            st.json(error)
        return

    if raw_image_path is not None:
        st.image(Image.open(raw_image_path), caption="Uploaded MRI", width=300)

    if result and debug_mode:
        st.subheader("Technical Result")
        st.json(result)

    if first_prediction is None:
        if debug_mode:
            st.json(status)
        return

    top5_predictions = _parse_top5(first_prediction)

    st.subheader("Tumor Classification")
    pred_label = first_prediction.get("classifier_pred_label")
    confidence = first_prediction.get("classifier_confidence")

    class_metrics_col1, class_metrics_col2 = st.columns(2)
    with class_metrics_col1:
        st.metric("Predicted Class", pred_label)
    with class_metrics_col2:
        if confidence is not None:
            st.metric("Confidence", f"{float(confidence):.2%}")

    st.markdown("**Top 5 Predictions:**")
    for rank, prediction in enumerate(top5_predictions, 1):
        name = prediction["label_name"]
        prob = float(prediction["probability"])
        bar_width = int(prob * 100)
        st.markdown(
            f"{rank}. **{name}** — {prob:.2%} "
            f"`{'█' * max(1, bar_width // 5)}{'░' * (20 - max(1, bar_width // 5))}`"
        )

    gradcam_path = _artifact_path(
        request_dir,
        first_prediction,
        "gradcam_path",
    )
    gradcam_overlay_path = _artifact_path(
        request_dir,
        first_prediction,
        "gradcam_overlay_path",
    )
    if gradcam_path is not None or gradcam_overlay_path is not None:
        st.subheader("Grad-CAM")
        gradcam_col1, gradcam_col2 = st.columns(2)

        with gradcam_col1:
            if gradcam_path is not None:
                st.image(
                    Image.open(gradcam_path),
                    caption="Grad-CAM",
                    use_container_width=True,
                )

        with gradcam_col2:
            if gradcam_overlay_path is not None:
                st.image(
                    Image.open(gradcam_overlay_path),
                    caption="Grad-CAM Overlay",
                    use_container_width=True,
                )

    st.subheader("Anomaly Detection")
    is_anomalous = bool(first_prediction.get("anomaly_is_detected"))
    anomalous_pixels = first_prediction.get("anomalous_pixel_count")
    anomaly_ratio = first_prediction.get("anomaly_ratio")
    anomaly_max_error = first_prediction.get("anomaly_max_error")
    anomaly_threshold = first_prediction.get("anomaly_threshold")

    if is_anomalous:
        st.error(
            f"Anomaly detected. {anomalous_pixels:,} pixels "
            f"({float(anomaly_ratio):.2%}) exceed threshold."
        )
    else:
        st.success("No anomalous pixels detected (appears normal).")

    anomaly_metrics_col1, anomaly_metrics_col2, anomaly_metrics_col3 = st.columns(3)
    total_pixels = _resolve_total_pixels(first_prediction, threshold_info)
    with anomaly_metrics_col1:
        if anomaly_max_error is not None:
            st.metric("Max Pixel Error", f"{float(anomaly_max_error):.6f}")
    with anomaly_metrics_col2:
        if anomaly_threshold is not None:
            st.metric("Active Threshold", f"{float(anomaly_threshold):.6f}")
    with anomaly_metrics_col3:
        if anomalous_pixels is not None and total_pixels is not None:
            st.metric(
                "Anomalous Pixels",
                f"{int(anomalous_pixels):,} / {int(total_pixels):,}",
                delta=f"{float(anomaly_ratio):.2%}" if anomaly_ratio is not None else None,
            )
        elif anomalous_pixels is not None:
            st.metric(
                "Anomalous Pixels",
                f"{int(anomalous_pixels):,}",
                delta=f"{float(anomaly_ratio):.2%}" if anomaly_ratio is not None else None,
            )

    anomaly_overlay_path = _artifact_path(
        request_dir,
        first_prediction,
        "anomaly_overlay_path",
    )
    error_map_path = _artifact_path(
        request_dir,
        first_prediction,
        "error_map_path",
    )
    reconstruction_path = _artifact_path(
        request_dir,
        first_prediction,
        "reconstruction_path",
    )

    image_col1, image_col2, image_col3 = st.columns(3)

    with image_col1:
        if anomaly_overlay_path is not None:
            st.image(
                Image.open(anomaly_overlay_path),
                caption="Anomaly Overlay",
                use_container_width=True,
            )

    with image_col2:
        if error_map_path is not None:
            st.image(
                Image.open(error_map_path),
                caption="Reconstruction Error Map",
                use_container_width=True,
            )

    with image_col3:
        if reconstruction_path is not None:
            st.image(
                Image.open(reconstruction_path),
                caption="Autoencoder Reconstruction",
                use_container_width=True,
            )

    if debug_mode:
        st.subheader("Debug")
        st.json(status)
        if result:
            st.json(result)


def main() -> None:
    st.set_page_config(
        page_title="Brain Tumor MRI Analysis",
        page_icon="🧠",
        layout="wide",
    )
    st.title("🧠 Brain Tumor MRI Analysis")
    st.write(
        "This version uses the inference worker for prediction while preserving "
        "the same display style as the original app."
    )

    st.sidebar.header("Model Status")
    if CLASSIFIER_MODEL_PATH.exists():
        st.sidebar.success("✅ Classifier available")
    else:
        st.sidebar.error(f"❌ Classifier not found at `{CLASSIFIER_MODEL_PATH}`")

    threshold_info = _load_threshold_info()
    if AUTOENCODER_MODEL_PATH.exists() and threshold_info is not None:
        st.sidebar.success("✅ Anomaly autoencoder available")
        st.sidebar.info(
            f"Strict threshold: {threshold_info['threshold_strict']:.6f}\n\n"
            f"Normal images used for calibration: "
            f"{threshold_info['normal_images_evaluated']}"
        )
    else:
        st.sidebar.warning("⚠️ Anomaly model or threshold not found")

    threshold_mode = st.sidebar.selectbox(
        "Anomaly Threshold",
        ["strict", "p99", "p95", "custom"],
        help=(
            "**strict**: guarantees 0% false positives on normal scans. "
            "**p99/p95**: slightly more sensitive. "
            "**custom**: set your own value."
        ),
        key="threshold_mode_selector",
        on_change=_clear_active_request,
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
            key="custom_threshold_slider",
            on_change=_clear_active_request,
        )
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

    uploaded_file = st.file_uploader(
        "Upload an MRI image",
        type=["png", "jpg", "jpeg"],
        key="uploaded_mri",
        on_change=_clear_active_request,
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", width=320)

    if st.button("Submit to worker", disabled=uploaded_file is None):
        request_id = _submit_request(
            uploaded_file,
            threshold_mode,
            custom_threshold=custom_threshold,
        )
        st.session_state["request_id"] = request_id
        st.rerun()

    request_id = st.session_state.get("request_id")
    if request_id:
        st.subheader("Request Tracking")
        st.code(request_id)
        _show_request_status(
            request_id,
            debug_mode=debug_mode,
            refresh_interval_seconds=0.8,
            threshold_info=threshold_info,
        )


if __name__ == "__main__":
    main()
