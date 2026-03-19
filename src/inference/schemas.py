from __future__ import annotations

from pyspark.sql import types as T


def prediction_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("request_id", T.StringType(), nullable=True),
            T.StructField("image_id", T.StringType(), nullable=False),
            T.StructField("raw_path", T.StringType(), nullable=False),
            T.StructField("pathology", T.StringType(), nullable=True),
            T.StructField("label_idx", T.IntegerType(), nullable=True),
            T.StructField("transform_version", T.StringType(), nullable=True),
            T.StructField("classifier_pred_idx", T.IntegerType(), nullable=True),
            T.StructField("classifier_pred_label", T.StringType(), nullable=True),
            T.StructField("classifier_confidence", T.DoubleType(), nullable=True),
            T.StructField("classifier_topk_json", T.StringType(), nullable=True),
            T.StructField("anomaly_max_error", T.DoubleType(), nullable=True),
            T.StructField("anomaly_threshold", T.DoubleType(), nullable=True),
            T.StructField("anomaly_is_detected", T.BooleanType(), nullable=True),
            T.StructField("anomalous_pixel_count", T.LongType(), nullable=True),
            T.StructField("anomaly_ratio", T.DoubleType(), nullable=True),
            T.StructField("reconstruction_path", T.StringType(), nullable=True),
            T.StructField("error_map_path", T.StringType(), nullable=True),
            T.StructField("anomaly_overlay_path", T.StringType(), nullable=True),
            T.StructField("gradcam_path", T.StringType(), nullable=True),
            T.StructField("gradcam_overlay_path", T.StringType(), nullable=True),
        ]
    )
