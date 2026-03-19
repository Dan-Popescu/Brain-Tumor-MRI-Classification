from __future__ import annotations 

import io 
import json 
from typing import Any

import numpy as np
from PIL import Image

LABEL_NAMES = [
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

def decode_processed_bytes(image_bytes: bytes, image_height: int, image_width: int) -> np.ndarray: 
    
    with Image.open(io.BytesIO(image_bytes)) as image:
        gray = image.convert("L")
        arr = np.array(gray, dtype=np.float32) / 255.0

    if arr.shape != (image_height, image_width):
        raise ValueError(
            f"Processed image shape mismatch. Expected {(image_height, image_width)}, got {arr.shape}"
        )

    return arr.reshape(1, image_height, image_width, 1)



def predict_classifier(
        model: Any,
        image_array: np.ndarray,
        top_k: int = 5,
) -> dict[str, Any]:
    

    preds = model.predict(image_array, verbose=0)
    probs = preds[0]
    
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    top_indices = np.argsort(probs)[::-1][:top_k]
    topk = [
        {
            "label_idx": int(idx),
            "label_name": LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else f"idx={idx}",
            "probability": float(probs[idx]),
        }
        for idx in top_indices
    ]

    return {
        "classifier_pred_idx": pred_idx,
        "classifier_pred_label": LABEL_NAMES[pred_idx] if pred_idx < len(LABEL_NAMES) else f"idx={pred_idx}",
        "classifier_confidence": confidence,
        "classifier_topk_json": json.dumps(topk),
    }


