"""
Auto reference-width calibration using GroundingDINO.

Detects a vehicle (truck, car, van…) in the image, measures its bounding-box
pixel width, then computes the real-world width that corresponds to the full
image width:

    reference_width_cm = known_object_cm × (image_width_px / bbox_width_px)

GroundingDINO is already a dependency of the grounded_sam worker, so no new
model download is needed.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# Real-world widths (cm) for common reference objects on construction sites
_KNOWN_WIDTHS_CM: dict[str, float] = {
    "truck":   240.0,
    "lorry":   240.0,
    "camion":  240.0,
    "van":     200.0,
    "car":     180.0,
    "vehicle": 220.0,
    "person":   50.0,
    "door":     90.0,
    "gate":    120.0,
}

_REFERENCE_QUERY = (
    "truck. lorry. van. car. vehicle. person. door."
)

_BOX_THRESHOLD  = 0.30
_TEXT_THRESHOLD = 0.25


class AutoCalibrator:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor: AutoProcessor | None = None
        self._model: AutoModelForZeroShotObjectDetection | None = None

    def _load(self) -> None:
        if self._model is None:
            self._processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
            self._model = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(GDINO_MODEL_ID)
                .to(self.device)
            )
            self._model.eval()

    # ------------------------------------------------------------------

    def estimate_reference_width_cm(
        self,
        image: Image.Image,
    ) -> dict | None:
        """
        Returns a dict with keys:
            reference_width_cm  float   full-image real-world width
            detected_object     str     what was used for calibration
            known_width_cm      float   the known size of that object
            confidence          float   detection confidence score
        or None if no known reference object is found.
        """
        self._load()

        img_w = image.size[0]

        gdino_inputs = self._processor(
            images=image,
            text=_REFERENCE_QUERY,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            gdino_outputs = self._model(**gdino_inputs)

        results = self._processor.post_process_grounded_object_detection(
            gdino_outputs,
            gdino_inputs.input_ids,
            box_threshold=_BOX_THRESHOLD,
            text_threshold=_TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes  = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        if len(boxes) == 0:
            return None

        # Prefer the detection with highest confidence
        best_idx   = int(torch.argmax(scores))
        best_box   = boxes[best_idx].tolist()
        best_label = labels[best_idx].lower().strip()
        best_score = float(scores[best_idx])

        # Match to a known object width
        known_width: float | None = None
        matched_key = "vehicle"
        for key, w in _KNOWN_WIDTHS_CM.items():
            if key in best_label:
                known_width  = w
                matched_key  = key
                break
        if known_width is None:
            known_width = _KNOWN_WIDTHS_CM["vehicle"]

        bbox_width_px = best_box[2] - best_box[0]
        if bbox_width_px <= 0:
            return None

        ref_width = known_width * (img_w / bbox_width_px)

        return {
            "reference_width_cm": round(ref_width, 1),
            "detected_object":    matched_key,
            "known_width_cm":     known_width,
            "confidence":         round(best_score, 3),
        }
