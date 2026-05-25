"""
SegFormer detector (legacy method, kept for benchmarking).

Uses nvidia/segformer-b0-finetuned-ade-512-512 and picks the dominant
predicted class as the debris region. This is the original notebook approach,
refactored into a class so it can be compared against GroundedSAM via the API.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.nn.functional import interpolate
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"


class SegformerDetector:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor: SegformerImageProcessor | None = None
        self._model: SegformerForSemanticSegmentation | None = None

    def _load_model(self) -> None:
        if self._model is None:
            self._processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
            self._model = (
                SegformerForSemanticSegmentation
                .from_pretrained(MODEL_ID)
                .to(self.device)
            )
            self._model.eval()

    def _dynamic_contrast(self, image: Image.Image) -> Image.Image:
        brightness = float(np.mean(np.array(image)))
        factor = 1.0 + (128.0 - brightness) / 128.0
        return ImageEnhance.Contrast(image).enhance(factor)

    def detect_and_segment(self, image: Image.Image) -> dict:
        """
        Returns the same dict schema as GroundedSAMDetector for API parity.

        Limitation: picks the *most frequent* ADE20K class as debris.
        Works when debris dominates the frame; fails otherwise.
        """
        self._load_model()

        enhanced = self._dynamic_contrast(image)
        resized = enhanced.resize((512, 512))

        inputs = self._processor(images=resized, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self._model(**inputs).logits  # (1, 150, H/4, W/4)

        upsampled = interpolate(
            logits,
            size=image.size[::-1],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()  # (H, W)

        flat = seg_map.flatten()
        unique, counts = np.unique(flat, return_counts=True)
        dominant_class = int(unique[np.argmax(counts)])

        binary_mask = (seg_map == dominant_class).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        refined = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        return {
            "mask": refined,
            "boxes": [],
            "scores": [1.0],
            "labels": [f"ade20k_class_{dominant_class}"],
            "n_detections": 1,
        }
