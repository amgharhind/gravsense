"""
Grounded SAM detector: text prompt → bounding boxes (GroundingDINO) → masks (SAM).

This replaces the fragile "dominant class" heuristic from the SegFormer approach.
Instead of segmenting everything and guessing which class is debris, we explicitly
tell the model what we're looking for via a text query.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.ops as tv_ops
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL_ID = "facebook/sam-vit-base"

_VEGETATION_TERMS = {
    "tree", "plant", "vegetation", "grass", "bush", "shrub",
    "flower", "leaf", "leaves", "garden", "hedge", "weed", "foliage",
}

# Comma-separated phrases improve GroundingDINO recall across different scenes.
DEFAULT_TEXT_QUERY = (
    "construction debris. rubble. gravel. building waste. "
    "gravat. demolition waste. rubble pile. sand pile. "
    "broken concrete. concrete rubble. crushed stone. aggregate. "
    "debris in truck. rubble in truck bed. construction waste. "
    "excavation debris. mixed debris. rock debris."
)


class GroundedSAMDetector:
    """
    Lazy-loading detector: models are downloaded and cached on first call,
    not at import time. Safe to instantiate in the FastAPI startup.
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._gdino_processor: AutoProcessor | None = None
        self._gdino_model: AutoModelForZeroShotObjectDetection | None = None
        self._sam_processor: SamProcessor | None = None
        self._sam_model: SamModel | None = None

    def _load_models(self) -> None:
        if self._gdino_model is None:
            self._gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
            self._gdino_model = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(GDINO_MODEL_ID)
                .to(self.device)
            )
            self._gdino_model.eval()

        if self._sam_model is None:
            self._sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
            self._sam_model = (
                SamModel
                .from_pretrained(SAM_MODEL_ID)
                .to(self.device)
            )
            self._sam_model.eval()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect_and_segment(
        self,
        image: Image.Image,
        text_query: str = DEFAULT_TEXT_QUERY,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        nms_iou_threshold: float = 0.50,
    ) -> dict:
        """
        Run the full pipeline on a PIL image.

        Returns
        -------
        dict with keys:
            mask          np.ndarray uint8 (H, W) — union of all debris masks
            boxes         list of [x0,y0,x1,y1] in pixel coords
            scores        list of float confidence scores
            labels        list of str matched phrases
            n_detections  int
        """
        self._load_models()

        # ── Step 1: GroundingDINO → bounding boxes ──────────────────────
        gdino_inputs = self._gdino_processor(
            images=image,
            text=text_query,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            gdino_outputs = self._gdino_model(**gdino_inputs)

        detections = self._gdino_processor.post_process_grounded_object_detection(
            gdino_outputs,
            gdino_inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (H, W)
        )[0]

        boxes = detections["boxes"]   # Tensor (N, 4)
        scores = detections["scores"] # Tensor (N,)
        labels = detections["labels"] # list[str]

        # Drop vegetation false positives
        keep = [
            i for i, lbl in enumerate(labels)
            if not any(v in lbl.lower() for v in _VEGETATION_TERMS)
        ]
        if len(keep) < len(labels):
            boxes  = boxes[keep]
            scores = scores[keep]
            labels = [labels[i] for i in keep]

        # NMS — merge overlapping boxes from different text phrases
        # (GroundingDINO fires once per phrase; the same pile gets 5-10 boxes)
        if len(boxes) > 1:
            keep_nms = tv_ops.nms(boxes.float(), scores, iou_threshold=nms_iou_threshold)
            boxes  = boxes[keep_nms]
            scores = scores[keep_nms]
            labels = [labels[i] for i in keep_nms.tolist()]

        if len(boxes) == 0:
            return {
                "mask": np.zeros((image.size[1], image.size[0]), dtype=np.uint8),
                "boxes": [],
                "scores": [],
                "labels": [],
                "n_detections": 0,
            }

        # ── Step 2: SAM → pixel-precise masks from boxes ────────────────
        sam_inputs = self._sam_processor(
            image,
            input_boxes=[boxes.tolist()],  # expects list[list[list[float]]]
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            sam_outputs = self._sam_model(**sam_inputs)

        # post_process_masks returns list[Tensor(N, 3, H, W)]
        # 3 candidates per box — we take index 0 (highest IoU score)
        masks_per_image = self._sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )
        masks = masks_per_image[0]  # (N, 3, H, W)
        best_masks = masks[:, 0]    # (N, H, W) — best candidate per detection

        # Union all debris masks into a single binary mask
        union_mask = best_masks.any(dim=0).numpy().astype(np.uint8)

        return {
            "mask": union_mask,
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "labels": labels,
            "n_detections": len(boxes),
        }
