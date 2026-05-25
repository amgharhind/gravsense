from enum import Enum
from pydantic import BaseModel


class DetectionMethod(str, Enum):
    grounded_sam = "grounded_sam"
    segformer = "segformer"


class AnalysisResult(BaseModel):
    method: DetectionMethod
    n_detections: int
    mask_pixel_count: int
    surface_area_cm2: float
    volume_cm3: float
    # Effective values used for the calculation
    reference_width_cm: float
    assumed_height_cm: float
    detected_labels: list[str]
    detection_scores: list[float]
    processing_time_s: float
    # Auto-calibration results
    auto_reference_width_cm: float | None = None
    auto_reference_object: str | None = None
    auto_reference_confidence: float | None = None
    auto_pile_height_cm: float | None = None
    # Images (base64 PNG) — present when include_overlay=true
    overlay_b64: str | None = None
    depth_map_b64: str | None = None
    depth_on_debris_b64: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
