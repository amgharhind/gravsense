"""
GravSense FastAPI application.

Endpoints:
    GET  /               — frontend UI
    GET  /health         — liveness probe
    POST /analyze        — detection + depth + auto-calibration + volume
    GET  /docs           — Swagger UI
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from gravsense.api.schemas import AnalysisResult, DetectionMethod, HealthResponse
from gravsense.core.auto_calibrate import AutoCalibrator
from gravsense.core.depth_estimator import DepthEstimator
from gravsense.core.grounded_sam import GroundedSAMDetector
from gravsense.core.segformer_detector import SegformerDetector
from gravsense.core.volume import estimate_volume, mask_to_surface_area

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="GravSense",
    description="Construction debris detection and volume estimation.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _grounded_sam() -> GroundedSAMDetector:
    return GroundedSAMDetector()

@lru_cache(maxsize=1)
def _segformer() -> SegformerDetector:
    return SegformerDetector()

@lru_cache(maxsize=1)
def _depth_estimator() -> DepthEstimator:
    return DepthEstimator()

@lru_cache(maxsize=1)
def _auto_calibrator() -> AutoCalibrator:
    return AutoCalibrator()

def _get_detector(method: DetectionMethod):
    return _grounded_sam() if method == DetectionMethod.grounded_sam else _segformer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_overlay_b64(image: Image.Image, mask: np.ndarray) -> str:
    img_arr = np.array(image.convert("RGB"))
    green   = np.zeros_like(img_arr)
    green[mask > 0] = [34, 197, 94]
    blended = (0.55 * img_arr + 0.45 * green).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def frontend():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/analyze", response_model=AnalysisResult, tags=["inference"])
async def analyze(
    file: UploadFile = File(..., description="Construction site image (JPEG/PNG)"),
    method: DetectionMethod = Query(DetectionMethod.grounded_sam),
    reference_width_cm: float = Query(100.0, gt=0),
    assumed_height_cm:  float = Query(30.0,  gt=0),
    auto_calibrate: bool = Query(
        True,
        description=(
            "Run Depth Anything V2 (pile height) + GroundingDINO (reference width) "
            "in parallel. Auto values override the manual defaults when found."
        ),
    ),
    include_overlay: bool = Query(True),
) -> AnalysisResult:

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()

    # ── Step 1: debris detection ──────────────────────────────────────────
    detector  = _get_detector(method)
    detection = await loop.run_in_executor(None, detector.detect_and_segment, image)
    mask: np.ndarray = detection["mask"]

    # ── Step 2: depth estimation + reference-width calibration (parallel) ─
    depth_result: dict       = {}
    ref_result:   dict | None = None

    if auto_calibrate:
        depth_fn = lambda: _depth_estimator().estimate(image, mask)
        ref_fn   = lambda: _auto_calibrator().estimate_reference_width_cm(image)

        depth_result, ref_result = await asyncio.gather(
            loop.run_in_executor(None, depth_fn),
            loop.run_in_executor(None, ref_fn),
        )

    elapsed = time.perf_counter() - t0

    # ── Step 3: resolve effective calibration values ──────────────────────
    auto_height    = depth_result.get("height_cm")
    eff_height     = auto_height if auto_height is not None else assumed_height_cm
    eff_ref_width  = (
        ref_result["reference_width_cm"]
        if ref_result is not None
        else reference_width_cm
    )

    surface_area = mask_to_surface_area(mask, eff_ref_width)
    volume       = estimate_volume(surface_area, eff_height)

    # ── Step 4: build images ──────────────────────────────────────────────
    overlay_b64         = _build_overlay_b64(image, mask) if include_overlay else None
    depth_map_b64       = depth_result.get("depth_map_b64")       if auto_calibrate else None
    depth_on_debris_b64 = depth_result.get("depth_on_debris_b64") if auto_calibrate else None

    return AnalysisResult(
        method=method,
        n_detections=detection["n_detections"],
        mask_pixel_count=int(np.sum(mask > 0)),
        surface_area_cm2=round(surface_area, 2),
        volume_cm3=round(volume, 2),
        reference_width_cm=round(eff_ref_width, 1),
        assumed_height_cm=round(eff_height, 1),
        detected_labels=detection.get("labels", []),
        detection_scores=[round(s, 3) for s in detection.get("scores", [])],
        processing_time_s=round(elapsed, 3),
        auto_reference_width_cm=(
            ref_result["reference_width_cm"] if ref_result else None
        ),
        auto_reference_object=(
            ref_result["detected_object"] if ref_result else None
        ),
        auto_reference_confidence=(
            ref_result["confidence"] if ref_result else None
        ),
        auto_pile_height_cm=auto_height,
        overlay_b64=overlay_b64,
        depth_map_b64=depth_map_b64,
        depth_on_debris_b64=depth_on_debris_b64,
    )
