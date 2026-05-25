"""
API tests — all model inference is mocked so CI runs without a GPU or
model downloads. Tests verify routing, response schema, and error handling.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jpeg_bytes(w: int = 120, h: int = 80) -> bytes:
    img = Image.fromarray(
        np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _fake_detection_result(h: int = 80, w: int = 120) -> dict:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:60, 30:90] = 1  # simulate a debris region
    return {
        "mask": mask,
        "boxes": [[30, 20, 90, 60]],
        "scores": [0.87],
        "labels": ["pile of construction debris"],
        "n_detections": 1,
    }


# ---------------------------------------------------------------------------
# Fixtures — patch heavy model loading before app import
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with (
        patch("gravsense.core.grounded_sam.GroundedSAMDetector._load_models"),
        patch("gravsense.core.segformer_detector.SegformerDetector._load_model"),
    ):
        from gravsense.api.main import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health_ok(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_serves_html(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# /analyze — grounded_sam
# ---------------------------------------------------------------------------

def test_analyze_grounded_sam_returns_schema(client: TestClient):
    with patch("gravsense.api.main._grounded_sam") as mock_factory:
        detector = MagicMock()
        detector.detect_and_segment.return_value = _fake_detection_result()
        mock_factory.return_value = detector

        resp = client.post(
            "/analyze?method=grounded_sam&reference_width_cm=200",
            files={"file": ("site.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["method"] == "grounded_sam"
    assert data["n_detections"] == 1
    assert data["mask_pixel_count"] > 0
    assert data["surface_area_cm2"] > 0
    assert data["volume_cm3"] > 0
    assert isinstance(data["processing_time_s"], float)
    # overlay is included by default
    assert data["overlay_b64"] is not None
    assert len(data["overlay_b64"]) > 0


def test_analyze_no_overlay_when_disabled(client: TestClient):
    with patch("gravsense.api.main._grounded_sam") as mock_factory:
        detector = MagicMock()
        detector.detect_and_segment.return_value = _fake_detection_result()
        mock_factory.return_value = detector

        resp = client.post(
            "/analyze?include_overlay=false",
            files={"file": ("site.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert resp.status_code == 200
    assert resp.json()["overlay_b64"] is None


# ---------------------------------------------------------------------------
# /analyze — segformer
# ---------------------------------------------------------------------------

def test_analyze_segformer_returns_schema(client: TestClient):
    with patch("gravsense.api.main._segformer") as mock_factory:
        detector = MagicMock()
        detector.detect_and_segment.return_value = {
            **_fake_detection_result(),
            "labels": ["ade20k_class_14"],
            "n_detections": 1,
        }
        mock_factory.return_value = detector

        resp = client.post(
            "/analyze?method=segformer",
            files={"file": ("site.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert resp.status_code == 200
    assert resp.json()["method"] == "segformer"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_analyze_rejects_non_image(client: TestClient):
    resp = client.post(
        "/analyze",
        files={"file": ("report.pdf", b"%PDF-1.4 content", "application/pdf")},
    )
    assert resp.status_code == 400


def test_analyze_rejects_corrupt_image(client: TestClient):
    resp = client.post(
        "/analyze",
        files={"file": ("bad.jpg", b"not_an_image", "image/jpeg")},
    )
    assert resp.status_code == 422


def test_analyze_invalid_method(client: TestClient):
    resp = client.post(
        "/analyze?method=unknown_model",
        files={"file": ("site.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 422


def test_analyze_zero_reference_width_rejected(client: TestClient):
    resp = client.post(
        "/analyze?reference_width_cm=0",
        files={"file": ("site.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Volume math (unit test — no API, no mocks)
# ---------------------------------------------------------------------------

def test_volume_calculation():
    from gravsense.core.volume import estimate_volume, mask_to_surface_area

    mask = np.ones((100, 200), dtype=np.uint8)  # all pixels are debris
    area = mask_to_surface_area(mask, reference_width_cm=200.0)
    assert abs(area - 100.0 * 200.0) < 1.0  # 20 000 cm² (200cm × 100cm)

    vol = estimate_volume(area, assumed_height_cm=50.0)
    assert vol == pytest.approx(area * 50.0)
