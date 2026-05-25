"""
Depth Anything V2 — automatic pile height estimation + depth visualization.

Strategy: the debris pile sits CLOSER to the camera than the surrounding
ground. Height = median(ground_depth_around_pile) − median(pile_depth).
Both values come from the metric depth map (in metres).
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline as hf_pipeline

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
_GROUND_BORDER_PX = 40


# ---------------------------------------------------------------------------
# Colormap (plasma-like, no matplotlib needed)
# ---------------------------------------------------------------------------

def _colorize_depth(depth_norm: np.ndarray) -> np.ndarray:
    """
    Map a normalized [0, 1] depth array to an RGB image.
    0 = close (bright yellow) → 1 = far (dark purple).
    Uses a plasma-inspired palette interpolated with numpy.
    """
    ctrl_pos = np.array([0.0,  0.25,  0.5,   0.75,  1.0])
    ctrl_r   = np.array([253,  94,    33,     59,    68], dtype=np.float32)
    ctrl_g   = np.array([231,  201,   145,    82,    1],  dtype=np.float32)
    ctrl_b   = np.array([37,   98,    140,    139,   84], dtype=np.float32)

    flat = depth_norm.flatten()
    r = np.interp(flat, ctrl_pos, ctrl_r)
    g = np.interp(flat, ctrl_pos, ctrl_g)
    b = np.interp(flat, ctrl_pos, ctrl_b)

    rgb = np.stack([r, g, b], axis=-1).reshape(*depth_norm.shape, 3).astype(np.uint8)
    return rgb


def _to_b64_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class DepthEstimator:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is None:
            self._pipe = hf_pipeline(
                "depth-estimation",
                model=DEPTH_MODEL,
                device=0 if self.device == "cuda" else -1,
            )

    def estimate(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> dict:
        """
        Returns a dict with:
            height_cm       float | None   estimated pile height
            depth_map_b64   str            full-image plasma-coloured depth PNG
            depth_on_debris_b64 str        depth map masked to debris region only
        """
        if int(mask.sum()) == 0:
            return {"height_cm": None, "depth_map_b64": None, "depth_on_debris_b64": None}

        self._load()

        result = self._pipe(image)
        depth_tensor = result["predicted_depth"]

        # Tensor → numpy (H, W) in metres
        depth_np = (
            depth_tensor.numpy()
            if hasattr(depth_tensor, "numpy")
            else np.array(depth_tensor)
        )
        if depth_np.ndim == 3:
            depth_np = depth_np[0]

        # Resize to match image / mask dimensions
        h, w = mask.shape
        if depth_np.shape != (h, w):
            depth_np = np.array(
                Image.fromarray(depth_np.astype(np.float32)).resize(
                    (w, h), Image.BILINEAR
                )
            )

        # ── Height estimation ────────────────────────────────────────────
        mask_u8 = mask.astype(np.uint8)
        kernel  = np.ones((_GROUND_BORDER_PX, _GROUND_BORDER_PX), np.uint8)
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        ground_ring = dilated.astype(bool) & ~mask_u8.astype(bool)

        pile_depths   = depth_np[mask_u8 > 0]
        ground_depths = depth_np[ground_ring] if ground_ring.sum() > 50 else None

        if ground_depths is not None and len(ground_depths) > 0:
            height_m = abs(float(np.median(ground_depths)) - float(np.median(pile_depths)))
        else:
            height_m = float(
                np.percentile(pile_depths, 95) - np.percentile(pile_depths, 5)
            )

        height_cm = max(5.0, round(height_m * 100.0, 1))

        # ── Depth visualisation ──────────────────────────────────────────
        d_min, d_max = depth_np.min(), depth_np.max()
        depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-8)

        # Full-image coloured depth
        depth_rgb = _colorize_depth(depth_norm)
        depth_map_b64 = _to_b64_png(depth_rgb)

        # Depth overlaid on debris region only (greyed out outside mask)
        grey = (depth_norm * 60 + 30).astype(np.uint8)          # dim grey outside
        grey_rgb = np.stack([grey, grey, grey], axis=-1)
        debris_depth_rgb = grey_rgb.copy()
        debris_depth_rgb[mask_u8 > 0] = depth_rgb[mask_u8 > 0]  # colour inside mask
        depth_on_debris_b64 = _to_b64_png(debris_depth_rgb)

        return {
            "height_cm":           height_cm,
            "depth_map_b64":       depth_map_b64,
            "depth_on_debris_b64": depth_on_debris_b64,
        }
