from __future__ import annotations

import numpy as np


def mask_to_surface_area(mask: np.ndarray, reference_width_cm: float = 100.0) -> float:
    """
    Convert a binary mask to real-world surface area.

    Assumes the image width corresponds to `reference_width_cm` in the real world.
    This is the only calibration parameter needed — measure one known dimension
    in the photo (e.g. a truck bed that is 200 cm wide) and pass it here.
    """
    image_width_px = mask.shape[1]
    pixel_scale_cm = reference_width_cm / image_width_px
    pixel_area_cm2 = pixel_scale_cm ** 2
    white_pixels = int(np.sum(mask > 0))
    return white_pixels * pixel_area_cm2


def estimate_volume(surface_area_cm2: float, assumed_height_cm: float = 30.0) -> float:
    """
    Rough volume from surface area × assumed uniform pile height.

    Replace `assumed_height_cm` with a Depth Anything v2 estimate for
    a more accurate result (see the monocular depth upgrade path).
    """
    return surface_area_cm2 * assumed_height_cm
