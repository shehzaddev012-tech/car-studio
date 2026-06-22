"""
AI Car Studio Background Pipeline — dealership-grade compositing.

Pipeline order:
  1. segment_vehicle     Vertex AI → SAM2 → merge → edge refine
  2. (pixels locked)       Original vehicle RGB never modified
  3. generate_background   Premium cyclorama studio backdrop
  4. generate_shadow       Footprint + contact + progressive blur
  5. composite_layers      background → shadow → vehicle
  6. glass_cleanup         Subtle windscreen brightening (studio reflection sim)
  7. validate_output       SSIM + LPIPS + crop + missing-part gates
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from app.services.quality import QualityReport, validate_output
from app.services.segmentation import segment_vehicle
from app.services.shadow import generate_shadow
from app.services.studio_background import generate_studio_background
from app.services.vehicle_preservation import (
    REJECTION_MESSAGE,
    VehiclePreservationReport,
    validate_vehicle_preservation,
)

# Re-export for backward compatibility
from app.services.segmentation.mask_ops import mask_coverage  # noqa: F401

logger = logging.getLogger(__name__)


def composite_layers(background: Image.Image, shadow: Image.Image, car_rgba: Image.Image) -> Image.Image:
    """Stack layers: background → shadow → original vehicle pixels."""
    result = background.convert("RGBA")
    result = Image.alpha_composite(result, shadow.convert("RGBA"))
    result = Image.alpha_composite(result, car_rgba)
    return result.convert("RGB")


def glass_cleanup_stub(
    image: Image.Image,
    car_rgba: Image.Image,
    mask: Image.Image | None = None,
) -> Image.Image:
    """
    Windscreen / glass cleanup for automotive studio composites.

    Detects dark interior-visible areas within the upper-centre of the car
    silhouette (typical windscreen zone) and blends them with a light studio
    grey, simulating the clean sky/studio-light reflection seen in professional
    dealership photography.

    The effect is deliberately subtle (~30 % blend) so paint, wheels, and
    other dark car details are not affected.
    """
    if mask is None:
        return image

    try:
        composite_arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        car_arr = np.array(car_rgba.convert("RGBA"), dtype=np.float32) / 255.0
        mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0

        h, w = mask_arr.shape

        in_car_mask = mask_arr > 0.5
        if not in_car_mask.any():
            return image

        ys, xs = np.where(in_car_mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        bh = max(y1 - y0, 1)
        bw = max(x1 - x0, 1)

        # Luminance of original car pixels (before compositing)
        car_rgb = car_arr[:, :, :3]
        luminance = (
            0.299 * car_rgb[:, :, 0]
            + 0.587 * car_rgb[:, :, 1]
            + 0.114 * car_rgb[:, :, 2]
        )

        # Windscreen zone: upper 65 % of car height, central 80 % of width
        # Excludes wheels/mirrors on the outer edges and the bonnet below
        wind_zone = np.zeros((h, w), dtype=bool)
        wy1 = int(y0 + bh * 0.65)
        wx0 = int(x0 + bw * 0.10)
        wx1 = int(x0 + bw * 0.90)
        wind_zone[y0:wy1, wx0:wx1] = True

        # Glass candidates: very dark (< 22 % luma) within car mask & windscreen zone
        glass_px = (luminance < 0.22) & in_car_mask & wind_zone

        if not glass_px.any():
            return image

        # Studio-grey target (matches the light grey studio background)
        studio_grey = np.array([0.91, 0.91, 0.92], dtype=np.float32)

        blend = 0.30  # subtle — keep car detail visible
        result = composite_arr.copy()
        result[glass_px] = (
            composite_arr[glass_px] * (1.0 - blend) + studio_grey * blend
        )

        glass_pct = glass_px.sum() / max(in_car_mask.sum(), 1) * 100
        logger.info("Glass cleanup applied: %.1f%% of car pixels brightened", glass_pct)

        return Image.fromarray(
            (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8), "RGB"
        )

    except Exception as exc:
        logger.warning("Glass cleanup skipped (error: %s)", exc)
        return image


__all__ = [
    "composite_layers",
    "generate_shadow",
    "generate_studio_background",
    "glass_cleanup_stub",
    "segment_vehicle",
    "validate_output",
    "validate_vehicle_preservation",
    "QualityReport",
    "VehiclePreservationReport",
    "REJECTION_MESSAGE",
]
