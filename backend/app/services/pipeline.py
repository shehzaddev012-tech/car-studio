"""
AI Car Studio Background Pipeline — dealership-grade compositing.

Pipeline order:
  1. segment_vehicle     Vertex AI → SAM2 → merge → edge refine
  2. (pixels locked)       Original vehicle RGB never modified
  3. generate_background   Premium cyclorama studio backdrop
  4. generate_shadow       Footprint + contact + progressive blur
  5. composite_layers      background → shadow → vehicle
  6. validate_output       SSIM + LPIPS + crop + missing-part gates
"""
from __future__ import annotations

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


def composite_layers(background: Image.Image, shadow: Image.Image, car_rgba: Image.Image) -> Image.Image:
    """Stack layers: background → shadow → original vehicle pixels."""
    result = background.convert("RGBA")
    result = Image.alpha_composite(result, shadow.convert("RGBA"))
    result = Image.alpha_composite(result, car_rgba)
    return result.convert("RGB")


def glass_cleanup_stub(image: Image.Image, car_rgba: Image.Image) -> Image.Image:  # noqa: ARG001
    """No-op — vehicle pixels are never inpainted or regenerated."""
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
