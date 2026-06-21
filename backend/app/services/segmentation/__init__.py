"""
Strict dealership segmentation pipeline.

Vertex AI image-segmentation-001 is the ONLY segmentation provider.
SAM2 refines the Vertex mask — it never replaces Vertex as primary segmentation.

No rembg, U2-Net, GrabCut, or silent fallbacks.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from PIL import Image

from app.config import settings
from app.services.segmentation.mask_ops import (
    MaskCandidate,
    extract_car_rgba,
    mask_coverage,
    merge_and_select_mask,
    refine_mask_edges,
)
from app.services.segmentation.sam2_refine import SAM2RefinementError, refine_mask_sam2
from app.services.segmentation.vertex import VertexSegmentationError, segment_vehicle_vertex

logger = logging.getLogger(__name__)


def _validate_mask_coverage(mask: np.ndarray) -> float:
    coverage = mask_coverage(mask)
    if coverage < settings.mask_min_coverage:
        raise VertexSegmentationError(
            f"Mask coverage {coverage:.1%} is below minimum {settings.mask_min_coverage:.1%}. "
            "No vehicle detected reliably in this image."
        )
    if coverage > settings.mask_max_coverage:
        raise VertexSegmentationError(
            f"Mask coverage {coverage:.1%} exceeds maximum {settings.mask_max_coverage:.1%}. "
            "Segmentation may include too much background."
        )
    return coverage


def segment_vehicle(image: Image.Image) -> Tuple[Image.Image, Image.Image, dict]:
    """
    Strict pipeline: Vertex AI → SAM2 → merge → edge refine.

    Raises VertexSegmentationError or SAM2RefinementError on any failure.
    """
    rgb = np.array(image.convert("RGB"))

    # ── Step 1: Vertex AI (sole segmentation provider) ────────────────────────
    vertex_mask, vertex_conf, vertex_ms = segment_vehicle_vertex(image)

    # ── Step 2: SAM2 refinement (required) ────────────────────────────────────
    try:
        sam2_mask, sam2_conf = refine_mask_sam2(image, vertex_mask)
    except SAM2RefinementError as exc:
        logger.error(
            "SAM2 refinement failed after Vertex success: confidence=%.3f response_time_ms=%.0f reason=%s",
            vertex_conf,
            vertex_ms,
            exc,
        )
        raise

    # ── Step 3: Merge + select highest-confidence mask ──────────────────────────
    candidates = [
        MaskCandidate(mask=vertex_mask, confidence=vertex_conf, source="vertex"),
        MaskCandidate(mask=sam2_mask, confidence=sam2_conf, source="sam2"),
    ]
    merged, selected_source, merge_score = merge_and_select_mask(
        candidates,
        min_coverage=settings.mask_min_coverage,
        max_coverage=settings.mask_max_coverage,
    )

    # ── Step 4: Edge refinement + anti-aliasing ─────────────────────────────────
    refined = refine_mask_edges(merged, rgb)
    coverage = _validate_mask_coverage(refined)

    car_rgba, mask_pil = extract_car_rgba(image, refined)
    metadata = {
        "segmentation_provider": "vertex-ai",
        "primary_source": "vertex",
        "selected_source": selected_source,
        "vertex_confidence": vertex_conf,
        "vertex_response_time_ms": vertex_ms,
        "sam2_confidence": sam2_conf,
        "merge_score": merge_score,
        "mask_coverage": coverage,
        # Mask stages for vehicle-preservation validation (not serialised to client)
        "vertex_mask": vertex_mask,
        "sam2_mask": sam2_mask,
        "merged_mask": merged,
        "final_mask": refined,
        "bootstrap_mask": vertex_mask,
    }

    logger.info(
        "segment_vehicle complete: provider=vertex-ai confidence=%.3f coverage=%.1f%% "
        "vertex_ms=%.0f sam2_conf=%.3f selected=%s",
        vertex_conf,
        coverage * 100,
        vertex_ms,
        sam2_conf,
        selected_source,
    )
    return car_rgba, mask_pil, metadata
