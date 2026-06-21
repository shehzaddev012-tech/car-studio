"""
Dealership segmentation pipeline.

Primary:  Vertex AI image-segmentation-001 → SAM2 → merge → edge refine
Fallback: rembg u2net (local, no API)      → SAM2 → edge refine

The fallback activates ONLY when Vertex AI returns a model-unavailable error
(404 / "unavailable"). Auth failures and network errors are still fatal.
SAM2 refinement, mask merging, and edge refinement run identically in both paths.
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
from app.services.segmentation.rembg_segment import RembgSegmentationError, segment_vehicle_rembg
from app.services.segmentation.sam2_refine import SAM2RefinementError, refine_mask_sam2
from app.services.segmentation.vertex import VertexSegmentationError, segment_vehicle_vertex

logger = logging.getLogger(__name__)


def _is_model_unavailable(exc: Exception) -> bool:
    """Return True for 404/unavailable and 429 quota errors — both use rembg fallback."""
    msg = str(exc)
    return (
        "404" in msg
        or "NOT_FOUND" in msg
        or "unavailable" in msg.lower()
        or "429" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "quota" in msg.lower()
    )


def _validate_mask_coverage(mask: np.ndarray, provider: str) -> float:
    coverage = mask_coverage(mask)
    if coverage < settings.mask_min_coverage:
        raise VertexSegmentationError(
            f"[{provider}] Mask coverage {coverage:.1%} is below minimum "
            f"{settings.mask_min_coverage:.1%}. No vehicle detected reliably."
        )
    if coverage > settings.mask_max_coverage:
        raise VertexSegmentationError(
            f"[{provider}] Mask coverage {coverage:.1%} exceeds maximum "
            f"{settings.mask_max_coverage:.1%}. Segmentation may include too much background."
        )
    return coverage


def segment_vehicle(image: Image.Image) -> Tuple[Image.Image, Image.Image, dict]:
    """
    Primary pipeline:  Vertex AI → SAM2 → merge → edge refine.
    Fallback pipeline: rembg      → SAM2 → edge refine.

    The fallback is used transparently when the Vertex model is unavailable.
    Raises VertexSegmentationError, RembgSegmentationError, or SAM2RefinementError on failure.
    """
    rgb = np.array(image.convert("RGB"))

    # ── Step 1: Initial segmentation ──────────────────────────────────────────
    using_fallback = False
    primary_mask: np.ndarray
    primary_conf: float
    primary_ms: float

    try:
        primary_mask, primary_conf, primary_ms = segment_vehicle_vertex(image)
        primary_source = "vertex"
    except VertexSegmentationError as exc:
        if not _is_model_unavailable(exc):
            raise  # auth / network failure — still fatal

        logger.warning(
            "Vertex AI model unavailable (%s) — switching to rembg fallback for this job",
            exc,
        )
        using_fallback = True
        try:
            primary_mask, primary_conf, primary_ms = segment_vehicle_rembg(image)
            primary_source = "rembg"
        except RembgSegmentationError as rembg_exc:
            raise RembgSegmentationError(
                f"Both Vertex AI and rembg fallback failed. "
                f"Vertex: {exc}. rembg: {rembg_exc}"
            ) from rembg_exc

    # ── Step 2: SAM2 refinement (runs on whichever mask we have) ──────────────
    try:
        sam2_mask, sam2_conf = refine_mask_sam2(image, primary_mask)
    except SAM2RefinementError as exc:
        logger.error(
            "SAM2 refinement failed after %s: confidence=%.3f ms=%.0f reason=%s",
            primary_source,
            primary_conf,
            primary_ms,
            exc,
        )
        raise

    # ── Step 3: Merge + select best mask ──────────────────────────────────────
    candidates = [
        MaskCandidate(mask=primary_mask, confidence=primary_conf, source=primary_source),
        MaskCandidate(mask=sam2_mask, confidence=sam2_conf, source="sam2"),
    ]
    merged, selected_source, merge_score = merge_and_select_mask(
        candidates,
        min_coverage=settings.mask_min_coverage,
        max_coverage=settings.mask_max_coverage,
    )

    # ── Step 4: Edge refinement + anti-aliasing ────────────────────────────────
    refined = refine_mask_edges(merged, rgb)
    provider_label = "rembg+sam2" if using_fallback else "vertex-ai"
    coverage = _validate_mask_coverage(refined, provider_label)

    car_rgba, mask_pil = extract_car_rgba(image, refined)
    metadata = {
        "segmentation_provider": provider_label,
        "primary_source": primary_source,
        "selected_source": selected_source,
        "primary_confidence": primary_conf,
        "primary_response_time_ms": primary_ms,
        # Keep vertex_confidence / vertex_response_time_ms as stable keys so
        # ai_background.py and vehicle_preservation code don't need changes.
        "vertex_confidence": primary_conf,
        "vertex_response_time_ms": primary_ms,
        "sam2_confidence": sam2_conf,
        "merge_score": merge_score,
        "mask_coverage": coverage,
        "vertex_fallback_active": using_fallback,
        # Mask stages for vehicle-preservation validation (not serialised to client)
        "vertex_mask": primary_mask,
        "sam2_mask": sam2_mask,
        "merged_mask": merged,
        "final_mask": refined,
        "bootstrap_mask": primary_mask,
    }

    logger.info(
        "segment_vehicle complete: provider=%s confidence=%.3f coverage=%.1f%% "
        "primary_ms=%.0f sam2_conf=%.3f selected=%s",
        provider_label,
        primary_conf,
        coverage * 100,
        primary_ms,
        sam2_conf,
        selected_source,
    )
    return car_rgba, mask_pil, metadata
