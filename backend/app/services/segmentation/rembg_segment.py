"""
rembg-based vehicle segmentation — fallback when Vertex AI model is unavailable.

This module is ONLY used when image-segmentation-001 returns a 404 (model
unavailable in the GCP project). The rest of the pipeline — SAM2 refinement,
mask merging, edge refinement, quality gates — runs unchanged after this step.
"""
from __future__ import annotations

import io
import logging
import time

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_rembg_session = None


class RembgSegmentationError(ValueError):
    """rembg segmentation failed."""


def _get_session():
    global _rembg_session
    if _rembg_session is None:
        try:
            from rembg import new_session  # type: ignore[import-untyped]

            logger.info("Loading rembg u2net model (first-time download may take a moment)")
            _rembg_session = new_session("u2net")
            logger.info("rembg u2net model loaded")
        except Exception as exc:
            raise RembgSegmentationError(f"rembg model failed to load: {exc}") from exc
    return _rembg_session


def segment_vehicle_rembg(image: Image.Image) -> tuple[np.ndarray, float, float]:
    """
    Run rembg u2net background removal and extract the foreground mask.

    Returns:
        (mask float32 0–1, confidence score, response_time_ms)

    Raises:
        RembgSegmentationError on any failure.
    """
    try:
        session = _get_session()
    except RembgSegmentationError:
        raise

    from rembg import remove  # type: ignore[import-untyped]

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    png_bytes = buf.getvalue()

    started = time.perf_counter()
    try:
        output_bytes = remove(png_bytes, session=session)
    except Exception as exc:
        raise RembgSegmentationError(f"rembg inference failed: {exc}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    try:
        output_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    except Exception as exc:
        raise RembgSegmentationError(f"rembg output parse failed: {exc}") from exc

    w, h = image.size
    if output_img.size != (w, h):
        output_img = output_img.resize((w, h), Image.LANCZOS)

    alpha = np.array(output_img.split()[3], dtype=np.float32) / 255.0

    coverage = float((alpha > 0.5).mean())
    if coverage < 0.01:
        raise RembgSegmentationError(
            "rembg produced an empty mask — no foreground object detected."
        )

    # Derive a pseudo-confidence from how well coverage fits the expected vehicle range.
    # Vehicles typically occupy 10–80% of a well-framed photo.
    ideal_coverage = 0.40
    confidence = max(0.0, 1.0 - abs(coverage - ideal_coverage) / ideal_coverage)
    confidence = max(confidence, 0.55)  # floor — u2net is generally reliable

    logger.info(
        "rembg segmentation: coverage=%.1f%% pseudo_confidence=%.3f response_time_ms=%.0f",
        coverage * 100,
        confidence,
        elapsed_ms,
    )
    return alpha, confidence, elapsed_ms
