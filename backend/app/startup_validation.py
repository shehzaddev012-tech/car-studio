"""
Strict startup validation for the dealership compositing pipeline.

The application refuses to start unless Vertex AI is correctly configured
and reachable. No segmentation fallbacks exist at runtime.
"""
from __future__ import annotations

import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class StartupValidationError(RuntimeError):
    """Raised when mandatory pipeline prerequisites are missing or unreachable."""


def _credentials_path() -> Path:
    raw = (settings.google_application_credentials or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")).strip()
    if not raw:
        raise StartupValidationError(
            "GOOGLE_APPLICATION_CREDENTIALS is required. "
            "Set it to the path of your GCP service account JSON key file."
        )
    path = Path(raw)
    if not path.is_file():
        raise StartupValidationError(
            f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {path}"
        )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path.resolve())
    return path


def _verify_vertex_connectivity() -> float:
    """
    Perform a live Vertex AI segmentation probe.

    Returns:
        Response time in milliseconds.

    Raises:
        StartupValidationError on auth, network, timeout, or empty-mask failure.
    """
    from google import genai  # type: ignore[import-untyped]
    from google.genai import types  # type: ignore[import-untyped]

    if not settings.google_cloud_project.strip():
        raise StartupValidationError(
            "GOOGLE_CLOUD_PROJECT is required. Set your GCP project ID."
        )

    client = genai.Client(
        vertexai=True,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )

    probe = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    probe.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    def _call_vertex():
        return client.models.segment_image(
            model=settings.vertex_segmentation_model,
            source=types.SegmentImageSource(
                image=types.Image(image_bytes=png_bytes, mime_type="image/png"),
                prompt=settings.vertex_segmentation_prompt,
            ),
            config=types.SegmentImageConfig(
                mode=types.SegmentMode.PROMPT,
                mask_dilation=settings.vertex_mask_dilation,
                confidence_threshold=settings.vertex_confidence_threshold,
            ),
        )

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_vertex)
            response = future.result(timeout=settings.vertex_request_timeout_seconds)
    except FuturesTimeout as exc:
        raise StartupValidationError(
            f"Vertex AI connectivity probe timed out after {settings.vertex_request_timeout_seconds}s. "
            "Check network access and Vertex AI API enablement."
        ) from exc
    except Exception as exc:
        raise StartupValidationError(
            f"Vertex AI connectivity probe failed: {exc}"
        ) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not response.generated_masks:
        raise StartupValidationError(
            "Vertex AI connectivity probe returned no masks — API may be misconfigured."
        )
    best = response.generated_masks[0]
    if not best.mask or not best.mask.image_bytes:
        raise StartupValidationError(
            "Vertex AI connectivity probe returned an empty mask."
        )

    logger.info(
        "Vertex AI connectivity verified: model=%s response_time_ms=%.0f project=%s",
        settings.vertex_segmentation_model,
        elapsed_ms,
        settings.google_cloud_project,
    )
    return elapsed_ms


def _verify_sam2_available() -> None:
    """Ensure SAM2 refinement model loads — required for the pipeline."""
    try:
        from app.services.segmentation.sam2_refine import ensure_sam2_loaded

        ensure_sam2_loaded()
    except Exception as exc:
        raise StartupValidationError(
            f"SAM2 refinement model failed to load: {exc}"
        ) from exc


def validate_dealership_pipeline(*, context: str = "application") -> None:
    """
    Run all mandatory startup checks.

    Set SKIP_VERTEX_STARTUP_VALIDATION=1 only in automated tests.
    """
    if os.environ.get("SKIP_VERTEX_STARTUP_VALIDATION", "").lower() in ("1", "true", "yes"):
        logger.warning("Skipping Vertex startup validation (%s)", context)
        return

    if settings.ai_provider.lower().strip() != "compositing":
        logger.warning("Startup validation skipped: AI_PROVIDER is not compositing")
        return

    logger.info("Running strict dealership pipeline startup validation (%s)…", context)
    creds = _credentials_path()
    logger.info("GCP credentials file verified: %s", creds)

    response_ms = _verify_vertex_connectivity()
    _verify_sam2_available()

    logger.info(
        "Dealership pipeline startup validation passed (%s): vertex_probe_ms=%.0f",
        context,
        response_ms,
    )
