"""
Startup validation for the dealership compositing pipeline.

Vertex AI is the primary segmentation provider. If the model is unavailable
(404 / GCP project not enrolled), the pipeline falls back to rembg at runtime
and the application starts with a warning. Auth failures and network errors
are still fatal — they indicate a misconfiguration, not a missing GCP feature.
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


class _VertexModelUnavailable(Exception):
    """Sentinel: Vertex API is reachable but the model isn't in this GCP project."""


def _verify_vertex_connectivity() -> float | None:
    """
    Probe Vertex AI.

    Returns:
        Response time in milliseconds, or None if the model is unavailable
        (indicating rembg fallback will be used at runtime).

    Raises:
        StartupValidationError on auth, network, or timeout failures.
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
        http_options={"api_version": "v1"},
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
        err_str = str(exc)
        if "404" in err_str or "NOT_FOUND" in err_str or "unavailable" in err_str.lower():
            # Model not in this GCP project — NOT a fatal misconfiguration.
            # rembg fallback will handle runtime segmentation.
            raise _VertexModelUnavailable(str(exc)) from exc
        # Auth / network / quota — operator must fix this before serving traffic.
        raise StartupValidationError(
            f"Vertex AI connectivity probe failed (auth or network error): {exc}"
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

    Vertex model unavailability (404) is a warning — rembg fallback activates.
    Auth / network failures are fatal — the app refuses to start.
    Set SKIP_VERTEX_STARTUP_VALIDATION=1 only in automated tests.
    """
    if os.environ.get("SKIP_VERTEX_STARTUP_VALIDATION", "").lower() in ("1", "true", "yes"):
        logger.warning("Skipping Vertex startup validation (%s)", context)
        return

    if settings.ai_provider.lower().strip() != "compositing":
        logger.warning("Startup validation skipped: AI_PROVIDER is not compositing")
        return

    logger.info("Running dealership pipeline startup validation (%s)…", context)
    creds = _credentials_path()
    logger.info("GCP credentials file verified: %s", creds)

    vertex_ms: float | None = None
    try:
        vertex_ms = _verify_vertex_connectivity()
    except _VertexModelUnavailable as exc:
        logger.warning(
            "Vertex AI model '%s' is not available in project '%s' / location '%s'. "
            "rembg (u2net) will be used as the segmentation provider at runtime. "
            "To enable Vertex AI, visit: https://console.cloud.google.com/vertex-ai/model-garden. "
            "Details: %s",
            settings.vertex_segmentation_model,
            settings.google_cloud_project,
            settings.google_cloud_location,
            exc,
        )

    _verify_sam2_available()

    if vertex_ms is not None:
        logger.info(
            "Pipeline startup validation passed (%s): provider=vertex-ai vertex_probe_ms=%.0f",
            context,
            vertex_ms,
        )
    else:
        logger.info(
            "Pipeline startup validation passed (%s): provider=rembg-fallback "
            "(Vertex model unavailable — enable image-segmentation-001 in GCP to use primary provider)",
            context,
        )
