"""Vertex AI image-segmentation-001 — sole segmentation provider."""
from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

_genai_client = None


class VertexSegmentationError(ValueError):
    """Vertex AI segmentation failed — job must fail; no fallback permitted."""


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        from google import genai  # type: ignore[import-untyped]

        if not settings.google_cloud_project.strip():
            raise VertexSegmentationError(
                "GOOGLE_CLOUD_PROJECT is not configured. "
                "Vertex AI image-segmentation-001 is the only permitted segmentation provider."
            )
        creds = settings.resolved_credentials_path()
        if not creds:
            raise VertexSegmentationError(
                "GOOGLE_APPLICATION_CREDENTIALS is not configured."
            )

        logger.info(
            "Initialising Vertex AI client (project=%s, location=%s)",
            settings.google_cloud_project,
            settings.google_cloud_location,
        )
        _genai_client = genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
            http_options={"api_version": "v1"},
        )
    return _genai_client


def segment_vehicle_vertex(image: Image.Image) -> tuple[np.ndarray, float, float]:
    """
    Run Vertex AI image-segmentation-001 with a vehicle prompt.

    Returns:
        (mask float32 0–1, confidence score, response_time_ms)

    Raises:
        VertexSegmentationError on any failure — never falls back to another provider.
    """
    from google.genai import types  # type: ignore[import-untyped]

    client = _get_genai_client()
    w, h = image.size

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    png_bytes = buf.getvalue()

    def _call():
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
            response = pool.submit(_call).result(timeout=settings.vertex_request_timeout_seconds)
    except FuturesTimeout as exc:
        raise VertexSegmentationError(
            f"Vertex AI segmentation timed out after {settings.vertex_request_timeout_seconds}s."
        ) from exc
    except Exception as exc:
        raise VertexSegmentationError(
            f"Vertex AI segmentation request failed: {exc}"
        ) from exc

    response_ms = (time.perf_counter() - started) * 1000.0

    if not response.generated_masks:
        raise VertexSegmentationError(
            "Vertex AI returned no segmentation masks for this image. "
            "Ensure the photo shows a clear vehicle."
        )

    best = max(
        response.generated_masks,
        key=lambda m: (m.labels[0].score if m.labels else 0.0),
    )
    mask_bytes = best.mask.image_bytes if best.mask else None
    if not mask_bytes:
        raise VertexSegmentationError("Vertex AI returned an empty segmentation mask.")

    confidence = float(best.labels[0].score) if best.labels else 0.0
    if confidence < settings.vertex_min_confidence:
        raise VertexSegmentationError(
            f"Vertex AI segmentation confidence {confidence:.3f} is below the required "
            f"minimum of {settings.vertex_min_confidence:.2f}. "
            "Upload a clearer vehicle photo or adjust lighting."
        )

    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    if mask_img.size != (w, h):
        mask_img = mask_img.resize((w, h), Image.LANCZOS)

    mask = np.array(mask_img, dtype=np.float32) / 255.0
    coverage = float((mask > 0.5).mean())

    logger.info(
        "Vertex segmentation: model=%s confidence=%.3f coverage=%.1f%% response_time_ms=%.0f",
        settings.vertex_segmentation_model,
        confidence,
        coverage * 100,
        response_ms,
    )
    return mask, confidence, response_ms
