"""
AIBackgroundService — high-level interface and production implementation.

Architecture note:
  The ``AIBackgroundService`` ABC defines the *contract*.  Business logic
  (routes, workers) depends only on this interface.  The underlying models
  (segmentation, compositing, shadow) are encapsulated inside
  ``StudioBackgroundService`` and can be swapped at any time by:
    1. Implementing a new subclass of ``AIBackgroundService``.
    2. Updating ``get_ai_background_service()`` to return it.
    Nothing else changes.

Current production implementation: rembg (U²-Net) + PIL compositing.
Planned upgrade path:  SAM 2 segmentation + diffusion-based shadow generation.
"""
from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from app.config import settings
from app.services.pipeline import (
    composite_layers,
    generate_shadow,
    generate_studio_background,
    glass_cleanup_stub,
    quality_check,
    segment_car,
)

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class ProcessingResult:
    success: bool
    image_bytes: Optional[bytes] = None   # JPEG bytes of the composited result
    ssim_score: Optional[float] = None
    error: Optional[str] = None


# ── Progress callback protocol ─────────────────────────────────────────────────

class ProgressCallback(ABC):
    @abstractmethod
    def __call__(self, percent: int, message: str) -> None: ...


class _NoOpProgress(ProgressCallback):
    def __call__(self, percent: int, message: str) -> None:
        logger.debug("progress %d%%: %s", percent, message)


# ── Abstract interface ─────────────────────────────────────────────────────────

class AIBackgroundService(ABC):
    """
    Transforms a car photo into a studio-background composite.

    Contract:
      - Must not alter the car's body, paint, wheels, lights, plate, or mirrors.
      - Only the background (and optionally glass-only cleanup) may change.
      - Must return ``ProcessingResult.success=False`` rather than raise on
        recoverable errors (no car detected, QC failure).
      - Unrecoverable errors (OOM, I/O failure) should raise ``RuntimeError``.
    """

    @abstractmethod
    def process(
        self,
        image_bytes: bytes,
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ProcessingResult:
        """Run the full pipeline and return a ``ProcessingResult``."""


# ── Production implementation ──────────────────────────────────────────────────

class StudioBackgroundService(AIBackgroundService):
    """
    Full pipeline:
      1. rembg U²-Net segmentation (or GrabCut in mock mode)
      2. Deterministic studio background generation (PIL)
      3. Three-layer soft shadow synthesis (OpenCV Gaussian blend)
      4. Alpha-composite:  background → shadow → original car pixels
      5. Glass-cleanup stub (no-op — see pipeline.py)
      6. SSIM quality gate on the masked car region

    To swap the segmentation model, edit ``pipeline.segment_car``.
    To swap the background style, edit ``pipeline.generate_studio_background``.
    """

    def __init__(self, *, qc_threshold: Optional[float] = None, mock: Optional[bool] = None) -> None:
        self._qc_threshold = qc_threshold if qc_threshold is not None else settings.qc_ssim_threshold
        self._mock = mock if mock is not None else settings.mock_ai

    def process(
        self,
        image_bytes: bytes,
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ProcessingResult:
        cb = on_progress or _NoOpProgress()

        try:
            # ── Load ───────────────────────────────────────────────────────────
            cb(5, "Loading image")
            original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            w, h = original.size
            logger.info("Processing image %dx%d (mock=%s)", w, h, self._mock)

            # ── Step 1: Segmentation ───────────────────────────────────────────
            cb(10, "Segmenting car from background")
            car_rgba, mask = segment_car(original, mock=self._mock)

            # ── Step 3: Background ─────────────────────────────────────────────
            cb(40, "Generating studio background")
            background = generate_studio_background(w, h)

            # ── Step 4: Shadow ─────────────────────────────────────────────────
            cb(58, "Synthesising contact shadow")
            shadow = generate_shadow(mask, w, h)

            # ── Step 5: Composite ──────────────────────────────────────────────
            cb(72, "Compositing layers")
            composite = composite_layers(background, shadow, car_rgba)

            # ── Step 6: Glass cleanup (stub) ───────────────────────────────────
            cb(80, "Finalising")
            composite = glass_cleanup_stub(composite, car_rgba)

            # ── Step 7: Quality gate ───────────────────────────────────────────
            cb(88, "Running quality verification")
            passed, ssim = quality_check(original, composite, mask, threshold=self._qc_threshold)

            if not passed:
                return ProcessingResult(
                    success=False,
                    ssim_score=ssim,
                    error=(
                        f"Quality check failed: car region SSIM {ssim:.3f} is below the "
                        f"required threshold of {self._qc_threshold:.2f}. "
                        "The compositing may have altered the vehicle's appearance. "
                        "This job has been rejected to prevent a degraded result from being delivered."
                    ),
                )

            # ── Encode output ──────────────────────────────────────────────────
            cb(95, "Encoding result")
            out_buf = io.BytesIO()
            composite.save(out_buf, format="JPEG", quality=95, subsampling=0)

            cb(100, "Complete")
            return ProcessingResult(success=True, image_bytes=out_buf.getvalue(), ssim_score=ssim)

        except ValueError as exc:
            # Business-logic errors (no car detected, segmentation failure, etc.)
            logger.warning("Pipeline validation error: %s", exc)
            return ProcessingResult(success=False, error=str(exc))

        except Exception as exc:
            logger.exception("Unexpected pipeline error")
            raise RuntimeError(f"Pipeline failed unexpectedly: {exc}") from exc


# ── DashScope cloud implementation ─────────────────────────────────────────────

class DashScopeBackgroundService(AIBackgroundService):
    """
    Replaces the car's background using Alibaba Cloud DashScope
    qwen-image-edit-max via the OpenAI-compatible /images/edits endpoint.
    No local model is loaded — all inference runs remotely.
    Requires DASHSCOPE_API_KEY set to an international DashScope key.

    Flow:
      1. Resize input image to ≤1024px and convert to PNG
      2. POST to /compatible-mode/v1/images/edits with image + prompt
      3. Decode the returned b64_json (or download URL)
      4. Re-encode as JPEG and return
    """

    _BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    _MODEL = "qwen-image-edit-max"

    _PROMPT = (
        "Replace only the car's background with a professional automotive dealership "
        "studio setting: neutral light grey seamless paper backdrop, soft diffused "
        "overhead studio lighting, clean minimal environment. "
        "Keep the car body, paint, wheels, windows, lights, and all details completely unchanged."
    )

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def _prepare_png(self, image_bytes: bytes) -> bytes:
        """Resize to ≤1024px on the longest edge and convert to PNG."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        max_side = 1024
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def process(
        self,
        image_bytes: bytes,
        *,
        on_progress=None,
    ) -> ProcessingResult:
        import base64
        import requests as req

        cb = on_progress or _NoOpProgress()

        try:
            # ── 1. Prepare ─────────────────────────────────────────────────────
            cb(5, "Preparing image for DashScope…")
            png_bytes = self._prepare_png(image_bytes)
            logger.info("DashScope: PNG prepared (%d bytes)", len(png_bytes))

            # ── 2. Call images/edits ───────────────────────────────────────────
            cb(20, "Calling qwen-image-edit-max for background replacement…")
            resp = req.post(
                f"{self._BASE_URL}/images/edits",
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"image": ("car.png", io.BytesIO(png_bytes), "image/png")},
                data={
                    "model": self._MODEL,
                    "prompt": self._PROMPT,
                    "n": "1",
                    "response_format": "b64_json",
                },
                timeout=120,
            )

            if resp.status_code != 200:
                body = resp.json()
                err = (body.get("error") or {}).get("message") or resp.text
                raise ValueError(f"DashScope HTTP {resp.status_code}: {err}")

            body = resp.json()
            data = body.get("data", [])
            if not data:
                raise RuntimeError(f"DashScope returned no image data: {body}")

            result = data[0]
            cb(82, "Decoding result from DashScope…")

            if "b64_json" in result:
                raw = base64.b64decode(result["b64_json"])
            elif "url" in result:
                cb(82, "Downloading result from DashScope…")
                img_resp = req.get(result["url"], timeout=90)
                img_resp.raise_for_status()
                raw = img_resp.content
            else:
                raise RuntimeError(
                    f"DashScope returned unexpected data format: {list(result.keys())}"
                )

            if len(raw) < 5_000:
                raise ValueError(
                    "DashScope returned an unexpectedly small image — possible generation error"
                )

            # ── 3. Re-encode as JPEG ───────────────────────────────────────────
            cb(95, "Finalising output…")
            out = io.BytesIO()
            Image.open(io.BytesIO(raw)).convert("RGB").save(
                out, format="JPEG", quality=95, subsampling=0
            )
            cb(100, "Complete")
            logger.info(
                "DashScope background generation complete (%d bytes)", len(out.getvalue())
            )
            return ProcessingResult(
                success=True,
                image_bytes=out.getvalue(),
                ssim_score=None,
            )

        except ValueError as exc:
            logger.warning("DashScope validation error: %s", exc)
            return ProcessingResult(success=False, error=str(exc))

        except Exception as exc:
            logger.exception("DashScope pipeline error")
            raise RuntimeError(f"DashScope pipeline failed: {exc}") from exc


# ── Factory ────────────────────────────────────────────────────────────────────

def get_ai_background_service() -> AIBackgroundService:
    """
    Return the active AIBackgroundService.

    Priority:
      1. DASHSCOPE_API_KEY set → DashScopeBackgroundService (cloud, no local model)
      2. MOCK_AI=true          → StudioBackgroundService with GrabCut (fast, rough edges)
      3. default               → StudioBackgroundService with rembg (best quality)
    """
    if settings.dashscope_api_key:
        logger.info(
            "AI provider: DashScope wanx-background-generation-v2 (cloud, no local model)"
        )
        return DashScopeBackgroundService(api_key=settings.dashscope_api_key)
    logger.info(
        "AI provider: local pipeline (rembg=%s)", not settings.mock_ai
    )
    return StudioBackgroundService()
