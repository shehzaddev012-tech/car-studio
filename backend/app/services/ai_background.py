"""
AIBackgroundService — strict dealership compositing pipeline.

Vehicle preservation policy: a perfect failure is better than an incorrect vehicle.
Never regenerate, repair, or reconstruct vehicle pixels.
"""
from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image

from app.config import settings
from app.services.pipeline import (
    composite_layers,
    generate_shadow,
    generate_studio_background,
    glass_cleanup_stub,
    segment_vehicle,
    validate_output,
)
from app.services.segmentation.sam2_refine import SAM2RefinementError
from app.services.segmentation.vertex import VertexSegmentationError
from app.services.vehicle_preservation import (
    REJECTION_MESSAGE,
    VehiclePreservationReport,
    preservation_failure_result,
    validate_vehicle_preservation,
)

logger = logging.getLogger(__name__)

_MASK_METADATA_KEYS = frozenset({
    "vertex_mask",
    "sam2_mask",
    "merged_mask",
    "final_mask",
    "bootstrap_mask",
})


@dataclass
class ProcessingResult:
    success: bool
    image_bytes: Optional[bytes] = None
    ssim_score: Optional[float] = None
    lpips_score: Optional[float] = None
    quality_metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ProgressCallback(ABC):
    @abstractmethod
    def __call__(self, percent: int, message: str) -> None: ...


class _NoOpProgress(ProgressCallback):
    def __call__(self, percent: int, message: str) -> None:
        logger.debug("progress %d%%: %s", percent, message)


class AIBackgroundService(ABC):
    @abstractmethod
    def process(
        self,
        image_bytes: bytes,
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ProcessingResult:
        ...


class StudioBackgroundService(AIBackgroundService):
    """
    Strict pipeline with vehicle-preservation gate BEFORE compositing.

    Policy: never return output if any vehicle part may have been removed.
    """

    def process(
        self,
        image_bytes: bytes,
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ProcessingResult:
        cb = on_progress or _NoOpProgress()

        try:
            cb(5, "Loading image")
            original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            w, h = original.size
            logger.info("Processing image %dx%d", w, h)

            cb(10, "Vertex AI image-segmentation-001")
            car_rgba, mask, seg_meta = segment_vehicle(original)

            # ── Vehicle preservation gate (before any output generation) ────────
            cb(32, "Validating vehicle preservation")
            provider = seg_meta.get("provider", "vertex+sam2")
            if provider.startswith("rembg"):
                # rembg mask is the reference — comparing SAM2 output back to rembg
                # produces false positives (SAM2 legitimately cleans rembg edges).
                # SAM2 confidence >= 0.50 is the quality gate in this path.
                sam2_conf = float(seg_meta.get("sam2_confidence", 0))
                if sam2_conf >= 0.50:
                    preservation = VehiclePreservationReport(passed=True)
                    logger.info("Preservation gate passed (rembg provider, SAM2 conf=%.3f)", sam2_conf)
                else:
                    preservation = VehiclePreservationReport(
                        passed=False,
                        user_message=REJECTION_MESSAGE,
                        internal_reasons=[f"SAM2 confidence too low for rembg path: {sam2_conf:.3f} < 0.50"],
                    )
            else:
                preservation = validate_vehicle_preservation(
                    vertex_mask=seg_meta["vertex_mask"],
                    sam2_mask=seg_meta["sam2_mask"],
                    merged_mask=seg_meta["merged_mask"],
                    final_mask=seg_meta["final_mask"],
                    vertex_confidence=float(seg_meta["vertex_confidence"]),
                    sam2_confidence=float(seg_meta["sam2_confidence"]),
                    provider=provider,
                )

            if not preservation.passed:
                logger.error(
                    "Job rejected (preservation, pre-composite): vertex_conf=%.3f "
                    "sam2_conf=%.3f edge_loss=%.1f%% stage=%s zones=%s reason=%s",
                    seg_meta.get("vertex_confidence", 0),
                    seg_meta.get("sam2_confidence", 0),
                    preservation.edge_loss_ratio * 100,
                    preservation.stage_retention,
                    preservation.zone_retention,
                    "; ".join(preservation.internal_reasons),
                )
                return ProcessingResult(
                    success=False,
                    quality_metadata=_public_metadata(seg_meta, preservation=preservation),
                    error=preservation_failure_result(preservation),
                )

            cb(45, "Generating premium studio backdrop")
            background = generate_studio_background(w, h)

            cb(58, "Synthesising ground-contact shadow")
            shadow = generate_shadow(mask, w, h)

            cb(72, "Compositing vehicle onto studio background")
            composite = composite_layers(background, shadow, car_rgba)

            cb(78, "Finalising")
            composite = glass_cleanup_stub(composite, car_rgba)

            cb(88, "Running quality validation")
            report = validate_output(
                original,
                composite,
                mask,
                primary_mask=seg_meta.get("vertex_mask"),
                vertex_confidence=None if provider.startswith("rembg") else seg_meta.get("vertex_confidence"),
            )

            quality_metadata = _public_metadata(
                seg_meta,
                preservation=preservation,
                report=report,
            )

            if not report.passed:
                logger.error(
                    "Job rejected (quality): vertex_conf=%.3f SSIM=%.3f LPIPS=%s "
                    "coverage=%.1f%% internal_reason=%s",
                    seg_meta.get("vertex_confidence", 0),
                    report.ssim_score,
                    f"{report.lpips_score:.3f}" if report.lpips_score is not None else "n/a",
                    report.mask_coverage * 100,
                    "; ".join(report.errors),
                )
                return ProcessingResult(
                    success=False,
                    ssim_score=report.ssim_score,
                    lpips_score=report.lpips_score,
                    quality_metadata=quality_metadata,
                    error=report.user_message or REJECTION_MESSAGE,
                )

            cb(95, "Encoding result")
            out_buf = io.BytesIO()
            composite.save(out_buf, format="JPEG", quality=96, subsampling=0)

            logger.info(
                "Job complete: vertex_conf=%.3f vertex_ms=%.0f coverage=%.1f%% "
                "SSIM=%.3f LPIPS=%s preservation=passed",
                seg_meta.get("vertex_confidence", 0),
                seg_meta.get("vertex_response_time_ms", 0),
                report.mask_coverage * 100,
                report.ssim_score,
                f"{report.lpips_score:.3f}" if report.lpips_score is not None else "n/a",
            )

            cb(100, "Complete")
            return ProcessingResult(
                success=True,
                image_bytes=out_buf.getvalue(),
                ssim_score=report.ssim_score,
                lpips_score=report.lpips_score,
                quality_metadata=quality_metadata,
            )

        except (VertexSegmentationError, SAM2RefinementError, ValueError) as exc:
            logger.error("Pipeline failed (non-retryable): %s", exc)
            return ProcessingResult(success=False, error=REJECTION_MESSAGE)

        except Exception as exc:
            logger.exception("Unexpected pipeline error")
            raise RuntimeError(f"Pipeline failed unexpectedly: {exc}") from exc


def _public_metadata(seg_meta: dict, *, preservation=None, report=None) -> dict:
    """Strip numpy masks; expose scores only."""
    out = {
        k: v
        for k, v in seg_meta.items()
        if k not in _MASK_METADATA_KEYS
    }
    if preservation is not None:
        out["preservation_passed"] = preservation.passed
        out["preservation_stage_retention"] = preservation.stage_retention
        out["preservation_zone_retention"] = preservation.zone_retention
        out["preservation_edge_loss"] = preservation.edge_loss_ratio
    if report is not None:
        out["ssim_score"] = report.ssim_score
        out["lpips_score"] = report.lpips_score
        out["vehicle_cropped"] = report.vehicle_cropped
        out["missing_parts"] = report.missing_parts
    return out


def get_ai_background_service() -> AIBackgroundService:
    if settings.ai_provider.lower().strip() != "compositing":
        raise RuntimeError(
            f"AI_PROVIDER={settings.ai_provider!r} is not supported. "
            "Only AI_PROVIDER=compositing (Vertex AI + SAM2) is permitted."
        )
    logger.info(
        "AI provider: strict vehicle preservation (Vertex + SAM2, SSIM≥%.2f, "
        "global_retention≥%.1f%%)",
        settings.qc_ssim_threshold,
        settings.preservation_min_global_retention * 100,
    )
    return StudioBackgroundService()
