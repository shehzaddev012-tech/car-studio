"""Multi-metric quality validation — rejects any sub-standard output."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app.config import settings
from app.services.vehicle_preservation import (
    REJECTION_MESSAGE,
    detect_missing_vehicle_regions,
)

logger = logging.getLogger(__name__)

_lpips_model = None

# Internal detail is logged; user always sees REJECTION_MESSAGE for preservation failures.
PRESERVATION_USER_MESSAGE = REJECTION_MESSAGE


@dataclass
class QualityReport:
    passed: bool
    ssim_score: float
    lpips_score: float | None
    mask_coverage: float
    vertex_confidence: float | None
    vehicle_cropped: bool
    missing_parts: bool
    preservation_failure: bool
    user_message: str | None = None
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"SSIM={self.ssim_score:.3f}"]
        if self.lpips_score is not None:
            parts.append(f"LPIPS={self.lpips_score:.3f}")
        parts.append(f"coverage={self.mask_coverage:.1%}")
        if self.vertex_confidence is not None:
            parts.append(f"vertex_conf={self.vertex_confidence:.3f}")
        return ", ".join(parts)


def _get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        import lpips  # type: ignore[import-untyped]
        import torch  # type: ignore[import-untyped]

        _lpips_model = lpips.LPIPS(net="alex", verbose=False)
        _lpips_model.eval()
        if not torch.cuda.is_available():
            _lpips_model = _lpips_model.cpu()
    return _lpips_model


def _compute_ssim(orig_arr: np.ndarray, res_arr: np.ndarray) -> float:
    from skimage.metrics import structural_similarity  # type: ignore[import-untyped]

    return float(
        structural_similarity(
            orig_arr,
            res_arr,
            data_range=1.0,
            channel_axis=2,
            win_size=7,
        )
    )


def _compute_lpips(orig_arr: np.ndarray, res_arr: np.ndarray, mask_arr: np.ndarray) -> float:
    import torch  # type: ignore[import-untyped]

    model = _get_lpips_model()
    binary = mask_arr > 0.5
    if not binary.any():
        raise ValueError("Cannot compute LPIPS: mask has no foreground pixels.")

    ys, xs = np.where(binary)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    o_crop = orig_arr[y0:y1, x0:x1]
    r_crop = res_arr[y0:y1, x0:x1]
    m_crop = mask_arr[y0:y1, x0:x1, np.newaxis]

    o = torch.from_numpy(o_crop * m_crop).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
    r = torch.from_numpy(r_crop * m_crop).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
    with torch.no_grad():
        dist = model(o, r)
    return float(dist.item())


def detect_vehicle_crop(mask: Image.Image, margin_ratio: float) -> bool:
    arr = np.array(mask.convert("L"))
    binary = arr > 10
    if not binary.any():
        return True
    h, w = arr.shape
    margin_x = max(int(w * margin_ratio), 4)
    margin_y = max(int(h * margin_ratio), 4)
    ys, xs = np.where(binary)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (
        x0 <= margin_x
        or y0 <= margin_y
        or x1 >= w - margin_x - 1
        or y1 >= h - margin_y - 1
    )


def validate_output(
    original: Image.Image,
    result: Image.Image,
    mask: Image.Image,
    *,
    primary_mask: np.ndarray | None = None,
    vertex_confidence: float | None = None,
    ssim_threshold: float | None = None,
    lpips_threshold: float | None = None,
) -> QualityReport:
    """
    Post-composite quality gate.

    Any preservation-related failure returns the standard user message while
    logging detailed internal reasons.
    """
    ssim_threshold = ssim_threshold if ssim_threshold is not None else settings.qc_ssim_threshold
    lpips_threshold = lpips_threshold if lpips_threshold is not None else settings.qc_lpips_threshold

    orig = original.convert("RGB")
    res = result.convert("RGB")
    if orig.size != res.size:
        orig = orig.resize(res.size, Image.LANCZOS)

    mask_resized = mask.resize(res.size, Image.LANCZOS).convert("L")
    mask_arr = np.array(mask_resized, dtype=np.float32) / 255.0
    orig_arr = np.array(orig, dtype=np.float32) / 255.0
    res_arr = np.array(res, dtype=np.float32) / 255.0

    orig_masked = orig_arr * mask_arr[:, :, np.newaxis]
    res_masked = res_arr * mask_arr[:, :, np.newaxis]

    ssim = _compute_ssim(orig_masked, res_masked)
    coverage = float((mask_arr > 0.5).mean())

    lpips: float | None = None
    if settings.lpips_enabled:
        try:
            lpips = _compute_lpips(orig_arr, res_arr, mask_arr)
        except Exception as exc:
            logger.error("LPIPS computation failed: %s", exc)
            lpips = None

    internal_errors: list[str] = []
    preservation_failure = False

    if vertex_confidence is not None and vertex_confidence < settings.vertex_min_confidence:
        preservation_failure = True
        internal_errors.append(
            f"Vertex confidence {vertex_confidence:.3f} below {settings.vertex_min_confidence:.2f}"
        )

    if coverage < settings.mask_min_coverage:
        preservation_failure = True
        internal_errors.append(f"Coverage {coverage:.1%} below minimum")

    if coverage > settings.mask_max_coverage:
        preservation_failure = True
        internal_errors.append(f"Coverage {coverage:.1%} above maximum")

    cropped = detect_vehicle_crop(mask_resized, settings.crop_edge_margin_ratio)
    if cropped:
        preservation_failure = True
        internal_errors.append("Vehicle cropped at frame edge")

    missing = False
    if primary_mask is not None:
        if primary_mask.shape != mask_arr.shape:
            primary_pil = Image.fromarray((primary_mask * 255).astype(np.uint8), "L")
            primary_pil = primary_pil.resize(res.size, Image.LANCZOS)
            primary_mask = np.array(primary_pil, dtype=np.float32) / 255.0
        missing = detect_missing_vehicle_regions(primary_mask, mask_arr)
        if missing:
            preservation_failure = True
            internal_errors.append("Missing vehicle regions in final composite")

    if ssim < ssim_threshold:
        preservation_failure = True
        internal_errors.append(f"SSIM {ssim:.3f} below {ssim_threshold:.2f}")

    if settings.lpips_enabled:
        if lpips is None:
            preservation_failure = True
            internal_errors.append("LPIPS could not be computed")
        elif lpips > lpips_threshold:
            preservation_failure = True
            internal_errors.append(f"LPIPS {lpips:.3f} above {lpips_threshold:.2f}")

    passed = len(internal_errors) == 0
    user_message = None if passed else PRESERVATION_USER_MESSAGE

    report = QualityReport(
        passed=passed,
        ssim_score=ssim,
        lpips_score=lpips,
        mask_coverage=coverage,
        vertex_confidence=vertex_confidence,
        vehicle_cropped=cropped,
        missing_parts=missing,
        preservation_failure=preservation_failure,
        user_message=user_message,
        errors=internal_errors,
    )

    if passed:
        logger.info(
            "quality PASSED: SSIM=%.3f LPIPS=%s coverage=%.1f%% vertex_conf=%s",
            ssim,
            f"{lpips:.3f}" if lpips is not None else "n/a",
            coverage * 100,
            f"{vertex_confidence:.3f}" if vertex_confidence is not None else "n/a",
        )
    else:
        logger.error(
            "quality FAILED: SSIM=%.3f LPIPS=%s coverage=%.1f%% vertex_conf=%s "
            "preservation_failure=%s internal_reason=%s",
            ssim,
            f"{lpips:.3f}" if lpips is not None else "n/a",
            coverage * 100,
            f"{vertex_confidence:.3f}" if vertex_confidence is not None else "n/a",
            preservation_failure,
            "; ".join(internal_errors),
        )

    return report
