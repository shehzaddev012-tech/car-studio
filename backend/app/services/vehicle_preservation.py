"""
Strict vehicle-preservation policy.

The vehicle is the most important asset. A perfect failure is better than
an incorrect dealership image. Never regenerate, repair, or reconstruct vehicle pixels.

User-facing rejection (all preservation failures):
  "Vehicle could not be isolated with sufficient accuracy. Please upload a clearer image."
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

REJECTION_MESSAGE = (
    "Vehicle could not be isolated with sufficient accuracy. "
    "Please upload a clearer image."
)


@dataclass
class ZoneSpec:
    name: str
    x0: float  # fraction of bbox width
    y0: float  # fraction of bbox height
    x1: float
    y1: float
    min_retention: float


@dataclass
class VehiclePreservationReport:
    passed: bool
    user_message: str | None = None
    internal_reasons: list[str] = field(default_factory=list)
    zone_retention: dict[str, float] = field(default_factory=dict)
    stage_retention: dict[str, float] = field(default_factory=dict)
    edge_loss_ratio: float = 0.0
    aggressive_edge_cut: bool = False

    def summary(self) -> str:
        parts = [f"passed={self.passed}"]
        if self.stage_retention:
            parts.append(
                "stages="
                + ",".join(f"{k}={v:.3f}" for k, v in self.stage_retention.items())
            )
        if self.zone_retention:
            weak = {k: v for k, v in self.zone_retention.items() if v < 1.0}
            if weak:
                parts.append(
                    "zones=" + ",".join(f"{k}={v:.3f}" for k, v in weak.items())
                )
        return " ".join(parts)


# Critical part zones relative to the Vertex mask bounding box.
# Zones with no Vertex foreground pixels are skipped (part not visible).
CRITICAL_ZONES: tuple[ZoneSpec, ...] = (
    ZoneSpec("mirrors_left", 0.00, 0.22, 0.14, 0.58, 0.93),
    ZoneSpec("mirrors_right", 0.86, 0.22, 1.00, 0.58, 0.93),
    ZoneSpec("wheels_tires_left", 0.00, 0.68, 0.28, 1.00, 0.90),
    ZoneSpec("wheels_tires_right", 0.72, 0.68, 1.00, 1.00, 0.90),
    ZoneSpec("headlights_left", 0.00, 0.55, 0.22, 0.88, 0.88),
    ZoneSpec("headlights_right", 0.78, 0.55, 1.00, 0.88, 0.88),
    ZoneSpec("tail_lights_left", 0.00, 0.48, 0.20, 0.78, 0.88),
    ZoneSpec("tail_lights_right", 0.80, 0.48, 1.00, 0.78, 0.88),
    ZoneSpec("license_plate", 0.32, 0.72, 0.68, 0.96, 0.85),
    ZoneSpec("roof_rails", 0.08, 0.00, 0.92, 0.18, 0.88),
    ZoneSpec("antenna", 0.40, 0.00, 0.60, 0.12, 0.82),
    ZoneSpec("vehicle_edges_left", 0.00, 0.10, 0.08, 0.90, 0.91),
    ZoneSpec("vehicle_edges_right", 0.92, 0.10, 1.00, 0.90, 0.91),
)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    binary = mask > 0.5
    if not binary.any():
        return 0, 0, mask.shape[1] - 1, mask.shape[0] - 1
    ys, xs = np.where(binary)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _align_masks(*masks: np.ndarray) -> list[np.ndarray]:
    """Ensure all masks share the same spatial dimensions."""
    h = max(m.shape[0] for m in masks)
    w = max(m.shape[1] for m in masks)
    aligned: list[np.ndarray] = []
    for m in masks:
        if m.shape[0] == h and m.shape[1] == w:
            aligned.append(m.astype(np.float32))
        else:
            from PIL import Image

            pil = Image.fromarray((m * 255).astype(np.uint8), "L").resize((w, h), Image.LANCZOS)
            aligned.append(np.array(pil, dtype=np.float32) / 255.0)
    return aligned


def _global_retention(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference > 0.5
    if not ref.any():
        return 0.0
    cand = candidate > 0.5
    return float(np.logical_and(ref, cand).sum() / ref.sum())


def _zone_slice(
    bbox: tuple[int, int, int, int],
    spec: ZoneSpec,
    shape: tuple[int, int],
) -> tuple[slice, slice]:
    x0, y0, x1, y1 = bbox
    bw = max(x1 - x0, 1)
    bh = max(y1 - y0, 1)
    h, w = shape
    zx0 = max(0, int(x0 + spec.x0 * bw))
    zx1 = min(w, int(x0 + spec.x1 * bw) + 1)
    zy0 = max(0, int(y0 + spec.y0 * bh))
    zy1 = min(h, int(y0 + spec.y1 * bh) + 1)
    return slice(zy0, zy1), slice(zx0, zx1)


def _zone_retention(
    reference: np.ndarray,
    candidate: np.ndarray,
    bbox: tuple[int, int, int, int],
    spec: ZoneSpec,
) -> tuple[float, bool]:
    """
    Retention in a critical zone.

    Returns:
        (retention 0–1, applicable) — applicable=False when Vertex detected nothing there.
    """
    ys, xs = _zone_slice(bbox, spec, reference.shape)
    ref_zone = reference[ys, xs] > 0.5
    if ref_zone.sum() < 8:
        return 1.0, False
    cand_zone = candidate[ys, xs] > 0.5
    retained = float(np.logical_and(ref_zone, cand_zone).sum() / ref_zone.sum())
    return retained, True


def _edge_loss_ratio(vertex: np.ndarray, final: np.ndarray) -> tuple[float, bool]:
    """
    Measure foreground loss concentrated on the vehicle boundary band.

    Returns:
        (loss_ratio, aggressive_cut_detected)
    """
    v_bin = (vertex > 0.5).astype(np.uint8) * 255
    f_bin = (final > 0.5).astype(np.uint8) * 255
    if v_bin.sum() == 0:
        return 1.0, True

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(v_bin, kernel, iterations=2)
    eroded = cv2.erode(v_bin, kernel, iterations=1)
    boundary_band = (dilated > 0) & (eroded == 0)

    vertex_fg = vertex > 0.5
    final_fg = final > 0.5
    lost = vertex_fg & ~final_fg
    boundary_lost = lost & boundary_band

    total_lost = float(lost.sum() / max(vertex_fg.sum(), 1))
    boundary_lost_ratio = float(boundary_lost.sum() / max(boundary_band.sum(), 1))

    aggressive = total_lost > settings.preservation_max_edge_loss_ratio
    return total_lost, aggressive


def _check_zones(
    reference: np.ndarray,
    candidate: np.ndarray,
    bbox: tuple[int, int, int, int],
    reasons: list[str],
    zone_scores: dict[str, float],
    *,
    stage_label: str,
) -> bool:
    """Validate all critical zones. Returns True if all applicable zones pass."""
    all_ok = True
    for spec in CRITICAL_ZONES:
        retention, applicable = _zone_retention(reference, candidate, bbox, spec)
        if not applicable:
            continue
        key = f"{stage_label}:{spec.name}"
        zone_scores[key] = retention
        threshold = max(spec.min_retention, settings.preservation_min_zone_retention)
        if retention < threshold:
            all_ok = False
            reasons.append(
                f"{stage_label} lost {spec.name.replace('_', ' ')} "
                f"(retention {retention:.1%} < {threshold:.1%})"
            )
    return all_ok


def validate_vehicle_preservation(
    *,
    vertex_mask: np.ndarray,
    sam2_mask: np.ndarray,
    merged_mask: np.ndarray,
    final_mask: np.ndarray,
    vertex_confidence: float,
    sam2_confidence: float,
) -> VehiclePreservationReport:
    """
    Compare Vertex → SAM2 → merged → final masks before any compositing.

    Rejects when any risk of removed vehicle parts is detected.
    """
    reasons: list[str] = []
    zone_scores: dict[str, float] = {}
    stage_scores: dict[str, float] = {}

    vertex_mask, sam2_mask, merged_mask, final_mask = _align_masks(
        vertex_mask, sam2_mask, merged_mask, final_mask
    )

    bbox = _bbox(vertex_mask)

    # ── Confidence gates ──────────────────────────────────────────────────────
    if vertex_confidence < settings.vertex_min_confidence:
        reasons.append(
            f"Vertex confidence {vertex_confidence:.3f} < {settings.vertex_min_confidence:.2f}"
        )
    if sam2_confidence < settings.preservation_min_sam2_confidence:
        reasons.append(
            f"SAM2 confidence {sam2_confidence:.3f} < {settings.preservation_min_sam2_confidence:.2f}"
        )

    # ── Cross-stage global retention (Vertex is ground truth) ─────────────────
    for label, candidate in (
        ("vertex_to_sam2", sam2_mask),
        ("vertex_to_merged", merged_mask),
        ("vertex_to_final", final_mask),
    ):
        retention = _global_retention(vertex_mask, candidate)
        stage_scores[label] = retention
        if retention < settings.preservation_min_global_retention:
            reasons.append(
                f"{label} global retention {retention:.1%} "
                f"< {settings.preservation_min_global_retention:.1%}"
            )

    # ── SAM2 must not aggressively trim Vertex ────────────────────────────────
    sam2_trim = _global_retention(vertex_mask, sam2_mask)
    if sam2_trim < settings.preservation_min_sam2_retention:
        reasons.append(
            f"SAM2 removed vehicle regions (retention {sam2_trim:.1%} "
            f"< {settings.preservation_min_sam2_retention:.1%})"
        )

    # ── Critical part zones (Vertex → final) ──────────────────────────────────
    _check_zones(
        vertex_mask, final_mask, bbox, reasons, zone_scores, stage_label="final"
    )
    _check_zones(
        vertex_mask, sam2_mask, bbox, reasons, zone_scores, stage_label="sam2"
    )

    # ── Aggressive edge cutting ───────────────────────────────────────────────
    edge_loss, aggressive = _edge_loss_ratio(vertex_mask, final_mask)
    if aggressive:
        reasons.append(
            f"Aggressive edge cutting detected (edge loss {edge_loss:.1%})"
        )

    # ── Final mask must retain almost all Vertex foreground ───────────────────
    if detect_missing_vehicle_regions(vertex_mask, final_mask):
        reasons.append("Missing vehicle regions detected in final mask")

    passed = len(reasons) == 0
    report = VehiclePreservationReport(
        passed=passed,
        user_message=None if passed else REJECTION_MESSAGE,
        internal_reasons=reasons,
        zone_retention=zone_scores,
        stage_retention=stage_scores,
        edge_loss_ratio=edge_loss,
        aggressive_edge_cut=aggressive,
    )

    if passed:
        logger.info(
            "vehicle preservation PASSED: vertex_conf=%.3f sam2_conf=%.3f %s",
            vertex_confidence,
            sam2_confidence,
            report.summary(),
        )
    else:
        logger.error(
            "vehicle preservation FAILED: vertex_conf=%.3f sam2_conf=%.3f "
            "edge_loss=%.1f%% zones=%s reason=%s",
            vertex_confidence,
            sam2_confidence,
            edge_loss * 100,
            zone_scores,
            "; ".join(reasons),
        )

    return report


def detect_missing_vehicle_regions(
    reference: np.ndarray,
    final: np.ndarray,
    *,
    min_retention: float | None = None,
) -> bool:
    """True when the final mask lost a significant portion of the reference."""
    threshold = (
        min_retention
        if min_retention is not None
        else settings.preservation_min_global_retention
    )
    return _global_retention(reference, final) < threshold


def preservation_failure_result(report: VehiclePreservationReport) -> str:
    """User-facing message — always the same; details are logged internally."""
    return report.user_message or REJECTION_MESSAGE
