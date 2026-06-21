"""Mask merging, edge refinement, and anti-aliasing for vehicle segmentation."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class MaskCandidate:
    mask: np.ndarray  # float32, 0–1, H×W
    confidence: float
    source: str


def mask_bbox(mask: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, int]:
    """Return (x_min, y_min, x_max, y_max) for a binary mask."""
    binary = mask > threshold
    if not binary.any():
        return 0, 0, mask.shape[1] - 1, mask.shape[0] - 1
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    y_min, y_max = int(np.where(rows)[0][[0, -1]].tolist()[0]), int(np.where(rows)[0][[0, -1]].tolist()[-1])
    x_min, x_max = int(np.where(cols)[0][[0, -1]].tolist()[0]), int(np.where(cols)[0][[0, -1]].tolist()[-1])
    return x_min, y_min, x_max, y_max


def mask_iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> float:
    a_bin = a > threshold
    b_bin = b > threshold
    inter = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    return float(inter / union) if union else 0.0


def mask_coverage(mask: np.ndarray, threshold: float = 0.5) -> float:
    return float((mask > threshold).sum() / mask.size)


def _edge_quality_score(mask: np.ndarray) -> float:
    """Higher when mask boundary aligns with image edges (clean cut)."""
    binary = (mask > 0.5).astype(np.uint8) * 255
    edges = cv2.Canny(binary, 50, 150)
    if edges.sum() == 0:
        return 0.0
    return min(float(edges.sum()) / (mask.size * 0.02), 1.0)


def _coverage_score(coverage: float, min_cov: float, max_cov: float) -> float:
    if coverage < min_cov or coverage > max_cov:
        return 0.0
    ideal = (min_cov + max_cov) / 2.0
    spread = (max_cov - min_cov) / 2.0
    return max(0.0, 1.0 - abs(coverage - ideal) / spread)


def score_mask_candidate(
    candidate: MaskCandidate,
    *,
    reference: MaskCandidate | None = None,
    min_coverage: float = 0.03,
    max_coverage: float = 0.88,
) -> float:
    """Composite confidence for mask selection."""
    cov = mask_coverage(candidate.mask)
    cov_score = _coverage_score(cov, min_coverage, max_coverage)
    edge_score = _edge_quality_score(candidate.mask)
    agreement = mask_iou(candidate.mask, reference.mask) if reference else 0.5
    return (
        candidate.confidence * 0.45
        + cov_score * 0.25
        + edge_score * 0.15
        + agreement * 0.15
    )


def merge_and_select_mask(
    candidates: list[MaskCandidate],
    *,
    min_coverage: float = 0.03,
    max_coverage: float = 0.88,
) -> tuple[np.ndarray, str, float]:
    """
    Score all candidates plus a conservative union, return the best mask.

    The union candidate prevents SAM2 from dropping mirrors/spoilers when
    Vertex captured them but SAM2 trimmed too aggressively.
    """
    if not candidates:
        raise ValueError("No segmentation mask candidates to merge.")

    primary = candidates[0]
    scored: list[tuple[np.ndarray, str, float]] = []

    for cand in candidates:
        score = score_mask_candidate(
            cand,
            reference=primary,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
        )
        scored.append((cand.mask, cand.source, score))

    if len(candidates) >= 2:
        union = np.clip(np.maximum(candidates[0].mask, candidates[1].mask), 0.0, 1.0)
        union_cov = mask_coverage(union)
        union_conf = min(candidates[0].confidence, candidates[1].confidence) * 0.95
        union_cand = MaskCandidate(mask=union, confidence=union_conf, source="vertex+sam2_union")
        union_score = score_mask_candidate(
            union_cand,
            reference=primary,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
        )
        scored.append((union, union_cand.source, union_score))

    best = max(scored, key=lambda x: x[2])
    logger.info(
        "merge_and_select_mask: selected=%s score=%.3f (candidates=%s)",
        best[1],
        best[2],
        [s[1] for s in scored],
    )
    return best[0], best[1], best[2]


def refine_mask_edges(mask: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    """
    Morphological cleanup + boundary-aware anti-aliasing.

    Preserves hard body edges while feathering only a 3 px boundary band.
    """
    h, w = mask.shape
    binary = (mask > 0.5).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove small isolated blobs (background noise)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cleaned = np.where(labels == largest, 255, 0).astype(np.uint8)

    refined = cleaned.astype(np.float32) / 255.0

    # Anti-alias: feather only the boundary band using the image luminance as guide
    dilated = cv2.dilate(cleaned, np.ones((3, 3), np.uint8), iterations=1)
    eroded = cv2.erode(cleaned, np.ones((3, 3), np.uint8), iterations=1)
    boundary = ((dilated > 0) & (eroded == 0)).astype(np.float32)

    if boundary.any() and image_rgb is not None:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        edge_strength = cv2.Laplacian(gray, cv2.CV_32F)
        edge_strength = np.abs(edge_strength)
        edge_strength = edge_strength / (edge_strength.max() + 1e-6)
        # Sharper alpha on strong image edges (body panel boundaries)
        band = refined.copy()
        band[boundary > 0] = np.clip(
            refined[boundary > 0] * (0.85 + 0.15 * edge_strength[boundary > 0]),
            0.0,
            1.0,
        )
        refined = cv2.GaussianBlur(band, (0, 0), sigmaX=0.8)
    else:
        refined = cv2.GaussianBlur(refined, (0, 0), sigmaX=0.8)

    return np.clip(refined, 0.0, 1.0)


def mask_to_pil_alpha(mask: np.ndarray) -> Image.Image:
    """Convert float mask to L-mode PIL image (0–255)."""
    return Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8), "L")


def extract_car_rgba(image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, Image.Image]:
    """Build RGBA car layer — original vehicle pixels, never regenerated."""
    alpha = mask_to_pil_alpha(mask)
    car_rgba = image.convert("RGBA")
    car_rgba.putalpha(alpha)
    return car_rgba, alpha
