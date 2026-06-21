"""SAM 2 mask refinement — required stage after Vertex AI segmentation."""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.services.segmentation.vertex import VertexSegmentationError

logger = logging.getLogger(__name__)

_sam2_predictor = None


class SAM2RefinementError(RuntimeError):
    """SAM2 refinement failed — job must fail; no fallback permitted."""


def _resolve_sam2_config() -> str:
    import sam2  # type: ignore[import-untyped]
    from pathlib import Path

    base = Path(sam2.__file__).resolve().parent
    candidate = base / settings.sam2_model_config
    if candidate.is_file():
        return str(candidate)
    alt = base / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml"
    if alt.is_file():
        return str(alt)
    raise SAM2RefinementError(f"SAM2 config not found: {settings.sam2_model_config}")


def ensure_sam2_loaded() -> None:
    """Load SAM2 predictor — used by startup validation."""
    _get_sam2_predictor()


def _get_sam2_predictor():
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor

    try:
        import torch  # type: ignore[import-untyped]
        from sam2.build_sam import build_sam2  # type: ignore[import-untyped]
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import-untyped]

        checkpoint = settings.sam2_checkpoint_path
        if not checkpoint:
            from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

            logger.info("Downloading SAM2 checkpoint %s", settings.sam2_model_id)
            checkpoint = hf_hub_download(
                repo_id=settings.sam2_model_id,
                filename=settings.sam2_checkpoint_filename,
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cfg = _resolve_sam2_config()
        logger.info("Loading SAM2 model=%s device=%s", model_cfg, device)
        sam2 = build_sam2(model_cfg, checkpoint, device=device)
        _sam2_predictor = SAM2ImagePredictor(sam2)
        logger.info("SAM2 model loaded")
    except SAM2RefinementError:
        raise
    except Exception as exc:
        raise SAM2RefinementError(f"SAM2 model failed to load: {exc}") from exc

    return _sam2_predictor


def _bootstrap_points_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    binary = mask > 0.5
    ys, xs = np.where(binary)
    if len(xs) == 0:
        raise SAM2RefinementError(
            "Cannot refine mask: Vertex AI mask contains no foreground pixels."
        )

    cx, cy = int(xs.mean()), int(ys.mean())
    bottom_y = int(np.percentile(ys, 92))
    bottom_xs = xs[ys >= bottom_y - 2]
    if len(bottom_xs) >= 2:
        lx, rx = int(np.percentile(bottom_xs, 20)), int(np.percentile(bottom_xs, 80))
        points = np.array([[cx, cy], [lx, bottom_y], [rx, bottom_y]])
        labels = np.array([1, 1, 1])
    else:
        points = np.array([[cx, cy]])
        labels = np.array([1])
    return points, labels


def refine_mask_sam2(image: Image.Image, vertex_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Refine the Vertex AI mask using SAM2 box + point prompts.

    Raises SAM2RefinementError on failure — never returns the unrefined mask.
    """
    try:
        predictor = _get_sam2_predictor()
    except SAM2RefinementError:
        raise
    except Exception as exc:
        raise SAM2RefinementError(f"SAM2 predictor unavailable: {exc}") from exc

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = mask_bbox(vertex_mask)

    pad_x = int((x1 - x0) * 0.04) + 4
    pad_y = int((y1 - y0) * 0.04) + 4
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)
    box = np.array([x0, y0, x1, y1])

    points, point_labels = _bootstrap_points_from_mask(vertex_mask)

    try:
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
    except Exception as exc:
        raise SAM2RefinementError(f"SAM2 prediction failed: {exc}") from exc

    if len(masks) == 0 or len(scores) == 0:
        raise SAM2RefinementError("SAM2 returned no masks.")

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(np.float32)
    best_score = float(scores[best_idx])

    binary = (best_mask > 0.5).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    refined = binary.astype(np.float32) / 255.0

    if not (refined > 0.5).any():
        raise SAM2RefinementError("SAM2 produced an empty refined mask.")

    logger.info(
        "SAM2 refinement: score=%.3f coverage=%.1f%%",
        best_score,
        float((refined > 0.5).mean()) * 100,
    )
    return refined, best_score
