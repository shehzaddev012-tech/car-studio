"""
AI Car Studio Background Pipeline — individual processing steps.

Pipeline order (must not be reordered — accuracy depends on this sequence):
  1. segment_car            rembg U²-Net → RGBA car layer + L-mode alpha mask
  2. (car pixels preserved) The car_rgba layer is never modified after step 1
  3. generate_background    Deterministic PIL studio backdrop
  4. generate_shadow        Perspective-squashed silhouette + layered Gaussian blur
  5. composite_layers       background → shadow → car (PIL alpha_composite)
  6. glass_cleanup_stub     No-op stub; see docstring for production path
  7. quality_check          SSIM on masked car region only

Model choices & production upgrade notes are inline in each function.
"""
from __future__ import annotations

import io
import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

# Module-level rembg session cache — loaded once per worker process.
# new_session() downloads / deserialises the ~170 MB ONNX model; calling it
# per-job would double or triple peak RAM and cause OOM in constrained containers.
_rembg_session = None


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session  # type: ignore[import]
        logger.info("Loading rembg u2net model (first job in this process)…")
        _rembg_session = new_session("u2net")
        logger.info("rembg u2net model loaded")
    return _rembg_session


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Car Segmentation
# ──────────────────────────────────────────────────────────────────────────────

def segment_car(image: Image.Image, mock: bool = False) -> Tuple[Image.Image, Image.Image]:
    """
    Segment the car from its background, returning an RGBA car layer and an L-mode mask.

    Production model:  rembg with u2net (170 MB, cached after first run).
    Production upgrade: SAM 2 (Meta) + a "vehicle" category prompt gives sub-pixel
                        accuracy on complex edges (mirrors, antennae, spoilers).
                        Swap by replacing this function's body with a SAM 2 call and
                        keeping the same (car_rgba, mask) return contract.

    Mock mode (MOCK_AI=true):  uses OpenCV GrabCut for a rough cut — good enough
                               to test the full pipeline without downloading models.
    """
    if mock:
        return _grabcut_segment(image)

    try:
        from rembg import remove  # type: ignore[import]
    except ImportError:
        raise RuntimeError(
            "rembg is not installed. Run: pip install 'rembg[gpu]'  (or rembg for CPU)."
        )

    session = _get_rembg_session()

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    output_bytes = remove(
        buf.getvalue(),
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    )

    car_rgba = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    mask = car_rgba.split()[3]  # alpha channel

    _validate_mask(mask)
    logger.info("segment_car: rembg complete")
    return car_rgba, mask


def _grabcut_segment(image: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """
    Mock segmentation via OpenCV GrabCut.
    Assumes the car occupies roughly the central 70 % of the frame.
    """
    cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
    mask_gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(cv_img, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg_pil = Image.fromarray(fg_mask, "L").filter(ImageFilter.GaussianBlur(radius=3))

    car_rgba = image.convert("RGBA")
    car_rgba.putalpha(fg_pil)

    _validate_mask(fg_pil)
    logger.info("segment_car: GrabCut (mock) complete")
    return car_rgba, fg_pil


def _validate_mask(mask: Image.Image) -> None:
    arr = np.array(mask)
    coverage = (arr > 10).sum() / arr.size
    if coverage < 0.02:
        raise ValueError(
            "No car detected in the image. "
            "Ensure the photo shows a clear vehicle from a front, side, or 3/4 angle "
            "and that the car is not obscured or cropped."
        )
    if coverage > 0.97:
        raise ValueError(
            "Segmentation failed: the entire image was detected as foreground. "
            "The car may be too close to the frame edges, or the background colour "
            "is too similar to the car's body."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Deterministic Studio Background
# ──────────────────────────────────────────────────────────────────────────────

def generate_studio_background(width: int, height: int) -> Image.Image:
    """
    Generate a deterministic automotive studio backdrop — identical output every call.

    Visual recipe (matches standard dealership studio photography):
      • Base tone:  neutral light grey #EBEBED (no colour cast)
      • Radial vignette: centre ~245,245,247 → edge ~195,195,197 (simulates soft-box from above)
      • "Infinity curve" at the wall/floor join: subtle darkened band, bell-curve profile
      • Floor zone (lower 35 %): slight warm-shift and brightness boost (simulates glossy floor)

    No generative AI involved — this is pure maths/PIL.
    Modify BACKDROP_* constants to tune the look without touching business logic.
    """
    BACKDROP_BASE = (235, 235, 237)
    VIGNETTE_STRENGTH = 0.22   # 0 = flat; 1 = very dark edges
    CURVE_BAND_PX = 80         # Height of the wall-floor "sweep" curve
    FLOOR_RATIO = 0.65         # Floor starts at this fraction of frame height
    FLOOR_WARMTH = 9.0         # How warm (red/green push) the floor gets at the bottom

    bg = np.full((height, width, 3), BACKDROP_BASE, dtype=np.float32)

    # ── Radial gradient ───────────────────────────────────────────────────────
    cx, cy = width / 2.0, height * 0.42  # Light source slightly above centre
    max_dist = np.sqrt((width / 2.0) ** 2 + (height * 0.42) ** 2)
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radial = np.clip(1.0 - (dist / (max_dist * 1.25)) * VIGNETTE_STRENGTH, 1.0 - VIGNETTE_STRENGTH, 1.0)

    bg[:, :, 0] = np.clip(bg[:, :, 0] * radial + 10 * radial, 0, 255)
    bg[:, :, 1] = np.clip(bg[:, :, 1] * radial + 10 * radial, 0, 255)
    bg[:, :, 2] = np.clip(bg[:, :, 2] * radial + 12 * radial, 0, 255)  # slight cool highlight

    # ── Infinity curve shadow band ────────────────────────────────────────────
    floor_start = int(height * FLOOR_RATIO)
    curve_start = max(0, floor_start - CURVE_BAND_PX // 2)
    curve_end = min(height, floor_start + CURVE_BAND_PX // 2)
    for row in range(curve_start, curve_end):
        t = (row - curve_start) / max(CURVE_BAND_PX, 1)
        shadow = float(np.sin(t * np.pi) * 14.0)
        bg[row, :, :] = np.clip(bg[row, :, :] - shadow, 0, 255)

    # ── Floor zone ────────────────────────────────────────────────────────────
    for row in range(floor_start, height):
        t = (row - floor_start) / max(height - floor_start, 1)
        bg[row, :, 0] = np.clip(bg[row, :, 0] + FLOOR_WARMTH * 1.3 * t, 0, 255)
        bg[row, :, 1] = np.clip(bg[row, :, 1] + FLOOR_WARMTH * t,       0, 255)
        bg[row, :, 2] = np.clip(bg[row, :, 2] - FLOOR_WARMTH * 0.3 * t, 0, 255)

    return Image.fromarray(bg.astype(np.uint8), "RGB")


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Shadow Synthesis
# ──────────────────────────────────────────────────────────────────────────────

def generate_shadow(mask: Image.Image, width: int, height: int) -> Image.Image:
    """
    Synthesize a soft contact shadow beneath the car.

    Technique: perspective-project the lower silhouette onto the floor plane,
    then apply three-layer Gaussian blur to simulate area-light diffusion.
    Three blur radii (wide/medium/tight) are blended to produce a shadow that
    is sharp directly under the car and diffuses outward — physically consistent
    with a large overhead soft-box.

    Production upgrade: a lightweight shadow-prediction network (e.g. ShadowNet or
    a fine-tuned pix2pix model) could generate more photo-realistic contact shadows
    that account for car undercarriage geometry. Swap this function body only.
    """
    mask_resized = np.array(mask.resize((width, height), Image.LANCZOS), dtype=np.float32)

    rows_with_car = np.any(mask_resized > 10, axis=1)
    cols_with_car = np.any(mask_resized > 10, axis=0)
    if not rows_with_car.any():
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))

    row_min, row_max = int(np.where(rows_with_car)[0][[0, -1]].tolist()[0]), \
                       int(np.where(rows_with_car)[0][[0, -1]].tolist()[-1])
    col_min = int(np.where(cols_with_car)[0][[0, -1]].tolist()[0])
    col_max = int(np.where(cols_with_car)[0][[0, -1]].tolist()[-1])

    car_h = row_max - row_min
    car_w = col_max - col_min

    # Keep only the bottom 35 % of the car silhouette (ground contact zone)
    contact_sil = mask_resized.copy()
    cutoff_row = row_min + int(car_h * 0.65)
    contact_sil[:cutoff_row, :] = 0

    # Squash to shadow thickness (~14 % of car height)
    shadow_h = max(int(car_h * 0.14), 12)
    sil_crop = contact_sil[row_min:row_max, col_min:col_max]
    sil_img = Image.fromarray(sil_crop.astype(np.uint8), "L")
    shadow_sil = sil_img.resize((car_w, shadow_h), Image.LANCZOS)

    # Place on canvas just below the car's bottom edge
    canvas = np.zeros((height, width), dtype=np.float32)
    paste_y = min(row_max - shadow_h // 2 + 4, height - shadow_h)
    paste_y = max(paste_y, 0)

    sil_arr = np.array(shadow_sil, dtype=np.float32)
    canvas[paste_y: paste_y + shadow_h, col_min: col_min + car_w] = sil_arr

    # Three-layer blur blend
    sigma_wide   = max(car_w * 0.10, 8)
    sigma_medium = max(car_w * 0.04, 4)
    sigma_tight  = max(car_w * 0.012, 1)

    layer_wide   = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sigma_wide,   sigmaY=sigma_wide   * 0.4)
    layer_medium = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sigma_medium, sigmaY=sigma_medium * 0.4)
    layer_tight  = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sigma_tight,  sigmaY=sigma_tight  * 0.4)

    combined = layer_wide * 0.35 + layer_medium * 0.40 + layer_tight * 0.25
    combined = np.clip(combined * 0.60, 0, 255)  # max ~60 % opacity

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = 18   # shadow colour — very dark cool grey
    rgba[:, :, 1] = 18
    rgba[:, :, 2] = 22
    rgba[:, :, 3] = combined.astype(np.uint8)

    logger.info("generate_shadow: complete")
    return Image.fromarray(rgba, "RGBA")


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Compositing
# ──────────────────────────────────────────────────────────────────────────────

def composite_layers(
    background: Image.Image,
    shadow: Image.Image,
    car_rgba: Image.Image,
) -> Image.Image:
    """
    Stack layers: background → shadow → original car pixels.

    The car_rgba layer is the untouched RGBA output from segment_car.
    PIL.Image.alpha_composite preserves the exact original car pixels —
    no blending, no colour shift.
    """
    result = background.convert("RGBA")
    result = Image.alpha_composite(result, shadow.convert("RGBA"))
    result = Image.alpha_composite(result, car_rgba)
    logger.info("composite_layers: complete")
    return result.convert("RGB")


# ──────────────────────────────────────────────────────────────────────────────
# Step 6 — Glass Cleanup (stub)
# ──────────────────────────────────────────────────────────────────────────────

def glass_cleanup_stub(image: Image.Image, car_rgba: Image.Image) -> Image.Image:  # noqa: ARG001
    """
    STUB — returns the composite unchanged.

    Production implementation path:
      1. Detect glass/window regions using a vehicle-part segmentation model
         (e.g. VehiclePartsNet, or a SAM 2 prompt with "windshield", "side window").
      2. Create a tight mask covering glass pixels only — NEVER the body panels.
      3. Run masked inpainting (e.g. SDXL-Inpaint or OpenCV's cv2.inpaint)
         restricted to the glass mask, removing unwanted reflections / dirt.
      4. Blend the inpainted patch back with a feathered mask boundary.

    The non-negotiable constraint: if this step runs, the quality_check SSIM gate
    in Step 7 must still pass — if the glass cleanup shifts the car region's SSIM
    below the threshold, the job is failed rather than returning a degraded result.
    """
    logger.info("glass_cleanup_stub: no-op (see pipeline.py for production notes)")
    return image


# ──────────────────────────────────────────────────────────────────────────────
# Step 7 — Quality / Consistency Gate
# ──────────────────────────────────────────────────────────────────────────────

def quality_check(
    original: Image.Image,
    result: Image.Image,
    mask: Image.Image,
    threshold: float = 0.82,
) -> Tuple[bool, float]:
    """
    Verify the car region in the output matches the original within tolerance.

    Uses SSIM (Structural Similarity Index) computed exclusively on the masked
    car region — background changes (by design) do not affect the score.

    Returns:
        (passed, ssim_score)  — passed=True when ssim_score >= threshold.

    Calibration notes:
      - 0.82 is a conservative default that catches real degradations (colour shifts,
        blurring, hallucinated details) while tolerating minor JPEG re-compression artefacts.
      - Reduce to ~0.75 if alpha-matting edge feathering is causing false failures.
      - Never set below 0.70 — the gate would become meaningless.
    """
    from skimage.metrics import structural_similarity  # type: ignore[import]

    # Resize original to match result dimensions (original and result are same size, but guard anyway)
    orig = original.convert("RGB")
    res = result.convert("RGB")
    if orig.size != res.size:
        orig = orig.resize(res.size, Image.LANCZOS)

    mask_resized = mask.resize(res.size, Image.LANCZOS).convert("L")

    orig_arr  = np.array(orig,  dtype=np.float32) / 255.0
    res_arr   = np.array(res,   dtype=np.float32) / 255.0
    mask_arr  = np.array(mask_resized, dtype=np.float32) / 255.0

    # Apply mask — only car pixels contribute to SSIM
    orig_masked = orig_arr * mask_arr[:, :, np.newaxis]
    res_masked  = res_arr  * mask_arr[:, :, np.newaxis]

    score = float(
        structural_similarity(
            orig_masked,
            res_masked,
            data_range=1.0,
            channel_axis=2,
            win_size=7,
        )
    )

    passed = score >= threshold
    logger.info(
        "quality_check: SSIM=%.4f  threshold=%.2f  → %s",
        score, threshold, "PASSED" if passed else "FAILED",
    )
    return passed, score
