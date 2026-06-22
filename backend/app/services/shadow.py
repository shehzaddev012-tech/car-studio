"""Realistic ground-contact shadow for studio vehicle compositing."""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def generate_shadow(mask: Image.Image, width: int, height: int) -> Image.Image:
    """
    Vehicle footprint shadow with strong tire contact and progressive falloff.

    Layers:
      1. Full footprint — perspective-compressed vehicle silhouette on floor
      2. Contact band — high-opacity strip at tire line
      3. Multi-scale blur — tight under body, wide outward diffusion
    """
    mask_arr = np.array(mask.resize((width, height), Image.LANCZOS), dtype=np.float32) / 255.0
    binary = mask_arr > 0.1
    if not binary.any():
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))

    ys, xs = np.where(binary)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    car_h = y1 - y0
    car_w = x1 - x0

    floor_y = int(height * 0.62)

    # ── 1. Footprint: squash lower 45 % of silhouette onto floor plane ────────
    footprint = np.zeros((height, width), dtype=np.float32)
    contact_start = y0 + int(car_h * 0.55)
    contact_zone = mask_arr.copy()
    contact_zone[:contact_start, :] = 0

    shadow_depth = max(int(car_h * 0.12), 14)
    sil_crop = contact_zone[y0 : y1 + 1, x0 : x1 + 1]
    if sil_crop.size == 0:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))

    sil_img = Image.fromarray((sil_crop * 255).astype(np.uint8), "L")
    squashed = np.array(
        sil_img.resize((car_w, shadow_depth), Image.LANCZOS),
        dtype=np.float32,
    ) / 255.0

    paste_y = min(max(y1 - shadow_depth // 3, floor_y - shadow_depth), height - shadow_depth)
    paste_y = max(paste_y, 0)
    footprint[paste_y : paste_y + shadow_depth, x0 : x0 + car_w] = squashed

    # ── 2. Contact shadow — dark band directly under tires ────────────────────
    contact = np.zeros((height, width), dtype=np.float32)
    tire_row = y1
    tire_band_h = max(int(car_h * 0.04), 6)
    tire_slice = mask_arr[max(tire_row - tire_band_h, 0) : tire_row + 1, x0 : x1 + 1]
    if tire_slice.size:
        tire_profile = tire_slice.max(axis=0)
        contact_h = max(int(car_h * 0.025), 4)
        cy = min(tire_row + 2, height - contact_h - 1)
        contact[cy : cy + contact_h, x0 : x0 + len(tire_profile)] = tire_profile * 1.0

    # ── 3. Progressive blur (contact → ambient) ───────────────────────────────
    sigma_tight = max(car_w * 0.008, 1.5)
    sigma_mid = max(car_w * 0.035, 5.0)
    sigma_wide = max(car_w * 0.12, 12.0)

    layer_tight = cv2.GaussianBlur(footprint + contact * 1.4, (0, 0), sigmaX=sigma_tight, sigmaY=sigma_tight * 0.35)
    layer_mid = cv2.GaussianBlur(footprint, (0, 0), sigmaX=sigma_mid, sigmaY=sigma_mid * 0.4)
    layer_wide = cv2.GaussianBlur(footprint, (0, 0), sigmaX=sigma_wide, sigmaY=sigma_wide * 0.45)

    combined = layer_tight * 0.45 + layer_mid * 0.35 + layer_wide * 0.20

    # Opacity falloff: stronger near contact, lighter outward
    yy = np.arange(height, dtype=np.float32)
    dist_from_contact = np.clip(yy - paste_y, 0, height) / max(height - paste_y, 1)
    falloff = 1.0 - dist_from_contact * 0.55
    combined *= falloff[:, np.newaxis]

    # Light-direction bias: shadow slightly offset toward camera-right
    shift = max(int(car_w * 0.01), 2)
    combined = np.roll(combined, shift, axis=1)

    alpha = np.clip(combined * 0.72, 0, 1.0)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = 16
    rgba[:, :, 1] = 16
    rgba[:, :, 2] = 20
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)

    logger.info("generate_shadow: footprint shadow complete")
    return Image.fromarray(rgba, "RGBA")
