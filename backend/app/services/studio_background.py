"""Premium automotive studio backdrop — dealership cyclorama style."""
from __future__ import annotations

import numpy as np
from PIL import Image


def generate_studio_background(width: int, height: int) -> Image.Image:
    """
    Deterministic premium studio environment.

    Visual spec (Carvana / OEM inventory style):
      • Neutral light grey wall (#ECECEE → #D8D8DC vertical gradient)
      • Seamless infinity curve — no visible wall/floor seam
      • Soft floor with subtle warm lift, no texture or reflections
      • Even overhead lighting — no vignette drama, no lens effects
    """
    # Wall: cool neutral grey, slightly brighter at top (overhead soft-box)
    wall_top = np.array([238.0, 238.0, 240.0], dtype=np.float32)
    wall_bottom = np.array([224.0, 224.0, 226.0], dtype=np.float32)

    # Floor: marginally warmer, very subtle gradient
    floor_near = np.array([226.0, 225.0, 224.0], dtype=np.float32)
    floor_far = np.array([218.0, 217.0, 216.0], dtype=np.float32)

    floor_line = int(height * 0.62)  # horizon / sweep line
    curve_half = max(int(height * 0.045), 24)

    bg = np.zeros((height, width, 3), dtype=np.float32)

    for row in range(height):
        if row < floor_line - curve_half:
            t = row / max(floor_line - curve_half, 1)
            bg[row, :, :] = wall_top * (1 - t) + wall_bottom * t
        elif row < floor_line + curve_half:
            # Bell-curve sweep — simulates seamless paper cyclorama bend
            t = (row - (floor_line - curve_half)) / max(2 * curve_half, 1)
            sweep = float(np.sin(t * np.pi) * 6.0)
            base = wall_bottom - sweep
            bg[row, :, :] = base
        else:
            t = (row - (floor_line + curve_half)) / max(height - floor_line - curve_half, 1)
            bg[row, :, :] = floor_near * (1 - t) + floor_far * t

    # Horizontal centre lift — very subtle, mimics even studio lighting
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    horizontal_lift = 1.0 + 0.018 * (1.0 - xx ** 2)
    bg *= horizontal_lift[np.newaxis, :, np.newaxis]

    return Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), "RGB")
