"""Unit tests for vehicle preservation policy."""
from __future__ import annotations

import numpy as np
import pytest

from app.services.vehicle_preservation import (
    REJECTION_MESSAGE,
    validate_vehicle_preservation,
)


def _solid_mask(h: int, w: int, fill: float = 1.0) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    m[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = fill
    return m


def test_preservation_passes_when_masks_agree():
    vertex = _solid_mask(400, 600)
    report = validate_vehicle_preservation(
        vertex_mask=vertex,
        sam2_mask=vertex.copy(),
        merged_mask=vertex.copy(),
        final_mask=vertex.copy(),
        vertex_confidence=0.85,
        sam2_confidence=0.90,
    )
    assert report.passed is True
    assert report.user_message is None


def test_preservation_fails_when_final_loses_wheels():
    vertex = _solid_mask(400, 600)
    # Aggressively erode final mask — simulates lost wheel zones
    final = vertex.copy()
    final[int(400 * 0.68) :, :] = 0.0
    report = validate_vehicle_preservation(
        vertex_mask=vertex,
        sam2_mask=vertex.copy(),
        merged_mask=vertex.copy(),
        final_mask=final,
        vertex_confidence=0.85,
        sam2_confidence=0.90,
    )
    assert report.passed is False
    assert report.user_message == REJECTION_MESSAGE
    assert len(report.internal_reasons) > 0


def test_preservation_fails_on_low_vertex_confidence():
    vertex = _solid_mask(200, 300)
    report = validate_vehicle_preservation(
        vertex_mask=vertex,
        sam2_mask=vertex.copy(),
        merged_mask=vertex.copy(),
        final_mask=vertex.copy(),
        vertex_confidence=0.30,
        sam2_confidence=0.90,
    )
    assert report.passed is False
    assert report.user_message == REJECTION_MESSAGE
