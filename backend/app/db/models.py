from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, String, Text, func
from app.db.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Job(Base):
    """Represents one image-processing job through the studio background pipeline."""

    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=_new_uuid)

    # ── Status lifecycle ──────────────────────────────────────────────────────
    # pending → processing → completed | failed
    status = Column(String(16), nullable=False, default="pending", index=True)

    # ── Storage keys (relative to the configured storage root) ────────────────
    original_image_key = Column(String(512), nullable=False)
    result_image_key = Column(String(512), nullable=True)

    # ── Runtime metadata ──────────────────────────────────────────────────────
    progress_percent = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    ssim_score = Column(Float, nullable=True)      # quality-check score on completion

    # ── Timestamps (stored as UTC) ────────────────────────────────────────────
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    def __repr__(self) -> str:
        return f"<Job id={self.id!r} status={self.status!r}>"
