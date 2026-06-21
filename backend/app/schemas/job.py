from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


# ── Serialised API shape ───────────────────────────────────────────────────────

class JobOut(BaseModel):
    """Wire-format representation returned by all job endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str                        # pending | processing | completed | failed
    originalImageUrl: str              # served via GET /api/jobs/{id}/original
    resultImageUrl: Optional[str]      # served via GET /api/jobs/{id}/download
    errorMessage: Optional[str]
    progressPercent: Optional[float]
    createdAt: datetime
    updatedAt: datetime

    @classmethod
    def from_orm_with_urls(cls, job) -> "JobOut":
        # Use relative paths so the browser resolves them against its own origin
        # (localhost:5173), not against the Docker-internal api:8000 hostname.
        return cls(
            id=job.id,
            status=job.status,
            originalImageUrl=f"/api/jobs/{job.id}/original",
            resultImageUrl=(
                f"/api/jobs/{job.id}/download"
                if job.result_image_key
                else None
            ),
            errorMessage=job.error_message,
            progressPercent=job.progress_percent,
            createdAt=job.created_at,
            updatedAt=job.updated_at,
        )


class JobListOut(BaseModel):
    jobs: list[JobOut]
    total: int


class JobProgressEvent(BaseModel):
    """Payload published to Redis pub/sub and forwarded over WebSocket."""

    jobId: str
    status: str
    progressPercent: Optional[float]
    message: str = ""
    resultImageUrl: Optional[str] = None
    errorMessage: Optional[str] = None
