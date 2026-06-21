from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy.orm import Session

from app.db.models import Job


class JobRepository:
    """
    Data-access layer for Job records.

    All write methods call session.commit() — callers are responsible for
    passing a session that was created *for this transaction* (i.e. the FastAPI
    ``get_db`` dependency or a fresh ``SessionLocal()`` in Celery workers).

    The interface is intentionally storage-agnostic: swapping to Postgres only
    requires changing the engine URL in config, not this class.
    """

    def __init__(self, db: Session) -> None:
        self._db = db

    # ── Reads ──────────────────────────────────────────────────────────────────

    def get(self, job_id: str) -> Optional[Job]:
        return self._db.get(Job, job_id)

    def get_many(self, job_ids: Sequence[str]) -> list[Job]:
        if not job_ids:
            return []
        return self._db.query(Job).filter(Job.id.in_(job_ids)).all()

    def list_pending(self) -> list[Job]:
        return self._db.query(Job).filter(Job.status == "pending").all()

    # ── Writes ─────────────────────────────────────────────────────────────────

    def create(self, job_id: str, original_image_key: str) -> Job:
        job = Job(id=job_id, original_image_key=original_image_key, status="pending")
        self._db.add(job)
        self._db.commit()
        self._db.refresh(job)
        return job

    def update_status(
        self,
        job_id: str,
        status: str,
        *,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
        result_image_key: Optional[str] = None,
        ssim_score: Optional[float] = None,
    ) -> Optional[Job]:
        job = self._db.get(Job, job_id)
        if job is None:
            return None

        job.status = status
        job.updated_at = datetime.now(timezone.utc)

        if progress is not None:
            job.progress_percent = progress
        if error_message is not None:
            job.error_message = error_message
        if result_image_key is not None:
            job.result_image_key = result_image_key
        if ssim_score is not None:
            job.ssim_score = ssim_score

        self._db.commit()
        self._db.refresh(job)
        return job

    def cancel(self, job_id: str) -> Optional[Job]:
        """Mark a *pending* job as failed/cancelled. Returns None if not in pending state."""
        job = self._db.get(Job, job_id)
        if job is None or job.status != "pending":
            return None
        return self.update_status(job_id, "failed", error_message="Cancelled by user")

    def delete(self, job_id: str) -> bool:
        job = self._db.get(Job, job_id)
        if job is None:
            return False
        self._db.delete(job)
        self._db.commit()
        return True
