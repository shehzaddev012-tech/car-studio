"""
Celery task: process_job

Runs the full AI pipeline for a single job.  The task is idempotent:
re-queuing after a transient failure will re-read the original image from
storage and reprocess from scratch, writing to the same output key.

Error taxonomy (determines retry vs. immediate failure):
  ValueError      → business-logic error (no car, QC fail) — fail immediately, no retry
  SoftTimeLimitExceeded → job exceeded timeout — fail immediately
  RuntimeError    → pipeline infrastructure error — retry up to max_retries
  Exception       → unexpected error — retry up to max_retries
"""
from __future__ import annotations

import json
import logging

import redis
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from app.config import settings
from app.db.database import SessionLocal
from app.db.repository import JobRepository
from app.services.ai_background import get_ai_background_service
from app.services.storage import get_storage
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _publish(job_id: str, status: str, progress: float, message: str = "", **extra: object) -> None:
    """Publish a job progress event to the Redis pub/sub channel for this job."""
    payload = {
        "jobId": job_id,
        "status": status,
        "progressPercent": progress,
        "message": message,
        **extra,
    }
    try:
        r = redis.from_url(settings.redis_url)
        r.publish(f"job:{job_id}", json.dumps(payload))
        r.close()
    except Exception as exc:
        # Never let pub/sub failure abort the pipeline
        logger.warning("Failed to publish progress for job %s: %s", job_id, exc)


@celery_app.task(
    bind=True,
    name="app.workers.tasks.process_job",
    max_retries=3,
    default_retry_delay=8,
    acks_late=True,
)
def process_job(self: Task, job_id: str) -> dict:
    """
    Process a car photo through the studio background pipeline.

    Publishes incremental progress to Redis pub/sub so the FastAPI WebSocket
    handler can forward updates to connected browser clients in real time.
    """
    db = SessionLocal()
    storage = get_storage()
    repo = JobRepository(db)

    try:
        # ── Guard: skip if already in a terminal state (idempotency) ──────────
        job = repo.get(job_id)
        if job is None:
            logger.error("process_job: job %s not found", job_id)
            return {"status": "not_found"}

        if job.status in ("completed", "failed"):
            logger.info("process_job: job %s already %s — skipping", job_id, job.status)
            return {"status": job.status}

        # ── Mark as processing ────────────────────────────────────────────────
        repo.update_status(job_id, "processing", progress=0)
        _publish(job_id, "processing", 0, "Pipeline started")
        logger.info("[%s] Pipeline started", job_id)

        # ── Load original image ───────────────────────────────────────────────
        _publish(job_id, "processing", 5, "Loading image from storage")
        image_bytes = storage.get(job.original_image_key)

        # ── Run AI pipeline ───────────────────────────────────────────────────
        service = get_ai_background_service()

        def on_progress(percent: int, message: str) -> None:
            # Map service progress (0-100) to task progress (5-95)
            task_pct = 5 + int(percent * 0.90)
            repo.update_status(job_id, "processing", progress=task_pct)
            _publish(job_id, "processing", task_pct, message)

        result = service.process(image_bytes, on_progress=on_progress)

        # ── Handle pipeline result ────────────────────────────────────────────
        if not result.success:
            error = result.error or "Pipeline failed for an unknown reason."
            repo.update_status(job_id, "failed", error_message=error)
            _publish(job_id, "failed", 0, error, errorMessage=error)
            logger.warning("[%s] Pipeline failed (non-retryable): %s", job_id, error)
            return {"status": "failed", "error": error}

        # ── Save result ───────────────────────────────────────────────────────
        repo.update_status(job_id, "processing", progress=96)
        _publish(job_id, "processing", 96, "Saving result image")

        result_key = f"results/{job_id}.jpg"
        storage.put(result_key, result.image_bytes, content_type="image/jpeg")

        # ── Complete ──────────────────────────────────────────────────────────
        repo.update_status(
            job_id, "completed",
            progress=100,
            result_image_key=result_key,
            ssim_score=result.ssim_score,
        )
        result_url = f"/api/jobs/{job_id}/download"
        _publish(
            job_id, "completed", 100,
            (
                f"Done (SSIM {result.ssim_score:.3f}, LPIPS {result.lpips_score:.3f})"
                if result.ssim_score and result.lpips_score
                else f"Done (SSIM {result.ssim_score:.3f})"
                if result.ssim_score
                else "Done"
            ),
            resultImageUrl=result_url,
        )
        logger.info("[%s] Pipeline complete. SSIM=%.4f", job_id, result.ssim_score or 0)
        return {"status": "completed", "ssim": result.ssim_score}

    except SoftTimeLimitExceeded:
        error = (
            f"Job timed out after {settings.job_timeout_seconds}s. "
            "The AI pipeline took longer than the allowed limit."
        )
        repo.update_status(job_id, "failed", error_message=error)
        _publish(job_id, "failed", 0, error, errorMessage=error)
        logger.error("[%s] Timeout", job_id)
        return {"status": "failed", "error": error}

    except Exception as exc:
        error = str(exc)
        logger.exception("[%s] Unexpected error: %s", job_id, error)

        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            final_error = f"Processing failed after {self.max_retries} retries: {error}"
            repo.update_status(job_id, "failed", error_message=final_error)
            _publish(job_id, "failed", 0, final_error, errorMessage=final_error)
            return {"status": "failed", "error": final_error}

    finally:
        db.close()
