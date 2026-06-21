"""
Job endpoints — implements the full API contract:

  POST   /api/jobs                → upload image(s), create jobs, return immediately
  GET    /api/jobs/:id            → single job status
  GET    /api/jobs?ids=...        → batch status
  DELETE /api/jobs/:id            → cancel pending job
  GET    /api/jobs/:id/original   → stream original image
  GET    /api/jobs/:id/download   → stream processed result image
"""
from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.db.repository import JobRepository
from app.schemas.job import JobListOut, JobOut
from app.services.storage import get_storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/jobs", tags=["jobs"])

ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp",
}


def _enqueue(job_id: str) -> None:
    """
    Submit a job to the Celery queue.
    Imported lazily so the API layer starts even when Redis is down
    (the import succeeds; the actual publish raises a connection error).
    """
    try:
        from app.workers.tasks import process_job  # type: ignore[import]
        process_job.delay(job_id)
    except Exception as exc:
        logger.error("Failed to enqueue job %s: %s", job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The job queue is currently unavailable. "
                "This is likely a transient infrastructure issue — please retry in a few seconds."
            ),
        ) from exc


# ── POST /api/jobs ─────────────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED, response_model=JobListOut)
async def create_jobs(
    files: List[UploadFile] = File(..., description="One or more car photos (JPEG / PNG / WebP)"),
    db: Session = Depends(get_db),
) -> JobListOut:
    """
    Upload one or more images and create an independent processing job for each.

    Returns immediately — AI processing happens asynchronously in the worker pool.
    Poll GET /api/jobs?ids=... or subscribe via WebSocket for status updates.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    storage = get_storage()
    repo = JobRepository(db)
    created: list[JobOut] = []

    for upload in files:
        # ── Validate ───────────────────────────────────────────────────────────
        if upload.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{upload.content_type}' for '{upload.filename}'. "
                    f"Accepted: JPEG, PNG, WebP."
                ),
            )

        data = await upload.read()

        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"'{upload.filename}' exceeds the {settings.max_upload_size_mb} MB limit "
                    f"({len(data) / 1_048_576:.1f} MB uploaded)."
                ),
            )

        if len(data) < 1_024:
            raise HTTPException(
                status_code=400,
                detail=f"'{upload.filename}' appears to be empty or corrupt (< 1 KB).",
            )

        # ── Persist ────────────────────────────────────────────────────────────
        job_id = str(uuid.uuid4())
        ext = (upload.filename or "image.jpg").rsplit(".", 1)[-1].lower()
        if ext not in settings.allowed_extensions:
            ext = "jpg"
        original_key = f"uploads/{job_id}.{ext}"

        storage.put(original_key, data, content_type=upload.content_type or "image/jpeg")
        job = repo.create(job_id=job_id, original_image_key=original_key)
        logger.info("Created job %s for '%s' (%d bytes)", job_id, upload.filename, len(data))

        # ── Enqueue ────────────────────────────────────────────────────────────
        _enqueue(job_id)

        created.append(JobOut.from_orm_with_urls(job))

    return JobListOut(jobs=created, total=len(created))


# ── GET /api/jobs/:id ──────────────────────────────────────────────────────────

@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    repo = JobRepository(db)
    job = repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobOut.from_orm_with_urls(job)


# ── GET /api/jobs?ids=... ──────────────────────────────────────────────────────

@router.get("", response_model=JobListOut)
def list_jobs(
    ids: Optional[str] = Query(None, description="Comma-separated job IDs"),
    db: Session = Depends(get_db),
) -> JobListOut:
    if not ids:
        raise HTTPException(status_code=400, detail="Provide at least one job ID via ?ids=id1,id2")
    repo = JobRepository(db)
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    jobs = repo.get_many(id_list)
    return JobListOut(
        jobs=[JobOut.from_orm_with_urls(j) for j in jobs],
        total=len(jobs),
    )


# ── DELETE /api/jobs/:id ───────────────────────────────────────────────────────

@router.delete("/{job_id}", status_code=status.HTTP_200_OK)
def cancel_job(job_id: str, db: Session = Depends(get_db)) -> dict:
    repo = JobRepository(db)
    job = repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status not in ("pending",):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job '{job_id}' is in status '{job.status}' and cannot be cancelled. "
                "Only pending jobs can be cancelled."
            ),
        )
    repo.cancel(job_id)
    logger.info("Cancelled job %s", job_id)
    return {"cancelled": True, "jobId": job_id}


# ── GET /api/jobs/:id/original ────────────────────────────────────────────────

@router.get("/{job_id}/original")
def get_original_image(job_id: str, db: Session = Depends(get_db)) -> Response:
    repo = JobRepository(db)
    job = repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    storage = get_storage()
    try:
        data = storage.get(job.original_image_key)
    except KeyError:
        raise HTTPException(status_code=404, detail="Original image file not found in storage.")
    ext = job.original_image_key.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    return Response(content=data, media_type=mime, headers={"Cache-Control": "private, max-age=3600"})


# ── GET /api/jobs/:id/download ────────────────────────────────────────────────

@router.get("/{job_id}/download")
def download_result(job_id: str, db: Session = Depends(get_db)) -> Response:
    repo = JobRepository(db)
    job = repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is not yet completed (current status: {job.status}).",
        )
    if not job.result_image_key:
        raise HTTPException(status_code=404, detail="Result image not found for this job.")
    storage = get_storage()
    try:
        data = storage.get(job.result_image_key)
    except KeyError:
        raise HTTPException(status_code=404, detail="Result image file not found in storage.")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="studio_{job_id}.jpg"',
            "Cache-Control": "private, max-age=86400",
        },
    )
