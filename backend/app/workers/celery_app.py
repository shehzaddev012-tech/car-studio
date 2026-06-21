"""
Celery application configuration.

Design decisions:
  - ``task_acks_late=True``: the task is acknowledged only *after* it completes,
    so a worker crash mid-task will re-queue the message (idempotent retry).
  - ``worker_prefetch_multiplier=1``: each worker process holds at most one task
    at a time.  AI processing is CPU-bound; prefetching extra tasks would starve
    memory without improving throughput.
  - ``task_time_limit`` / ``task_soft_time_limit``: hard kill and graceful shutdown
    thresholds prevent hung AI tasks from blocking workers forever.
"""
from __future__ import annotations

from celery import Celery
from celery.signals import worker_process_init

from app.config import settings
from app.startup_validation import StartupValidationError, validate_dealership_pipeline

celery_app = Celery(
    "car_studio",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Safety limits
    task_time_limit=settings.job_timeout_seconds,
    task_soft_time_limit=settings.job_timeout_seconds - 30,
    # Retry / ack behaviour
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    # Keep results for 24 h (used for Celery result backend; job status is in the DB)
    result_expires=86_400,
)


@worker_process_init.connect
def _validate_worker_on_start(**_kwargs: object) -> None:
    """Each worker process must pass the same strict validation as the API."""
    try:
        validate_dealership_pipeline(context="celery-worker")
    except StartupValidationError as exc:
        raise RuntimeError(f"Worker startup validation failed: {exc}") from exc
