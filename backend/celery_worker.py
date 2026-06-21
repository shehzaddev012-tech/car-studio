"""
Celery worker entry point.

Start with:
    celery -A celery_worker.celery_app worker --loglevel=info --concurrency=2

The ``--concurrency=2`` flag runs 2 parallel worker processes.  Each process
holds one AI task at a time (``worker_prefetch_multiplier=1`` in config).
Increase concurrency only if your machine has enough RAM for multiple SAM2 sessions.
"""
from app.workers.celery_app import celery_app  # noqa: F401 — re-export for Celery CLI

if __name__ == "__main__":
    celery_app.start()
