"""
Unit tests for JobRepository — covers all state transitions and edge cases.
"""
from __future__ import annotations

import uuid

import pytest

from app.db.repository import JobRepository


class TestJobRepositoryCreate:
    def test_creates_job_with_pending_status(self, db_session):
        repo = JobRepository(db_session)
        job_id = str(uuid.uuid4())
        job = repo.create(job_id=job_id, original_image_key="uploads/test.jpg")

        assert job.id == job_id
        assert job.status == "pending"
        assert job.original_image_key == "uploads/test.jpg"
        assert job.result_image_key is None
        assert job.error_message is None
        assert job.progress_percent is None
        assert job.created_at is not None
        assert job.updated_at is not None

    def test_get_returns_created_job(self, db_session):
        repo = JobRepository(db_session)
        job_id = str(uuid.uuid4())
        repo.create(job_id=job_id, original_image_key="uploads/x.jpg")
        fetched = repo.get(job_id)
        assert fetched is not None
        assert fetched.id == job_id

    def test_get_returns_none_for_unknown_id(self, db_session):
        repo = JobRepository(db_session)
        assert repo.get("does-not-exist") is None


class TestJobRepositoryStatusTransitions:
    @pytest.fixture()
    def job(self, db_session):
        repo = JobRepository(db_session)
        return repo.create(job_id=str(uuid.uuid4()), original_image_key="uploads/car.jpg")

    def test_pending_to_processing(self, db_session, job):
        repo = JobRepository(db_session)
        updated = repo.update_status(job.id, "processing", progress=0)
        assert updated.status == "processing"
        assert updated.progress_percent == 0

    def test_processing_to_completed(self, db_session, job):
        repo = JobRepository(db_session)
        repo.update_status(job.id, "processing", progress=50)
        done = repo.update_status(
            job.id, "completed",
            progress=100,
            result_image_key="results/car.jpg",
            ssim_score=0.95,
        )
        assert done.status == "completed"
        assert done.result_image_key == "results/car.jpg"
        assert done.ssim_score == pytest.approx(0.95)

    def test_pending_to_failed_with_message(self, db_session, job):
        repo = JobRepository(db_session)
        failed = repo.update_status(job.id, "failed", error_message="No car detected")
        assert failed.status == "failed"
        assert "No car" in failed.error_message

    def test_cancel_pending_job(self, db_session, job):
        repo = JobRepository(db_session)
        cancelled = repo.cancel(job.id)
        assert cancelled is not None
        assert cancelled.status == "failed"
        assert "Cancelled" in cancelled.error_message

    def test_cancel_non_pending_job_returns_none(self, db_session, job):
        repo = JobRepository(db_session)
        repo.update_status(job.id, "processing", progress=10)
        result = repo.cancel(job.id)
        assert result is None  # can only cancel pending jobs

    def test_update_nonexistent_job_returns_none(self, db_session):
        repo = JobRepository(db_session)
        assert repo.update_status("ghost-id", "processing") is None


class TestJobRepositoryGetMany:
    def test_get_many_returns_all_matching(self, db_session):
        repo = JobRepository(db_session)
        ids = [str(uuid.uuid4()) for _ in range(3)]
        for i, jid in enumerate(ids):
            repo.create(job_id=jid, original_image_key=f"uploads/{i}.jpg")

        results = repo.get_many(ids)
        assert len(results) == 3
        assert {j.id for j in results} == set(ids)

    def test_get_many_empty_list(self, db_session):
        repo = JobRepository(db_session)
        assert repo.get_many([]) == []

    def test_get_many_ignores_unknown_ids(self, db_session):
        repo = JobRepository(db_session)
        jid = str(uuid.uuid4())
        repo.create(job_id=jid, original_image_key="uploads/a.jpg")
        results = repo.get_many([jid, "unknown-id"])
        assert len(results) == 1


class TestJobRepositoryDelete:
    def test_delete_existing_job(self, db_session):
        repo = JobRepository(db_session)
        jid = str(uuid.uuid4())
        repo.create(job_id=jid, original_image_key="uploads/del.jpg")
        assert repo.delete(jid) is True
        assert repo.get(jid) is None

    def test_delete_nonexistent_returns_false(self, db_session):
        repo = JobRepository(db_session)
        assert repo.delete("ghost") is False
