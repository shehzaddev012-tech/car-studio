"""
Integration tests for the upload → job creation flow.

Uses FastAPI TestClient + in-memory SQLite (via conftest fixtures).
The Celery task is patched to avoid needing a running Redis/worker.
"""
from __future__ import annotations

import io
import uuid
from unittest.mock import MagicMock, patch

import pytest


class TestCreateJobs:
    def test_upload_single_image_returns_201(self, test_app, sample_jpeg):
        with patch("app.api.routes.jobs._enqueue"):
            resp = test_app.post(
                "/api/jobs",
                files=[("files", ("car.jpg", sample_jpeg, "image/jpeg"))],
            )
        assert resp.status_code == 201
        body = resp.json()
        assert body["total"] == 1
        assert len(body["jobs"]) == 1
        job = body["jobs"][0]
        assert job["status"] == "pending"
        assert "id" in job
        assert job["originalImageUrl"].endswith("/original")
        assert job["resultImageUrl"] is None

    def test_upload_multiple_images_creates_multiple_jobs(self, test_app, sample_jpeg, sample_png):
        with patch("app.api.routes.jobs._enqueue"):
            resp = test_app.post(
                "/api/jobs",
                files=[
                    ("files", ("a.jpg", sample_jpeg, "image/jpeg")),
                    ("files", ("b.png", sample_png, "image/png")),
                ],
            )
        assert resp.status_code == 201
        body = resp.json()
        assert body["total"] == 2
        ids = {j["id"] for j in body["jobs"]}
        assert len(ids) == 2  # distinct job IDs

    def test_upload_unsupported_file_type_returns_400(self, test_app):
        with patch("app.api.routes.jobs._enqueue"):
            resp = test_app.post(
                "/api/jobs",
                files=[("files", ("doc.pdf", b"%PDF-1.4", "application/pdf"))],
            )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_upload_empty_file_returns_400(self, test_app):
        with patch("app.api.routes.jobs._enqueue"):
            resp = test_app.post(
                "/api/jobs",
                files=[("files", ("empty.jpg", b"", "image/jpeg"))],
            )
        assert resp.status_code == 400

    def test_upload_no_files_returns_422(self, test_app):
        resp = test_app.post("/api/jobs")
        assert resp.status_code == 422


class TestGetJob:
    def test_get_existing_job(self, test_app, sample_jpeg):
        with patch("app.api.routes.jobs._enqueue"):
            create_resp = test_app.post(
                "/api/jobs",
                files=[("files", ("car.jpg", sample_jpeg, "image/jpeg"))],
            )
        job_id = create_resp.json()["jobs"][0]["id"]

        resp = test_app.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == job_id
        assert resp.json()["status"] == "pending"

    def test_get_nonexistent_job_returns_404(self, test_app):
        resp = test_app.get(f"/api/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404


class TestListJobs:
    def test_list_by_ids(self, test_app, sample_jpeg):
        with patch("app.api.routes.jobs._enqueue"):
            r1 = test_app.post("/api/jobs", files=[("files", ("a.jpg", sample_jpeg, "image/jpeg"))])
            r2 = test_app.post("/api/jobs", files=[("files", ("b.jpg", sample_jpeg, "image/jpeg"))])
        id1 = r1.json()["jobs"][0]["id"]
        id2 = r2.json()["jobs"][0]["id"]

        resp = test_app.get(f"/api/jobs?ids={id1},{id2}")
        assert resp.status_code == 200
        returned_ids = {j["id"] for j in resp.json()["jobs"]}
        assert {id1, id2} == returned_ids

    def test_list_missing_ids_returns_400(self, test_app):
        resp = test_app.get("/api/jobs")
        assert resp.status_code == 400


class TestCancelJob:
    def test_cancel_pending_job(self, test_app, sample_jpeg):
        with patch("app.api.routes.jobs._enqueue"):
            create_resp = test_app.post(
                "/api/jobs",
                files=[("files", ("car.jpg", sample_jpeg, "image/jpeg"))],
            )
        job_id = create_resp.json()["jobs"][0]["id"]

        resp = test_app.delete(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["cancelled"] is True

        # Verify the job is now failed
        get_resp = test_app.get(f"/api/jobs/{job_id}")
        assert get_resp.json()["status"] == "failed"

    def test_cancel_nonexistent_job_returns_404(self, test_app):
        resp = test_app.delete(f"/api/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404


class TestStorageLayer:
    def test_local_storage_put_and_get(self, tmp_storage):
        tmp_storage.put("test/hello.txt", b"hello world")
        assert tmp_storage.get("test/hello.txt") == b"hello world"

    def test_local_storage_exists(self, tmp_storage):
        assert not tmp_storage.exists("nope.jpg")
        tmp_storage.put("yes.jpg", b"\xff\xd8")
        assert tmp_storage.exists("yes.jpg")

    def test_local_storage_delete(self, tmp_storage):
        tmp_storage.put("del.jpg", b"data")
        tmp_storage.delete("del.jpg")
        assert not tmp_storage.exists("del.jpg")

    def test_local_storage_get_missing_raises_key_error(self, tmp_storage):
        with pytest.raises(KeyError):
            tmp_storage.get("nonexistent.jpg")

    def test_local_storage_path_traversal_blocked(self, tmp_storage):
        with pytest.raises(ValueError):
            tmp_storage.get("../../etc/passwd")
