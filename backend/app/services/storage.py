"""
Storage abstraction layer.

Swap implementations by changing ``settings.storage_backend``:
  - "local"  → LocalStorageAdapter  (writes to disk; default for the prototype)
  - "s3"     → S3StorageAdapter     (writes to AWS S3 / MinIO / Cloudflare R2)

The interface intentionally mirrors the S3 object model (put / get / delete / exists)
so the swap is a one-line config change.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import settings


class StorageAdapter(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Write ``data`` under ``key``."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Return the bytes stored at ``key``. Raises ``KeyError`` if not found."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the object at ``key``."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if ``key`` exists."""

    @abstractmethod
    def public_url(self, key: str) -> str:
        """Return a URL that can serve the file (may be relative for local)."""


# ── Local-disk implementation ──────────────────────────────────────────────────

class LocalStorageAdapter(StorageAdapter):
    """
    Stores files on the local filesystem under ``root``.

    Directory structure:
        {root}/uploads/{job_id}.{ext}   ← original uploads
        {root}/results/{job_id}.jpg     ← processed outputs
    """

    def __init__(self, root: str | None = None) -> None:
        self.root = Path(root or settings.storage_path)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "uploads").mkdir(exist_ok=True)
        (self.root / "results").mkdir(exist_ok=True)

    def _path(self, key: str) -> Path:
        # Guard against path traversal
        resolved = (self.root / key).resolve()
        if not str(resolved).startswith(str(self.root.resolve())):
            raise ValueError(f"Path traversal attempt detected for key: {key!r}")
        return resolved

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, key: str) -> bytes:
        p = self._path(key)
        if not p.exists():
            raise KeyError(f"Storage key not found: {key!r}")
        return p.read_bytes()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def public_url(self, key: str) -> str:
        # In the prototype the API layer streams files — this is informational only
        return f"/storage/{key}"


# ── S3 implementation stub ─────────────────────────────────────────────────────

class S3StorageAdapter(StorageAdapter):
    """
    Production implementation — wraps boto3.

    Set these env vars (or .env) to activate:
        STORAGE_BACKEND=s3
        S3_BUCKET=my-bucket
        S3_REGION=us-east-1
        S3_ACCESS_KEY=...
        S3_SECRET_KEY=...
        S3_ENDPOINT_URL=         # blank for AWS; set for MinIO / R2
    """

    def __init__(self) -> None:
        try:
            import boto3
        except ImportError as e:
            raise RuntimeError("boto3 is required for S3 storage. Run: pip install boto3") from e

        self._client = boto3.client(
            "s3",
            region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            endpoint_url=settings.s3_endpoint_url or None,
        )
        self._bucket = settings.s3_bucket

    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes:
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            return resp["Body"].read()
        except self._client.exceptions.NoSuchKey:
            raise KeyError(f"S3 key not found: {key!r}")

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def public_url(self, key: str) -> str:
        if settings.s3_endpoint_url:
            return f"{settings.s3_endpoint_url}/{self._bucket}/{key}"
        return f"https://{self._bucket}.s3.{settings.s3_region}.amazonaws.com/{key}"


# ── Factory ────────────────────────────────────────────────────────────────────

def get_storage() -> StorageAdapter:
    """Return the configured storage adapter (singleton-ish; safe to call repeatedly)."""
    if settings.storage_backend == "s3":
        return S3StorageAdapter()
    return LocalStorageAdapter()
