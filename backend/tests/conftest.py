"""
Shared pytest fixtures for the car-studio backend test suite.
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ── Patch settings before any app import ─────────────────────────────────────
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")  # isolated test DB
os.environ.setdefault("MOCK_AI", "true")

from app.db.database import Base, get_db  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.storage import LocalStorageAdapter  # noqa: E402


# ── In-memory SQLite engine for tests ────────────────────────────────────────

@pytest.fixture(scope="function")
def db_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


# ── Temp-dir storage adapter ──────────────────────────────────────────────────

@pytest.fixture(scope="function")
def tmp_storage(tmp_path) -> Generator[LocalStorageAdapter, None, None]:
    yield LocalStorageAdapter(root=str(tmp_path))


# ── FastAPI test client ───────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def test_app(db_session, tmp_storage):
    app = create_app()

    # Override DB dependency
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    # Override storage to use tmp directory
    import app.api.routes.jobs as jobs_module
    original_get_storage = jobs_module.get_storage
    jobs_module.get_storage = lambda: tmp_storage

    with TestClient(app) as client:
        yield client

    jobs_module.get_storage = original_get_storage


# ── Minimal valid JPEG factory ────────────────────────────────────────────────

@pytest.fixture()
def sample_jpeg() -> bytes:
    """Returns a minimal 10×10 white JPEG image."""
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (10, 10), color=(240, 240, 240))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def sample_png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (10, 10), color=(200, 200, 255))
    img.save(buf, format="PNG")
    return buf.getvalue()
