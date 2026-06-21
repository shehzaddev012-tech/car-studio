"""
FastAPI application entry point.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.jobs import router as jobs_router
from app.api.websocket import handle_jobs_websocket
from app.config import settings
from app.db.database import create_tables

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Async car photo → studio background composite pipeline. "
            "Upload car photos, get back the same vehicle on a professional dealership backdrop."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ───────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Startup ────────────────────────────────────────────────────────────────
    @app.on_event("startup")
    def on_startup() -> None:
        create_tables()
        logger.info("Database tables verified / created")
        logger.info("Storage backend: %s", settings.storage_backend)
        logger.info("Mock AI mode: %s", settings.mock_ai)

    # ── REST routes ────────────────────────────────────────────────────────────
    app.include_router(jobs_router)

    # ── WebSocket ──────────────────────────────────────────────────────────────
    @app.websocket("/ws/jobs")
    async def ws_jobs(websocket: WebSocket) -> None:
        await handle_jobs_websocket(websocket)

    # ── Health check ───────────────────────────────────────────────────────────
    @app.get("/health", tags=["infra"])
    def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "mock_ai": settings.mock_ai})

    return app


app = create_app()
