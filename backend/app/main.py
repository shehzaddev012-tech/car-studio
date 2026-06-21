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
from app.startup_validation import StartupValidationError, validate_dealership_pipeline

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Strict dealership-grade car photo → studio background pipeline. "
            "Vertex AI image-segmentation-001 + SAM2 refinement. No fallbacks."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def on_startup() -> None:
        create_tables()
        logger.info("Database tables verified / created")
        logger.info("Storage backend: %s", settings.storage_backend)
        try:
            validate_dealership_pipeline(context="api")
        except StartupValidationError as exc:
            logger.critical("Startup validation failed — refusing to start: %s", exc)
            raise RuntimeError(str(exc)) from exc

    app.include_router(jobs_router)

    @app.websocket("/ws/jobs")
    async def ws_jobs(websocket: WebSocket) -> None:
        await handle_jobs_websocket(websocket)

    @app.get("/health", tags=["infra"])
    def health() -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "pipeline": "vertex-ai+sam2",
            "segmentation_provider": "image-segmentation-001",
        })

    return app


app = create_app()
