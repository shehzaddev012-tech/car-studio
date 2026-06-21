from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Car Studio Background Generator"
    debug: bool = False

    # Storage — swap storage_backend to "s3" and fill S3 vars for production
    storage_backend: str = "local"
    storage_path: str = str(Path(__file__).parent.parent / "storage")
    max_upload_size_mb: int = 25
    allowed_extensions: list[str] = ["jpg", "jpeg", "png", "webp"]

    # S3 (used only when storage_backend == "s3")
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_endpoint_url: str = ""  # blank = AWS; set for MinIO / R2

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Database
    database_url: str = "sqlite:///./car_studio.db"

    # CORS origins — stored as str to avoid pydantic-settings' JSON-decode requirement
    # for list fields.  Accepts both comma-separated and JSON-array formats:
    #   CORS_ORIGINS=http://localhost:5173,http://localhost:3000
    #   CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
    # Call settings.cors_origins_list() wherever a list[str] is needed.
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    def cors_origins_list(self) -> list[str]:
        v = self.cors_origins.strip()
        if v.startswith("["):
            import json
            return json.loads(v)
        return [o.strip() for o in v.split(",") if o.strip()]

    # Pipeline
    qc_ssim_threshold: float = 0.82  # SSIM threshold for the quality gate
    job_timeout_seconds: int = 300   # Hard limit per Celery task

    # AI mock mode — set MOCK_AI=true to run the full pipeline without rembg
    # (uses OpenCV GrabCut for a rough segmentation; good for local dev without model download)
    mock_ai: bool = False

    # DashScope (Alibaba Cloud) API key.
    # When set, the pipeline calls wanx-background-generation-v2 via the cloud
    # instead of loading any local model — no RAM overhead.
    # Get yours at: https://dashscope.console.aliyun.com
    dashscope_api_key: str = ""


settings = Settings()
