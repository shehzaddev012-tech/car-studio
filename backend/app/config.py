from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Car Studio Background Generator"
    debug: bool = False

    storage_backend: str = "local"
    storage_path: str = str(Path(__file__).parent.parent / "storage")
    max_upload_size_mb: int = 25
    allowed_extensions: list[str] = ["jpg", "jpeg", "png", "webp"]

    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_endpoint_url: str = ""

    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "sqlite:///./car_studio.db"

    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    def cors_origins_list(self) -> list[str]:
        v = self.cors_origins.strip()
        if v.startswith("["):
            import json
            return json.loads(v)
        return [o.strip() for o in v.split(",") if o.strip()]

    # Pipeline — compositing only (no fallbacks)
    ai_provider: str = "compositing"

    # Quality gates
    qc_ssim_threshold: float = 0.82
    qc_lpips_threshold: float = 0.08
    lpips_enabled: bool = True
    job_timeout_seconds: int = 600

    mask_min_coverage: float = 0.03
    mask_max_coverage: float = 0.88
    mask_retention_min: float = 0.88
    crop_edge_margin_ratio: float = 0.015
    reject_cropped_vehicles: bool = False

    # Production-calibrated defaults — strict preservation, <15% false reject target
    preservation_min_global_retention: float = 0.88
    preservation_min_zone_retention: float = 0.85
    preservation_min_sam2_retention: float = 0.86
    preservation_max_edge_loss_ratio: float = 0.07
    preservation_max_boundary_loss_ratio: float = 0.30
    preservation_min_sam2_confidence: float = 0.62

    # Vertex AI — sole segmentation provider (required)
    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"
    google_application_credentials: str = ""
    vertex_segmentation_model: str = "image-segmentation-001"
    vertex_segmentation_prompt: str = "car"
    vertex_mask_dilation: float = 0.02
    vertex_confidence_threshold: float = 0.1
    vertex_min_confidence: float = 0.50
    vertex_request_timeout_seconds: int = 60

    # SAM2 — required refinement stage (not a segmentation fallback)
    sam2_model_id: str = "facebook/sam2.1-hiera-base-plus"
    sam2_checkpoint_filename: str = "sam2.1_hiera_base_plus.pt"
    sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2_checkpoint_path: str = ""

    def resolved_credentials_path(self) -> str:
        """Return the active GOOGLE_APPLICATION_CREDENTIALS path."""
        raw = (self.google_application_credentials or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")).strip()
        return raw


settings = Settings()
