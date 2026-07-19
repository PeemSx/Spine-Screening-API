"""Environment-backed application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Validated settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SPINE_API_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Spine Screening API"
    environment: Literal["development", "test", "production"] = "development"
    api_v1_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    model_path: Path = PROJECT_ROOT / "app" / "weight" / "best_center_f1.pt"
    model_device: Literal["auto", "cpu", "cuda"] = "auto"
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=50_000_000, gt=0)
    inference_concurrency: int = Field(default=1, gt=0)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide immutable settings instance."""
    return Settings()
