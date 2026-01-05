"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # Database (raw from env, may need conversion)
    database_url: str = "postgresql+asyncpg://betting:betting_dev@localhost:5432/nba_betting"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Auth
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 1 week

    # API Keys
    odds_api_key: str = ""
    balldontlie_api_key: str = ""

    # API Settings
    api_v1_prefix: str = "/api/v1"

    # CORS - include production URLs
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://nba-value.vercel.app",
        "https://*.vercel.app",
    ]

    # Logging
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def async_database_url(self) -> str:
        """Convert database URL to async driver format."""
        url = self.database_url
        # Railway uses postgresql://, SQLAlchemy async needs postgresql+asyncpg://
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
