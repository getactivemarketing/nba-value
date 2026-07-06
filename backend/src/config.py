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

    # Twitter/X API (legacy direct posting — kept for fallback)
    twitter_api_key: str = ""
    twitter_api_secret: str = ""
    twitter_access_token: str = ""
    twitter_access_token_secret: str = ""
    twitter_bearer_token: str = ""

    # Typefully (preferred posting service)
    typefully_api_key: str = ""
    typefully_social_set_id: int = 296324  # @trulineapp
    typefully_auto_share: bool = True  # Auto-publish drafts (vs save as draft)

    # Blotato (primary posting service — replaces Typefully)
    blotato_api_key: str = ""
    blotato_twitter_account_id: str = ""  # @trulineapp account ID on Blotato

    # Posting safety switch (applies to both direct Twitter and Typefully)
    twitter_posting_enabled: bool = False

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

    # Model Flags
    # Suppress totals (over/under) betting - model has 41% win rate (need 52.4% to break even)
    # Set to False when a proper totals model is trained
    suppress_totals: bool = True

    # Allow totals (over/under) to be chosen as the overall best_bet.
    # Independent of suppress_totals (which stops totals being scored at all).
    # Totals as best bets ran -29u / 48.7% Apr-Jul 2026. Flip to True only via
    # the re-entry gate: >=100 graded best_total picks under the retrained
    # model with >=53% WR and positive cumulative units.
    totals_in_best_bet: bool = False

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
