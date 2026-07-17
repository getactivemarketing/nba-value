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
    # NOTE: the shadow best_total record (this gate's input) only accumulates
    # while the Railway env keeps SUPPRESS_TOTALS=false — the code default
    # suppress_totals=True above would stop totals being scored at all.
    totals_in_best_bet: bool = False

    # Path to the MLB totals model. v2 (trained through 2026-07-06 via
    # retrain_mlb_totals) passed the holdout gate vs v1: MAE 3.538 vs 3.564,
    # hit-rate 55.6% (Jun 1 - Jul 5 holdout, 468 games). Serving in SHADOW
    # mode: powers best_total only; excluded from best_bet until the
    # re-entry gate passes (see totals_in_best_bet).
    # Missing file -> scorer falls back to v1, then to the heuristic.
    mlb_totals_model_path: str = "models/mlb_totals_v2.joblib"

    # NFL MOV + totals models (Phase 2, v1) -- trained via src/tasks/nfl_train.py
    # on 2010-2023 seasons; see the walk-forward backtest report for the
    # Phase 2 exit-gate numbers (spread/totals ATS%, reliability, saturation).
    nfl_mov_model_path: str = "models/nfl_mov_v1.joblib"
    nfl_totals_model_path: str = "models/nfl_totals_v1.joblib"

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
