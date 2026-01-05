"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.database import init_db
from src.api import health, markets, bets, evaluation, admin

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting NBA Value Betting API", environment=settings.environment)
    await init_db()
    yield
    # Shutdown
    logger.info("Shutting down NBA Value Betting API")


app = FastAPI(
    title="NBA Value Betting API",
    description="Sharp-focused betting intelligence platform with Value Score ranking",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(markets.router, prefix=settings.api_v1_prefix, tags=["Markets"])
app.include_router(bets.router, prefix=settings.api_v1_prefix, tags=["Bets"])
app.include_router(evaluation.router, prefix=settings.api_v1_prefix, tags=["Evaluation"])
app.include_router(admin.router, prefix=settings.api_v1_prefix, tags=["Admin"])


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "name": "NBA Value Betting API",
        "version": "0.1.0",
        "docs": "/docs",
    }
