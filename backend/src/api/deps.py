"""Shared FastAPI dependencies."""
from fastapi import Header, HTTPException

from src.api.admin import _verify_cron_key


def require_admin(x_cron_key: str | None = Header(None)) -> None:
    """404 (not 401) so probes cannot tell an admin route exists."""
    if not _verify_cron_key(x_cron_key):
        raise HTTPException(status_code=404, detail="Not Found")
