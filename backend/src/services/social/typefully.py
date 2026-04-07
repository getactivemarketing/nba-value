"""Typefully posting service for TruLine daily picks.

Typefully is a tweet scheduling/drafting platform with a clean API.
Better than direct Twitter API for: image attachments, scheduling,
content management, and avoiding rate limits.

API docs: https://typefully.com/docs/api
"""

import structlog
import httpx
from datetime import datetime

from src.config import settings

logger = structlog.get_logger()

API_BASE = "https://api.typefully.com/v2"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.typefully_api_key}",
        "Content-Type": "application/json",
    }


def post_tweet(text: str, schedule_at: datetime | None = None) -> dict | None:
    """
    Post a single tweet via Typefully.

    Args:
        text: Tweet text (max 280 chars per post)
        schedule_at: Optional UTC datetime to schedule. If None, publishes immediately.

    Returns:
        Dict with draft_id and other metadata, or None on failure.
    """
    return post_thread([text], schedule_at=schedule_at)


def post_thread(posts: list[str], schedule_at: datetime | None = None) -> dict | None:
    """
    Post a thread of tweets via Typefully.

    Args:
        posts: List of tweet texts (each max 280 chars)
        schedule_at: Optional UTC datetime to schedule. If None, publishes immediately.

    Returns:
        Dict with draft_id and other metadata, or None on failure.
    """
    if not posts:
        return None

    if not settings.twitter_posting_enabled:
        log_msg = posts[0][:80] + ("..." if len(posts[0]) > 80 else "")
        logger.info("Typefully posting disabled (dry run)", first_post=log_msg, post_count=len(posts))
        print(f"[TYPEFULLY-DRY-RUN] Would post {len(posts)} post(s):", flush=True)
        for i, p in enumerate(posts):
            print(f"  [{i+1}] {p}", flush=True)
        return {"dry_run": True, "post_count": len(posts)}

    if not settings.typefully_api_key:
        logger.warning("Typefully API key not configured")
        return None

    payload = {
        "platforms": {
            "x": {
                "enabled": True,
                "posts": [{"text": text} for text in posts],
            }
        },
        "share": settings.typefully_auto_share,
    }

    if schedule_at:
        # Typefully expects ISO 8601 with Z suffix for UTC
        payload["publish_at"] = schedule_at.strftime("%Y-%m-%dT%H:%M:%SZ")

    url = f"{API_BASE}/social-sets/{settings.typefully_social_set_id}/drafts"

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=_headers(), json=payload)
            response.raise_for_status()
            data = response.json()
            draft_id = data.get("id")
            logger.info(
                "Typefully draft created",
                draft_id=draft_id,
                post_count=len(posts),
                scheduled=bool(schedule_at),
            )
            return data
    except httpx.HTTPStatusError as e:
        logger.error(f"Typefully API error: {e.response.status_code} {e.response.text[:200]}")
        print(f"[TYPEFULLY-ERROR] {e.response.status_code}: {e.response.text[:200]}", flush=True)
        return None
    except Exception as e:
        logger.error(f"Failed to post via Typefully: {e}")
        print(f"[TYPEFULLY-ERROR] {e}", flush=True)
        return None
