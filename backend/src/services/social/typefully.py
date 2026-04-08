"""Typefully posting service for TruLine daily picks.

Typefully is a tweet scheduling/drafting platform with a clean API.
Better than direct Twitter API for: image attachments, scheduling,
content management, and avoiding rate limits.

API docs: https://typefully.com/docs/api
"""

import time
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


def upload_media(image_bytes: bytes, filename: str = "image.png") -> str | None:
    """
    Upload an image to Typefully and return the media_id.

    Flow:
    1. POST to /media/upload to get a presigned S3 URL
    2. PUT the raw bytes to the S3 URL
    3. Poll status until 'ready'

    Args:
        image_bytes: Raw PNG/JPEG bytes
        filename: Filename for Typefully's records

    Returns:
        media_id string, or None on failure.
    """
    if not settings.typefully_api_key:
        return None

    try:
        with httpx.Client(timeout=60.0) as client:
            # Step 1: Get presigned upload URL
            resp = client.post(
                f"{API_BASE}/social-sets/{settings.typefully_social_set_id}/media/upload",
                headers=_headers(),
                json={"file_name": filename},
            )
            resp.raise_for_status()
            data = resp.json()
            media_id = data.get("media_id")
            upload_url = data.get("upload_url")

            if not media_id or not upload_url:
                logger.error("Typefully media upload: missing media_id or upload_url")
                return None

            # Step 2: PUT the raw bytes (no headers)
            upload_resp = client.put(upload_url, content=image_bytes)
            if upload_resp.status_code not in (200, 204):
                logger.error(f"Typefully S3 upload failed: {upload_resp.status_code}")
                return None

            # Step 3: Poll until ready (max 10 seconds)
            for _ in range(10):
                time.sleep(1)
                status_resp = client.get(
                    f"{API_BASE}/social-sets/{settings.typefully_social_set_id}/media/{media_id}",
                    headers=_headers(),
                )
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    if status_data.get("status") == "ready":
                        logger.info("Typefully media ready", media_id=media_id)
                        return media_id

            logger.warning(f"Typefully media {media_id} still processing after 10s, returning anyway")
            return media_id

    except httpx.HTTPStatusError as e:
        logger.error(f"Typefully media upload error: {e.response.status_code} {e.response.text[:200]}")
        return None
    except Exception as e:
        logger.error(f"Failed to upload media to Typefully: {e}")
        return None


def post_tweet(
    text: str,
    schedule_at: datetime | str | None = "next-free-slot",
    media_ids: list[str] | None = None,
) -> dict | None:
    """
    Post a single tweet via Typefully.

    Args:
        text: Tweet text (max 280 chars per post)
        schedule_at: Either a datetime, "next-free-slot" (default — uses Typefully queue),
                     or None (saves as draft only without publishing).
        media_ids: Optional list of Typefully media_ids to attach.

    Returns:
        Dict with draft_id and other metadata, or None on failure.
    """
    return post_thread([text], schedule_at=schedule_at, media_ids_per_post=[media_ids] if media_ids else None)


def post_thread(
    posts: list[str],
    schedule_at: datetime | str | None = "next-free-slot",
    media_ids_per_post: list[list[str] | None] | None = None,
) -> dict | None:
    """
    Post a thread of tweets via Typefully.

    Args:
        posts: List of tweet texts (each max 280 chars)
        schedule_at: Either a datetime, "next-free-slot" (default — uses Typefully queue),
                     or None (saves as draft only without publishing).
        media_ids_per_post: Optional list (same length as posts) of media_id lists
                            to attach to each post. Use None for posts without media.

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

    # Build per-post payloads, optionally with media_ids
    post_objs = []
    for i, text in enumerate(posts):
        obj = {"text": text}
        if media_ids_per_post and i < len(media_ids_per_post) and media_ids_per_post[i]:
            obj["media_ids"] = media_ids_per_post[i]
        post_objs.append(obj)

    payload = {
        "platforms": {
            "x": {
                "enabled": True,
                "posts": post_objs,
            }
        },
    }

    # Schedule the post — defaults to "next-free-slot" using Typefully's queue
    if schedule_at is not None:
        if isinstance(schedule_at, str):
            payload["publish_at"] = schedule_at
        else:
            # ISO 8601 with Z suffix for UTC
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
