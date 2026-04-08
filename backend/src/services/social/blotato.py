"""Blotato posting service for TruLine.

Blotato supports X/Twitter, TikTok, Instagram, LinkedIn, etc.
Media can be attached by passing public URLs (no upload step needed for
images already on a public CDN).

API docs: https://help.blotato.com/api/
"""

import time
import structlog
import httpx
from datetime import datetime

from src.config import settings

logger = structlog.get_logger()

API_BASE = "https://backend.blotato.com/v2"


def _headers() -> dict:
    return {
        "blotato-api-key": settings.blotato_api_key,
        "Content-Type": "application/json",
    }


def upload_media(image_bytes: bytes, filename: str = "image.png") -> str | None:
    """
    Upload an image to Blotato and return a public URL.

    Flow:
    1. POST to /media/uploads to get a presigned URL + publicUrl
    2. PUT the raw bytes to presignedUrl
    3. Return the publicUrl which can be attached to posts

    Args:
        image_bytes: Raw PNG/JPEG bytes
        filename: Filename for Blotato's records

    Returns:
        Public URL string, or None on failure.
    """
    if not settings.blotato_api_key:
        logger.warning("Blotato API key not configured")
        return None

    try:
        with httpx.Client(timeout=60.0) as client:
            # Step 1: Get presigned upload URL
            resp = client.post(
                f"{API_BASE}/media/uploads",
                headers=_headers(),
                json={"filename": filename},
            )
            resp.raise_for_status()
            data = resp.json()
            presigned_url = data.get("presignedUrl")
            public_url = data.get("publicUrl")

            if not presigned_url or not public_url:
                logger.error("Blotato upload: missing presignedUrl or publicUrl")
                return None

            # Step 2: PUT raw bytes to presigned URL
            upload_resp = client.put(presigned_url, content=image_bytes)
            if upload_resp.status_code not in (200, 204):
                logger.error(f"Blotato S3 upload failed: {upload_resp.status_code}")
                return None

            logger.info("Blotato media uploaded", public_url=public_url)
            return public_url

    except httpx.HTTPStatusError as e:
        logger.error(f"Blotato upload error: {e.response.status_code} {e.response.text[:200]}")
        return None
    except Exception as e:
        logger.error(f"Failed to upload media to Blotato: {e}")
        return None


def post_tweet(
    text: str,
    schedule_at: datetime | str | None = "next-free-slot",
    media_urls: list[str] | None = None,
) -> dict | None:
    """
    Post a single tweet via Blotato.

    Args:
        text: Tweet text (max 280 chars per post)
        schedule_at: Either a datetime, "next-free-slot" (uses Blotato's queue),
                     or None (publishes immediately).
        media_urls: Optional list of public image URLs to attach.

    Returns:
        Dict with postSubmissionId, or None on failure.
    """
    return post_thread([text], schedule_at=schedule_at, media_urls_per_post=[media_urls] if media_urls else None)


def post_thread(
    posts: list[str],
    schedule_at: datetime | str | None = "next-free-slot",
    media_urls_per_post: list[list[str] | None] | None = None,
) -> dict | None:
    """
    Post a thread (or single post) via Blotato.

    NOTE: Blotato's POST /posts creates ONE post per call. For threads,
    we use the `additionalPosts` field within content.

    Args:
        posts: List of tweet texts (each max 280 chars). First is main tweet,
               rest become `additionalPosts` (thread continuation).
        schedule_at: datetime, "next-free-slot", or None for immediate.
        media_urls_per_post: Optional per-post media URL lists.

    Returns:
        Dict with postSubmissionId, or None on failure.
    """
    if not posts:
        return None

    if not settings.twitter_posting_enabled:
        log_msg = posts[0][:80] + ("..." if len(posts[0]) > 80 else "")
        logger.info("Blotato posting disabled (dry run)", first_post=log_msg, post_count=len(posts))
        print(f"[BLOTATO-DRY-RUN] Would post {len(posts)} post(s):", flush=True)
        for i, p in enumerate(posts):
            print(f"  [{i+1}] {p}", flush=True)
        return {"dry_run": True, "post_count": len(posts)}

    if not settings.blotato_api_key:
        logger.warning("Blotato API key not configured")
        return None

    if not settings.blotato_twitter_account_id:
        logger.warning("Blotato Twitter account ID not configured")
        return None

    # Build main post content
    main_media = media_urls_per_post[0] if media_urls_per_post and media_urls_per_post[0] else []
    content = {
        "text": posts[0],
        "mediaUrls": main_media,
        "platform": "twitter",
    }

    # Add thread continuation as additionalPosts
    if len(posts) > 1:
        additional = []
        for i in range(1, len(posts)):
            add_media = (
                media_urls_per_post[i]
                if media_urls_per_post and i < len(media_urls_per_post) and media_urls_per_post[i]
                else []
            )
            additional.append({"text": posts[i], "mediaUrls": add_media})
        content["additionalPosts"] = additional

    payload = {
        "post": {
            "accountId": settings.blotato_twitter_account_id,
            "content": content,
            "target": {
                "targetType": "twitter",
            },
        },
    }

    # Handle scheduling
    if schedule_at is not None:
        if isinstance(schedule_at, str):
            if schedule_at == "next-free-slot":
                payload["useNextFreeSlot"] = True
            else:
                payload["scheduledTime"] = schedule_at
        else:
            # datetime object — ISO 8601 with Z suffix for UTC
            payload["scheduledTime"] = schedule_at.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{API_BASE}/posts",
                headers=_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            submission_id = data.get("postSubmissionId")
            logger.info(
                "Blotato post submitted",
                submission_id=submission_id,
                post_count=len(posts),
                scheduled=bool(schedule_at),
            )
            return data
    except httpx.HTTPStatusError as e:
        logger.error(f"Blotato API error: {e.response.status_code} {e.response.text[:300]}")
        print(f"[BLOTATO-ERROR] {e.response.status_code}: {e.response.text[:300]}", flush=True)
        return None
    except Exception as e:
        logger.error(f"Failed to post via Blotato: {e}")
        print(f"[BLOTATO-ERROR] {e}", flush=True)
        return None
