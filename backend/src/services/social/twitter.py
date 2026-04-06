"""Twitter/X posting service for TruLine daily picks."""

import structlog
from datetime import datetime, timezone, timedelta, date

from src.config import settings

logger = structlog.get_logger()


def _get_client():
    """Get authenticated Twitter API v2 client."""
    import tweepy

    client = tweepy.Client(
        bearer_token=settings.twitter_bearer_token,
        consumer_key=settings.twitter_api_key,
        consumer_secret=settings.twitter_api_secret,
        access_token=settings.twitter_access_token,
        access_token_secret=settings.twitter_access_token_secret,
    )
    return client


def post_tweet(text: str, reply_to: str | None = None) -> str | None:
    """
    Post a tweet. Returns the tweet ID or None on failure.

    Args:
        text: Tweet text (max 280 chars)
        reply_to: Optional tweet ID to reply to (for threads)
    """
    if not settings.twitter_posting_enabled:
        logger.info("Twitter posting disabled", text=text[:80])
        print(f"[TWITTER-DRY-RUN] Would post: {text}", flush=True)
        return None

    if not settings.twitter_api_key:
        logger.warning("Twitter API keys not configured")
        return None

    try:
        client = _get_client()
        kwargs = {"text": text}
        if reply_to:
            kwargs["in_reply_to_tweet_id"] = reply_to

        response = client.create_tweet(**kwargs)
        tweet_id = response.data["id"]
        logger.info("Tweet posted", tweet_id=tweet_id, length=len(text))
        return tweet_id
    except Exception as e:
        logger.error(f"Failed to post tweet: {e}")
        print(f"[TWITTER-ERROR] {e}", flush=True)
        return None


def post_thread(tweets: list[str]) -> list[str]:
    """
    Post a thread of tweets. Returns list of tweet IDs.

    Args:
        tweets: List of tweet texts (each max 280 chars)
    """
    tweet_ids = []
    reply_to = None

    for i, text in enumerate(tweets):
        tweet_id = post_tweet(text, reply_to=reply_to)
        if tweet_id:
            tweet_ids.append(tweet_id)
            reply_to = tweet_id
        else:
            if settings.twitter_posting_enabled:
                logger.error(f"Thread broken at tweet {i+1}")
                break

    return tweet_ids
