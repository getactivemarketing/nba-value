"""
Social media scheduler for TruLine daily content.

Posts to Twitter/X:
1. Morning: Yesterday's results recap + NRFI recap
2. Pre-game: Today's picks thread + NRFI plays


Schedule (all times ET):
- 9:00 AM: Results recap from yesterday
- 9:15 AM: NRFI results from yesterday
- 10:00 AM: Today's picks thread
- 10:15 AM: Today's NRFI plays
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta, date

import structlog
import schedule

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from src.config import settings

logger = structlog.get_logger()


def log_task(message: str, **kwargs):
    """Log social scheduler output."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    print(f"[SOCIAL-SCHEDULER] {timestamp} | {message} {extra}", flush=True)


def _today_et() -> date:
    """Get today's date in US Eastern time."""
    eastern = timedelta(hours=-5)
    return (datetime.now(timezone.utc) + eastern).date()


# Lazy-initialized engine for social scheduler.
# Created inside start_scheduler() to avoid blocking app startup.
_social_engine = None
_social_session_factory = None


def _init_engine():
    """Initialize the social scheduler's DB engine. Called once from start_scheduler()."""
    global _social_engine, _social_session_factory
    if _social_engine is None:
        _social_engine = create_async_engine(
            settings.async_database_url,
            pool_pre_ping=True,
            pool_size=1,
            max_overflow=1,
            pool_recycle=300,
            pool_timeout=30,
        )
        _social_session_factory = async_sessionmaker(
            _social_engine, class_=AsyncSession, expire_on_commit=False
        )


# Persistent event loop
_social_loop: asyncio.AbstractEventLoop | None = None


def _run_async(coro):
    """Run async on the social scheduler's event loop."""
    global _social_loop
    if _social_loop is None:
        _social_loop = asyncio.new_event_loop()
    return _social_loop.run_until_complete(coro)


async def _post_results_async() -> dict:
    """Post yesterday's results recap."""
    from src.services.social.content import generate_results_tweet
    from src.services.social.typefully import post_tweet

    yesterday = _today_et() - timedelta(days=1)

    async with _social_session_factory() as session:
        tweet_text = await generate_results_tweet(session, yesterday)

    if tweet_text:
        tweet_id = post_tweet(tweet_text)
        return {"posted": True, "tweet_id": tweet_id, "type": "results"}
    else:
        log_task("No results to post for yesterday")
        return {"posted": False, "reason": "no_results"}


async def _post_nrfi_results_async() -> dict:
    """Post yesterday's NRFI recap."""
    from src.services.social.content import generate_nrfi_results_tweet
    from src.services.social.typefully import post_tweet

    yesterday = _today_et() - timedelta(days=1)

    async with _social_session_factory() as session:
        tweet_text = await generate_nrfi_results_tweet(session, yesterday)

    if tweet_text:
        tweet_id = post_tweet(tweet_text)
        return {"posted": True, "tweet_id": tweet_id, "type": "nrfi_results"}
    else:
        log_task("No NRFI results to post")
        return {"posted": False, "reason": "no_nrfi_data"}


async def _post_daily_picks_async() -> dict:
    """Post today's picks thread."""
    from src.services.social.content import generate_daily_picks_thread
    from src.services.social.typefully import post_thread

    today = _today_et()

    async with _social_session_factory() as session:
        tweets = await generate_daily_picks_thread(session, today)

    if tweets:
        tweet_ids = post_thread(tweets)
        return {"posted": True, "tweets": len(tweets), "ids": tweet_ids, "type": "picks"}
    else:
        log_task("No picks to post for today")
        return {"posted": False, "reason": "no_picks"}


async def _post_nrfi_plays_async() -> dict:
    """Post today's NRFI plays."""
    from src.services.social.content import generate_nrfi_tweet
    from src.services.social.typefully import post_tweet

    today = _today_et()

    async with _social_session_factory() as session:
        tweet_text = await generate_nrfi_tweet(session, today)

    if tweet_text:
        tweet_id = post_tweet(tweet_text)
        return {"posted": True, "tweet_id": tweet_id, "type": "nrfi_plays"}
    else:
        log_task("No NRFI plays to post")
        return {"posted": False, "reason": "no_nrfi_data"}


def run_post_results():
    """Post yesterday's results."""
    log_task("Posting results recap...")
    try:
        result = _run_async(_post_results_async())
        log_task("Results post complete", **{k: str(v) for k, v in result.items()})
        return result
    except Exception as e:
        log_task(f"Results post FAILED: {e}")
        return {"status": "failed", "error": str(e)}


def run_post_nrfi_results():
    """Post yesterday's NRFI recap."""
    log_task("Posting NRFI results...")
    try:
        result = _run_async(_post_nrfi_results_async())
        log_task("NRFI results post complete", **{k: str(v) for k, v in result.items()})
        return result
    except Exception as e:
        log_task(f"NRFI results post FAILED: {e}")
        return {"status": "failed", "error": str(e)}


def run_post_daily_picks():
    """Post today's picks thread."""
    log_task("Posting daily picks...")
    try:
        result = _run_async(_post_daily_picks_async())
        log_task("Daily picks post complete", **{k: str(v) for k, v in result.items()})
        return result
    except Exception as e:
        log_task(f"Daily picks post FAILED: {e}")
        return {"status": "failed", "error": str(e)}


def run_post_nrfi_plays():
    """Post today's NRFI plays."""
    log_task("Posting NRFI plays...")
    try:
        result = _run_async(_post_nrfi_plays_async())
        log_task("NRFI plays post complete", **{k: str(v) for k, v in result.items()})
        return result
    except Exception as e:
        log_task(f"NRFI plays post FAILED: {e}")
        return {"status": "failed", "error": str(e)}


def start_scheduler():
    """Start the social media posting scheduler.

    All times are UTC. ET = UTC - 4 (EDT) or UTC - 5 (EST).
    9:00 AM ET = 13:00 UTC (EDT)
    """
    global _social_loop
    _social_loop = asyncio.new_event_loop()

    log_task("Starting social media scheduler...")
    log_task("Waiting 180s for other schedulers to finish startup...")
    time.sleep(180)

    # Initialize DB engine after startup delay so it doesn't compete with healthcheck
    _init_engine()

    social_scheduler = schedule.Scheduler()

    # Morning results (9:00-9:15 AM ET = 13:00-13:15 UTC in EDT)
    social_scheduler.every().day.at("13:00").do(run_post_results)
    social_scheduler.every().day.at("13:15").do(run_post_nrfi_results)

    # Pre-game picks (10:00-10:15 AM ET = 14:00-14:15 UTC in EDT)
    social_scheduler.every().day.at("14:00").do(run_post_daily_picks)
    social_scheduler.every().day.at("14:15").do(run_post_nrfi_plays)

    log_task("Social scheduler configured:")
    log_task("  - Results recap: 9:00 AM ET (13:00 UTC)")
    log_task("  - NRFI results: 9:15 AM ET (13:15 UTC)")
    log_task("  - Daily picks: 10:00 AM ET (14:00 UTC)")
    log_task("  - NRFI plays: 10:15 AM ET (14:15 UTC)")

    while True:
        social_scheduler.run_pending()
        time.sleep(60)
