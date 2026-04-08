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


async def _post_pregame_nrfi_picks_async() -> dict:
    """Post per-game NRFI pregame picks for games starting within 90 minutes.

    Each post includes a branded card image with team logos, NRFI %,
    and pitcher matchup.
    """
    from sqlalchemy import select, and_, text
    from src.models import MLBGame
    from src.services.social.content import (
        generate_pregame_nrfi_tweet,
        _get_team_first_inning_pct,
        _get_pitcher_era,
        TEAM_NAMES,
        TEAM_HANDLES,
    )
    from src.services.social.typefully import post_tweet, upload_media
    from src.services.social.image_generator import generate_nrfi_card

    today = _today_et()
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc + timedelta(minutes=90)

    posted_count = 0
    skipped = 0

    async with _social_session_factory() as session:
        stmt = select(MLBGame).where(
            and_(
                MLBGame.game_date == today,
                MLBGame.status == "scheduled",
                MLBGame.pregame_tweet_posted == False,  # noqa: E712
                MLBGame.game_time.isnot(None),
                MLBGame.game_time <= cutoff,
                MLBGame.game_time >= now_utc,
            )
        ).order_by(MLBGame.game_time)
        games = list((await session.execute(stmt)).scalars().all())

        # Score each by NRFI% and keep top 5
        scored = []
        for g in games:
            home_off, home_def = await _get_team_first_inning_pct(session, g.home_team)
            away_off, away_def = await _get_team_first_inning_pct(session, g.away_team)
            if home_off is None or away_off is None or home_def is None or away_def is None:
                continue
            p_away_scores = (away_off + home_def) / 2.0
            p_home_scores = (home_off + away_def) / 2.0
            nrfi = (1.0 - p_away_scores) * (1.0 - p_home_scores)
            scored.append((nrfi, g, home_off, away_off))
        scored.sort(key=lambda x: x[0], reverse=True)

        for nrfi, game, home_pct, away_pct in scored[:5]:
            tweet = await generate_pregame_nrfi_tweet(session, game)
            if not tweet:
                skipped += 1
                continue

            # Generate the card image
            media_ids = None
            try:
                away_last, away_era = await _get_pitcher_era(session, game.away_starter_id)
                home_last, home_era = await _get_pitcher_era(session, game.home_starter_id)

                game_time_str = None
                if game.game_time:
                    et = game.game_time - timedelta(hours=4)
                    try:
                        game_time_str = et.strftime("%-I:%M %p ET")
                    except Exception:
                        game_time_str = et.strftime("%I:%M %p ET").lstrip("0")

                png_bytes = generate_nrfi_card(
                    away_team=game.away_team,
                    home_team=game.home_team,
                    away_name=TEAM_NAMES.get(game.away_team, game.away_team),
                    home_name=TEAM_NAMES.get(game.home_team, game.home_team),
                    nrfi_pct=nrfi * 100,
                    away_pitcher=away_last,
                    away_era=away_era,
                    home_pitcher=home_last,
                    home_era=home_era,
                    away_handle=TEAM_HANDLES.get(game.away_team),
                    home_handle=TEAM_HANDLES.get(game.home_team),
                    game_time=game_time_str,
                )

                media_id = upload_media(png_bytes, filename=f"nrfi_{game.game_id}.png")
                if media_id:
                    media_ids = [media_id]
            except Exception as e:
                log_task(f"Failed to generate/upload NRFI card for {game.game_id}: {e}")

            result = post_tweet(tweet, schedule_at="next-free-slot", media_ids=media_ids)
            if result:
                await session.execute(
                    text("UPDATE mlb_games SET pregame_tweet_posted = TRUE WHERE game_id = :gid"),
                    {"gid": game.game_id},
                )
                posted_count += 1
            else:
                skipped += 1
        await session.commit()

    return {"posted": posted_count, "skipped": skipped, "type": "pregame_nrfi"}


async def _post_first_inning_recaps_async() -> dict:
    """Post per-game 1st inning recaps after the 1st inning ends."""
    from sqlalchemy import select, and_, or_, text
    from src.models import MLBGame
    from src.services.social.content import (
        generate_first_inning_recap_tweet,
        _get_team_first_inning_pct,
        TEAM_NAMES,
    )
    from src.services.social.typefully import post_tweet, upload_media
    from src.services.social.image_generator import generate_recap_card

    posted_count = 0
    skipped = 0

    async with _social_session_factory() as session:
        stmt = select(MLBGame).where(
            and_(
                MLBGame.first_inning_tweet_posted == False,  # noqa: E712
                MLBGame.home_first_inning_runs.isnot(None),
                MLBGame.away_first_inning_runs.isnot(None),
                or_(
                    and_(MLBGame.status == "in_progress", MLBGame.inning >= 2),
                    MLBGame.status == "final",
                ),
            )
        )
        games = list((await session.execute(stmt)).scalars().all())

        for game in games:
            tweet = generate_first_inning_recap_tweet(game)
            if not tweet:
                skipped += 1
                continue

            # Generate the recap card image
            media_ids = None
            try:
                away_fi = game.away_first_inning_runs or 0
                home_fi = game.home_first_inning_runs or 0
                is_nrfi = (away_fi + home_fi) == 0

                # Get the model's pre-game NRFI prediction for context
                home_off, home_def = await _get_team_first_inning_pct(session, game.home_team)
                away_off, away_def = await _get_team_first_inning_pct(session, game.away_team)
                predicted_nrfi_pct = None
                if (home_off is not None and away_off is not None
                        and home_def is not None and away_def is not None):
                    p_away_scores = (away_off + home_def) / 2.0
                    p_home_scores = (home_off + away_def) / 2.0
                    predicted_nrfi_pct = (1.0 - p_away_scores) * (1.0 - p_home_scores) * 100

                png_bytes = generate_recap_card(
                    away_team=game.away_team,
                    home_team=game.home_team,
                    away_name=TEAM_NAMES.get(game.away_team, game.away_team),
                    home_name=TEAM_NAMES.get(game.home_team, game.home_team),
                    away_first=away_fi,
                    home_first=home_fi,
                    is_nrfi=is_nrfi,
                    predicted_nrfi_pct=predicted_nrfi_pct,
                )

                media_id = upload_media(png_bytes, filename=f"recap_{game.game_id}.png")
                if media_id:
                    media_ids = [media_id]
            except Exception as e:
                log_task(f"Failed to generate/upload recap card for {game.game_id}: {e}")

            result = post_tweet(tweet, schedule_at="next-free-slot", media_ids=media_ids)
            if result:
                await session.execute(
                    text("UPDATE mlb_games SET first_inning_tweet_posted = TRUE WHERE game_id = :gid"),
                    {"gid": game.game_id},
                )
                posted_count += 1
            else:
                skipped += 1
        await session.commit()

    return {"posted": posted_count, "skipped": skipped, "type": "first_inning_recap"}


async def _post_final_recaps_async() -> dict:
    """Post per-game final recaps for completed games."""
    from sqlalchemy import select, and_, text
    from src.models import MLBGame
    from src.services.social.content import generate_final_recap_tweet
    from src.services.social.typefully import post_tweet

    posted_count = 0
    skipped = 0

    async with _social_session_factory() as session:
        stmt = select(MLBGame).where(
            and_(
                MLBGame.status == "final",
                MLBGame.final_tweet_posted == False,  # noqa: E712
                MLBGame.home_score.isnot(None),
                MLBGame.away_score.isnot(None),
            )
        )
        games = list((await session.execute(stmt)).scalars().all())

        for game in games:
            tweet = generate_final_recap_tweet(game)
            if not tweet:
                skipped += 1
                continue
            result = post_tweet(tweet, schedule_at="next-free-slot")
            if result:
                await session.execute(
                    text("UPDATE mlb_games SET final_tweet_posted = TRUE WHERE game_id = :gid"),
                    {"gid": game.game_id},
                )
                posted_count += 1
            else:
                skipped += 1
        await session.commit()

    return {"posted": posted_count, "skipped": skipped, "type": "final_recap"}


def run_post_pregame_nrfi():
    """Post per-game pregame NRFI picks."""
    log_task("Posting pregame NRFI picks...")
    try:
        result = _run_async(_post_pregame_nrfi_picks_async())
        log_task("Pregame NRFI post complete", **{k: str(v) for k, v in result.items()})
        return result
    except Exception as e:
        log_task(f"Pregame NRFI post FAILED: {e}")
        return {"status": "failed", "error": str(e)}


def run_post_first_inning_recaps():
    """Post per-game 1st inning recaps."""
    log_task("Posting 1st inning recaps...")
    try:
        result = _run_async(_post_first_inning_recaps_async())
        log_task("1st inning recaps post complete", **{k: str(v) for k, v in result.items()})
        return result
    except Exception as e:
        log_task(f"1st inning recaps post FAILED: {e}")
        return {"status": "failed", "error": str(e)}


def run_post_final_recaps():
    """Post per-game final recaps."""
    log_task("Posting final recaps...")
    try:
        result = _run_async(_post_final_recaps_async())
        log_task("Final recaps post complete", **{k: str(v) for k, v in result.items()})
        return result
    except Exception as e:
        log_task(f"Final recaps post FAILED: {e}")
        return {"status": "failed", "error": str(e)}


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

    # Per-game posting tasks
    # Pregame NRFI picks — runs every 30 min; posts games within 90 min of start
    social_scheduler.every(30).minutes.do(run_post_pregame_nrfi)
    # First inning recaps — runs every 10 min during game hours
    social_scheduler.every(10).minutes.do(run_post_first_inning_recaps)
    # Final recaps — runs every 30 min, mostly evening
    social_scheduler.every(30).minutes.do(run_post_final_recaps)

    log_task("Social scheduler configured:")
    log_task("  - Results recap: 9:00 AM ET (13:00 UTC)")
    log_task("  - NRFI results: 9:15 AM ET (13:15 UTC)")
    log_task("  - Daily picks: 10:00 AM ET (14:00 UTC)")
    log_task("  - NRFI plays: 10:15 AM ET (14:15 UTC)")
    log_task("  - Pregame NRFI picks: every 30 min (games within 90 min of start)")
    log_task("  - 1st inning recaps: every 10 min")
    log_task("  - Final recaps: every 30 min")

    while True:
        social_scheduler.run_pending()
        time.sleep(60)
