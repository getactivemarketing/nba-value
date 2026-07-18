#!/usr/bin/env python3
"""
NFL Scheduler for the weekly prediction cycle (Phase 4, Task 4).

Ships DISABLED (`settings.nfl_scheduler_enabled = False`). Nothing calls
`start_scheduler()` anywhere in the app yet — flipping it on is a deliberate
go-live switch for ~Sept 2026, once the 2026 season actually has games to
snapshot/grade against.

Mirrors `src/tasks/mlb_scheduler.py`'s structure: a dedicated engine/session
factory (so this scheduler's connection pool never touches the MLB/NBA
schedulers), sync wrappers around async task functions, and a dedicated
`schedule.Scheduler()` instance.

Task functions:
- `snapshot_due_games(session, minutes_before, mov_bundle=None, totals_bundle=None)`
  — freeze predictions for games kicking off soon.
- `grade_finals(session)` — grade snapshots for completed games.
- `weekly_refresh(session)` — pull schedule + recompute team stats (Tue cadence).
- `refresh_odds(session)` — pull current lines (more frequent cadence).

Each is safe to run out of season: 0 due/final/upcoming games -> a summary
dict with zero counts, no error.

Commit convention: each async task function commits its own session before
returning (matches `mlb_scheduler`'s `snapshot_predictions_async` /
`grade_predictions_async`, which both call `session.commit()` internally
rather than leaving it to the caller). `season_update.refresh_schedule` /
`recompute_team_stats` / `odds_to_markets` do NOT commit internally (caller
owns the transaction), so `weekly_refresh`/`refresh_odds` commit after
calling them.
"""
import asyncio
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import schedule
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Add parent to path for imports (mirrors mlb_scheduler.py, for standalone execution).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import settings
from src.models import NFLGame, NFLGameContext, NFLMarket, NFLPredictionSnapshot, NFLTeamStats
from src.services.nfl import model_training, season_update
from src.services.nfl.live_features import build_live_feature_row
from src.services.nfl.scorer import score_game
from src.services.nfl.snapshot import build_snapshot, grade_snapshot

logger = structlog.get_logger()

# Track last run times for health monitoring (mirrors mlb_scheduler).
_last_run_times: dict[str, datetime] = {}

# Lazy-initialized engine for the NFL scheduler thread. Created inside
# start_scheduler()/_init_engine() to avoid blocking app startup and to
# never share a connection pool (and its event loop) with the MLB/NBA
# schedulers -- asyncpg connections are bound to the loop that created them.
_nfl_engine = None
_nfl_session_factory = None

# Model column set, computed once, used to filter build_snapshot()'s dict
# down to constructor-safe kwargs for NFLPredictionSnapshot.
_SNAPSHOT_COLUMNS = {c.name for c in NFLPredictionSnapshot.__table__.columns}


def _init_engine():
    """Initialize the NFL scheduler's DB engine. Called once from start_scheduler()."""
    global _nfl_engine, _nfl_session_factory
    if _nfl_engine is None:
        _nfl_engine = create_async_engine(
            settings.async_database_url,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=1,
            pool_recycle=180,
            pool_timeout=30,
        )
        _nfl_session_factory = async_sessionmaker(
            _nfl_engine, class_=AsyncSession, expire_on_commit=False
        )


def nfl_session():
    """Get an async session from the NFL scheduler's dedicated engine."""
    if _nfl_session_factory is None:
        _init_engine()
    return _nfl_session_factory()


def log_task(message: str, **kwargs):
    """Log scheduler task output."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    print(f"[NFL-SCHEDULER] {timestamp} | {message} {extra}", flush=True)


def _today_et() -> date:
    """Today's date in US Eastern time (Railway runs in UTC)."""
    eastern = timedelta(hours=-5)  # EDT approximation (close enough for scheduling cadence)
    return (datetime.now(timezone.utc) + eastern).date()


def _current_season(today: date) -> int:
    """NFL season label for a given date (PURE).

    The 2026 season spans Sep 2026 - Feb 2027 and is labeled 2026:
    - Aug-Dec -> the calendar year (season is underway/about to start).
    - Jan-Feb -> year - 1 (still finishing last season, e.g. Super Bowl).
    - Mar-Jul -> the calendar year (offseason; points at the upcoming season).
    """
    if today.month in (1, 2):
        return today.year - 1
    return today.year


# ---------------------------------------------------------------------------
# Per-game scoring: pure helper (no I/O) factored out of snapshot_due_games
# so it can be unit-tested directly without mocking a DB session.
# ---------------------------------------------------------------------------
def _score_one(game: dict, home_stats: dict | None, away_stats: dict | None,
                context: dict | None, markets: list[dict],
                mov_bundle: dict, totals_bundle: dict) -> dict | None:
    """Score one game and build its snapshot row, or None if it can't be scored.

    `game`: dict with game_id/home_team/away_team/kickoff_utc/game_date/week/
    is_divisional/is_primetime. `home_stats`/`away_stats`: nfl_team_stats rows
    (as dicts) at through_week = game.week - 1, or None if unavailable.
    `markets`: nfl_markets rows (as dicts, one per market_type, latest capture)
    shaped like NFLMarket (market_type/line/home_odds/away_odds/over_odds/
    under_odds).
    """
    spread_line = next((m["line"] for m in markets if m.get("market_type") == "spread"), None)
    total_line = next((m["line"] for m in markets if m.get("market_type") == "total"), None)

    feat = build_live_feature_row(game, home_stats, away_stats, context, spread_line, total_line)
    if feat is None:
        return None

    scored = score_game(feat, markets, mov_bundle, totals_bundle)

    snap_game = {
        "game_id": game["game_id"],
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "kickoff_utc": game.get("kickoff_utc"),
        "game_date": game.get("game_date"),
        "snapshot_time": datetime.now(timezone.utc),
    }
    return build_snapshot(snap_game, scored)


def _stats_dict(row: NFLTeamStats) -> dict:
    return {
        "off_epa_play": row.off_epa_play, "def_epa_play": row.def_epa_play,
        "pass_epa": row.pass_epa, "rush_epa": row.rush_epa,
        "success_rate": row.success_rate, "pace": row.pace,
        "power_rating": row.power_rating,
    }


def _context_dict(row: NFLGameContext | None) -> dict:
    if row is None:
        return {}
    return {
        "home_rest_days": row.home_rest_days, "away_rest_days": row.away_rest_days,
        "wind_mph": row.wind_mph, "temp_f": row.temp_f, "is_dome": row.is_dome,
    }


def _latest_markets(rows: list[NFLMarket]) -> list[dict]:
    """Reduce raw nfl_markets rows to one dict per market_type: the row with
    the latest `captured_at`."""
    latest_captured: dict[str, datetime | None] = {}
    latest_row: dict[str, NFLMarket] = {}
    for r in rows:
        mt = r.market_type
        captured = r.captured_at
        prior = latest_captured.get(mt, "__unset__")
        if prior == "__unset__" or (captured is not None and (prior is None or captured > prior)):
            latest_captured[mt] = captured
            latest_row[mt] = r

    return [
        {
            "market_type": mt, "line": r.line,
            "home_odds": r.home_odds, "away_odds": r.away_odds,
            "over_odds": r.over_odds, "under_odds": r.under_odds,
        }
        for mt, r in latest_row.items()
    ]


async def snapshot_due_games(session: AsyncSession, minutes_before: int,
                              mov_bundle: dict | None = None,
                              totals_bundle: dict | None = None) -> dict:
    """Snapshot predictions for scheduled games kicking off within the window.

    Due window: status == "scheduled" AND kickoff_utc in
    [now_utc, now_utc + minutes_before]. Games that already have a snapshot
    are skipped. Bundles are loaded once (only if not injected) and reused
    across every due game in this run.
    """
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc + timedelta(minutes=minutes_before)

    already_snapshotted = select(NFLPredictionSnapshot.game_id)
    stmt = select(NFLGame).where(
        NFLGame.status == "scheduled",
        NFLGame.kickoff_utc >= now_utc,
        NFLGame.kickoff_utc <= cutoff,
        ~NFLGame.game_id.in_(already_snapshotted),
    )
    games = (await session.execute(stmt)).scalars().all()

    if not games:
        return {"snapshotted": 0}

    if mov_bundle is None:
        mov_bundle = model_training.load_bundle(settings.nfl_mov_model_path)
    if totals_bundle is None:
        totals_bundle = model_training.load_bundle(settings.nfl_totals_model_path)

    snapshotted = 0
    for game in games:
        home_stats_row = (await session.execute(
            select(NFLTeamStats).where(
                NFLTeamStats.team == game.home_team,
                NFLTeamStats.season == game.season,
                NFLTeamStats.through_week == game.week - 1,
            )
        )).scalar_one_or_none()
        away_stats_row = (await session.execute(
            select(NFLTeamStats).where(
                NFLTeamStats.team == game.away_team,
                NFLTeamStats.season == game.season,
                NFLTeamStats.through_week == game.week - 1,
            )
        )).scalar_one_or_none()
        context_row = (await session.execute(
            select(NFLGameContext).where(NFLGameContext.game_id == game.game_id)
        )).scalar_one_or_none()
        market_rows = (await session.execute(
            select(NFLMarket).where(NFLMarket.game_id == game.game_id)
        )).scalars().all()

        game_dict = {
            "game_id": game.game_id, "home_team": game.home_team, "away_team": game.away_team,
            "kickoff_utc": game.kickoff_utc,
            "game_date": game.kickoff_utc.date() if game.kickoff_utc else None,
            "week": game.week, "is_divisional": game.is_divisional,
            "is_primetime": game.is_primetime,
        }
        home_stats = _stats_dict(home_stats_row) if home_stats_row else None
        away_stats = _stats_dict(away_stats_row) if away_stats_row else None
        context = _context_dict(context_row)
        markets = _latest_markets(market_rows)

        snap = _score_one(game_dict, home_stats, away_stats, context, markets,
                           mov_bundle, totals_bundle)
        if snap is None:
            log_task("Skipping game (missing prior-week stats or line)", game_id=game.game_id)
            continue

        filtered = {k: v for k, v in snap.items() if k in _SNAPSHOT_COLUMNS}
        session.add(NFLPredictionSnapshot(**filtered))
        snapshotted += 1

    await session.commit()
    return {"snapshotted": snapshotted}


async def grade_finals(session: AsyncSession) -> dict:
    """Grade ungraded snapshots for completed games.

    "Completed" = nfl_games.home_score is not null; "ungraded" = the
    snapshot's best_bet_result is null. `snapshot.grade_snapshot` grades off
    the snapshot's own stored best_*_line/team/direction; its
    spread_line/total_line params are currently unused (reserved for future
    CLV), so we pass the snapshot's own stored best_spread_line/
    best_total_line for them.
    """
    stmt = (
        select(NFLPredictionSnapshot, NFLGame)
        .join(NFLGame, NFLGame.game_id == NFLPredictionSnapshot.game_id)
        .where(
            NFLGame.home_score.is_not(None),
            NFLPredictionSnapshot.best_bet_result.is_(None),
        )
    )
    rows = (await session.execute(stmt)).all()

    graded = 0
    for snap, game in rows:
        snap_dict = {
            "best_total_direction": snap.best_total_direction,
            "best_total_line": snap.best_total_line,
            "best_spread_team": snap.best_spread_team,
            "best_spread_line": snap.best_spread_line,
            "best_ml_team": snap.best_ml_team,
            "best_ml_odds": snap.best_ml_odds,
            "best_bet_type": snap.best_bet_type,
        }
        result = grade_snapshot(
            snap_dict, game.home_score, game.away_score,
            snap.best_spread_line, snap.best_total_line,
        )
        for key, value in result.items():
            setattr(snap, key, value)
        graded += 1

    await session.commit()
    return {"graded": graded}


async def weekly_refresh(session: AsyncSession) -> dict:
    """Pull the current season's schedule + recompute team stats.

    Weekly cadence (Tuesday, once Monday-night results have settled).
    `season_update.refresh_schedule`/`recompute_team_stats` don't commit
    internally -- this function owns the transaction.
    """
    season = _current_season(_today_et())
    n1 = await season_update.refresh_schedule(session, season)
    n2 = await season_update.recompute_team_stats(session, season)
    await session.commit()
    return {"games": n1, "stats": n2}


async def refresh_odds(session: AsyncSession) -> dict:
    """Pull current NFL odds -> nfl_markets. Runs more often than weekly_refresh
    (kept modest to respect the Odds API budget -- see start_scheduler cadence).
    """
    season = _current_season(_today_et())
    n = await season_update.odds_to_markets(session, season)
    await session.commit()
    return {"markets": n}


# ---------------------------------------------------------------------------
# Sync wrappers + persistent event loop (mirrors mlb_scheduler._run_async)
# ---------------------------------------------------------------------------
_nfl_loop: asyncio.AbstractEventLoop | None = None


def _run_async(coro):
    """Run an async coroutine on the NFL scheduler's persistent event loop."""
    global _nfl_loop
    if _nfl_loop is None:
        _nfl_loop = asyncio.new_event_loop()
    return _nfl_loop.run_until_complete(coro)


async def _snapshot_task_async() -> dict:
    async with nfl_session() as session:
        return await snapshot_due_games(session, settings.nfl_snapshot_minutes_before)


async def _grade_task_async() -> dict:
    async with nfl_session() as session:
        return await grade_finals(session)


async def _weekly_refresh_task_async() -> dict:
    async with nfl_session() as session:
        return await weekly_refresh(session)


async def _refresh_odds_task_async() -> dict:
    async with nfl_session() as session:
        return await refresh_odds(session)


def run_snapshot():
    """Sync wrapper for snapshot_due_games."""
    log_task("Running NFL snapshot...")
    try:
        result = _run_async(_snapshot_task_async())
        log_task("Snapshot complete", **result)
        _last_run_times["snapshot"] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Snapshot FAILED: {e}")
        _last_run_times["snapshot"] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


def run_grade():
    """Sync wrapper for grade_finals."""
    log_task("Running NFL grading...")
    try:
        result = _run_async(_grade_task_async())
        log_task("Grading complete", **result)
        _last_run_times["grade"] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Grading FAILED: {e}")
        _last_run_times["grade"] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


def run_weekly_refresh():
    """Sync wrapper for weekly_refresh."""
    log_task("Running NFL weekly refresh...")
    try:
        result = _run_async(_weekly_refresh_task_async())
        log_task("Weekly refresh complete", **result)
        _last_run_times["weekly_refresh"] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Weekly refresh FAILED: {e}")
        _last_run_times["weekly_refresh"] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


def run_refresh_odds():
    """Sync wrapper for refresh_odds."""
    log_task("Running NFL odds refresh...")
    try:
        result = _run_async(_refresh_odds_task_async())
        log_task("Odds refresh complete", **result)
        _last_run_times["refresh_odds"] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Odds refresh FAILED: {e}")
        _last_run_times["refresh_odds"] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


def start_scheduler():
    """Start the NFL scheduler loop. Ships DISABLED.

    Guarded on `settings.nfl_scheduler_enabled` (default False) -- flipping
    it on is a deliberate go-live switch for ~Sept 2026. Nothing calls this
    function anywhere in the app; it exists so the switch is a one-line flip
    (+ wiring a call to it, e.g. from main.py's startup) when the season
    actually starts.

    Uses its own `schedule.Scheduler()` instance + a dedicated engine/event
    loop so it never conflicts with the MLB/NBA schedulers running in the
    same process.
    """
    if not settings.nfl_scheduler_enabled:
        log_task("NFL scheduler disabled (nfl_scheduler_enabled=False) -- not starting.")
        return

    global _nfl_loop
    _nfl_loop = asyncio.new_event_loop()

    log_task("Starting NFL prediction scheduler...")
    log_task("Waiting 120s before initial run to avoid connection pressure during startup...")
    time.sleep(120)

    _init_engine()

    nfl_scheduler_instance = schedule.Scheduler()

    # Results settle Monday night -> pull schedule/stats fresh Tuesday.
    nfl_scheduler_instance.every().tuesday.do(run_weekly_refresh)
    # Keep the Odds API budget modest: once/day, Wed-Sun (not gameday-only,
    # so lines are fresh heading into the week too).
    for day in ("wednesday", "thursday", "friday", "saturday", "sunday"):
        getattr(nfl_scheduler_instance.every(), day).do(run_refresh_odds)
    nfl_scheduler_instance.every(1).hour.do(run_snapshot)
    nfl_scheduler_instance.every(1).hour.do(run_grade)

    log_task("NFL Scheduler configured:")
    log_task("  - Weekly refresh (schedule + team stats): every Tuesday")
    log_task("  - Odds refresh: daily Wed-Sun")
    log_task("  - Prediction snapshot: every 1 hour")
    log_task("  - Grading: every 1 hour")

    while True:
        nfl_scheduler_instance.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "snapshot":
            run_snapshot()
        elif command == "grade":
            run_grade()
        elif command == "weekly":
            run_weekly_refresh()
        elif command == "odds":
            run_refresh_odds()
        elif command == "daemon":
            start_scheduler()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python nfl_scheduler.py [snapshot|grade|weekly|odds|daemon]")
    else:
        print("Usage: python nfl_scheduler.py [snapshot|grade|weekly|odds|daemon]")
