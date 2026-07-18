"""Season update: upcoming schedule + current-season team stats + odds->markets.

Thin async orchestrations of already-built + reviewed Phase-1 functions
(nfl_data / features / ingest / tasks.nfl_backfill), plus one pure
event->game_id matcher used to resolve Odds API rows to `nfl_games`.

None of the three async functions here call ``session.commit()`` — the
caller owns the transaction (the P4 scheduler wraps multiple season-update
steps in one session/commit), mirroring the MLB season-update convention in
`services/mlb/ingest.py` where the orchestrator (not the low-level ingest
step) controls commit boundaries for a scheduler run.
"""
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import NFLGame, NFLMarket
from src.services.nfl.nfl_data import load_schedules, load_pbp, schedule_to_game_rows
from src.services.nfl.features import team_game_epa, rolling_team_stats
from src.services.nfl.ingest import upsert_games, upsert_game_context, upsert_team_stats
from src.services.nfl.odds_client import (
    NFLOddsClient, parse_nfl_odds_to_markets, NFL_TEAM_NAME_TO_ABBR,
)
from src.tasks.nfl_backfill import _clean_nan, _compute_candidate_features, _load_injury_depth

logger = structlog.get_logger()


def match_event_to_game(event_row: dict, games: list[dict]) -> str | None:
    """Pure: resolve an Odds API parsed-market row to an `nfl_games.game_id`.

    Matches on (home_team_abbr, away_team_abbr, date) — ALL THREE must agree.
    Date is part of the key (not just teams) because two teams can play twice
    in a season (e.g. a divisional rematch); matching on teams alone would
    pick the wrong game or collide.

    `event_row`: dict with `home_team_abbr`, `away_team_abbr`, `commence_date`
    (ISO `YYYY-MM-DD`).
    `games`: list of dicts with `game_id`, `home_team`, `away_team`,
    `kickoff_date` (ISO date string).

    Returns the matching `game_id`, or `None` if no game matches.
    """
    home = event_row.get("home_team_abbr")
    away = event_row.get("away_team_abbr")
    ev_date = event_row.get("commence_date")
    for g in games:
        if g.get("home_team") == home and g.get("away_team") == away and g.get("kickoff_date") == ev_date:
            return g.get("game_id")
    return None


async def refresh_schedule(session: AsyncSession, season: int) -> int:
    """Pull the current season's nflverse schedule (played + upcoming) and
    upsert `nfl_games` + `nfl_game_context`.

    Mirrors `tasks.nfl_backfill.backfill_season`'s schedule+context half
    (REG-season filter, `_clean_nan` sanitize, candidate-feature maps via
    the same best-effort helpers) but imports those helpers rather than
    duplicating them, and — unlike `backfill_season` — does NOT commit; the
    caller owns the transaction so the scheduler can batch this with
    `recompute_team_stats`/`odds_to_markets` in one commit.

    Returns the number of game rows upserted.
    """
    sched = load_schedules([season])
    sched = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    game_rows = [_clean_nan(r) for r in schedule_to_game_rows(sched)]

    injuries, depth = _load_injury_depth(season)
    home_so_map, away_so_map, home_stakes_map, away_stakes_map = _compute_candidate_features(
        sched, season, injuries, depth
    )

    await upsert_games(session, game_rows)
    await upsert_game_context(
        session, sched,
        home_starters_out_map=home_so_map,
        away_starters_out_map=away_so_map,
        home_stakes_map=home_stakes_map,
        away_stakes_map=away_stakes_map,
    )
    logger.info("nfl_refresh_schedule_done", season=season, games=len(game_rows))
    return len(game_rows)


async def recompute_team_stats(session: AsyncSession, season: int) -> int:
    """Recompute `nfl_team_stats` for the season from played games' pbp.

    REUSE: `nfl_data.load_pbp` -> `features.team_game_epa` ->
    `features.rolling_team_stats` -> `ingest.upsert_team_stats`. Idempotent
    (upsert on conflict of (team, season, through_week)) — running this twice
    for the same season yields the same row count, no dup-key error. No
    commit inside; the caller owns the transaction.

    Before a season's first game, nflverse has no play-by-play rows for it
    yet at all (not even an empty-but-columned frame) — `load_pbp`'s internal
    `pbp["season_type"] == "REG"` filter raises `KeyError` on that shape.
    That's a legitimate "nothing to recompute yet" state (mirrors
    `odds_to_markets` legitimately returning 0 out of season), not a bug in
    the reused Phase-1 loader, so it's caught here rather than crashing the
    scheduler run.

    Returns the number of team-stat rows written.
    """
    try:
        pbp = load_pbp([season])
    except KeyError:
        logger.info("nfl_recompute_team_stats_no_pbp_yet", season=season)
        return 0
    tg = team_game_epa(pbp)
    stats = rolling_team_stats(tg)
    n = await upsert_team_stats(session, stats)
    logger.info("nfl_recompute_team_stats_done", season=season, rows=n)
    return n


async def odds_to_markets(session: AsyncSession, season: int) -> int:
    """Fetch current NFL odds, match each parsed row to a `game_id`, and
    write `nfl_markets` rows.

    REUSE: `NFLOddsClient().get_nfl_odds()` -> `parse_nfl_odds_to_markets`.
    Each parsed row is matched against the season's `nfl_games`
    (game_id/home_team/away_team + kickoff_utc, from which we derive
    `kickoff_date` as the UTC ISO date) via `match_event_to_game`. Rows with
    no match are dropped (and the drop count logged) rather than written
    with a null/garbage game_id.

    `nfl_markets` has no natural unique key beyond the autoincrement
    `market_id` (unlike `nfl_games`/`nfl_team_stats`, which upsert on a
    natural key) — each capture is inserted as a new row, with `captured_at`
    (default utcnow) distinguishing successive runs. This mirrors
    `captured_at` being a snapshot timestamp rather than a "last updated"
    column.

    Out of season this legitimately returns 0 (no live odds events). No
    commit inside; the caller owns the transaction.

    Returns the number of market rows written.
    """
    events = await NFLOddsClient().get_nfl_odds()
    rows = parse_nfl_odds_to_markets(events, NFL_TEAM_NAME_TO_ABBR)
    if not rows:
        logger.info("nfl_odds_to_markets_no_events", season=season)
        return 0

    result = await session.execute(
        select(NFLGame.game_id, NFLGame.home_team, NFLGame.away_team, NFLGame.kickoff_utc)
        .where(NFLGame.season == season)
    )
    games = [
        {
            "game_id": r.game_id,
            "home_team": r.home_team,
            "away_team": r.away_team,
            # Odds API commence_time is a true UTC instant; nflverse
            # kickoff_utc is ET-wall-clock-as-UTC (see nfl_data._kickoff_utc),
            # i.e. it's mislabeled, not converted. CONFIRMED via prod smoke
            # (2026-07-18, live 75-event slate): this is NOT just a boundary
            # edge case — every primetime/evening kickoff (TNF/SNF/MNF, ET
            # kickoff late enough that +4/+5h true UTC crosses midnight) has
            # a true commence_date one day LATER than this kickoff_date, so
            # match_event_to_game legitimately (and per spec, correctly)
            # drops those rows rather than mis-matching them. See the P4
            # Task 3 report for the count. Fixing this needs a real UTC
            # kickoff_utc in nfl_data._kickoff_utc (Phase-1, out of scope
            # here — not one of this task's staged files).
            "kickoff_date": r.kickoff_utc.date().isoformat() if r.kickoff_utc else None,
        }
        for r in result.all()
    ]

    matched_rows = []
    dropped = 0
    for row in rows:
        game_id = match_event_to_game(row, games)
        if game_id is None:
            dropped += 1
            continue
        matched_rows.append({
            "game_id": game_id,
            "market_type": row["market_type"],
            "line": row["line"],
            "home_odds": row["home_odds"],
            "away_odds": row["away_odds"],
            "over_odds": row["over_odds"],
            "under_odds": row["under_odds"],
            "book": row["book"],
        })

    logger.info(
        "nfl_odds_to_markets_matched",
        season=season, matched=len(matched_rows), dropped=dropped,
    )

    if not matched_rows:
        return 0

    session.add_all([NFLMarket(**r) for r in matched_rows])
    return len(matched_rows)
