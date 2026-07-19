"""NFL API endpoints: picks, upcoming games, and an odds debug probe.

Reads the same `nfl_prediction_snapshots` / `nfl_games` tables the weekly
scheduler writes. Picks are the frozen `best_bet` selections, which are
totals-only live (spread + moneyline are shadow-recorded, never best_bet).
Mirrors `src/api/mlb.py`'s conventions (router prefix, `async_session()`
usage, min_value_score default 40, key-prefix masking).
"""
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import asc, select

from src.database import async_session
from src.models import NFLGame, NFLPredictionSnapshot

router = APIRouter(prefix="/nfl", tags=["NFL"])


# --- Response models -------------------------------------------------------

class NFLPick(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    kickoff_utc: datetime | None
    best_bet_type: str | None
    best_bet_team: str | None
    best_bet_line: float | None
    best_bet_odds: float | None
    best_bet_value_score: float | None
    best_bet_edge: float | None
    predicted_margin: float | None
    predicted_total: float | None


class NFLPicksResponse(BaseModel):
    picks: list[NFLPick]
    total: int
    min_value_score: float


class NFLGameSummary(BaseModel):
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    kickoff_utc: datetime | None
    is_divisional: bool | None
    is_primetime: bool | None
    # best_bet summary from the snapshot, when one exists (else null = not yet snapshotted)
    best_bet_type: str | None = None
    best_bet_team: str | None = None
    best_bet_line: float | None = None
    best_bet_value_score: float | None = None


class NFLGamesResponse(BaseModel):
    games: list[NFLGameSummary]
    total: int


# --- Endpoints -------------------------------------------------------------

@router.get("/picks", response_model=NFLPicksResponse)
async def get_picks(
    min_value_score: float = Query(
        40, ge=0, le=100,
        description="Minimum best_bet value score (display scale, same default as MLB).",
    ),
    limit: int = Query(20, ge=1, le=50, description="Maximum picks to return"),
) -> NFLPicksResponse:
    """Frozen best_bet picks with value_score >= min_value_score, soonest kickoff first.

    best_bet is totals-only while spread/ML are shadow-gated, so these are the
    live-bettable totals selections.
    """
    async with async_session() as session:
        stmt = (
            select(NFLPredictionSnapshot)
            .where(NFLPredictionSnapshot.best_bet_value_score >= min_value_score)
            .order_by(asc(NFLPredictionSnapshot.kickoff_utc))
            .limit(limit)
        )
        snapshots = (await session.execute(stmt)).scalars().all()

    picks = [
        NFLPick(
            game_id=s.game_id,
            home_team=s.home_team,
            away_team=s.away_team,
            kickoff_utc=s.kickoff_utc,
            best_bet_type=s.best_bet_type,
            best_bet_team=s.best_bet_team,
            best_bet_line=s.best_bet_line,
            best_bet_odds=s.best_bet_odds,
            best_bet_value_score=s.best_bet_value_score,
            best_bet_edge=s.best_bet_edge,
            predicted_margin=s.predicted_margin,
            predicted_total=s.predicted_total,
        )
        for s in snapshots
        if s.best_bet_type and s.best_bet_value_score is not None
    ]
    return NFLPicksResponse(picks=picks, total=len(picks), min_value_score=min_value_score)


@router.get("/games", response_model=NFLGamesResponse)
async def get_games(
    season: int | None = Query(None, description="Filter by season (e.g. 2026)"),
    week: int | None = Query(None, ge=1, le=22, description="Filter by week"),
    limit: int = Query(32, ge=1, le=64, description="Maximum games to return"),
) -> NFLGamesResponse:
    """Upcoming NFL games (soonest kickoff first) with their snapshot best_bet, if any.

    Defaults to the nearest upcoming scheduled games; pass season+week to pin a slate.
    """
    async with async_session() as session:
        stmt = select(NFLGame)
        if season is not None:
            stmt = stmt.where(NFLGame.season == season)
        if week is not None:
            stmt = stmt.where(NFLGame.week == week)
        if season is None and week is None:
            # nearest upcoming slate: only games not yet final
            stmt = stmt.where(NFLGame.status == "scheduled")
        stmt = stmt.order_by(asc(NFLGame.kickoff_utc)).limit(limit)
        games = (await session.execute(stmt)).scalars().all()

        game_ids = [g.game_id for g in games]
        snaps_by_id: dict[str, NFLPredictionSnapshot] = {}
        if game_ids:
            snap_rows = (await session.execute(
                select(NFLPredictionSnapshot).where(NFLPredictionSnapshot.game_id.in_(game_ids))
            )).scalars().all()
            # if a game has multiple snapshots, keep the latest by snapshot_time
            for s in snap_rows:
                cur = snaps_by_id.get(s.game_id)
                if cur is None or (s.snapshot_time and cur.snapshot_time
                                   and s.snapshot_time > cur.snapshot_time):
                    snaps_by_id[s.game_id] = s

    summaries = []
    for g in games:
        s = snaps_by_id.get(g.game_id)
        summaries.append(NFLGameSummary(
            game_id=g.game_id, season=g.season, week=g.week,
            home_team=g.home_team, away_team=g.away_team, kickoff_utc=g.kickoff_utc,
            is_divisional=g.is_divisional, is_primetime=g.is_primetime,
            best_bet_type=s.best_bet_type if s else None,
            best_bet_team=s.best_bet_team if s else None,
            best_bet_line=s.best_bet_line if s else None,
            best_bet_value_score=s.best_bet_value_score if s else None,
        ))
    return NFLGamesResponse(games=summaries, total=len(summaries))


@router.get("/debug/odds")
async def debug_odds(live: bool = Query(
    True, description="Also hit The Odds API for a live event count (set false to skip the network call)")
) -> dict:
    """Odds/data sanity probe for the ~Sept scheduler go-live.

    Reports masked key presence, current nfl_* table counts, and (when
    live=true) a live The Odds API event/market count so odds ingestion can be
    verified before enabling the scheduler. Never exposes the full API key
    (8-char prefix only). A DB or Odds-API failure is reported in-band, not as
    a 500 — this is a diagnostic endpoint.
    """
    from sqlalchemy import text
    from src.config import settings

    results: dict = {
        "odds_api_key_set": bool(settings.odds_api_key),
        "odds_api_key_prefix": (settings.odds_api_key[:8] + "...") if settings.odds_api_key else None,
    }
    try:
        async with async_session() as session:
            for label, table in (
                ("existing_markets", "nfl_markets"),
                ("existing_games", "nfl_games"),
                ("existing_snapshots", "nfl_prediction_snapshots"),
            ):
                row = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                results[label] = row.scalar()
    except Exception as e:  # noqa: BLE001 - debug probe must never 500 the caller
        raise HTTPException(status_code=503, detail=f"NFL data probe failed: {e}")

    if live:
        # Live odds probe (network). Isolated so an Odds-API hiccup still returns
        # the DB counts above rather than failing the whole endpoint.
        from src.services.nfl.odds_client import (
            NFLOddsClient, NFL_TEAM_NAME_TO_ABBR, parse_nfl_odds_to_markets,
        )
        try:
            events = await NFLOddsClient().get_nfl_odds()
            rows = parse_nfl_odds_to_markets(events, NFL_TEAM_NAME_TO_ABBR)
            results["live_event_count"] = len(events)
            results["live_market_row_count"] = len(rows)
        except Exception as e:  # noqa: BLE001 - network probe is best-effort
            results["live_odds_error"] = str(e)

    return results
