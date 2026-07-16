"""Backfill NFL games + team stats from nflverse for a season range."""
import asyncio
import math
import sys

import pandas as pd
import structlog

from src.database import async_session_maker, init_db
from src.services.nfl.nfl_data import load_schedules, load_pbp, schedule_to_game_rows
from src.services.nfl.features import (
    team_game_epa, rolling_team_stats, starters_out, playoff_stakes,
)
from src.services.nfl.ingest import upsert_games, upsert_game_context, upsert_team_stats

logger = structlog.get_logger()


def _clean_nan(row: dict) -> dict:
    """schedule_to_game_rows can leave raw NaN floats (not None) on optional
    string fields (e.g. surface, roof, QB names) when nflverse data is
    missing for a game; asyncpg rejects NaN for VARCHAR params. Sanitize
    defensively here rather than touching the already-reviewed ingest layer.
    """
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


def _load_injury_depth(season: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Best-effort load of nflverse injuries + depth charts for a season.

    These datasets have gaps and schema drift for older seasons (they don't
    go back nearly as far as schedules/pbp), so any failure here is caught
    and logged rather than allowed to fail the season backfill — starters_out
    is a Phase-2 candidate feature, not a correctness-critical one.

    Column-name note: import_injuries gives ``team``/``gsis_id``/
    ``report_status`` matching the pure starters_out contract directly.
    import_depth_charts gives ``club_code`` (renamed to ``team`` here) and a
    STRING ``depth_team`` ("1"/"2"/"3"), coerced to numeric here so the
    ``== 1`` comparison in starters_out works.
    """
    try:
        import nfl_data_py as nfl
        injuries = nfl.import_injuries([season])
        depth = nfl.import_depth_charts([season])
        if "club_code" in depth.columns:
            depth = depth.rename(columns={"club_code": "team"})
        if "depth_team" in depth.columns:
            depth = depth.copy()
            depth["depth_team"] = pd.to_numeric(depth["depth_team"], errors="coerce")
        return injuries, depth
    except Exception as e:   # older seasons / schema drift — candidate feature only
        logger.warning("nfl_injury_data_unavailable", season=season, error=str(e))
        return None, None


def _standings_through_week(sched: pd.DataFrame, week: int) -> pd.DataFrame:
    """Wins per team from completed games with week <= `week` (pure tally
    over the season schedule, which carries final scores for played games).
    """
    teams = pd.concat([sched["home_team"], sched["away_team"]]).unique()
    wins = {t: 0 for t in teams}
    played = sched[
        (sched["week"] <= week) & sched["home_score"].notna() & sched["away_score"].notna()
    ]
    for _, g in played.iterrows():
        if g["home_score"] > g["away_score"]:
            wins[g["home_team"]] += 1
        elif g["away_score"] > g["home_score"]:
            wins[g["away_team"]] += 1
    return pd.DataFrame([{"team": t, "wins": w} for t, w in wins.items()])


def _compute_candidate_features(
    sched: pd.DataFrame,
    season: int,
    injuries: pd.DataFrame | None,
    depth: pd.DataFrame | None,
) -> tuple[dict | None, dict | None, dict, dict]:
    """Best-effort per-game starters_out + playoff_stakes, keyed by game_id.

    Returns (home_starters_out, away_starters_out, home_stakes, away_stakes).
    The starters_out maps are None (not {}) when injury/depth data was
    entirely unavailable this run, so upsert_game_context knows to leave
    those columns untouched rather than writing NULL over a prior good run.
    playoff_stakes only depends on the already-loaded schedule, so it's
    always attempted; per-game failures are caught individually.
    """
    have_injury_data = injuries is not None and depth is not None
    home_starters_out: dict[str, int] | None = {} if have_injury_data else None
    away_starters_out: dict[str, int] | None = {} if have_injury_data else None
    home_stakes: dict[str, str] = {}
    away_stakes: dict[str, str] = {}
    stakes_cache: dict[int, dict[str, str]] = {}

    for _, g in sched.iterrows():
        try:
            game_id, week = g["game_id"], int(g["week"])
            home, away = g["home_team"], g["away_team"]
        except (KeyError, TypeError, ValueError) as e:
            # Malformed schedule row (e.g. NaN week, missing team/id) — skip
            # this game rather than aborting the whole season backfill.
            logger.warning("nfl_candidate_features_bad_row", error=str(e))
            continue

        if have_injury_data:
            try:
                wk_inj = injuries[injuries["week"] == week] if "week" in injuries else injuries
                wk_depth = depth[depth["week"] == week] if "week" in depth else depth
                home_starters_out[game_id] = starters_out(wk_inj, wk_depth, home)
                away_starters_out[game_id] = starters_out(wk_inj, wk_depth, away)
            except Exception as e:
                logger.warning("nfl_starters_out_failed", game_id=game_id, error=str(e))

        try:
            if week not in stakes_cache:
                standings = _standings_through_week(sched, week)
                stakes_cache[week] = playoff_stakes(standings, season, week)
            stakes = stakes_cache[week]
            home_stakes[game_id] = stakes.get(home)
            away_stakes[game_id] = stakes.get(away)
        except Exception as e:
            logger.warning("nfl_playoff_stakes_failed", game_id=game_id, error=str(e))

    return home_starters_out, away_starters_out, home_stakes, away_stakes


async def backfill_season(season: int) -> None:
    sched = load_schedules([season])
    sched = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    game_rows = [_clean_nan(r) for r in schedule_to_game_rows(sched)]

    pbp = load_pbp([season])
    tg = team_game_epa(pbp)
    stats = rolling_team_stats(tg)

    injuries, depth = _load_injury_depth(season)
    home_so_map, away_so_map, home_stakes_map, away_stakes_map = _compute_candidate_features(
        sched, season, injuries, depth
    )

    async with async_session_maker() as session:
        await upsert_games(session, game_rows)
        await upsert_game_context(
            session, sched,
            home_starters_out_map=home_so_map,
            away_starters_out_map=away_so_map,
            home_stakes_map=home_stakes_map,
            away_stakes_map=away_stakes_map,
        )
        await upsert_team_stats(session, stats)
        await session.commit()
    logger.info("nfl_backfill_season_done", season=season,
                games=len(game_rows), stat_rows=len(stats))


async def main(start: int, end: int) -> None:
    await init_db()   # ensure nfl_* tables exist
    for season in range(start, end + 1):
        await backfill_season(season)


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2010
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    asyncio.run(main(start, end))
