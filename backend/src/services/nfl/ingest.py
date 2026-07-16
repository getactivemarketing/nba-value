"""Persist nflverse-derived DataFrames into nfl_* tables (idempotent upserts).

Upserts are batched into chunked multi-row ``INSERT ... ON CONFLICT DO UPDATE``
statements rather than one round-trip per row. A per-row loop is fine for the
live weekly slate (a dozen games) but crippling for the historical backfill
(~4k games x 15 seasons) over a remote DB's latency. Every batch here has
unique conflict keys within it (a season's schedule has unique game_ids; the
rolling-stats output has unique (team, season, through_week)), so the multi-row
form is safe from Postgres' "cannot affect row a second time" error.
"""
import pandas as pd
import structlog
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import NFLGame, NFLGameContext, NFLTeamStats

logger = structlog.get_logger()

# Keep each statement well under Postgres' 65535 bound parameter limit
# (largest row here is nfl_games at ~22 cols -> 500*22 = 11k params).
_CHUNK = 500


def _clean(row: dict) -> dict:
    """NaN floats -> None so asyncpg accepts them for non-float columns."""
    return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}


async def _bulk_upsert(
    session: AsyncSession, model, rows: list[dict], index_elements: list[str]
) -> int:
    """Chunked multi-row upsert. Updates exactly the columns present in the
    source rows (minus the conflict keys and the autoincrement ``id``), so the
    semantics match a per-row upsert of the same dicts."""
    if not rows:
        return 0
    rows = [_clean(r) for r in rows]
    update_keys = [k for k in rows[0] if k not in index_elements and k != "id"]
    for start in range(0, len(rows), _CHUNK):
        batch = rows[start:start + _CHUNK]
        stmt = insert(model).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=index_elements,
            set_={k: getattr(stmt.excluded, k) for k in update_keys},
        )
        await session.execute(stmt)
    return len(rows)


async def upsert_games(session: AsyncSession, game_rows: list[dict]) -> int:
    n = await _bulk_upsert(session, NFLGame, game_rows, ["game_id"])
    logger.info("nfl_upsert_games", count=n)
    return n


def _is_dome(roof) -> bool:
    """True for enclosed venues; safe against NaN/None/non-str roof values."""
    return isinstance(roof, str) and roof.strip().lower() in ("dome", "closed")


async def upsert_game_context(
    session: AsyncSession,
    sched: pd.DataFrame,
    home_starters_out_map: dict[str, int] | None = None,
    away_starters_out_map: dict[str, int] | None = None,
    home_stakes_map: dict[str, str] | None = None,
    away_stakes_map: dict[str, str] | None = None,
) -> int:
    """Upsert game context rows.

    The four *_map params are OPTIONAL and keyed by game_id — best-effort
    candidate features (starters_out / playoff_stakes; see spec). Passing
    ``None`` for a map (the default) omits that pair of columns from the
    upsert entirely, leaving any previously-written values untouched rather
    than clobbering them with NULL when this run's data happened to be
    unavailable (e.g. injury feed down for this backfill pass). When a map IS
    provided, a game_id missing from it is written as NULL for that column —
    "we tried, no data for this game" rather than "we didn't try."
    """
    rows = []
    for _, g in sched.iterrows():
        game_id = g["game_id"]
        row = {
            "game_id": game_id,
            "home_rest_days": None if pd.isna(g.get("home_rest")) else int(g["home_rest"]),
            "away_rest_days": None if pd.isna(g.get("away_rest")) else int(g["away_rest"]),
            "temp_f": None if pd.isna(g.get("temp")) else float(g["temp"]),
            "wind_mph": None if pd.isna(g.get("wind")) else float(g["wind"]),
            "is_dome": _is_dome(g.get("roof")),
        }
        if home_starters_out_map is not None:
            row["home_starters_out"] = home_starters_out_map.get(game_id)
        if away_starters_out_map is not None:
            row["away_starters_out"] = away_starters_out_map.get(game_id)
        if home_stakes_map is not None:
            row["home_playoff_stakes"] = home_stakes_map.get(game_id)
        if away_stakes_map is not None:
            row["away_playoff_stakes"] = away_stakes_map.get(game_id)
        rows.append(row)
    n = await _bulk_upsert(session, NFLGameContext, rows, ["game_id"])
    logger.info("nfl_upsert_game_context", count=n)
    return n


async def upsert_team_stats(session: AsyncSession, stats: pd.DataFrame) -> int:
    rows = stats.to_dict("records")
    n = await _bulk_upsert(session, NFLTeamStats, rows, ["team", "season", "through_week"])
    logger.info("nfl_upsert_team_stats", count=n)
    return n
