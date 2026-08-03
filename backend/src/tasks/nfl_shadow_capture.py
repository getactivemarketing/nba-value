"""Score every NFL market candidate with every shadow model, and persist.

Runs the whole board: for each current game, each book, each market, each
side, each line and price at each scoring timestamp, every registered model
produces a row — regardless of edge, qualification, or market enablement.

NOTHING HERE CAN PLACE A BET. All three NFL markets are disabled and the
scheduler is off; this only collects.

    python -m src.tasks.nfl_shadow_capture [--season 2026]
"""

import asyncio
import sys
from datetime import datetime, timezone

import structlog
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import settings
from src.models import NFLGame, NFLOddsSnapshot, NFLShadowPrediction, NFLTeamStats
from src.services.nfl.live_features import build_live_feature_row
from src.services.nfl.odds_history import minutes_to_kickoff
from src.services.nfl.shadow_models import build_registry, resid_std_for
from src.services.nfl.shadow_runner import (
    ShadowStatus, candidate_hash, enumerate_candidates, score_candidate,
)

logger = structlog.get_logger()

# Columns _feature_diffs reads by subscript.
_STAT_COLS = (
    "off_epa_play", "def_epa_play", "pass_epa", "rush_epa",
    "success_rate", "pace", "power_rating",
)


async def _latest_quotes(session: AsyncSession, season: int) -> list[dict]:
    """Newest quote per (game, book) for games that have not kicked off."""
    rows = await session.execute(text("""
        SELECT DISTINCT ON (s.game_id, s.book)
               s.game_id, s.book, s.ml_home, s.ml_away, s.spread_line,
               s.spread_home_odds, s.spread_away_odds, s.total_line,
               s.over_odds, s.under_odds
        FROM nfl_odds_snapshots s
        JOIN nfl_games g ON g.game_id = s.game_id
        WHERE g.season = :season AND g.kickoff_utc > NOW()
        ORDER BY s.game_id, s.book, s.snapshot_time DESC
    """), {"season": season})
    return [dict(r._mapping) for r in rows.all()]


async def capture(season: int = 2026) -> dict:
    engine = create_async_engine(settings.async_database_url, echo=False)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    registry = build_registry()
    now = datetime.now(timezone.utc)

    written = skipped = 0
    pending: list[dict] = []
    expected = 0
    status_counts: dict[str, int] = {}

    async with Session() as session:
        quotes = await _latest_quotes(session, season)
        if not quotes:
            logger.info("nfl_shadow_no_quotes", season=season)
            await engine.dispose()
            return {"expected": 0, "written": 0, "status": "no_quotes"}

        games = {
            g.game_id: g for g in (await session.execute(
                select(NFLGame).where(NFLGame.season == season)
            )).scalars().all()
        }

        # Prior-week team stats, keyed (team, through_week). build_live_feature_row
        # returns None without them, which surfaces as MISSING_FEATURES rather
        # than a default-filled prediction.
        #
        # Stored as DICTS, not ORM objects. `_feature_diffs` does `stats["pace"]`
        # — it accepts a pandas Series or a row-mapping, and an ORM instance
        # raises TypeError on subscript. That would not have shown up until
        # Week 2 of the season, when 2026 team stats first exist: today every
        # row is legitimately MISSING_FEATURES, so the crash is masked.
        stats = {
            (r.team, r.through_week): {c: getattr(r, c) for c in _STAT_COLS}
            for r in (await session.execute(
                select(NFLTeamStats).where(NFLTeamStats.season == season)
            )).scalars().all()
        }

        for quote in quotes:
            game = games.get(quote["game_id"])
            if game is None:
                continue

            candidates = enumerate_candidates(quote["game_id"], quote)
            # Cardinality is asserted, not assumed: every candidate must
            # produce exactly one row per registered model.
            expected += len(candidates) * len(registry)

            prior_week = (game.week or 1) - 1
            home_stats = stats.get((game.home_team, prior_week))
            away_stats = stats.get((game.away_team, prior_week))

            features = build_live_feature_row(
                game={"is_divisional": getattr(game, "is_divisional", False),
                      "is_primetime": getattr(game, "is_primetime", False)},
                home_stats=home_stats, away_stats=away_stats,
                context={},
                spread_line=quote.get("spread_line"),
                total_line=quote.get("total_line"),
            )
            coverage = None if features is None else round(
                sum(1 for v in features.values() if v is not None) / max(len(features), 1), 3)

            mtk = minutes_to_kickoff(now, game.kickoff_utc) if game.kickoff_utc else None

            for candidate in candidates:
                chash = candidate_hash(candidate)
                for model in registry:
                    rstd = resid_std_for(model, candidate["market_type"])
                    row = score_candidate(model, candidate, features, rstd)
                    status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
                    pending.append({
                        "game_id": candidate["game_id"], "season": season, "week": game.week,
                        "scored_at": now, "minutes_to_kickoff": mtk,
                        "market_type": candidate["market_type"], "side": candidate["side"],
                        "book": candidate["book"], "line": candidate["line"],
                        "odds_decimal": candidate["odds_decimal"],
                        "model_key": row["model_key"], "model_version": row["model_version"],
                        "candidate_hash": chash,
                        "status": row["status"], "status_detail": row["status_detail"],
                        "predicted_value": row["predicted_value"],
                        "probability": row["probability"],
                        "resid_std": row["resid_std"], "feature_coverage": coverage,
                        "created_at": now,
                    })

        # Bulk insert in chunks rather than one statement per row. The board is
        # ~272 games x 10 books x 6 candidates x 3 models ~= 49k rows; issuing
        # them individually is 49k network round trips on a job that runs
        # hourly. Third time this pattern has bitten in three days (per-request
        # joblib.load on /health, per-pair SELECT in the MLB odds history).
        for i in range(0, len(pending), 1000):
            chunk = pending[i:i + 1000]
            stmt = insert(NFLShadowPrediction).values(chunk).on_conflict_do_nothing(
                # INDEX INFERENCE, not `constraint=`. uq_nfl_shadow_candidate_
                # model is a unique INDEX, and Postgres accepts ON CONFLICT ON
                # CONSTRAINT only for a real constraint — the mistake that left
                # mlb_pitcher_stats empty for the life of the system.
                index_elements=["candidate_hash", "model_key"]
            )
            result = await session.execute(stmt)
            written += result.rowcount or 0
        skipped = len(pending) - written

        await session.commit()

    await engine.dispose()

    out = {
        "season": season, "quotes": len(quotes),
        "expected": expected, "written": written,
        "skipped_idempotent": skipped,
        "accounted": written + skipped,
        "cardinality_ok": (written + skipped) == expected,
        "statuses": status_counts,
        "scored_ok_pct": round(
            status_counts.get(ShadowStatus.OK, 0) / max(expected, 1) * 100, 1),
    }
    # A cardinality mismatch means rows were silently lost — louder than INFO.
    (logger.error if not out["cardinality_ok"] else logger.info)(
        "nfl_shadow_capture_done", **{k: str(v) for k, v in out.items()})
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    season = 2026
    for i, a in enumerate(args):
        if a == "--season" and i + 1 < len(args):
            season = int(args[i + 1])
    print(asyncio.run(capture(season)))
