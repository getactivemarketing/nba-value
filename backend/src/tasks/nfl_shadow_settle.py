"""Attach closing lines and settle CLV for shadow predictions. Also: coverage.

Fills ONLY settlement columns. The frozen candidate and model fields are a
historical record of what was forecast and are never rewritten.

Two closes per row — same-book (what the bettor could have got) and median
consensus (robust to one book going stale) — with explicit staleness and an
explicit fallback label, so a missing book close degrades to something
readable rather than a silent gap.

    python -m src.tasks.nfl_shadow_settle [--season 2026] [--coverage-only]
"""

import asyncio
import sys
from datetime import datetime, timezone

import structlog
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import settings
from src.models import NFLGame, NFLShadowPrediction
from src.services.nfl.settlement import (
    CloseSource, consensus_close, select_book_close, settle_candidate,
)

logger = structlog.get_logger()

_QUOTE_COLS = (
    "book", "snapshot_time", "ml_home", "ml_away", "spread_line",
    "spread_home_odds", "spread_away_odds", "total_line", "over_odds", "under_odds",
)


def _actual_value(market: str, home_score: int, away_score: int) -> float:
    """Home margin for spread/moneyline, combined points for totals."""
    return float(home_score + away_score) if market == "total" else float(home_score - away_score)


async def settle(season: int = 2026) -> dict:
    engine = create_async_engine(settings.async_database_url, echo=False)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    settled = 0
    no_close = 0
    by_source: dict[str, int] = {}

    async with Session() as session:
        games = (await session.execute(
            select(NFLGame).where(
                NFLGame.season == season,
                NFLGame.home_score.isnot(None),
                NFLGame.kickoff_utc.isnot(None),
            )
        )).scalars().all()

        for game in games:
            rows = (await session.execute(
                select(NFLShadowPrediction).where(
                    NFLShadowPrediction.game_id == game.game_id,
                    NFLShadowPrediction.settled_at.is_(None),
                )
            )).scalars().all()
            if not rows:
                continue

            quotes = [dict(r._mapping) for r in (await session.execute(text(f"""
                SELECT {', '.join(_QUOTE_COLS)}
                FROM nfl_odds_snapshots WHERE game_id = :gid
            """), {"gid": game.game_id})).all()]

            # Consensus per (market, side) — computed once, reused across books
            # and models so every row sees the same robust close.
            consensus_cache: dict[tuple[str, str], dict | None] = {}

            for row in rows:
                key = (row.market_type, row.side)
                if key not in consensus_cache:
                    consensus_cache[key] = consensus_close(
                        quotes, game.kickoff_utc, market=row.market_type, side=row.side)

                book_quotes = [q for q in quotes if q["book"] == row.book]
                close = select_book_close(book_quotes, game.kickoff_utc)
                if close is not None:
                    line_key = {"spread": "spread_line", "total": "total_line"}.get(row.market_type)
                    own = {
                        "spread": "spread_home_odds" if row.side == "home" else "spread_away_odds",
                        "total": "over_odds" if row.side == "over" else "under_odds",
                        "moneyline": "ml_home" if row.side == "home" else "ml_away",
                    }[row.market_type]
                    opp = {
                        "spread": "spread_away_odds" if row.side == "home" else "spread_home_odds",
                        "total": "under_odds" if row.side == "over" else "over_odds",
                        "moneyline": "ml_away" if row.side == "home" else "ml_home",
                    }[row.market_type]
                    raw_line = close.get(line_key) if line_key else None
                    # The away spread row carries the mirrored line, so the
                    # close must be mirrored to match its perspective.
                    if row.market_type == "spread" and raw_line is not None and row.side == "away":
                        raw_line = -float(raw_line)
                    close = {
                        "line": None if raw_line is None else float(raw_line),
                        "odds": close.get(own), "opposing_odds": close.get(opp),
                        "staleness_min": close["staleness_min"],
                    }
                    if close["odds"] is None and close["line"] is None:
                        close = None

                result = settle_candidate(
                    candidate={"market_type": row.market_type, "side": row.side,
                               "line": float(row.line) if row.line is not None else None,
                               "odds_decimal": float(row.odds_decimal) if row.odds_decimal else None},
                    book_close=close,
                    consensus=consensus_cache[key],
                    actual_value=_actual_value(row.market_type, game.home_score, game.away_score),
                )

                by_source[result["closing_source"]] = by_source.get(result["closing_source"], 0) + 1
                if result["closing_source"] == CloseSource.NONE:
                    no_close += 1

                await session.execute(
                    update(NFLShadowPrediction)
                    .where(NFLShadowPrediction.id == row.id)
                    .values(settled_at=datetime.now(timezone.utc), **{
                        k: result[k] for k in (
                            "closing_source", "closing_line", "closing_odds",
                            "closing_novig_prob", "closing_staleness_min",
                            "consensus_line", "consensus_novig_prob", "consensus_book_count",
                            "line_clv", "price_clv", "line_clv_consensus",
                            "price_clv_consensus", "crossed_key", "actual_value", "side_won",
                        )
                    })
                )
                settled += 1

        await session.commit()

    await engine.dispose()
    out = {"season": season, "settled": settled, "no_close": no_close, "by_source": by_source}
    logger.info("nfl_shadow_settle_done", **{k: str(v) for k, v in out.items()})
    return out


async def coverage(season: int = 2026) -> dict:
    """Every count the reconciliation needs, in one place.

    Reported together on purpose: a healthy-looking "scored" number means
    nothing without the candidate count it came from, and a graded count means
    nothing without knowing how many rows had a close attached.
    """
    engine = create_async_engine(settings.async_database_url, echo=False)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with Session() as session:
        row = (await session.execute(text("""
            SELECT
              COUNT(DISTINCT candidate_hash)                                   AS candidates,
              COUNT(*)                                                         AS model_attempts,
              COUNT(*) FILTER (WHERE status = 'ok')                            AS scored_ok,
              COUNT(*) FILTER (WHERE status <> 'ok')                           AS failures,
              COUNT(*) FILTER (WHERE closing_source = 'book')                  AS book_closes,
              COUNT(*) FILTER (WHERE closing_source = 'consensus_fallback')    AS consensus_closes,
              COUNT(*) FILTER (WHERE closing_source = 'none' AND settled_at IS NOT NULL)
                                                                               AS closes_missing,
              COUNT(*) FILTER (WHERE settled_at IS NOT NULL)                   AS settled,
              COUNT(*) FILTER (WHERE side_won IS NOT NULL)                     AS graded,
              COUNT(*) FILTER (WHERE price_clv IS NOT NULL)                    AS price_clv_rows,
              COUNT(*) FILTER (WHERE line_clv IS NOT NULL)                     AS line_clv_rows,
              COUNT(DISTINCT model_key)                                        AS models,
              COUNT(DISTINCT game_id)                                          AS games
            FROM nfl_shadow_predictions WHERE season = :season
        """), {"season": season})).one()
        stats = dict(row._mapping)

        breakdown = (await session.execute(text("""
            SELECT model_key, status, COUNT(*) AS n
            FROM nfl_shadow_predictions WHERE season = :season
            GROUP BY model_key, status ORDER BY model_key, status
        """), {"season": season})).all()

    await engine.dispose()
    stats["by_model_status"] = {f"{r.model_key}/{r.status}": r.n for r in breakdown}
    # Ratios, so a shrinking denominator cannot masquerade as improvement.
    attempts = max(stats["model_attempts"], 1)
    stats["scored_pct"] = round(stats["scored_ok"] / attempts * 100, 1)
    stats["close_attach_pct"] = round(
        (stats["book_closes"] + stats["consensus_closes"]) / max(stats["settled"], 1) * 100, 1)
    return stats


if __name__ == "__main__":
    args = sys.argv[1:]
    season = 2026
    for i, a in enumerate(args):
        if a == "--season" and i + 1 < len(args):
            season = int(args[i + 1])
    if "--coverage-only" not in args:
        print(asyncio.run(settle(season)))
    import json
    print(json.dumps(asyncio.run(coverage(season)), indent=2, default=str))
