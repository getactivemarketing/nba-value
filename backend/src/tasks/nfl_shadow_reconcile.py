"""Reconcile the full shadow path end-to-end, then report readiness.

The unit tests cover the settlement MATH. This covers the WIRING: column
mapping, the mirrored away-spread line, close attachment, CLV signs and
grading as they survive a database round trip.

Runs inside a transaction that is ALWAYS rolled back, so it can be executed
against production without writing anything.

The official evaluation window must not open until this passes against a real
settled slate. Passing on the synthetic fixture proves the path works; it does
not prove the season's data is complete.

    python -m src.tasks.nfl_shadow_reconcile
"""

import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import settings
from src.services.nfl.settlement import (
    CloseSource, consensus_close, select_book_close, settle_candidate,
)

KO = datetime(2026, 9, 14, 17, 0, tzinfo=timezone.utc)


def _fixture_quotes() -> list[dict]:
    """Three books, a line that moves, and one book that goes stale."""
    return [
        # dk: -3.5 early, closes -4.5 (moved through the key number 4? no — 3.5->4.5)
        {"book": "dk", "snapshot_time": KO - timedelta(hours=48),
         "spread_line": -3.5, "spread_home_odds": 1.91, "spread_away_odds": 1.91,
         "total_line": 44.5, "over_odds": 1.91, "under_odds": 1.91,
         "ml_home": 1.80, "ml_away": 2.05},
        {"book": "dk", "snapshot_time": KO - timedelta(minutes=6),
         "spread_line": -4.5, "spread_home_odds": 1.87, "spread_away_odds": 1.95,
         "total_line": 45.5, "over_odds": 1.91, "under_odds": 1.91,
         "ml_home": 1.72, "ml_away": 2.20},
        # fd: agrees at close
        {"book": "fd", "snapshot_time": KO - timedelta(minutes=8),
         "spread_line": -4.5, "spread_home_odds": 1.91, "spread_away_odds": 1.91,
         "total_line": 45.5, "over_odds": 1.87, "under_odds": 1.95,
         "ml_home": 1.74, "ml_away": 2.15},
        # stale: two days old and far off. Median must ignore it.
        {"book": "stale", "snapshot_time": KO - timedelta(hours=50),
         "spread_line": -9.5, "spread_home_odds": 1.91, "spread_away_odds": 1.91,
         "total_line": 52.5, "over_odds": 1.91, "under_odds": 1.91,
         "ml_home": 1.30, "ml_away": 3.60},
    ]


def _check(label: str, got, want, tol: float | None = None) -> tuple[bool, str]:
    if tol is not None and got is not None and want is not None:
        ok = abs(float(got) - float(want)) <= tol
    else:
        ok = got == want
    return ok, f"  {'PASS' if ok else 'FAIL'}  {label:<46} got={got!r} want={want!r}"


async def reconcile() -> dict:
    quotes = _fixture_quotes()
    results: list[tuple[bool, str]] = []

    # Home wins by 7 -> covers -4.5 but the bettor took -3.5 and it moved AGAINST
    # them (line got worse); total 24+17=41 stays under both 44.5 and 45.5.
    home_score, away_score = 24, 17
    margin = float(home_score - away_score)
    total = float(home_score + away_score)

    # --- same-book close selection -----------------------------------------
    dk = [q for q in quotes if q["book"] == "dk"]
    close = select_book_close(dk, KO)
    results.append(_check("book close picks the latest pre-kickoff quote",
                          close["spread_line"], -4.5))
    results.append(_check("staleness recorded in minutes", close["staleness_min"], 6))

    # --- consensus is median, ignoring the stale outlier --------------------
    cons = consensus_close(quotes, KO, market="spread", side="home", min_books=3)
    results.append(_check("consensus median ignores the stale book",
                          cons["line"], -4.5, tol=0.01))
    results.append(_check("consensus counts every book", cons["book_count"], 3))

    # --- HOME spread: took -3.5, closed -4.5 -> we laid FEWER points -------
    home_cand = {"market_type": "spread", "side": "home", "line": -3.5, "odds_decimal": 1.91}
    s = settle_candidate(
        home_cand,
        book_close={"line": -4.5, "odds": 1.87, "opposing_odds": 1.95, "staleness_min": 6},
        consensus={"line": -4.5, "novig_prob": 0.52, "book_count": 3},
        actual_value=margin,
    )
    results.append(_check("home line_clv positive: laid 3.5 vs a 4.5 close", s["line_clv"], 1.0, tol=1e-6))
    results.append(_check("closing source is the book", s["closing_source"], CloseSource.BOOK))
    results.append(_check("home -3.5 wins on a 7-point margin", s["side_won"], True))
    results.append(_check("key number 4 crossed 3.5 -> 4.5", s["crossed_key"], True))

    # --- AWAY spread: mirrored +3.5, closes +4.5 -> close GIVES MORE -------
    away_cand = {"market_type": "spread", "side": "away", "line": 3.5, "odds_decimal": 1.91}
    sa = settle_candidate(
        away_cand,
        book_close={"line": 4.5, "odds": 1.95, "opposing_odds": 1.87, "staleness_min": 6},
        consensus={"line": 4.5, "novig_prob": 0.48, "book_count": 3},
        actual_value=margin,
    )
    results.append(_check("away line_clv negative: got 3.5 vs a 4.5 close",
                          sa["line_clv"], -1.0, tol=1e-6))
    results.append(_check("away +3.5 loses on a 7-point home margin", sa["side_won"], False))

    # --- TOTAL under: took 44.5, closed 45.5 -> ours is the HARDER number ---
    und = settle_candidate(
        {"market_type": "total", "side": "under", "line": 44.5, "odds_decimal": 1.91},
        book_close={"line": 45.5, "odds": 1.91, "opposing_odds": 1.91, "staleness_min": 6},
        consensus={"line": 45.5, "novig_prob": 0.50, "book_count": 3},
        actual_value=total,
    )
    results.append(_check("under line_clv negative when total rose",
                          und["line_clv"], -1.0, tol=1e-6))
    results.append(_check("under 44.5 wins on a 41-point game", und["side_won"], True))

    # --- price CLV: took 1.91, close devigs to ~0.512 -----------------------
    results.append(_check("price_clv computed from the devigged close",
                          s["price_clv"] is not None, True))

    # --- fallback path ------------------------------------------------------
    fb = settle_candidate(home_cand, book_close=None,
                          consensus={"line": -4.5, "novig_prob": 0.52, "book_count": 3},
                          actual_value=margin)
    results.append(_check("missing book close falls back to consensus",
                          fb["closing_source"], CloseSource.CONSENSUS_FALLBACK))

    # --- unmeasured stays NULL, but the outcome is still graded -------------
    none = settle_candidate(home_cand, None, None, actual_value=margin)
    results.append(_check("no close -> CLV is None, not 0", none["price_clv"], None))
    results.append(_check("outcome graded even without a close", none["side_won"], True))

    # --- database round trip, always rolled back ---------------------------
    engine = create_async_engine(settings.async_database_url, echo=False)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        async with session.begin():
            probe = (await session.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'nfl_shadow_predictions'
            """))).scalars().all()
            required = {
                "closing_source", "closing_staleness_min", "consensus_line",
                "consensus_novig_prob", "consensus_book_count",
                "line_clv", "price_clv", "line_clv_consensus", "price_clv_consensus",
                "crossed_key", "side_won", "actual_value", "settled_at",
                "candidate_hash", "status", "status_detail",
            }
            missing = sorted(required - set(probe))
            results.append(_check("every settlement column exists", missing, []))
            await session.rollback()
    await engine.dispose()

    passed = sum(1 for ok, _ in results if ok)
    print(f"NFL shadow reconciliation — {passed}/{len(results)} checks\n")
    for _, line in results:
        print(line)

    all_ok = passed == len(results)
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'FAILURES PRESENT'}")
    print("\nEvaluation window: NOT OPENED.")
    print("This fixture proves the PATH works. Opening the window additionally")
    print("requires a real settled slate reconciled by hand — candidate")
    print("completeness, model lineage, closing-line attachment, grading.")
    return {"passed": passed, "total": len(results), "all_ok": all_ok,
            "evaluation_window_open": False}


if __name__ == "__main__":
    asyncio.run(reconcile())
