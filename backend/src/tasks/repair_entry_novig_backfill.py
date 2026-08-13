"""Retract the derived `entry_novig_prob` backfill (and what depends on it).

`9f278a9` froze `entry_novig_prob` at snapshot time precisely BECAUSE deriving
it was unsafe — the derivation needed `winner_probability` minus edge, and
`winner_probability` is `max(p_home, p_away)`, the PREDICTED WINNER's
probability rather than the backed side's. That commit then used the same
derivation to backfill 964 historical rows.

Those rows do not calibrate. Bucketed against outcomes:

    stored entry_novig_prob, games before 2026-07-31   n=943
      bucket        n   implied   actual
      0.00-0.35   577     0.231     0.490      <-- impossible
      0.35-0.42   157     0.389     0.459
      0.42-0.48   154     0.441     0.519
      OVERALL     943     0.309     0.490

A vig-free market price of 0.231 cannot win 0.490. From 2026-07-31 — when
`mlb_odds_snapshots` began recording real pre-game prices — the stored values
agree with a rebuilt consensus to a median of 0.0068 (93.9% within two
points), so the live path is correct and only the backfill is wrong.

There is no way to repair the old values: no pre-game market price was ever
recorded for those games, so the true number is not recoverable. NULL is
therefore the correct value, per the rule these columns already document —
NULL means "not measured", never "neutral". Leaving a plausible-looking wrong
number in place is worse than an honest gap, because `market_move` and
`vig_paid` derive from it and CLV is the instrument model changes get promoted
on.

    python -m src.tasks.repair_entry_novig_backfill            # dry run
    python -m src.tasks.repair_entry_novig_backfill --apply
"""

import asyncio
import sys
from datetime import date

from sqlalchemy import select, update, and_, or_

from src.database import async_session
from src.models import MLBPredictionSnapshot

# First real pre-game price in mlb_odds_snapshots: 2026-07-31T14:49:52Z.
# Games before this date have no recorded market to reconstruct from.
ODDS_HISTORY_START = date(2026, 7, 31)

# The derived value and everything computed from it. `clv` is NOT here: it is
# closing_novig - 1/odds and never touched entry_novig, so it stays valid.
DERIVED_COLUMNS = ("entry_novig_prob", "market_move", "vig_paid")


def is_unverifiable(game_date: date | None, cutoff: date = ODDS_HISTORY_START) -> bool:
    """Whether a row's entry price could only have come from the backfill.

    A NULL game_date cannot be shown to fall after the cutoff, so it is
    treated as unverifiable — the conservative direction, since the cost of
    retracting a good value is a gap and the cost of keeping a bad one is a
    wrong CLV number.
    """
    if game_date is None:
        return True
    return game_date < cutoff


async def find_unverifiable(session, cutoff: date = ODDS_HISTORY_START) -> list[int]:
    """Ids of snapshots holding a derived entry price we cannot stand behind."""
    rows = (await session.execute(
        select(MLBPredictionSnapshot).where(
            or_(
                MLBPredictionSnapshot.entry_novig_prob.isnot(None),
                MLBPredictionSnapshot.market_move.isnot(None),
                MLBPredictionSnapshot.vig_paid.isnot(None),
            )
        )
    )).scalars().all()
    return [r.id for r in rows if is_unverifiable(r.game_date, cutoff)]


async def repair(apply: bool = False, cutoff: date = ODDS_HISTORY_START) -> dict:
    async with async_session() as session:
        ids = await find_unverifiable(session, cutoff)

        total_with_entry = len((await session.execute(
            select(MLBPredictionSnapshot.id)
            .where(MLBPredictionSnapshot.entry_novig_prob.isnot(None))
        )).scalars().all())

        if apply and ids:
            await session.execute(
                update(MLBPredictionSnapshot)
                .where(MLBPredictionSnapshot.id.in_(ids))
                .values(entry_novig_prob=None, market_move=None, vig_paid=None)
            )
            await session.commit()

        remaining = len((await session.execute(
            select(MLBPredictionSnapshot.id)
            .where(MLBPredictionSnapshot.entry_novig_prob.isnot(None))
        )).scalars().all())

    return {
        "cutoff": cutoff.isoformat(),
        "rows_with_entry_price_before": total_with_entry,
        "unverifiable": len(ids),
        "applied": apply,
        "rows_with_entry_price_after": remaining,
    }


def main() -> None:
    apply = "--apply" in sys.argv
    result = asyncio.run(repair(apply=apply))
    print(f"cutoff                     {result['cutoff']}")
    print(f"rows with an entry price   {result['rows_with_entry_price_before']}")
    print(f"unverifiable (backfilled)  {result['unverifiable']}")
    print(f"applied                    {result['applied']}")
    print(f"rows remaining after       {result['rows_with_entry_price_after']}")
    if not apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to retract.")


if __name__ == "__main__":
    main()
