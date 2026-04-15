"""Backfill ATS and O/U records for MLB teams from existing DB data.

For each completed MLB game in our DB, look at the runline + total markets we
ingested at game time and compute the ATS/O-U result for each team. Then write
season totals to the latest mlb_team_stats row per team.

Run via Railway CLI to access the prod DB:
    railway run python -m scripts.backfill_mlb_betting_records

Also runnable locally if you have prod DB credentials in your env:
    python -m scripts.backfill_mlb_betting_records
"""
import asyncio
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, and_, desc, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from src.config import settings


async def main():
    engine = create_async_engine(
        settings.async_database_url,
        pool_pre_ping=True,
        connect_args={"timeout": 30},
    )
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        # Pull all final games with their latest runline + total markets in one query.
        # We use DISTINCT ON to grab the most recent market per (game_id, market_type).
        result = await session.execute(text("""
            WITH latest_markets AS (
                SELECT DISTINCT ON (game_id, market_type)
                    game_id, market_type, line
                FROM mlb_markets
                WHERE market_type IN ('runline', 'total')
                  AND line IS NOT NULL
                ORDER BY game_id, market_type, updated_at DESC
            )
            SELECT
                g.game_id,
                g.home_team,
                g.away_team,
                g.home_score,
                g.away_score,
                MAX(CASE WHEN lm.market_type = 'runline' THEN lm.line END) AS runline,
                MAX(CASE WHEN lm.market_type = 'total' THEN lm.line END) AS total_line
            FROM mlb_games g
            LEFT JOIN latest_markets lm ON lm.game_id = g.game_id
            WHERE g.status = 'final'
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
              AND g.game_type = 'R'
            GROUP BY g.game_id, g.home_team, g.away_team, g.home_score, g.away_score
        """))
        rows = result.fetchall()

        print(f"Found {len(rows)} final regular-season games")

        # Track per-team running tallies
        ats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "pushes": 0}
        )
        ou: dict[str, dict[str, int]] = defaultdict(
            lambda: {"overs": 0, "unders": 0, "pushes": 0}
        )

        ats_games_counted = 0
        ou_games_counted = 0
        no_market = 0

        for row in rows:
            home, away = row.home_team, row.away_team
            hs, as_ = int(row.home_score), int(row.away_score)

            # ATS — home line is stored in mlb_markets.line.
            # Convention: negative = home favored (must win by abs(line))
            if row.runline is not None:
                home_line = float(row.runline)
                home_adjusted = hs + home_line
                if home_adjusted > as_:
                    ats[home]["wins"] += 1
                    ats[away]["losses"] += 1
                elif home_adjusted < as_:
                    ats[home]["losses"] += 1
                    ats[away]["wins"] += 1
                else:
                    ats[home]["pushes"] += 1
                    ats[away]["pushes"] += 1
                ats_games_counted += 1

            # O/U
            if row.total_line is not None:
                total_runs = hs + as_
                total_line = float(row.total_line)
                if total_runs > total_line:
                    ou[home]["overs"] += 1
                    ou[away]["overs"] += 1
                elif total_runs < total_line:
                    ou[home]["unders"] += 1
                    ou[away]["unders"] += 1
                else:
                    ou[home]["pushes"] += 1
                    ou[away]["pushes"] += 1
                ou_games_counted += 1

            if row.runline is None and row.total_line is None:
                no_market += 1

        print(f"\nATS games counted: {ats_games_counted}")
        print(f"O/U games counted: {ou_games_counted}")
        print(f"Games with no market data: {no_market}")
        print(f"\nSample team records:")
        for team in sorted(ats.keys())[:5]:
            a = ats[team]
            o = ou[team]
            print(f"  {team}: ATS {a['wins']}-{a['losses']}-{a['pushes']}, "
                  f"O/U {o['overs']}-{o['unders']}-{o['pushes']}")

        # Update mlb_team_stats — write to the LATEST row per team (the row
        # `get_team_card_stats` reads via ORDER BY stat_date DESC LIMIT 1).
        all_teams = set(ats.keys()) | set(ou.keys())
        updated = 0
        for team in all_teams:
            a = ats[team]
            o = ou[team]
            # Update the latest row by stat_date for this team
            await session.execute(text("""
                UPDATE mlb_team_stats
                SET ats_wins = :aw, ats_losses = :al, ats_pushes = :ap,
                    ou_overs = :oo, ou_unders = :ou_u, ou_pushes = :op
                WHERE stat_id = (
                    SELECT stat_id FROM mlb_team_stats
                    WHERE team_abbr = :team
                    ORDER BY stat_date DESC
                    LIMIT 1
                )
            """), {
                "team": team,
                "aw": a["wins"], "al": a["losses"], "ap": a["pushes"],
                "oo": o["overs"], "ou_u": o["unders"], "op": o["pushes"],
            })
            updated += 1

        await session.commit()
        print(f"\nUpdated {updated} team stat rows.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
