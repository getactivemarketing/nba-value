"""Generate a real card preview from today's DB data, save to /tmp.

Run with: python3 -m scripts.preview_card
"""
import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker


async def main():
    engine = create_async_engine(
        settings.async_database_url,
        pool_pre_ping=True,
        connect_args={"timeout": 30, "server_settings": {"statement_timeout": "30000"}},
    )
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        from sqlalchemy import select, and_, desc, or_
        from src.models import MLBGame, MLBPredictionSnapshot
        from src.services.social.content import (
            get_team_card_stats, _get_team_first_inning_pct,
            _get_pitcher_era, TEAM_NAMES, TEAM_HANDLES,
        )
        from src.services.social.image_generator import (
            generate_nrfi_card, generate_final_card, generate_recap_card,
        )

        eastern = timedelta(hours=-4)
        today_et = (datetime.now(timezone.utc) + eastern).date()
        yesterday_et = today_et - timedelta(days=1)

        # Try to find a good preview candidate:
        # 1. Scheduled game today (for NRFI card)
        # 2. Final game from today/yesterday (for final card)

        # --- NRFI preview ---
        print(f"Looking for NRFI card candidate on {today_et}...")
        result = await session.execute(
            select(MLBGame).where(
                and_(
                    MLBGame.game_date == today_et,
                    MLBGame.status == "scheduled",
                    MLBGame.home_starter_id.isnot(None),
                    MLBGame.away_starter_id.isnot(None),
                )
            ).order_by(MLBGame.game_time).limit(10)
        )
        scheduled = list(result.scalars().all())
        print(f"  Found {len(scheduled)} scheduled games with starters")

        for game in scheduled:
            home_off, home_def = await _get_team_first_inning_pct(session, game.home_team)
            away_off, away_def = await _get_team_first_inning_pct(session, game.away_team)
            if not (home_off and away_off and home_def and away_def):
                continue
            p_away_scores = (away_off + home_def) / 2.0
            p_home_scores = (home_off + away_def) / 2.0
            nrfi_pct = (1.0 - p_away_scores) * (1.0 - p_home_scores)
            if nrfi_pct < 0.55:
                continue

            away_last, away_era = await _get_pitcher_era(session, game.away_starter_id)
            home_last, home_era = await _get_pitcher_era(session, game.home_starter_id)
            if not away_last or not home_last:
                continue

            away_stats = await get_team_card_stats(session, game.away_team)
            home_stats = await get_team_card_stats(session, game.home_team)

            game_time_str = None
            if game.game_time:
                et = game.game_time - timedelta(hours=4)
                try:
                    game_time_str = et.strftime("%-I:%M %p ET")
                except Exception:
                    game_time_str = et.strftime("%I:%M %p ET").lstrip("0")

            print(f"  Using {game.away_team} @ {game.home_team} ({nrfi_pct*100:.0f}% NRFI)")
            print(f"  Away stats: {away_stats}")
            print(f"  Home stats: {home_stats}")

            png = generate_nrfi_card(
                away_team=game.away_team, home_team=game.home_team,
                away_name=TEAM_NAMES.get(game.away_team, game.away_team),
                home_name=TEAM_NAMES.get(game.home_team, game.home_team),
                nrfi_pct=nrfi_pct * 100,
                away_pitcher=away_last, away_era=away_era,
                home_pitcher=home_last, home_era=home_era,
                game_time=game_time_str,
                away_record=away_stats.get("record"), home_record=home_stats.get("record"),
                away_l10=away_stats.get("l10"), home_l10=home_stats.get("l10"),
                away_div_rank=away_stats.get("div_rank"), home_div_rank=home_stats.get("div_rank"),
                away_ats=away_stats.get("ats"), home_ats=home_stats.get("ats"),
                away_ou=away_stats.get("ou"), home_ou=home_stats.get("ou"),
            )
            path = "/tmp/preview_nrfi_live.png"
            with open(path, "wb") as f:
                f.write(png)
            print(f"  → {path}")
            break
        else:
            print("  No NRFI candidate found")

        # --- Final card preview ---
        print(f"\nLooking for final game from {yesterday_et} or {today_et}...")
        result = await session.execute(
            select(MLBGame).where(
                and_(
                    MLBGame.game_date.in_([yesterday_et, today_et]),
                    MLBGame.status == "final",
                    MLBGame.home_score.isnot(None),
                    MLBGame.away_score.isnot(None),
                )
            ).order_by(desc(MLBGame.game_time)).limit(5)
        )
        finals = list(result.scalars().all())
        print(f"  Found {len(finals)} final games")

        if finals:
            game = finals[0]
            # Look up pick snapshot
            snap_result = await session.execute(
                select(MLBPredictionSnapshot).where(
                    and_(
                        MLBPredictionSnapshot.game_id == game.game_id,
                        MLBPredictionSnapshot.best_bet_team.isnot(None),
                    )
                ).order_by(desc(MLBPredictionSnapshot.snapshot_time)).limit(1)
            )
            snap = snap_result.scalar_one_or_none()

            away_stats = await get_team_card_stats(session, game.away_team)
            home_stats = await get_team_card_stats(session, game.home_team)

            print(f"  Using {game.away_team} @ {game.home_team} ({game.away_score}-{game.home_score})")

            png = generate_final_card(
                away_team=game.away_team, home_team=game.home_team,
                away_name=TEAM_NAMES.get(game.away_team, game.away_team),
                home_name=TEAM_NAMES.get(game.home_team, game.home_team),
                away_score=game.away_score, home_score=game.home_score,
                away_first=game.away_first_inning_runs, home_first=game.home_first_inning_runs,
                pick_team=snap.best_bet_team if snap else None,
                pick_type=snap.best_bet_type if snap else None,
                pick_line=float(snap.best_bet_line) if snap and snap.best_bet_line is not None else None,
                pick_result=snap.best_bet_result if snap else None,
                away_record=away_stats.get("record"), home_record=home_stats.get("record"),
                away_l10=away_stats.get("l10"), home_l10=home_stats.get("l10"),
                away_div_rank=away_stats.get("div_rank"), home_div_rank=home_stats.get("div_rank"),
                away_ats=away_stats.get("ats"), home_ats=home_stats.get("ats"),
                away_ou=away_stats.get("ou"), home_ou=home_stats.get("ou"),
            )
            path = "/tmp/preview_final_live.png"
            with open(path, "wb") as f:
                f.write(png)
            print(f"  → {path}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
