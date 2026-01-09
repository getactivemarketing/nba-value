"""Trends API endpoints for betting analysis."""

from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db

router = APIRouter()


@router.get("/trends/ats")
async def get_ats_leaderboard(
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """
    Get all teams ranked by ATS (Against the Spread) performance.

    Returns teams sorted by ATS win percentage over the last 10 games.
    """
    # Get latest stats for each team
    result = await db.execute(text("""
        WITH latest_stats AS (
            SELECT DISTINCT ON (team_id)
                team_id,
                ats_wins_l10,
                ats_losses_l10,
                ats_pushes_l10,
                wins_l10,
                losses_l10,
                home_wins,
                home_losses,
                away_wins,
                away_losses,
                net_rtg_10,
                stat_date
            FROM team_stats
            ORDER BY team_id, stat_date DESC
        )
        SELECT
            team_id,
            COALESCE(ats_wins_l10, 0) as ats_wins,
            COALESCE(ats_losses_l10, 0) as ats_losses,
            COALESCE(ats_pushes_l10, 0) as ats_pushes,
            COALESCE(wins_l10, 0) as wins_l10,
            COALESCE(losses_l10, 0) as losses_l10,
            COALESCE(home_wins, 0) as home_wins,
            COALESCE(home_losses, 0) as home_losses,
            COALESCE(away_wins, 0) as away_wins,
            COALESCE(away_losses, 0) as away_losses,
            net_rtg_10
        FROM latest_stats
        ORDER BY
            CASE WHEN (COALESCE(ats_wins_l10, 0) + COALESCE(ats_losses_l10, 0)) > 0
                THEN COALESCE(ats_wins_l10, 0)::float / (COALESCE(ats_wins_l10, 0) + COALESCE(ats_losses_l10, 0))
                ELSE 0
            END DESC,
            ats_wins_l10 DESC
    """))

    rows = result.fetchall()

    leaderboard = []
    for i, row in enumerate(rows, 1):
        total_ats = row.ats_wins + row.ats_losses
        ats_pct = round(row.ats_wins / total_ats * 100, 1) if total_ats > 0 else 0

        leaderboard.append({
            "rank": i,
            "team": row.team_id,
            "ats_wins": row.ats_wins,
            "ats_losses": row.ats_losses,
            "ats_pushes": row.ats_pushes,
            "ats_record": f"{row.ats_wins}-{row.ats_losses}" + (f"-{row.ats_pushes}" if row.ats_pushes else ""),
            "ats_pct": ats_pct,
            "wins_l10": row.wins_l10,
            "losses_l10": row.losses_l10,
            "l10_record": f"{row.wins_l10}-{row.losses_l10}",
            "home_record": f"{row.home_wins}-{row.home_losses}",
            "away_record": f"{row.away_wins}-{row.away_losses}",
            "net_rtg": round(float(row.net_rtg_10), 1) if row.net_rtg_10 else None,
        })

    return leaderboard


@router.get("/trends/ou")
async def get_ou_leaderboard(
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """
    Get all teams ranked by Over/Under tendencies.

    Returns teams sorted by Over percentage (teams whose games go over most often).
    """
    result = await db.execute(text("""
        WITH latest_stats AS (
            SELECT DISTINCT ON (team_id)
                team_id,
                ou_overs_l10,
                ou_unders_l10,
                ou_pushes_l10,
                ppg_10,
                opp_ppg_10,
                ppg_season,
                opp_ppg_season,
                stat_date
            FROM team_stats
            ORDER BY team_id, stat_date DESC
        )
        SELECT
            team_id,
            COALESCE(ou_overs_l10, 0) as overs,
            COALESCE(ou_unders_l10, 0) as unders,
            COALESCE(ou_pushes_l10, 0) as pushes,
            ppg_10,
            opp_ppg_10,
            ppg_season,
            opp_ppg_season
        FROM latest_stats
        ORDER BY
            CASE WHEN (COALESCE(ou_overs_l10, 0) + COALESCE(ou_unders_l10, 0)) > 0
                THEN COALESCE(ou_overs_l10, 0)::float / (COALESCE(ou_overs_l10, 0) + COALESCE(ou_unders_l10, 0))
                ELSE 0.5
            END DESC
    """))

    rows = result.fetchall()

    leaderboard = []
    for i, row in enumerate(rows, 1):
        total_ou = row.overs + row.unders
        over_pct = round(row.overs / total_ou * 100, 1) if total_ou > 0 else 50

        ppg = float(row.ppg_10) if row.ppg_10 else 0
        opp_ppg = float(row.opp_ppg_10) if row.opp_ppg_10 else 0
        avg_total = round(ppg + opp_ppg, 1)

        # Determine pace label
        if avg_total >= 230:
            pace = "Very Fast"
        elif avg_total >= 220:
            pace = "Fast"
        elif avg_total >= 210:
            pace = "Average"
        elif avg_total >= 200:
            pace = "Slow"
        else:
            pace = "Very Slow"

        leaderboard.append({
            "rank": i,
            "team": row.team_id,
            "overs": row.overs,
            "unders": row.unders,
            "pushes": row.pushes,
            "ou_record": f"{row.overs}o-{row.unders}u" + (f"-{row.pushes}p" if row.pushes else ""),
            "over_pct": over_pct,
            "ppg": round(ppg, 1),
            "opp_ppg": round(opp_ppg, 1),
            "avg_total": avg_total,
            "pace": pace,
        })

    return leaderboard


@router.get("/trends/situational")
async def get_situational_trends(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get situational betting trends.

    Analyzes performance by rest days, back-to-back status, home/away.
    """
    # Get ATS performance by rest situation from game_results
    # This requires joining with team context at game time

    # For now, aggregate from team_stats
    result = await db.execute(text("""
        WITH latest_stats AS (
            SELECT DISTINCT ON (team_id)
                team_id,
                days_rest,
                is_back_to_back,
                ats_wins_l10,
                ats_losses_l10,
                home_wins,
                home_losses,
                away_wins,
                away_losses,
                net_rtg_10
            FROM team_stats
            ORDER BY team_id, stat_date DESC
        )
        SELECT
            -- B2B stats
            SUM(CASE WHEN is_back_to_back THEN 1 ELSE 0 END) as teams_on_b2b,
            -- Rest buckets
            SUM(CASE WHEN days_rest = 0 THEN 1 ELSE 0 END) as teams_0_rest,
            SUM(CASE WHEN days_rest = 1 THEN 1 ELSE 0 END) as teams_1_rest,
            SUM(CASE WHEN days_rest = 2 THEN 1 ELSE 0 END) as teams_2_rest,
            SUM(CASE WHEN days_rest >= 3 THEN 1 ELSE 0 END) as teams_3plus_rest,
            -- Total ATS
            SUM(ats_wins_l10) as total_ats_wins,
            SUM(ats_losses_l10) as total_ats_losses,
            -- Home/Away
            SUM(home_wins) as total_home_wins,
            SUM(home_losses) as total_home_losses,
            SUM(away_wins) as total_away_wins,
            SUM(away_losses) as total_away_losses
        FROM latest_stats
    """))

    row = result.fetchone()

    # Get historical ATS by rest from game_results
    rest_result = await db.execute(text("""
        SELECT
            CASE
                WHEN ts.days_rest = 0 THEN '0 days'
                WHEN ts.days_rest = 1 THEN '1 day'
                WHEN ts.days_rest = 2 THEN '2 days'
                ELSE '3+ days'
            END as rest_bucket,
            COUNT(*) as games,
            SUM(CASE WHEN
                (gr.spread_result = 'home_cover' AND gr.home_team_id = ts.team_id) OR
                (gr.spread_result = 'away_cover' AND gr.away_team_id = ts.team_id)
                THEN 1 ELSE 0 END) as covers,
            SUM(CASE WHEN gr.spread_result = 'push' THEN 1 ELSE 0 END) as pushes
        FROM game_results gr
        JOIN team_stats ts ON (
            (gr.home_team_id = ts.team_id OR gr.away_team_id = ts.team_id)
            AND gr.game_date = ts.stat_date
        )
        WHERE gr.spread_result IS NOT NULL
        GROUP BY rest_bucket
        ORDER BY rest_bucket
    """))

    rest_trends = []
    for r in rest_result.fetchall():
        games = r.games or 1
        covers = r.covers or 0
        cover_pct = round(covers / games * 100, 1) if games > 0 else 50
        rest_trends.append({
            "situation": r.rest_bucket,
            "games": games,
            "covers": covers,
            "cover_pct": cover_pct,
        })

    # Home vs Away ATS
    home_ats_result = await db.execute(text("""
        SELECT
            'Home' as location,
            COUNT(*) as games,
            SUM(CASE WHEN spread_result = 'home_cover' THEN 1 ELSE 0 END) as covers
        FROM game_results
        WHERE spread_result IS NOT NULL
        UNION ALL
        SELECT
            'Away' as location,
            COUNT(*) as games,
            SUM(CASE WHEN spread_result = 'away_cover' THEN 1 ELSE 0 END) as covers
        FROM game_results
        WHERE spread_result IS NOT NULL
    """))

    location_trends = []
    for r in home_ats_result.fetchall():
        games = r.games or 1
        covers = r.covers or 0
        cover_pct = round(covers / games * 100, 1) if games > 0 else 50
        location_trends.append({
            "situation": r.location,
            "games": games,
            "covers": covers,
            "cover_pct": cover_pct,
        })

    # B2B performance
    b2b_result = await db.execute(text("""
        SELECT
            COUNT(*) as games,
            SUM(CASE WHEN
                (gr.spread_result = 'home_cover' AND ts.is_back_to_back) OR
                (gr.spread_result = 'away_cover' AND ts.is_back_to_back)
                THEN 1 ELSE 0 END) as b2b_covers
        FROM game_results gr
        JOIN team_stats ts ON (
            (gr.home_team_id = ts.team_id OR gr.away_team_id = ts.team_id)
            AND gr.game_date = ts.stat_date
        )
        WHERE gr.spread_result IS NOT NULL
        AND ts.is_back_to_back = true
    """))

    b2b_row = b2b_result.fetchone()

    return {
        "by_rest": rest_trends if rest_trends else [
            {"situation": "0 days (B2B)", "games": 0, "covers": 0, "cover_pct": 50},
            {"situation": "1 day", "games": 0, "covers": 0, "cover_pct": 50},
            {"situation": "2 days", "games": 0, "covers": 0, "cover_pct": 50},
            {"situation": "3+ days", "games": 0, "covers": 0, "cover_pct": 50},
        ],
        "by_location": location_trends if location_trends else [
            {"situation": "Home", "games": 0, "covers": 0, "cover_pct": 50},
            {"situation": "Away", "games": 0, "covers": 0, "cover_pct": 50},
        ],
        "b2b_summary": {
            "games": b2b_row.games if b2b_row else 0,
            "covers": b2b_row.b2b_covers if b2b_row else 0,
            "cover_pct": round(b2b_row.b2b_covers / b2b_row.games * 100, 1) if b2b_row and b2b_row.games > 0 else 50,
        },
        "summary": {
            "total_home_wins": row.total_home_wins if row else 0,
            "total_home_losses": row.total_home_losses if row else 0,
            "total_away_wins": row.total_away_wins if row else 0,
            "total_away_losses": row.total_away_losses if row else 0,
        }
    }


@router.get("/trends/model")
async def get_model_performance(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get model performance trends.

    Shows algorithm comparison, performance by value bucket, and recent results.
    """
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Get graded predictions
    result = await db.execute(text("""
        SELECT
            winner_correct,
            best_bet_result,
            best_bet_profit,
            best_bet_value_score,
            algo_a_value_score,
            algo_b_value_score,
            algo_a_bet_result,
            algo_b_bet_result,
            algo_a_profit,
            algo_b_profit,
            snapshot_time
        FROM prediction_snapshots
        WHERE snapshot_time >= :cutoff
        AND winner_correct IS NOT NULL
        ORDER BY snapshot_time DESC
    """), {"cutoff": cutoff})

    rows = result.fetchall()

    if not rows:
        return {
            "days_analyzed": days,
            "total_predictions": 0,
            "winner_accuracy": {"correct": 0, "total": 0, "pct": 0},
            "algo_a": {"wins": 0, "losses": 0, "pushes": 0, "profit": 0, "roi": 0},
            "algo_b": {"wins": 0, "losses": 0, "pushes": 0, "profit": 0, "roi": 0},
            "by_bucket": [],
            "recent": [],
        }

    # Calculate stats
    total = len(rows)
    winner_correct = sum(1 for r in rows if r.winner_correct)

    # Algorithm performance
    algo_a = {"wins": 0, "losses": 0, "pushes": 0, "profit": 0}
    algo_b = {"wins": 0, "losses": 0, "pushes": 0, "profit": 0}

    # Buckets: 50-60, 60-70, 70-80, 80+
    buckets = {
        "50-60": {"wins": 0, "losses": 0, "pushes": 0, "profit": 0},
        "60-70": {"wins": 0, "losses": 0, "pushes": 0, "profit": 0},
        "70-80": {"wins": 0, "losses": 0, "pushes": 0, "profit": 0},
        "80+": {"wins": 0, "losses": 0, "pushes": 0, "profit": 0},
    }

    for r in rows:
        # Best bet performance by bucket
        score = r.best_bet_value_score or 0
        if score >= 80:
            bucket = "80+"
        elif score >= 70:
            bucket = "70-80"
        elif score >= 60:
            bucket = "60-70"
        else:
            bucket = "50-60"

        if r.best_bet_result == "win":
            buckets[bucket]["wins"] += 1
            buckets[bucket]["profit"] += float(r.best_bet_profit or 90.91)
        elif r.best_bet_result == "loss":
            buckets[bucket]["losses"] += 1
            buckets[bucket]["profit"] -= 100
        else:
            buckets[bucket]["pushes"] += 1

        # Algo A
        if r.algo_a_bet_result == "win":
            algo_a["wins"] += 1
            algo_a["profit"] += float(r.algo_a_profit or 90.91)
        elif r.algo_a_bet_result == "loss":
            algo_a["losses"] += 1
            algo_a["profit"] -= 100
        else:
            algo_a["pushes"] += 1

        # Algo B
        if r.algo_b_bet_result == "win":
            algo_b["wins"] += 1
            algo_b["profit"] += float(r.algo_b_profit or 90.91)
        elif r.algo_b_bet_result == "loss":
            algo_b["losses"] += 1
            algo_b["profit"] -= 100
        else:
            algo_b["pushes"] += 1

    # Calculate ROI
    def calc_roi(stats):
        total_bets = stats["wins"] + stats["losses"]
        if total_bets == 0:
            return 0
        return round(stats["profit"] / (total_bets * 100) * 100, 1)

    def calc_win_rate(stats):
        total = stats["wins"] + stats["losses"]
        if total == 0:
            return 0
        return round(stats["wins"] / total * 100, 1)

    # Format bucket data
    bucket_data = []
    for bucket_name, stats in buckets.items():
        total_bets = stats["wins"] + stats["losses"]
        if total_bets > 0:
            bucket_data.append({
                "bucket": bucket_name,
                "bets": total_bets,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": calc_win_rate(stats),
                "profit": round(stats["profit"], 2),
                "roi": calc_roi(stats),
            })

    # Recent results (last 10)
    recent = []
    for r in rows[:10]:
        recent.append({
            "date": r.snapshot_time.strftime("%m/%d") if r.snapshot_time else "",
            "winner_correct": r.winner_correct,
            "bet_result": r.best_bet_result,
            "value_score": r.best_bet_value_score,
        })

    return {
        "days_analyzed": days,
        "total_predictions": total,
        "winner_accuracy": {
            "correct": winner_correct,
            "total": total,
            "pct": round(winner_correct / total * 100, 1) if total > 0 else 0,
        },
        "algo_a": {
            "wins": algo_a["wins"],
            "losses": algo_a["losses"],
            "pushes": algo_a["pushes"],
            "profit": round(algo_a["profit"], 2),
            "win_rate": calc_win_rate(algo_a),
            "roi": calc_roi(algo_a),
        },
        "algo_b": {
            "wins": algo_b["wins"],
            "losses": algo_b["losses"],
            "pushes": algo_b["pushes"],
            "profit": round(algo_b["profit"], 2),
            "win_rate": calc_win_rate(algo_b),
            "roi": calc_roi(algo_b),
        },
        "by_bucket": bucket_data,
        "recent": recent,
    }
