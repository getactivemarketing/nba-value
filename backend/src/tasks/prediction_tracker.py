"""
Prediction Tracker - Snapshots predictions before games and grades them after.

This module handles:
1. Saving predictions to prediction_snapshots before tip-off
2. Grading predictions after games complete
3. Tracking model performance over time
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import psycopg2
import structlog

from src.services.injuries import get_all_team_injury_reports

logger = structlog.get_logger()

DB_URL = 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway'


def snapshot_predictions(hours_ahead: int = 6, db_url: str = None) -> dict:
    """
    Snapshot predictions for games starting in the next N hours.

    Should be run periodically (e.g., hourly) to capture predictions before tip-off.

    Args:
        hours_ahead: How many hours ahead to look for games
        db_url: Database connection string

    Returns:
        Summary of snapshots created
    """
    conn = psycopg2.connect(db_url or DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)

    # Find upcoming games that haven't been snapshotted yet
    cur.execute('''
        SELECT DISTINCT g.game_id, g.home_team_id, g.away_team_id, g.tip_time_utc
        FROM games g
        WHERE g.tip_time_utc > %s
        AND g.tip_time_utc < %s
        AND g.status = 'scheduled'
        AND NOT EXISTS (
            SELECT 1 FROM prediction_snapshots ps
            WHERE ps.game_id = g.game_id
        )
        ORDER BY g.tip_time_utc
    ''', (now, cutoff))

    games = cur.fetchall()

    if not games:
        logger.info("No new games to snapshot")
        cur.close()
        conn.close()
        return {"games_snapshotted": 0, "status": "no_new_games"}

    # Fetch injury reports for all teams
    try:
        injury_reports = asyncio.run(get_all_team_injury_reports())
        logger.info(f"Fetched injury reports for {len(injury_reports)} teams")
    except Exception as e:
        logger.warning(f"Failed to fetch injury reports: {e}")
        injury_reports = {}

    snapshots_created = 0

    for game_id, home_team, away_team, tip_time in games:
        # Get team stats for prediction factors
        home_stats = get_team_stats(cur, home_team)
        away_stats = get_team_stats(cur, away_team)

        # Get value scores for this game
        cur.execute('''
            SELECT
                m.market_type,
                m.outcome_label,
                m.line,
                m.odds_decimal,
                vs.algo_b_value_score,
                vs.algo_b_combined_edge,
                mp.p_true,
                mp.p_market,
                mp.raw_edge
            FROM value_scores vs
            JOIN markets m ON vs.market_id = m.market_id
            JOIN model_predictions mp ON vs.prediction_id = mp.prediction_id
            WHERE m.game_id = %s
            ORDER BY vs.algo_b_value_score DESC
        ''', (game_id,))

        scores = cur.fetchall()

        if not scores:
            logger.info(f"No value scores for game {game_id}, skipping")
            continue

        # Determine predicted winner from moneyline markets
        home_prob = 0.5
        away_prob = 0.5
        best_bet = None
        best_score = 0

        for mtype, outcome, line, odds, value_score, edge, p_true, p_market, raw_edge in scores:
            # Track best value bet
            if value_score and float(value_score) > best_score:
                best_score = float(value_score)
                is_home = 'home' in outcome.lower()
                best_bet = {
                    "type": mtype,
                    "team": home_team if is_home else away_team,
                    "line": float(line) if line else None,
                    "value_score": int(value_score),
                    "edge": float(edge) if edge else 0,
                    "odds": float(odds) if odds else None,
                    "p_true": float(p_true) if p_true else 0,
                    "p_market": float(p_market) if p_market else 0,
                }

            # Get moneyline probabilities
            if mtype == 'moneyline':
                if 'home' in outcome.lower():
                    home_prob = float(p_true) if p_true else 0.5
                else:
                    away_prob = float(p_true) if p_true else 0.5

        # Determine predicted winner
        if home_prob >= away_prob:
            predicted_winner = home_team
            winner_prob = home_prob
        else:
            predicted_winner = away_team
            winner_prob = away_prob

        # Confidence level
        if winner_prob >= 0.65:
            confidence = 'high'
        elif winner_prob >= 0.55:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Build explanation factors
        factors = build_factors(home_team, away_team, home_stats, away_stats, best_bet)

        # Get injury data for this game
        home_injury = injury_reports.get(home_team)
        away_injury = injury_reports.get(away_team)

        home_injury_score = home_injury.spread_injury_score if home_injury else 0
        away_injury_score = away_injury.spread_injury_score if away_injury else 0
        home_ppg_lost = home_injury.total_ppg_lost if home_injury else 0
        away_ppg_lost = away_injury.total_ppg_lost if away_injury else 0
        injury_edge = away_injury_score - home_injury_score  # Positive = home advantage

        # Add injury factor if significant
        if abs(injury_edge) >= 0.05:
            advantage_team = home_team if injury_edge > 0 else away_team
            factors.append(f"{advantage_team} injury advantage ({abs(injury_edge):.0%})")

        # Insert snapshot
        cur.execute('''
            INSERT INTO prediction_snapshots (
                game_id, snapshot_time, home_team, away_team, tip_time,
                predicted_winner, winner_probability, winner_confidence,
                best_bet_type, best_bet_team, best_bet_line,
                best_bet_value_score, best_bet_edge, best_bet_odds,
                factors,
                home_injury_score, away_injury_score, home_ppg_lost, away_ppg_lost, injury_edge
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            game_id, now, home_team, away_team, tip_time,
            predicted_winner, round(winner_prob * 100, 1), confidence,
            best_bet['type'] if best_bet else None,
            best_bet['team'] if best_bet else None,
            best_bet['line'] if best_bet else None,
            best_bet['value_score'] if best_bet else None,
            best_bet['edge'] if best_bet else None,
            best_bet['odds'] if best_bet else None,
            json.dumps(factors),
            round(home_injury_score, 2),
            round(away_injury_score, 2),
            round(home_ppg_lost, 1),
            round(away_ppg_lost, 1),
            round(injury_edge, 2)
        ))

        snapshots_created += 1
        logger.info(f"Snapshotted {away_team} @ {home_team}: {predicted_winner} to win ({confidence})")

        if best_bet and best_bet['value_score'] >= 50:
            logger.info(f"  Best bet: {best_bet['team']} {best_bet['type']} {best_bet['line'] or ''} ({best_bet['value_score']}%)")

    cur.close()
    conn.close()

    return {
        "games_snapshotted": snapshots_created,
        "snapshot_time": now.isoformat(),
        "status": "completed"
    }


def get_team_stats(cur, team_id: str) -> dict:
    """Get latest team stats for building prediction factors."""
    cur.execute('''
        SELECT
            wins, losses, net_rtg_10, days_rest, is_back_to_back,
            wins_l10, losses_l10, home_wins, home_losses, away_wins, away_losses,
            ats_wins_l10, ats_losses_l10
        FROM team_stats
        WHERE team_id = %s
        ORDER BY stat_date DESC
        LIMIT 1
    ''', (team_id,))

    row = cur.fetchone()
    if not row:
        return {}

    return {
        "wins": row[0] or 0,
        "losses": row[1] or 0,
        "net_rtg_l10": float(row[2]) if row[2] else None,
        "days_rest": row[3],
        "is_b2b": row[4] or False,
        "wins_l10": row[5] or 0,
        "losses_l10": row[6] or 0,
        "home_wins": row[7] or 0,
        "home_losses": row[8] or 0,
        "away_wins": row[9] or 0,
        "away_losses": row[10] or 0,
        "ats_wins_l10": row[11] or 0,
        "ats_losses_l10": row[12] or 0,
    }


def build_factors(home_team: str, away_team: str, home_stats: dict, away_stats: dict, best_bet: dict | None) -> list[str]:
    """Build explanation factors for the prediction."""
    factors = []

    # 1. Net rating comparison
    home_net = home_stats.get('net_rtg_l10')
    away_net = away_stats.get('net_rtg_l10')
    if home_net is not None and away_net is not None:
        diff = home_net - away_net
        if abs(diff) >= 1.0:
            better = home_team if diff > 0 else away_team
            factors.append(f"{better} +{abs(diff):.1f} Net Rating (L10)")

    # 2. Rest advantage
    home_b2b = home_stats.get('is_b2b', False)
    away_b2b = away_stats.get('is_b2b', False)
    home_rest = home_stats.get('days_rest') or 0
    away_rest = away_stats.get('days_rest') or 0

    if home_b2b and not away_b2b:
        factors.append(f"{away_team} rest advantage (vs B2B)")
    elif away_b2b and not home_b2b:
        factors.append(f"{home_team} rest advantage (vs B2B)")
    elif abs(home_rest - away_rest) >= 2:
        better = home_team if home_rest > away_rest else away_team
        factors.append(f"{better} +{abs(home_rest - away_rest)} days rest")

    # 3. Model edge on best bet
    if best_bet and best_bet.get('edge', 0) > 0:
        p_true = best_bet.get('p_true', 50)
        p_market = best_bet.get('p_market', 50)
        edge = best_bet.get('edge', 0)
        factors.append(f"Model: {p_true:.0f}% vs Market: {p_market:.0f}% (+{edge:.1f}% edge)")

    # 4. L10 record comparison
    home_l10_wins = home_stats.get('wins_l10', 0)
    away_l10_wins = away_stats.get('wins_l10', 0)
    if abs(home_l10_wins - away_l10_wins) >= 3:
        home_l10 = f"{home_stats.get('wins_l10', 0)}-{home_stats.get('losses_l10', 0)}"
        away_l10 = f"{away_stats.get('wins_l10', 0)}-{away_stats.get('losses_l10', 0)}"
        better = home_team if home_l10_wins > away_l10_wins else away_team
        better_record = home_l10 if home_l10_wins > away_l10_wins else away_l10
        factors.append(f"{better} is {better_record} in L10")

    return factors[:4]


def grade_predictions(db_url: str = None) -> dict:
    """
    Grade predictions for completed games.

    Looks for games that have:
    - A prediction snapshot
    - Final scores in game_results
    - Not yet graded

    Returns:
        Summary of grading results
    """
    conn = psycopg2.connect(db_url or DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    # Find ungraded predictions with completed games
    cur.execute('''
        SELECT
            ps.id, ps.game_id, ps.predicted_winner, ps.winner_probability,
            ps.best_bet_type, ps.best_bet_team, ps.best_bet_line,
            ps.best_bet_value_score, ps.best_bet_odds,
            gr.actual_winner, gr.home_score, gr.away_score,
            gr.closing_spread, gr.closing_total, gr.spread_result, gr.total_result,
            ps.home_team, ps.away_team
        FROM prediction_snapshots ps
        JOIN game_results gr ON ps.game_id = gr.game_id
        WHERE ps.winner_correct IS NULL
        AND gr.actual_winner IS NOT NULL
    ''')

    predictions = cur.fetchall()

    if not predictions:
        cur.close()
        conn.close()
        return {"predictions_graded": 0, "status": "no_predictions_to_grade"}

    graded = 0
    wins = 0
    losses = 0
    pushes = 0
    total_profit = 0

    for row in predictions:
        (pred_id, game_id, predicted_winner, winner_prob,
         bet_type, bet_team, bet_line, bet_value, bet_odds,
         actual_winner, home_score, away_score,
         closing_spread, closing_total, spread_result, total_result,
         home_team, away_team) = row

        # Grade winner prediction
        winner_correct = (predicted_winner == actual_winner)

        # Grade best bet
        bet_result = None
        bet_profit = None

        if bet_type and bet_team:
            if bet_type == 'spread':
                is_home_bet = (bet_team == home_team)
                if is_home_bet:
                    bet_result = 'win' if spread_result == 'home_cover' else ('push' if spread_result == 'push' else 'loss')
                else:
                    bet_result = 'win' if spread_result == 'away_cover' else ('push' if spread_result == 'push' else 'loss')

            elif bet_type == 'moneyline':
                bet_result = 'win' if bet_team == actual_winner else 'loss'

            elif bet_type == 'total':
                # For totals, bet_team might be 'over' or 'under' stored differently
                # Check the outcome label
                is_over = bet_line and bet_line > 0  # Simplified check
                if total_result:
                    if 'over' in bet_team.lower() or bet_line:
                        # Need to determine if this was over or under bet
                        # For now, use total_result directly
                        bet_result = 'push' if total_result == 'push' else total_result

            # Calculate profit (assuming $100 bet at -110 for spreads/totals)
            if bet_result == 'win':
                if bet_odds:
                    bet_profit = 100 * (float(bet_odds) - 1)
                else:
                    bet_profit = 90.91  # -110 payout
            elif bet_result == 'loss':
                bet_profit = -100
            else:
                bet_profit = 0  # Push

        # Update the snapshot
        cur.execute('''
            UPDATE prediction_snapshots SET
                actual_winner = %s,
                home_score = %s,
                away_score = %s,
                closing_spread = %s,
                closing_total = %s,
                winner_correct = %s,
                best_bet_result = %s,
                best_bet_profit = %s
            WHERE id = %s
        ''', (
            actual_winner, home_score, away_score,
            closing_spread, closing_total,
            winner_correct, bet_result, bet_profit,
            pred_id
        ))

        graded += 1
        if bet_result == 'win':
            wins += 1
            total_profit += (bet_profit or 0)
        elif bet_result == 'loss':
            losses += 1
            total_profit += (bet_profit or 0)
        else:
            pushes += 1

        result_str = 'WIN' if winner_correct else 'LOSS'
        bet_str = f", Best Bet: {bet_result.upper()}" if bet_result else ""
        logger.info(f"Graded {game_id}: Winner {result_str}{bet_str}")

    cur.close()
    conn.close()

    return {
        "predictions_graded": graded,
        "best_bet_wins": wins,
        "best_bet_losses": losses,
        "best_bet_pushes": pushes,
        "total_profit": round(total_profit, 2),
        "status": "completed"
    }


def get_performance_summary(days: int = 7, db_url: str = None) -> dict:
    """
    Get performance summary for the model over recent days.

    Args:
        days: Number of days to analyze
        db_url: Database connection string

    Returns:
        Performance metrics
    """
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Get all graded predictions
    cur.execute('''
        SELECT
            predicted_winner, actual_winner, winner_correct,
            best_bet_type, best_bet_team, best_bet_value_score,
            best_bet_result, best_bet_profit,
            snapshot_time
        FROM prediction_snapshots
        WHERE snapshot_time >= %s
        AND winner_correct IS NOT NULL
    ''', (cutoff,))

    predictions = cur.fetchall()
    cur.close()
    conn.close()

    if not predictions:
        return {
            "days_analyzed": days,
            "total_predictions": 0,
            "status": "no_data"
        }

    # Calculate winner accuracy
    total = len(predictions)
    winner_correct = sum(1 for p in predictions if p[2])

    # Calculate best bet performance
    best_bets = [p for p in predictions if p[6]]  # Has a bet result
    bet_wins = sum(1 for p in best_bets if p[6] == 'win')
    bet_losses = sum(1 for p in best_bets if p[6] == 'loss')
    total_profit = sum(p[7] or 0 for p in best_bets)

    # Performance by value score bucket
    buckets = {}
    for p in best_bets:
        value_score = p[5] or 0
        if value_score >= 70:
            bucket = "70+"
        elif value_score >= 60:
            bucket = "60-69"
        elif value_score >= 50:
            bucket = "50-59"
        else:
            bucket = "<50"

        if bucket not in buckets:
            buckets[bucket] = {"wins": 0, "losses": 0, "profit": 0}

        if p[6] == 'win':
            buckets[bucket]["wins"] += 1
        elif p[6] == 'loss':
            buckets[bucket]["losses"] += 1
        buckets[bucket]["profit"] += (p[7] or 0)

    return {
        "days_analyzed": days,
        "total_predictions": total,
        "winner_accuracy": {
            "correct": winner_correct,
            "total": total,
            "percentage": round(winner_correct / total * 100, 1) if total > 0 else 0
        },
        "best_bet_performance": {
            "wins": bet_wins,
            "losses": bet_losses,
            "win_rate": round(bet_wins / (bet_wins + bet_losses) * 100, 1) if (bet_wins + bet_losses) > 0 else 0,
            "total_profit": round(total_profit, 2),
            "roi": round(total_profit / (len(best_bets) * 100) * 100, 1) if best_bets else 0
        },
        "by_value_bucket": buckets,
        "status": "success"
    }


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python prediction_tracker.py [snapshot|grade|summary]")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'snapshot':
        hours = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        result = snapshot_predictions(hours_ahead=hours)
        print(f"\nResult: {result}")

    elif command == 'grade':
        result = grade_predictions()
        print(f"\nResult: {result}")

    elif command == 'summary':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        result = get_performance_summary(days=days)
        print(f"\nPerformance Summary ({result['days_analyzed']} days):")
        print(f"  Total Predictions: {result['total_predictions']}")
        if result['total_predictions'] > 0:
            wa = result['winner_accuracy']
            print(f"  Winner Accuracy: {wa['correct']}/{wa['total']} ({wa['percentage']}%)")
            bp = result['best_bet_performance']
            print(f"  Best Bet Record: {bp['wins']}-{bp['losses']} ({bp['win_rate']}%)")
            print(f"  Total Profit: ${bp['total_profit']:.2f}")
            print(f"  ROI: {bp['roi']}%")
            print(f"\n  By Value Bucket:")
            for bucket, stats in result.get('by_value_bucket', {}).items():
                print(f"    {bucket}: {stats['wins']}-{stats['losses']}, ${stats['profit']:.2f}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
