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

# Minimum value score to consider a bet worth tracking
# Below this threshold, we don't record a "best bet" - it's a pass
# Set to 65 after analysis showed low-value bets (40-60) had poor win rates
MIN_VALUE_THRESHOLD = 65


def snapshot_predictions(hours_ahead: float = 0.75, db_url: str = None) -> dict:
    """
    Snapshot predictions for games starting in the next N hours.

    Default is 0.75 hours (45 minutes) to capture predictions close to tip-off.
    Combined with a 15-minute schedule, this captures games ~30 min before tip.

    Args:
        hours_ahead: How many hours ahead to look for games (supports decimals)
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

        # Get value scores for this game - both algorithms
        cur.execute('''
            SELECT
                m.market_type,
                m.outcome_label,
                m.line,
                m.odds_decimal,
                vs.algo_a_value_score,
                vs.algo_a_edge_score,
                vs.algo_a_confidence,
                vs.algo_b_value_score,
                vs.algo_b_combined_edge,
                vs.algo_b_confidence,
                vs.active_algorithm,
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
        best_bet_a = None  # Best bet according to Algorithm A
        best_bet_b = None  # Best bet according to Algorithm B
        best_score_a = 0
        best_score_b = 0
        best_total = None  # Best total (over/under) bet
        best_total_score = 0
        active_algo = 'b'  # Default

        for row in scores:
            (mtype, outcome, line, odds,
             algo_a_score, algo_a_edge, algo_a_conf,
             algo_b_score, algo_b_edge, algo_b_conf,
             active_algorithm, p_true, p_market, raw_edge) = row

            is_home = 'home' in outcome.lower()
            active_algo = (active_algorithm or 'b').lower()

            # Track best value bet for Algorithm A (only if meets threshold)
            if algo_a_score and float(algo_a_score) > best_score_a and float(algo_a_score) >= MIN_VALUE_THRESHOLD:
                best_score_a = float(algo_a_score)
                best_bet_a = {
                    "type": mtype,
                    "team": home_team if is_home else away_team,
                    "line": float(line) if line else None,
                    "value_score": int(algo_a_score),
                    "edge_score": float(algo_a_edge) if algo_a_edge else 0,
                    "confidence": float(algo_a_conf) if algo_a_conf else 1.0,
                    "odds": float(odds) if odds else None,
                    "p_true": float(p_true) if p_true else 0,
                    "p_market": float(p_market) if p_market else 0,
                }

            # Track best value bet for Algorithm B (only if meets threshold)
            if algo_b_score and float(algo_b_score) > best_score_b and float(algo_b_score) >= MIN_VALUE_THRESHOLD:
                best_score_b = float(algo_b_score)
                best_bet_b = {
                    "type": mtype,
                    "team": home_team if is_home else away_team,
                    "line": float(line) if line else None,
                    "value_score": int(algo_b_score),
                    "combined_edge": float(algo_b_edge) if algo_b_edge else 0,
                    "confidence": float(algo_b_conf) if algo_b_conf else 1.0,
                    "odds": float(odds) if odds else None,
                    "p_true": float(p_true) if p_true else 0,
                    "p_market": float(p_market) if p_market else 0,
                }

            # Track best total bet separately (only if meets threshold)
            if mtype == 'total' and algo_a_score and float(algo_a_score) > best_total_score and float(algo_a_score) >= MIN_VALUE_THRESHOLD:
                best_total_score = float(algo_a_score)
                best_total = {
                    "direction": outcome.replace('_', ''),  # "over" or "under"
                    "line": float(line) if line else None,
                    "value_score": int(algo_a_score),
                    "edge": float(algo_a_edge) if algo_a_edge else 0,
                    "odds": float(odds) if odds else None,
                }

            # Get moneyline probabilities
            if mtype == 'moneyline':
                if is_home:
                    home_prob = float(p_true) if p_true else 0.5
                else:
                    away_prob = float(p_true) if p_true else 0.5

        # Use active algorithm's best bet as the "primary" best bet
        best_bet = best_bet_a if active_algo == 'a' else best_bet_b

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

        # Get line movement data
        line_movement = get_line_movement(cur, game_id)

        # Add line movement factor if significant
        if line_movement.get("spread_movement") and abs(line_movement["spread_movement"]) >= 0.5:
            direction = "toward home" if line_movement["spread_movement"] < 0 else "toward away"
            factors.append(f"Line moved {abs(line_movement['spread_movement'])} pts {direction}")

        # Calculate game_date from tip_time (use Eastern Time for NBA game dates)
        # tip_time is UTC, convert to ET for the game date
        eastern_offset = timedelta(hours=-5)
        tip_time_et = tip_time + eastern_offset if tip_time.tzinfo else tip_time.replace(tzinfo=timezone.utc) + eastern_offset
        game_date = tip_time_et.date()

        # Insert snapshot with both algorithm scores, total bet, and line movement
        cur.execute('''
            INSERT INTO prediction_snapshots (
                game_id, snapshot_time, home_team, away_team, tip_time, game_date,
                predicted_winner, winner_probability, winner_confidence,
                best_bet_type, best_bet_team, best_bet_line,
                best_bet_value_score, best_bet_edge, best_bet_odds,
                best_total_direction, best_total_line, best_total_value_score,
                best_total_edge, best_total_odds,
                factors,
                home_injury_score, away_injury_score, home_ppg_lost, away_ppg_lost, injury_edge,
                algo_a_value_score, algo_a_edge_score, algo_a_confidence,
                algo_b_value_score, algo_b_combined_edge, algo_b_confidence,
                active_algorithm,
                opening_spread, current_spread, spread_movement,
                opening_total, current_total, total_movement,
                line_movement_direction
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            game_id, now, home_team, away_team, tip_time, game_date,
            predicted_winner, round(winner_prob * 100, 1), confidence,
            best_bet['type'] if best_bet else None,
            best_bet['team'] if best_bet else None,
            best_bet['line'] if best_bet else None,
            best_bet['value_score'] if best_bet else None,
            best_bet.get('edge') or best_bet.get('edge_score') or best_bet.get('combined_edge') if best_bet else None,
            best_bet['odds'] if best_bet else None,
            # Total bet fields
            best_total['direction'] if best_total else None,
            best_total['line'] if best_total else None,
            best_total['value_score'] if best_total else None,
            best_total['edge'] if best_total else None,
            best_total['odds'] if best_total else None,
            json.dumps(factors),
            round(home_injury_score, 2),
            round(away_injury_score, 2),
            round(home_ppg_lost, 1),
            round(away_ppg_lost, 1),
            round(injury_edge, 2),
            # Algorithm A
            best_bet_a['value_score'] if best_bet_a else None,
            best_bet_a['edge_score'] if best_bet_a else None,
            best_bet_a['confidence'] if best_bet_a else None,
            # Algorithm B
            best_bet_b['value_score'] if best_bet_b else None,
            best_bet_b['combined_edge'] if best_bet_b else None,
            best_bet_b['confidence'] if best_bet_b else None,
            active_algo,
            # Line movement
            line_movement.get("opening_spread"),
            line_movement.get("current_spread"),
            line_movement.get("spread_movement"),
            line_movement.get("opening_total"),
            line_movement.get("current_total"),
            line_movement.get("total_movement"),
            line_movement.get("direction"),
        ))

        snapshots_created += 1
        logger.info(f"Snapshotted {away_team} @ {home_team}: {predicted_winner} to win ({confidence})")

        if best_bet and best_bet['value_score'] >= MIN_VALUE_THRESHOLD:
            logger.info(f"  Best bet: {best_bet['team']} {best_bet['type']} {best_bet['line'] or ''} ({best_bet['value_score']}%)")
        else:
            logger.info(f"  No qualifying bet (threshold: {MIN_VALUE_THRESHOLD})")

        if line_movement.get("spread_movement") and abs(line_movement["spread_movement"]) >= 0.5:
            logger.info(f"  Line move: {line_movement['opening_spread']} -> {line_movement['current_spread']} ({line_movement['direction']})")

    cur.close()
    conn.close()

    return {
        "games_snapshotted": snapshots_created,
        "snapshot_time": now.isoformat(),
        "status": "completed"
    }


def get_line_movement(cur, game_id: str) -> dict:
    """
    Get opening and current lines for a game to track line movement.

    Returns dict with opening/current spread and total, plus movement.
    Line movement helps identify sharp money:
    - Spread moving toward a team often indicates sharp action
    - Total movement shows where money is going on O/U
    """
    result = {
        "opening_spread": None,
        "current_spread": None,
        "spread_movement": None,
        "opening_total": None,
        "current_total": None,
        "total_movement": None,
        "direction": None,
    }

    # Get opening lines (earliest snapshot for this game)
    cur.execute('''
        SELECT home_spread, total_line, snapshot_time
        FROM odds_snapshots
        WHERE game_id = %s
        AND home_spread IS NOT NULL
        ORDER BY snapshot_time ASC
        LIMIT 1
    ''', (game_id,))

    opening = cur.fetchone()
    if opening:
        result["opening_spread"] = float(opening[0]) if opening[0] else None
        result["opening_total"] = float(opening[1]) if opening[1] else None

    # Get current lines (most recent snapshot)
    cur.execute('''
        SELECT home_spread, total_line, snapshot_time
        FROM odds_snapshots
        WHERE game_id = %s
        AND home_spread IS NOT NULL
        ORDER BY snapshot_time DESC
        LIMIT 1
    ''', (game_id,))

    current = cur.fetchone()
    if current:
        result["current_spread"] = float(current[0]) if current[0] else None
        result["current_total"] = float(current[1]) if current[1] else None

    # Calculate movement
    if result["opening_spread"] is not None and result["current_spread"] is not None:
        # Movement: current - opening
        # Positive = line moved toward away team (home got more points)
        # Negative = line moved toward home team (home gave more points)
        result["spread_movement"] = round(result["current_spread"] - result["opening_spread"], 1)

        # Determine direction (half point or more is significant)
        if result["spread_movement"] <= -0.5:
            result["direction"] = "sharp_home"  # Money on home
        elif result["spread_movement"] >= 0.5:
            result["direction"] = "sharp_away"  # Money on away

    if result["opening_total"] is not None and result["current_total"] is not None:
        result["total_movement"] = round(result["current_total"] - result["opening_total"], 1)

        # Combine with spread direction if significant total movement
        if result["total_movement"] <= -0.5:
            if result["direction"]:
                result["direction"] += "_under"
            else:
                result["direction"] = "steam_under"
        elif result["total_movement"] >= 0.5:
            if result["direction"]:
                result["direction"] += "_over"
            else:
                result["direction"] = "steam_over"

    return result


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


def grade_bet(bet_type: str, bet_team: str, home_team: str, actual_winner: str,
               spread_result: str, total_result: str, bet_odds: float = None) -> tuple:
    """Grade a single bet and return (result, profit)."""
    bet_result = None
    bet_profit = None

    if not bet_type or not bet_team:
        return None, None

    if bet_type == 'spread':
        is_home_bet = (bet_team == home_team)
        if is_home_bet:
            bet_result = 'win' if spread_result == 'home_cover' else ('push' if spread_result == 'push' else 'loss')
        else:
            bet_result = 'win' if spread_result == 'away_cover' else ('push' if spread_result == 'push' else 'loss')

    elif bet_type == 'moneyline':
        bet_result = 'win' if bet_team == actual_winner else 'loss'

    elif bet_type == 'total':
        if total_result:
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

    return bet_result, bet_profit


def grade_predictions(db_url: str = None) -> dict:
    """
    Grade predictions for completed games.

    Looks for games that have:
    - A prediction snapshot
    - Final scores in game_results
    - Not yet graded

    Grades both Algorithm A and Algorithm B picks for comparison.

    Returns:
        Summary of grading results including per-algorithm performance
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
            ps.home_team, ps.away_team,
            ps.algo_a_value_score, ps.algo_b_value_score, ps.active_algorithm,
            ps.best_total_direction, ps.best_total_line, ps.best_total_odds
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

    # Per-algorithm tracking
    algo_a_stats = {"wins": 0, "losses": 0, "pushes": 0, "profit": 0}
    algo_b_stats = {"wins": 0, "losses": 0, "pushes": 0, "profit": 0}

    for row in predictions:
        (pred_id, game_id, predicted_winner, winner_prob,
         bet_type, bet_team, bet_line, bet_value, bet_odds,
         actual_winner, home_score, away_score,
         closing_spread, closing_total, spread_result, total_result,
         home_team, away_team,
         algo_a_value, algo_b_value, active_algo,
         total_direction, total_line, total_odds) = row

        # Grade winner prediction
        winner_correct = (predicted_winner == actual_winner)

        # Grade total bet if present
        total_bet_result = None
        total_bet_profit = None
        if total_direction and total_result:
            if total_result == 'push':
                total_bet_result = 'push'
                total_bet_profit = 0
            elif total_direction == total_result:  # 'over' == 'over' or 'under' == 'under'
                total_bet_result = 'win'
                total_bet_profit = 90.91 if total_odds is None else 100 * (float(total_odds) - 1)
            else:
                total_bet_result = 'loss'
                total_bet_profit = -100

        # Grade the primary best bet (from active algorithm)
        bet_result, bet_profit = grade_bet(
            bet_type, bet_team, home_team, actual_winner,
            spread_result, total_result, bet_odds
        )

        # For now, use same bet for both algos (they pick same bet, just different scores)
        # In future, we could track separate best bets per algorithm
        algo_a_result = bet_result
        algo_b_result = bet_result
        algo_a_profit = bet_profit
        algo_b_profit = bet_profit

        # Track per-algorithm stats
        if algo_a_result == 'win':
            algo_a_stats["wins"] += 1
            algo_a_stats["profit"] += algo_a_profit or 0
        elif algo_a_result == 'loss':
            algo_a_stats["losses"] += 1
            algo_a_stats["profit"] += algo_a_profit or 0
        else:
            algo_a_stats["pushes"] += 1

        if algo_b_result == 'win':
            algo_b_stats["wins"] += 1
            algo_b_stats["profit"] += algo_b_profit or 0
        elif algo_b_result == 'loss':
            algo_b_stats["losses"] += 1
            algo_b_stats["profit"] += algo_b_profit or 0
        else:
            algo_b_stats["pushes"] += 1

        # Update the snapshot with both algorithm results and total bet
        cur.execute('''
            UPDATE prediction_snapshots SET
                actual_winner = %s,
                home_score = %s,
                away_score = %s,
                closing_spread = %s,
                closing_total = %s,
                winner_correct = %s,
                best_bet_result = %s,
                best_bet_profit = %s,
                best_total_result = %s,
                best_total_profit = %s,
                algo_a_bet_result = %s,
                algo_b_bet_result = %s,
                algo_a_profit = %s,
                algo_b_profit = %s
            WHERE id = %s
        ''', (
            actual_winner, home_score, away_score,
            closing_spread, closing_total,
            winner_correct, bet_result, bet_profit,
            total_bet_result, total_bet_profit,
            algo_a_result, algo_b_result,
            algo_a_profit, algo_b_profit,
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
        "algo_a_performance": algo_a_stats,
        "algo_b_performance": algo_b_stats,
        "status": "completed"
    }


def get_performance_summary(days: int = 7, db_url: str = None) -> dict:
    """
    Get performance summary for the model over recent days.

    Compares Algorithm A and Algorithm B performance.

    Args:
        days: Number of days to analyze
        db_url: Database connection string

    Returns:
        Performance metrics including per-algorithm comparison
    """
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Get all graded predictions with algorithm data
    cur.execute('''
        SELECT
            predicted_winner, actual_winner, winner_correct,
            best_bet_type, best_bet_team, best_bet_value_score,
            best_bet_result, best_bet_profit,
            snapshot_time,
            algo_a_value_score, algo_b_value_score,
            algo_a_bet_result, algo_b_bet_result,
            algo_a_profit, algo_b_profit
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

    # Algorithm A performance (columns: 9=algo_a_value, 11=algo_a_result, 13=algo_a_profit)
    algo_a_bets = [p for p in predictions if p[11]]  # Has algo_a result
    algo_a_wins = sum(1 for p in algo_a_bets if p[11] == 'win')
    algo_a_losses = sum(1 for p in algo_a_bets if p[11] == 'loss')
    algo_a_profit = sum(p[13] or 0 for p in algo_a_bets)

    # Algorithm B performance (columns: 10=algo_b_value, 12=algo_b_result, 14=algo_b_profit)
    algo_b_bets = [p for p in predictions if p[12]]  # Has algo_b result
    algo_b_wins = sum(1 for p in algo_b_bets if p[12] == 'win')
    algo_b_losses = sum(1 for p in algo_b_bets if p[12] == 'loss')
    algo_b_profit = sum(p[14] or 0 for p in algo_b_bets)

    # Performance by value score bucket (using best_bet_value_score)
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

    def calc_win_rate(wins, losses):
        return round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0

    def calc_roi(profit, bets):
        return round(profit / (len(bets) * 100) * 100, 1) if bets else 0

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
            "win_rate": calc_win_rate(bet_wins, bet_losses),
            "total_profit": round(total_profit, 2),
            "roi": calc_roi(total_profit, best_bets)
        },
        "algorithm_comparison": {
            "algo_a": {
                "wins": algo_a_wins,
                "losses": algo_a_losses,
                "win_rate": calc_win_rate(algo_a_wins, algo_a_losses),
                "profit": round(algo_a_profit, 2),
                "roi": calc_roi(algo_a_profit, algo_a_bets)
            },
            "algo_b": {
                "wins": algo_b_wins,
                "losses": algo_b_losses,
                "win_rate": calc_win_rate(algo_b_wins, algo_b_losses),
                "profit": round(algo_b_profit, 2),
                "roi": calc_roi(algo_b_profit, algo_b_bets)
            }
        },
        "by_value_bucket": buckets,
        "status": "success"
    }


def analyze_line_movement_performance(days: int = 30, db_url: str = None) -> dict:
    """
    Analyze how predictions perform based on line movement.

    Tracks:
    - Performance when betting WITH sharp money (following line movement)
    - Performance when betting AGAINST sharp money (fading line movement)
    - Performance by direction (sharp_home, sharp_away, etc.)

    Returns:
        Analysis of line movement correlation with results
    """
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    cur.execute('''
        SELECT
            best_bet_team, home_team, away_team,
            spread_movement, line_movement_direction,
            best_bet_result, best_bet_profit
        FROM prediction_snapshots
        WHERE snapshot_time >= %s
        AND best_bet_result IS NOT NULL
        AND spread_movement IS NOT NULL
    ''', (cutoff,))

    results = cur.fetchall()
    cur.close()
    conn.close()

    if not results:
        return {"status": "no_data", "days_analyzed": days}

    # Track performance by movement direction
    with_sharp = {"wins": 0, "losses": 0, "profit": 0}  # Betting same direction as line move
    against_sharp = {"wins": 0, "losses": 0, "profit": 0}  # Betting against line move
    by_direction = {}

    for row in results:
        bet_team, home, away, spread_move, direction, result, profit = row

        if not spread_move or abs(float(spread_move)) < 0.5:
            continue  # Ignore small movements

        # Determine if we bet WITH or AGAINST the line movement
        is_home_bet = (bet_team == home)
        line_moved_to_home = (float(spread_move) < 0)  # Negative = toward home

        if is_home_bet == line_moved_to_home:
            # We bet WITH the sharp money
            bucket = with_sharp
        else:
            # We bet AGAINST the sharp money
            bucket = against_sharp

        if result == 'win':
            bucket["wins"] += 1
            bucket["profit"] += float(profit or 0)
        elif result == 'loss':
            bucket["losses"] += 1
            bucket["profit"] += float(profit or 0)

        # Track by direction
        if direction:
            if direction not in by_direction:
                by_direction[direction] = {"wins": 0, "losses": 0, "profit": 0, "count": 0}
            by_direction[direction]["count"] += 1
            if result == 'win':
                by_direction[direction]["wins"] += 1
                by_direction[direction]["profit"] += float(profit or 0)
            elif result == 'loss':
                by_direction[direction]["losses"] += 1
                by_direction[direction]["profit"] += float(profit or 0)

    def calc_stats(bucket):
        total = bucket["wins"] + bucket["losses"]
        return {
            "wins": bucket["wins"],
            "losses": bucket["losses"],
            "win_rate": round(bucket["wins"] / total * 100, 1) if total > 0 else 0,
            "profit": round(bucket["profit"], 2),
            "roi": round(bucket["profit"] / (total * 100) * 100, 1) if total > 0 else 0,
        }

    return {
        "days_analyzed": days,
        "with_sharp_money": calc_stats(with_sharp),
        "against_sharp_money": calc_stats(against_sharp),
        "by_direction": {k: calc_stats(v) for k, v in by_direction.items()},
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

    elif command == 'line-movement':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        result = analyze_line_movement_performance(days=days)
        print(f"\nLine Movement Analysis ({result['days_analyzed']} days):")

        if result['status'] == 'no_data':
            print("  No data available with line movement tracking")
        else:
            ws = result['with_sharp_money']
            print(f"\n  WITH Sharp Money (following line moves):")
            print(f"    Record: {ws['wins']}-{ws['losses']} ({ws['win_rate']}%)")
            print(f"    Profit: ${ws['profit']:.2f} (ROI: {ws['roi']}%)")

            ag = result['against_sharp_money']
            print(f"\n  AGAINST Sharp Money (fading line moves):")
            print(f"    Record: {ag['wins']}-{ag['losses']} ({ag['win_rate']}%)")
            print(f"    Profit: ${ag['profit']:.2f} (ROI: {ag['roi']}%)")

            if result.get('by_direction'):
                print(f"\n  By Direction:")
                for direction, stats in result['by_direction'].items():
                    print(f"    {direction}: {stats['wins']}-{stats['losses']} ({stats['win_rate']}%) ${stats['profit']:.2f}")

    else:
        print(f"Unknown command: {command}")
        print("Available commands: snapshot, grade, summary, line-movement")
        sys.exit(1)
