"""
Re-grade predictions that were graded with incorrect closing lines.

This script:
1. Finds predictions where closing_spread doesn't match best_bet_line
2. Re-grades them using the correct pre-game snapshot values
3. Updates the prediction_snapshots table

Run with: python -m src.tasks.regrade_predictions
"""

import psycopg2
import structlog
from datetime import datetime, timezone, timedelta

logger = structlog.get_logger()

DB_URL = 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway'


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


def regrade_predictions(days_back: int = 7, db_url: str = None) -> dict:
    """
    Re-grade predictions that may have incorrect closing lines.

    Args:
        days_back: Number of days to look back
        db_url: Database connection string

    Returns:
        Summary of re-grading results
    """
    conn = psycopg2.connect(db_url or DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    # First, show current state for diagnostics
    print("\n=== DIAGNOSTIC: Checking for mismatched lines ===")
    cur.execute('''
        SELECT
            ps.id, ps.game_id, ps.home_team, ps.away_team,
            ps.best_bet_line, ps.closing_spread,
            ps.best_total_line, ps.closing_total,
            ps.best_bet_result, ps.best_bet_type,
            gr.home_score, gr.away_score
        FROM prediction_snapshots ps
        JOIN game_results gr ON ps.game_id = gr.game_id
        WHERE ps.snapshot_time >= %s
        AND ps.winner_correct IS NOT NULL
        ORDER BY ps.snapshot_time DESC
        LIMIT 20
    ''', (cutoff,))

    rows = cur.fetchall()
    print(f"\nFound {len(rows)} graded predictions in last {days_back} days:")
    print("-" * 100)

    mismatched = 0
    for row in rows:
        (pred_id, game_id, home, away, bet_line, closing_spread,
         total_line, closing_total, bet_result, bet_type, home_score, away_score) = row

        # Check if closing_spread looks wrong (not a valid betting line)
        spread_mismatch = False
        if closing_spread is not None:
            # Valid spreads are typically between -20 and +20
            if abs(float(closing_spread)) > 25:
                spread_mismatch = True
                mismatched += 1

        mismatch_flag = " *** MISMATCH ***" if spread_mismatch else ""
        print(f"{away} @ {home}: bet_line={bet_line}, closing_spread={closing_spread}, "
              f"result={bet_result}, score={away_score}-{home_score}{mismatch_flag}")

    print(f"\nMismatched predictions: {mismatched}")

    # Now re-grade ALL graded predictions using correct snapshot values
    print("\n=== RE-GRADING PREDICTIONS ===")

    cur.execute('''
        SELECT
            ps.id, ps.game_id, ps.predicted_winner, ps.winner_probability,
            ps.best_bet_type, ps.best_bet_team, ps.best_bet_line,
            ps.best_bet_value_score, ps.best_bet_odds,
            gr.actual_winner, gr.home_score, gr.away_score,
            ps.home_team, ps.away_team,
            ps.best_total_direction, ps.best_total_line, ps.best_total_odds
        FROM prediction_snapshots ps
        JOIN game_results gr ON ps.game_id = gr.game_id
        WHERE ps.snapshot_time >= %s
        AND gr.actual_winner IS NOT NULL
    ''', (cutoff,))

    predictions = cur.fetchall()

    if not predictions:
        cur.close()
        conn.close()
        return {"predictions_regraded": 0, "status": "no_predictions_to_regrade"}

    regraded = 0
    changed = 0
    wins = 0
    losses = 0

    for row in predictions:
        (pred_id, game_id, predicted_winner, winner_prob,
         bet_type, bet_team, bet_line, bet_value, bet_odds,
         actual_winner, home_score, away_score,
         home_team, away_team,
         total_direction, total_line, total_odds) = row

        # Use snapshot's pre-game lines for grading
        closing_spread = bet_line  # The line captured at snapshot time
        closing_total = total_line  # The total captured at snapshot time

        # Calculate spread_result using snapshot's betting line
        spread_result = None
        if closing_spread is not None and home_score is not None and away_score is not None:
            home_adjusted = home_score + float(closing_spread)
            if home_adjusted > away_score:
                spread_result = 'home_cover'
            elif home_adjusted < away_score:
                spread_result = 'away_cover'
            else:
                spread_result = 'push'

        # Calculate total_result using snapshot's total line
        total_result = None
        if closing_total is not None and home_score is not None and away_score is not None:
            actual_total = home_score + away_score
            if actual_total > float(closing_total):
                total_result = 'over'
            elif actual_total < float(closing_total):
                total_result = 'under'
            else:
                total_result = 'push'

        # Grade winner prediction
        winner_correct = (predicted_winner == actual_winner)

        # Grade total bet if present
        total_bet_result = None
        total_bet_profit = None
        if total_direction and total_result:
            if total_result == 'push':
                total_bet_result = 'push'
                total_bet_profit = 0
            elif total_direction == total_result:
                total_bet_result = 'win'
                total_bet_profit = 90.91 if total_odds is None else 100 * (float(total_odds) - 1)
            else:
                total_bet_result = 'loss'
                total_bet_profit = -100

        # Grade the primary best bet
        bet_result, bet_profit = grade_bet(
            bet_type, bet_team, home_team, actual_winner,
            spread_result, total_result, bet_odds
        )

        # Update the snapshot with corrected values
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
            RETURNING best_bet_result
        ''', (
            actual_winner, home_score, away_score,
            closing_spread, closing_total,  # NOW uses snapshot values
            winner_correct, bet_result, bet_profit,
            total_bet_result, total_bet_profit,
            bet_result, bet_result,  # Same for both algos
            bet_profit, bet_profit,
            pred_id
        ))

        old_result = cur.fetchone()
        regraded += 1

        if bet_result == 'win':
            wins += 1
        elif bet_result == 'loss':
            losses += 1

        print(f"Re-graded {away_team} @ {home_team}: {bet_type} {bet_team} {bet_line} -> {bet_result}")

    cur.close()
    conn.close()

    return {
        "predictions_regraded": regraded,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
        "status": "completed"
    }


if __name__ == '__main__':
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    result = regrade_predictions(days_back=days)
    print(f"\n=== RESULT ===")
    print(f"Re-graded: {result['predictions_regraded']}")
    print(f"Record: {result['wins']}-{result['losses']} ({result['win_rate']}%)")
