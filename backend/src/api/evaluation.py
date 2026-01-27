"""Evaluation and analytics API endpoints."""

from datetime import date, timedelta

import psycopg2
from fastapi import APIRouter, Query

router = APIRouter()

DB_URL = 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway'


@router.get("/evaluation/summary")
async def get_evaluation_summary(
    days: int = Query(14, ge=1, le=90),
    min_value: float = Query(65, ge=0, le=100),
) -> dict:
    """
    Get overall model performance summary.

    Returns metrics including win rate, ROI, and profit for bets above min_value threshold.
    Uses prediction_snapshots for accurate graded data.
    """
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cutoff = date.today() - timedelta(days=days)

    # Get performance from prediction_snapshots
    cur.execute('''
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN best_bet_result = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN best_bet_result = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN best_bet_result = 'push' THEN 1 ELSE 0 END) as pushes,
            SUM(COALESCE(best_bet_profit, 0)) as profit
        FROM prediction_snapshots
        WHERE game_date >= %s
        AND winner_correct IS NOT NULL
        AND best_bet_value_score >= %s
        AND best_bet_type IS NOT NULL
    ''', (cutoff, min_value))

    row = cur.fetchone()
    total, wins, losses, pushes, profit = row

    cur.close()
    conn.close()

    total_bets = (wins or 0) + (losses or 0)

    return {
        'period_days': days,
        'min_value_threshold': min_value,
        'total_bets': total_bets,
        'wins': wins or 0,
        'losses': losses or 0,
        'pushes': pushes or 0,
        'win_rate': round((wins or 0) / total_bets * 100, 1) if total_bets > 0 else None,
        'profit': round(float(profit or 0), 2),
        'roi': round(float(profit or 0) / (total_bets * 100) * 100, 1) if total_bets > 0 else None,
    }


def _performance_by_bucket(days: int) -> list[dict]:
    """Calculate performance by value score bucket using prediction_snapshots."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cutoff = date.today() - timedelta(days=days)

    # Query prediction_snapshots for accurate graded data
    cur.execute('''
        SELECT
            best_bet_value_score,
            best_bet_result,
            best_bet_profit,
            best_bet_odds
        FROM prediction_snapshots
        WHERE game_date >= %s
        AND winner_correct IS NOT NULL
        AND best_bet_type IS NOT NULL
        AND best_bet_value_score IS NOT NULL
    ''', (cutoff,))

    rows = cur.fetchall()

    # Buckets
    buckets = {
        '<50': {'wins': 0, 'losses': 0, 'profit': 0},
        '50-59': {'wins': 0, 'losses': 0, 'profit': 0},
        '60-69': {'wins': 0, 'losses': 0, 'profit': 0},
        '70-79': {'wins': 0, 'losses': 0, 'profit': 0},
        '80+': {'wins': 0, 'losses': 0, 'profit': 0},
    }

    for row in rows:
        value_score, result, profit, odds = row

        if value_score is None or result is None:
            continue

        value = float(value_score)

        # Determine bucket
        if value < 50:
            bucket = '<50'
        elif value < 60:
            bucket = '50-59'
        elif value < 70:
            bucket = '60-69'
        elif value < 80:
            bucket = '70-79'
        else:
            bucket = '80+'

        if result == 'push':
            continue
        elif result == 'win':
            buckets[bucket]['wins'] += 1
            buckets[bucket]['profit'] += float(profit) if profit else 90.91
        else:
            buckets[bucket]['losses'] += 1
            buckets[bucket]['profit'] += float(profit) if profit else -100

    cur.close()
    conn.close()

    # Format results
    results = []
    for bucket_name in ['80+', '70-79', '60-69', '50-59', '<50']:
        data = buckets[bucket_name]
        total = data['wins'] + data['losses']
        results.append({
            'bucket': bucket_name,
            'bet_count': total,
            'wins': data['wins'],
            'losses': data['losses'],
            'win_rate': round(data['wins'] / total * 100, 1) if total > 0 else None,
            'roi': round(data['profit'] / (total * 100) * 100, 1) if total > 0 else None,
            'profit': round(data['profit'], 2),
        })

    return results


@router.get("/evaluation/performance")
async def get_performance_by_bucket(
    days: int = Query(14, ge=1, le=90),
) -> list[dict]:
    """
    Get performance metrics grouped by Value Score buckets.

    Shows win rate, ROI, and profit for different score ranges.
    Uses prediction_snapshots for accurate graded data.
    """
    return _performance_by_bucket(days)


@router.get("/evaluation/daily")
async def get_daily_results(
    days: int = Query(7, ge=1, le=30),
    min_value: float = Query(65, ge=0, le=100),
) -> list[dict]:
    """
    Get daily performance breakdown for recent days.

    Shows each day's bets, wins, losses, and P/L.
    Uses prediction_snapshots for accurate line data.
    """
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cutoff = date.today() - timedelta(days=days)

    # Query prediction_snapshots for accurate graded data
    cur.execute('''
        SELECT
            ps.game_date,
            ps.home_team,
            ps.away_team,
            ps.best_bet_type,
            ps.best_bet_team,
            ps.best_bet_line,
            ps.best_bet_odds,
            ps.best_bet_value_score as value_score,
            ps.best_bet_result,
            ps.best_bet_profit,
            ps.home_score,
            ps.away_score
        FROM prediction_snapshots ps
        WHERE ps.game_date >= %s
        AND ps.winner_correct IS NOT NULL
        AND ps.best_bet_value_score >= %s
        AND ps.best_bet_type IS NOT NULL
        ORDER BY ps.game_date DESC, ps.tip_time DESC
    ''', (cutoff, min_value))

    rows = cur.fetchall()

    # Group by date
    daily = {}

    for row in rows:
        (game_date, home, away, bet_type, bet_team, bet_line, bet_odds,
         value_score, bet_result, bet_profit, home_score, away_score) = row

        if value_score is None:
            continue

        date_str = game_date.isoformat()
        if date_str not in daily:
            daily[date_str] = {
                'date': date_str,
                'bets': [],
                'wins': 0,
                'losses': 0,
                'pushes': 0,
                'profit': 0.0,
            }

        # Format bet description
        if bet_type == 'total':
            # For totals, bet_team would be 'over' or 'under'
            team_str = bet_team.capitalize() if bet_team else 'Total'
        else:
            team_str = bet_team or ''

        line_str = f'{float(bet_line):+.1f}' if bet_line and bet_type != 'moneyline' else ''

        # Use pre-calculated result from grading
        if bet_result == 'push':
            daily[date_str]['pushes'] += 1
            profit = 0.0
        elif bet_result == 'win':
            daily[date_str]['wins'] += 1
            profit = float(bet_profit) if bet_profit else 90.91
        else:
            daily[date_str]['losses'] += 1
            profit = float(bet_profit) if bet_profit else -100.0

        daily[date_str]['profit'] += profit

        daily[date_str]['bets'].append({
            'matchup': f'{away} @ {home}',
            'bet': f'{team_str} {bet_type} {line_str}'.strip(),
            'value_score': round(float(value_score)),
            'result': bet_result or 'pending',
            'profit': round(profit, 2),
            'final_score': f'{away_score}-{home_score}' if home_score else 'N/A',
        })

    cur.close()
    conn.close()

    # Format and sort by date
    results = []
    for d in sorted(daily.values(), key=lambda x: x['date'], reverse=True):
        total_bets = d['wins'] + d['losses']
        d['record'] = f"{d['wins']}-{d['losses']}" + (f"-{d['pushes']}" if d['pushes'] > 0 else "")
        d['roi'] = round(d['profit'] / (total_bets * 100) * 100, 1) if total_bets > 0 else 0
        d['profit'] = round(d['profit'], 2)
        results.append(d)

    return results


@router.get("/evaluation/predictions")
async def get_prediction_performance(
    days: int = Query(14, ge=1, le=90),
    min_value: int = Query(0, ge=0, le=100),
) -> dict:
    """
    Get performance of predictions from prediction_snapshots.

    This uses the pre-game snapshots that are captured ~30 min before tip-off
    and graded after games complete.
    """
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cutoff = date.today() - timedelta(days=days)

    # Get summary stats
    cur.execute('''
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN winner_correct = true THEN 1 ELSE 0 END) as winner_wins,
            SUM(CASE WHEN winner_correct = false THEN 1 ELSE 0 END) as winner_losses,
            SUM(CASE WHEN best_bet_result = 'win' THEN 1 ELSE 0 END) as bet_wins,
            SUM(CASE WHEN best_bet_result = 'loss' THEN 1 ELSE 0 END) as bet_losses,
            SUM(CASE WHEN best_bet_result = 'push' THEN 1 ELSE 0 END) as bet_pushes,
            SUM(COALESCE(best_bet_profit, 0)) as total_profit
        FROM prediction_snapshots
        WHERE game_date >= %s
        AND winner_correct IS NOT NULL
        AND (best_bet_value_score >= %s OR %s = 0)
    ''', (cutoff, min_value, min_value))

    row = cur.fetchone()
    total, winner_wins, winner_losses, bet_wins, bet_losses, bet_pushes, total_profit = row

    # Get performance by value bucket
    cur.execute('''
        SELECT
            CASE
                WHEN best_bet_value_score >= 80 THEN '80+'
                WHEN best_bet_value_score >= 70 THEN '70-79'
                WHEN best_bet_value_score >= 60 THEN '60-69'
                WHEN best_bet_value_score >= 50 THEN '50-59'
                ELSE '<50'
            END as bucket,
            COUNT(*) as total,
            SUM(CASE WHEN best_bet_result = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN best_bet_result = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(COALESCE(best_bet_profit, 0)) as profit
        FROM prediction_snapshots
        WHERE game_date >= %s
        AND winner_correct IS NOT NULL
        AND best_bet_value_score IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket DESC
    ''', (cutoff,))

    buckets = []
    for row in cur.fetchall():
        bucket, count, wins, losses, profit = row
        total_bets = (wins or 0) + (losses or 0)
        buckets.append({
            'bucket': bucket,
            'total': count,
            'wins': wins or 0,
            'losses': losses or 0,
            'win_rate': round((wins or 0) / total_bets * 100, 1) if total_bets > 0 else None,
            'profit': float(profit or 0),
            'roi': round(float(profit or 0) / (total_bets * 100) * 100, 1) if total_bets > 0 else None,
        })

    # Get recent predictions with results
    cur.execute('''
        SELECT
            home_team, away_team, tip_time,
            predicted_winner, winner_probability, winner_confidence,
            best_bet_type, best_bet_team, best_bet_line, best_bet_value_score,
            actual_winner, home_score, away_score,
            winner_correct, best_bet_result, best_bet_profit,
            home_injury_score, away_injury_score, injury_edge
        FROM prediction_snapshots
        WHERE game_date >= %s
        AND winner_correct IS NOT NULL
        ORDER BY game_date DESC, tip_time DESC
        LIMIT 50
    ''', (cutoff,))

    recent = []
    for row in cur.fetchall():
        (home, away, tip_time, pred_winner, winner_prob, confidence,
         bet_type, bet_team, bet_line, bet_value,
         actual_winner, home_score, away_score,
         winner_correct, bet_result, bet_profit,
         home_inj, away_inj, inj_edge) = row

        recent.append({
            'matchup': f'{away} @ {home}',
            'tip_time': tip_time.isoformat() if tip_time else None,
            'predicted_winner': pred_winner,
            'winner_prob': float(winner_prob) if winner_prob else None,
            'confidence': confidence,
            'best_bet': {
                'type': bet_type,
                'team': bet_team,
                'line': float(bet_line) if bet_line else None,
                'value_score': int(bet_value) if bet_value else None,
            } if bet_type else None,
            'actual_winner': actual_winner,
            'final_score': f'{away_score}-{home_score}' if home_score else None,
            'winner_correct': winner_correct,
            'bet_result': bet_result,
            'bet_profit': float(bet_profit) if bet_profit else None,
            'injury_edge': float(inj_edge) if inj_edge else None,
        })

    # Get pending predictions (not yet graded)
    cur.execute('''
        SELECT COUNT(*) FROM prediction_snapshots
        WHERE winner_correct IS NULL
        AND tip_time < NOW()
    ''')
    pending_grading = cur.fetchone()[0]

    cur.close()
    conn.close()

    total_bets = (bet_wins or 0) + (bet_losses or 0)

    return {
        'summary': {
            'total_predictions': total or 0,
            'winner_accuracy': {
                'wins': winner_wins or 0,
                'losses': winner_losses or 0,
                'rate': round((winner_wins or 0) / total * 100, 1) if total else None,
            },
            'best_bet_performance': {
                'wins': bet_wins or 0,
                'losses': bet_losses or 0,
                'pushes': bet_pushes or 0,
                'win_rate': round((bet_wins or 0) / total_bets * 100, 1) if total_bets > 0 else None,
                'profit': float(total_profit or 0),
                'roi': round(float(total_profit or 0) / (total_bets * 100) * 100, 1) if total_bets > 0 else None,
            },
            'pending_grading': pending_grading,
        },
        'by_value_bucket': buckets,
        'recent_predictions': recent,
    }
