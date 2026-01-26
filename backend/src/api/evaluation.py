"""Evaluation and analytics API endpoints."""

from datetime import date, timedelta
from typing import Literal

import psycopg2
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.schemas.evaluation import (
    AlgorithmComparisonResponse,
    CalibrationResponse,
    PerformanceByBucketResponse,
)

router = APIRouter()

DB_URL = 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway'


def _calculate_bet_result(market_type: str, outcome_label: str, line: float,
                          home_score: int, away_score: int) -> tuple[bool, bool, float]:
    """Calculate if bet won, pushed, and profit. Returns (won, pushed, profit)."""
    home_margin = home_score - away_score
    total_pts = home_score + away_score
    is_home = 'home' in outcome_label.lower()

    if market_type == 'spread':
        if is_home:
            margin = home_margin + (line or 0)
        else:
            margin = -home_margin + (line or 0)
        return (margin > 0, margin == 0, margin)

    elif market_type == 'moneyline':
        if is_home:
            return (home_margin > 0, False, home_margin)
        else:
            return (home_margin < 0, False, -home_margin)

    elif market_type == 'total':
        is_over = 'over' in outcome_label.lower()
        if is_over:
            return (total_pts > (line or 0), total_pts == (line or 0), total_pts - (line or 0))
        else:
            return (total_pts < (line or 0), total_pts == (line or 0), (line or 0) - total_pts)

    return (False, False, 0)


def _backtest_algorithm(days: int, min_value: float, algorithm: Literal['a', 'b']) -> dict:
    """Run backtest for an algorithm and return metrics."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cutoff = date.today() - timedelta(days=days)
    value_col = 'algo_a_value_score' if algorithm == 'a' else 'algo_b_value_score'

    # Query value scores with game results - JOIN in one query to avoid N+1
    cur.execute(f'''
        WITH ranked_bets AS (
            SELECT
                vs.{value_col} as value_score, mp.p_true, mp.raw_edge,
                m.market_type, m.outcome_label, m.line, m.odds_decimal,
                g.home_team_id, g.away_team_id, g.game_date,
                ROW_NUMBER() OVER (
                    PARTITION BY g.game_id, m.market_type,
                        CASE
                            WHEN m.market_type = 'total' THEN
                                CASE WHEN m.outcome_label ILIKE '%%over%%' THEN 'over' ELSE 'under' END
                            ELSE
                                CASE WHEN m.outcome_label ILIKE '%%home%%' THEN 'home' ELSE 'away' END
                        END
                    ORDER BY vs.{value_col} DESC, vs.calc_time DESC
                ) as rn
            FROM value_scores vs
            JOIN model_predictions mp ON mp.prediction_id = vs.prediction_id
            JOIN markets m ON m.market_id = vs.market_id
            JOIN games g ON g.game_id = m.game_id
            WHERE g.game_date >= %s
            AND vs.{value_col} >= %s
        )
        SELECT rb.value_score, rb.p_true, rb.raw_edge, rb.market_type, rb.outcome_label,
               rb.line, rb.odds_decimal, gr.home_score, gr.away_score
        FROM ranked_bets rb
        JOIN game_results gr ON gr.home_team_id = rb.home_team_id
                            AND gr.away_team_id = rb.away_team_id
                            AND gr.game_date = rb.game_date
        WHERE rb.rn = 1
    ''', (cutoff, min_value))

    scores = cur.fetchall()

    wins = 0
    losses = 0
    pushes = 0
    profit = 0
    brier_sum = 0
    brier_count = 0

    for row in scores:
        (value_score, p_true, raw_edge, market_type, outcome_label,
         line, odds, home_score, away_score) = row

        won, pushed, _ = _calculate_bet_result(market_type, outcome_label,
                                               float(line) if line else None,
                                               home_score, away_score)

        if pushed:
            pushes += 1
        elif won:
            wins += 1
            profit += 100 * (float(odds) - 1)
        else:
            losses += 1
            profit -= 100

        # Brier score: (p_true - actual)^2
        actual = 1.0 if won else 0.0
        brier_sum += (float(p_true) - actual) ** 2
        brier_count += 1

    cur.close()
    conn.close()

    total_bets = wins + losses

    return {
        'brier_score': brier_sum / brier_count if brier_count > 0 else None,
        'log_loss': None,  # TODO
        'clv_avg': None,  # TODO
        'roi': (profit / (total_bets * 100)) if total_bets > 0 else None,
        'win_rate': (wins / total_bets) if total_bets > 0 else None,
        'bet_count': total_bets,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'profit': profit,
    }


@router.get("/evaluation/compare", response_model=AlgorithmComparisonResponse)
async def compare_algorithms(
    start_date: date | None = None,
    end_date: date | None = None,
    market_type: str | None = None,
    days: int = Query(14, ge=1, le=90),
    min_value: float = Query(50, ge=0, le=100),
    db: AsyncSession = Depends(get_db),
) -> AlgorithmComparisonResponse:
    """
    Compare Algorithm A vs Algorithm B performance.

    Returns metrics including:
    - Brier score
    - ROI
    - Win rate
    - Bet count and P/L
    """
    algo_a = _backtest_algorithm(days, min_value, 'a')
    algo_b = _backtest_algorithm(days, min_value, 'b')

    # Determine recommendation
    if algo_a['bet_count'] < 10 or algo_b['bet_count'] < 10:
        recommendation = 'insufficient_data'
    elif algo_a['roi'] and algo_b['roi']:
        if algo_a['roi'] > algo_b['roi'] + 0.05:
            recommendation = 'algo_a'
        elif algo_b['roi'] > algo_a['roi'] + 0.05:
            recommendation = 'algo_b'
        else:
            recommendation = 'no_difference'
    else:
        recommendation = 'insufficient_data'

    return AlgorithmComparisonResponse(
        period_start=start_date or (date.today() - timedelta(days=days)),
        period_end=end_date or date.today(),
        algo_a_metrics={
            "brier_score": algo_a['brier_score'],
            "log_loss": algo_a['log_loss'],
            "clv_avg": algo_a['clv_avg'],
            "roi": algo_a['roi'],
            "win_rate": algo_a['win_rate'],
            "bet_count": algo_a['bet_count'],
        },
        algo_b_metrics={
            "brier_score": algo_b['brier_score'],
            "log_loss": algo_b['log_loss'],
            "clv_avg": algo_b['clv_avg'],
            "roi": algo_b['roi'],
            "win_rate": algo_b['win_rate'],
            "bet_count": algo_b['bet_count'],
        },
        recommendation=recommendation,
    )


@router.get("/evaluation/calibration", response_model=list[CalibrationResponse])
async def get_calibration_curves(
    market_type: str | None = None,
    algorithm: str = "a",
    db: AsyncSession = Depends(get_db),
) -> list[CalibrationResponse]:
    """
    Get calibration curve data for model evaluation.

    Returns predicted vs actual probabilities binned by prediction confidence.
    """
    # TODO: Implement
    return []


def _performance_by_bucket(days: int, algorithm: Literal['a', 'b']) -> list[dict]:
    """Calculate performance by value score bucket."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cutoff = date.today() - timedelta(days=days)
    value_col = 'algo_a_value_score' if algorithm == 'a' else 'algo_b_value_score'

    # Get all scored bets with results - JOIN to avoid N+1 queries
    cur.execute(f'''
        WITH ranked_bets AS (
            SELECT
                vs.{value_col} as value_score, mp.p_true,
                m.market_type, m.outcome_label, m.line, m.odds_decimal,
                g.home_team_id, g.away_team_id, g.game_date,
                ROW_NUMBER() OVER (
                    PARTITION BY g.game_id, m.market_type,
                        CASE
                            WHEN m.market_type = 'total' THEN
                                CASE WHEN m.outcome_label ILIKE '%%over%%' THEN 'over' ELSE 'under' END
                            ELSE
                                CASE WHEN m.outcome_label ILIKE '%%home%%' THEN 'home' ELSE 'away' END
                        END
                    ORDER BY vs.{value_col} DESC, vs.calc_time DESC
                ) as rn
            FROM value_scores vs
            JOIN model_predictions mp ON mp.prediction_id = vs.prediction_id
            JOIN markets m ON m.market_id = vs.market_id
            JOIN games g ON g.game_id = m.game_id
            WHERE g.game_date >= %s
        )
        SELECT rb.value_score, rb.p_true, rb.market_type, rb.outcome_label,
               rb.line, rb.odds_decimal, gr.home_score, gr.away_score
        FROM ranked_bets rb
        JOIN game_results gr ON gr.home_team_id = rb.home_team_id
                            AND gr.away_team_id = rb.away_team_id
                            AND gr.game_date = rb.game_date
        WHERE rb.rn = 1
    ''', (cutoff,))

    scores = cur.fetchall()

    # Buckets
    buckets = {
        '0-50': {'wins': 0, 'losses': 0, 'profit': 0},
        '50-60': {'wins': 0, 'losses': 0, 'profit': 0},
        '60-70': {'wins': 0, 'losses': 0, 'profit': 0},
        '70-80': {'wins': 0, 'losses': 0, 'profit': 0},
        '80-90': {'wins': 0, 'losses': 0, 'profit': 0},
        '90-100': {'wins': 0, 'losses': 0, 'profit': 0},
    }

    for row in scores:
        (value_score, p_true, market_type, outcome_label,
         line, odds, home_score, away_score) = row

        value = float(value_score)

        # Determine bucket
        if value < 50:
            bucket = '0-50'
        elif value < 60:
            bucket = '50-60'
        elif value < 70:
            bucket = '60-70'
        elif value < 80:
            bucket = '70-80'
        elif value < 90:
            bucket = '80-90'
        else:
            bucket = '90-100'

        won, pushed, _ = _calculate_bet_result(market_type, outcome_label,
                                               float(line) if line else None,
                                               home_score, away_score)

        if pushed:
            continue

        if won:
            buckets[bucket]['wins'] += 1
            buckets[bucket]['profit'] += 100 * (float(odds) - 1)
        else:
            buckets[bucket]['losses'] += 1
            buckets[bucket]['profit'] -= 100

    cur.close()
    conn.close()

    # Format results
    results = []
    for bucket_name, data in buckets.items():
        total = data['wins'] + data['losses']
        results.append({
            'bucket': bucket_name,
            'bet_count': total,
            'wins': data['wins'],
            'losses': data['losses'],
            'win_rate': data['wins'] / total if total > 0 else None,
            'roi': data['profit'] / (total * 100) if total > 0 else None,
            'profit': data['profit'],
            'clv_avg': None,
        })

    return results


@router.get("/evaluation/performance", response_model=list[PerformanceByBucketResponse])
async def get_performance_by_bucket(
    algorithm: str = "a",
    bucket_type: str = Query("score", pattern="^(score|edge|confidence)$"),
    days: int = Query(14, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
) -> list[PerformanceByBucketResponse]:
    """
    Get performance metrics grouped by Value Score buckets.

    Shows win rate, ROI, and CLV for different score ranges.
    """
    results = _performance_by_bucket(days, algorithm)  # type: ignore

    return [
        PerformanceByBucketResponse(**r)
        for r in results if r['bet_count'] > 0
    ]


@router.get("/evaluation/daily")
async def get_daily_results(
    days: int = Query(7, ge=1, le=30),
    algorithm: Literal['a', 'b'] = Query('b'),
    min_value: float = Query(50, ge=0, le=100),
) -> list[dict]:
    """
    Get daily performance breakdown for recent days.

    Shows each day's bets, wins, losses, and P/L.
    Uses prediction_snapshots for accurate line data (markets table may have stale data).
    """
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cutoff = date.today() - timedelta(days=days)
    value_col = 'algo_a_value_score' if algorithm == 'a' else 'algo_b_value_score'

    # Query prediction_snapshots for accurate graded data
    cur.execute(f'''
        SELECT
            ps.game_date,
            ps.home_team,
            ps.away_team,
            ps.best_bet_type,
            ps.best_bet_team,
            ps.best_bet_line,
            ps.best_bet_odds,
            ps.{value_col} as value_score,
            ps.best_bet_result,
            ps.best_bet_profit,
            ps.home_score,
            ps.away_score
        FROM prediction_snapshots ps
        WHERE ps.game_date >= %s
        AND ps.winner_correct IS NOT NULL
        AND ps.{value_col} >= %s
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


@router.get("/trends")
async def get_trends(
    trend_type: str = Query("team", pattern="^(team|market|time|situational)$"),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """
    Get trend analysis for edge patterns.

    Identifies situations where the model has historically found edge.
    """
    # TODO: Implement trend analysis
    return []


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
