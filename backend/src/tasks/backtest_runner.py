"""
Backtest Runner for Model Performance Evaluation

Evaluates model predictions against actual outcomes using frozen prediction_snapshots.
Stores results in backtest_runs table for historical tracking.

Usage:
    python -m src.tasks.backtest_runner --start 2026-01-01 --end 2026-01-26 --min-value 65
"""

import argparse
import json
import psycopg2
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

import structlog

logger = structlog.get_logger()

# Use environment variable or fallback
import os
DB_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway')


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    start_date: str
    end_date: str
    min_value_score: int = 65
    model_version: str = "spread_v2"
    bet_size: float = 100.0  # Flat bet size in dollars


@dataclass
class BacktestMetrics:
    """Metrics from a backtest run."""
    total_bets: int
    wins: int
    losses: int
    pushes: int
    profit: float
    roi: float
    win_rate: float
    max_drawdown: float
    max_profit: float
    avg_value_score: float


@dataclass
class BucketMetrics:
    """Metrics for a specific value score bucket."""
    bucket: str
    bets: int
    wins: int
    losses: int
    win_rate: float
    profit: float
    roi: float


@dataclass
class BacktestResult:
    """Complete result of a backtest run."""
    config: BacktestConfig
    overall: BacktestMetrics
    by_bucket: list[BucketMetrics]
    by_bet_type: dict
    daily_results: list[dict]
    created_at: str


def run_backtest(config: BacktestConfig) -> BacktestResult:
    """
    Run a backtest using prediction_snapshots data.

    This evaluates our model's performance by looking at:
    1. Predictions we made (captured in prediction_snapshots)
    2. Actual outcomes (from game_results)
    3. Calculate what our P/L would have been

    Args:
        config: BacktestConfig with date range and parameters

    Returns:
        BacktestResult with comprehensive metrics
    """
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Query all graded predictions in the date range
    cur.execute('''
        SELECT
            ps.game_id,
            ps.home_team,
            ps.away_team,
            DATE(ps.snapshot_time) as prediction_date,
            ps.best_bet_type,
            ps.best_bet_team,
            ps.best_bet_line,
            ps.best_bet_value_score,
            ps.best_bet_result,
            ps.best_bet_profit,
            gr.home_score,
            gr.away_score,
            gr.closing_spread,
            gr.spread_result,
            gr.total_result
        FROM prediction_snapshots ps
        JOIN game_results gr ON ps.game_id = gr.game_id
        WHERE DATE(ps.snapshot_time) >= %s
        AND DATE(ps.snapshot_time) <= %s
        AND ps.best_bet_value_score >= %s
        AND ps.best_bet_result IS NOT NULL
        ORDER BY ps.snapshot_time
    ''', (config.start_date, config.end_date, config.min_value_score))

    predictions = cur.fetchall()
    cur.close()
    conn.close()

    if not predictions:
        return BacktestResult(
            config=config,
            overall=BacktestMetrics(
                total_bets=0, wins=0, losses=0, pushes=0,
                profit=0.0, roi=0.0, win_rate=0.0,
                max_drawdown=0.0, max_profit=0.0, avg_value_score=0.0
            ),
            by_bucket=[],
            by_bet_type={},
            daily_results=[],
            created_at=datetime.now(timezone.utc).isoformat()
        )

    # Process predictions
    total_bets = 0
    wins = 0
    losses = 0
    pushes = 0
    profit = 0.0
    value_scores = []

    # Track max drawdown
    running_profit = 0.0
    max_profit = 0.0
    max_drawdown = 0.0

    # Track by bucket
    buckets = {
        '90+': {'wins': 0, 'losses': 0, 'pushes': 0, 'profit': 0.0},
        '80-89': {'wins': 0, 'losses': 0, 'pushes': 0, 'profit': 0.0},
        '70-79': {'wins': 0, 'losses': 0, 'pushes': 0, 'profit': 0.0},
        '60-69': {'wins': 0, 'losses': 0, 'pushes': 0, 'profit': 0.0},
        '50-59': {'wins': 0, 'losses': 0, 'pushes': 0, 'profit': 0.0},
        '<50': {'wins': 0, 'losses': 0, 'pushes': 0, 'profit': 0.0},
    }

    # Track by bet type
    bet_types = {
        'spread': {'wins': 0, 'losses': 0, 'pushes': 0, 'profit': 0.0},
        'total': {'wins': 0, 'losses': 0, 'pushes': 0, 'profit': 0.0},
        'moneyline': {'wins': 0, 'losses': 0, 'pushes': 0, 'profit': 0.0},
    }

    # Track daily results
    daily = {}

    for row in predictions:
        (game_id, home, away, pred_date, bet_type, bet_team, bet_line,
         value_score, result, bet_profit, home_score, away_score,
         closing_spread, spread_result, total_result) = row

        total_bets += 1
        value_scores.append(value_score or 0)
        bet_profit = float(bet_profit or 0)

        # Track result
        if result == 'win':
            wins += 1
            profit += bet_profit if bet_profit > 0 else 90.91
        elif result == 'loss':
            losses += 1
            profit += bet_profit if bet_profit < 0 else -100.0
        else:  # push
            pushes += 1

        # Track running P/L for drawdown
        running_profit = profit
        if running_profit > max_profit:
            max_profit = running_profit
        drawdown = max_profit - running_profit
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        # Bucket assignment
        vs = value_score or 0
        if vs >= 90:
            bucket = '90+'
        elif vs >= 80:
            bucket = '80-89'
        elif vs >= 70:
            bucket = '70-79'
        elif vs >= 60:
            bucket = '60-69'
        elif vs >= 50:
            bucket = '50-59'
        else:
            bucket = '<50'

        # Track by bucket
        if result == 'win':
            buckets[bucket]['wins'] += 1
            buckets[bucket]['profit'] += bet_profit if bet_profit > 0 else 90.91
        elif result == 'loss':
            buckets[bucket]['losses'] += 1
            buckets[bucket]['profit'] += bet_profit if bet_profit < 0 else -100.0
        else:
            buckets[bucket]['pushes'] += 1

        # Track by bet type
        bt = bet_type or 'spread'
        if result == 'win':
            bet_types[bt]['wins'] += 1
            bet_types[bt]['profit'] += bet_profit if bet_profit > 0 else 90.91
        elif result == 'loss':
            bet_types[bt]['losses'] += 1
            bet_types[bt]['profit'] += bet_profit if bet_profit < 0 else -100.0
        else:
            bet_types[bt]['pushes'] += 1

        # Track daily
        date_str = str(pred_date)
        if date_str not in daily:
            daily[date_str] = {'wins': 0, 'losses': 0, 'pushes': 0, 'profit': 0.0}
        if result == 'win':
            daily[date_str]['wins'] += 1
            daily[date_str]['profit'] += bet_profit if bet_profit > 0 else 90.91
        elif result == 'loss':
            daily[date_str]['losses'] += 1
            daily[date_str]['profit'] += bet_profit if bet_profit < 0 else -100.0
        else:
            daily[date_str]['pushes'] += 1

    # Calculate metrics
    total_wagered = (wins + losses) * config.bet_size
    win_rate = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0.0
    roi = round(profit / total_wagered * 100, 1) if total_wagered > 0 else 0.0
    avg_value = round(sum(value_scores) / len(value_scores), 1) if value_scores else 0.0

    # Build bucket metrics
    bucket_metrics = []
    for bucket_name, stats in buckets.items():
        total = stats['wins'] + stats['losses']
        bucket_metrics.append(BucketMetrics(
            bucket=bucket_name,
            bets=total + stats['pushes'],
            wins=stats['wins'],
            losses=stats['losses'],
            win_rate=round(stats['wins'] / total * 100, 1) if total > 0 else 0.0,
            profit=round(stats['profit'], 2),
            roi=round(stats['profit'] / (total * config.bet_size) * 100, 1) if total > 0 else 0.0
        ))

    # Build bet type stats
    bet_type_stats = {}
    for bt, stats in bet_types.items():
        total = stats['wins'] + stats['losses']
        bet_type_stats[bt] = {
            'bets': total + stats['pushes'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'win_rate': round(stats['wins'] / total * 100, 1) if total > 0 else 0.0,
            'profit': round(stats['profit'], 2),
            'roi': round(stats['profit'] / (total * config.bet_size) * 100, 1) if total > 0 else 0.0
        }

    # Build daily results list
    daily_results = []
    for date_str, stats in sorted(daily.items()):
        total = stats['wins'] + stats['losses']
        daily_results.append({
            'date': date_str,
            'bets': total + stats['pushes'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'profit': round(stats['profit'], 2),
        })

    overall = BacktestMetrics(
        total_bets=total_bets,
        wins=wins,
        losses=losses,
        pushes=pushes,
        profit=round(profit, 2),
        roi=roi,
        win_rate=win_rate,
        max_drawdown=round(max_drawdown, 2),
        max_profit=round(max_profit, 2),
        avg_value_score=avg_value
    )

    return BacktestResult(
        config=config,
        overall=overall,
        by_bucket=bucket_metrics,
        by_bet_type=bet_type_stats,
        daily_results=daily_results,
        created_at=datetime.now(timezone.utc).isoformat()
    )


def save_backtest_result(result: BacktestResult) -> int:
    """
    Save backtest result to database.

    Returns:
        ID of the created backtest_runs record
    """
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Ensure table exists
    cur.execute('''
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id SERIAL PRIMARY KEY,
            model_version VARCHAR(50),
            start_date DATE,
            end_date DATE,
            min_value_score INT,
            total_bets INT,
            wins INT,
            losses INT,
            pushes INT,
            profit DECIMAL(10,2),
            roi DECIMAL(5,2),
            max_drawdown DECIMAL(10,2),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            results_json JSONB
        )
    ''')

    # Insert result
    results_json = json.dumps({
        'overall': asdict(result.overall),
        'by_bucket': [asdict(b) for b in result.by_bucket],
        'by_bet_type': result.by_bet_type,
        'daily_results': result.daily_results,
    })

    cur.execute('''
        INSERT INTO backtest_runs (
            model_version, start_date, end_date, min_value_score,
            total_bets, wins, losses, pushes, profit, roi, max_drawdown,
            results_json
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    ''', (
        result.config.model_version,
        result.config.start_date,
        result.config.end_date,
        result.config.min_value_score,
        result.overall.total_bets,
        result.overall.wins,
        result.overall.losses,
        result.overall.pushes,
        result.overall.profit,
        result.overall.roi,
        result.overall.max_drawdown,
        results_json
    ))

    backtest_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return backtest_id


def get_backtest_results(limit: int = 10) -> list[dict]:
    """Get recent backtest results from database."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cur.execute('''
        SELECT id, model_version, start_date, end_date, min_value_score,
               total_bets, wins, losses, pushes, profit, roi, max_drawdown,
               created_at, results_json
        FROM backtest_runs
        ORDER BY created_at DESC
        LIMIT %s
    ''', (limit,))

    results = []
    for row in cur.fetchall():
        results.append({
            'id': row[0],
            'model_version': row[1],
            'start_date': str(row[2]),
            'end_date': str(row[3]),
            'min_value_score': row[4],
            'total_bets': row[5],
            'wins': row[6],
            'losses': row[7],
            'pushes': row[8],
            'profit': float(row[9]) if row[9] else 0,
            'roi': float(row[10]) if row[10] else 0,
            'max_drawdown': float(row[11]) if row[11] else 0,
            'created_at': row[12].isoformat() if row[12] else None,
            'details': row[13] if row[13] else {}
        })

    cur.close()
    conn.close()

    return results


def print_backtest_report(result: BacktestResult):
    """Print a formatted backtest report to console."""
    c = result.config
    o = result.overall

    print(f"\n{'='*60}")
    print(f"BACKTEST REPORT: {c.model_version}")
    print(f"{'='*60}")
    print(f"Period: {c.start_date} to {c.end_date}")
    print(f"Min Value Score: {c.min_value_score}")
    print(f"Bet Size: ${c.bet_size}")

    print(f"\n--- Overall Performance ---")
    print(f"Total Bets: {o.total_bets}")
    print(f"Record: {o.wins}-{o.losses}-{o.pushes}")
    print(f"Win Rate: {o.win_rate}%")
    print(f"Profit: ${o.profit:+.2f}")
    print(f"ROI: {o.roi:+.1f}%")
    print(f"Max Drawdown: ${o.max_drawdown:.2f}")
    print(f"Avg Value Score: {o.avg_value_score}")

    print(f"\n--- By Value Score Bucket ---")
    for b in result.by_bucket:
        if b.bets > 0:
            print(f"  {b.bucket}: {b.wins}-{b.losses} ({b.win_rate}%) | ${b.profit:+.2f} | ROI: {b.roi:+.1f}%")

    print(f"\n--- By Bet Type ---")
    for bt, stats in result.by_bet_type.items():
        if stats['bets'] > 0:
            print(f"  {bt.title()}: {stats['wins']}-{stats['losses']} ({stats['win_rate']}%) | ${stats['profit']:+.2f}")

    print(f"\n--- Daily Results (last 7 days) ---")
    for d in result.daily_results[-7:]:
        print(f"  {d['date']}: {d['wins']}-{d['losses']} | ${d['profit']:+.2f}")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model backtest')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--min-value', type=int, default=65, help='Minimum value score')
    parser.add_argument('--model', type=str, default='spread_v2', help='Model version name')
    parser.add_argument('--save', action='store_true', help='Save results to database')

    args = parser.parse_args()

    # Default date range: last 14 days
    if not args.end:
        args.end = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
    if not args.start:
        args.start = (datetime.now(timezone.utc) - timedelta(days=14)).strftime('%Y-%m-%d')

    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        min_value_score=args.min_value,
        model_version=args.model
    )

    print(f"Running backtest from {args.start} to {args.end} (min value: {args.min_value})...")
    result = run_backtest(config)

    print_backtest_report(result)

    if args.save:
        backtest_id = save_backtest_result(result)
        print(f"Results saved to database with ID: {backtest_id}")
