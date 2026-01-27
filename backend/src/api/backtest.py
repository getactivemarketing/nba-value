"""Backtest API endpoints for model performance evaluation."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta

from src.tasks.backtest_runner import (
    BacktestConfig,
    run_backtest,
    save_backtest_result,
    get_backtest_results,
)

router = APIRouter(prefix="/backtest", tags=["Backtest"])


class BacktestRequest(BaseModel):
    """Request to run a backtest."""
    start_date: Optional[str] = None  # YYYY-MM-DD, defaults to 14 days ago
    end_date: Optional[str] = None    # YYYY-MM-DD, defaults to yesterday
    min_value_score: int = 65
    model_version: str = "spread_v2"
    save_results: bool = False


class BacktestResponse(BaseModel):
    """Response from a backtest run."""
    status: str
    message: str
    backtest_id: Optional[int] = None
    summary: Optional[dict] = None


@router.post("/run", response_model=BacktestResponse)
async def run_backtest_endpoint(request: BacktestRequest) -> BacktestResponse:
    """
    Run a backtest against historical predictions.

    This evaluates how our model's picks would have performed over a date range.
    Uses frozen prediction_snapshots to ensure accurate point-in-time analysis.
    """
    try:
        # Set default dates
        if not request.end_date:
            end_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            end_date = request.end_date

        if not request.start_date:
            start_date = (datetime.now(timezone.utc) - timedelta(days=14)).strftime('%Y-%m-%d')
        else:
            start_date = request.start_date

        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            min_value_score=request.min_value_score,
            model_version=request.model_version
        )

        result = run_backtest(config)

        # Optionally save to database
        backtest_id = None
        if request.save_results:
            backtest_id = save_backtest_result(result)

        return BacktestResponse(
            status="success",
            message=f"Backtest completed: {result.overall.total_bets} bets analyzed",
            backtest_id=backtest_id,
            summary={
                "period": f"{start_date} to {end_date}",
                "min_value_score": request.min_value_score,
                "total_bets": result.overall.total_bets,
                "record": f"{result.overall.wins}-{result.overall.losses}-{result.overall.pushes}",
                "win_rate": result.overall.win_rate,
                "profit": result.overall.profit,
                "roi": result.overall.roi,
                "max_drawdown": result.overall.max_drawdown,
                "by_bucket": [
                    {
                        "bucket": b.bucket,
                        "bets": b.bets,
                        "win_rate": b.win_rate,
                        "profit": b.profit,
                        "roi": b.roi
                    }
                    for b in result.by_bucket if b.bets > 0
                ],
                "by_bet_type": result.by_bet_type,
                "daily_results": result.daily_results
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def get_backtest_history(
    limit: int = Query(10, ge=1, le=50, description="Number of results to return")
) -> dict:
    """
    Get recent backtest results from the database.

    Returns a list of past backtest runs with their summary metrics.
    """
    try:
        results = get_backtest_results(limit=limit)
        return {
            "status": "success",
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{backtest_id}")
async def get_backtest_detail(backtest_id: int) -> dict:
    """
    Get detailed results for a specific backtest run.
    """
    import psycopg2
    import os

    DB_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway')

    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        cur.execute('''
            SELECT id, model_version, start_date, end_date, min_value_score,
                   total_bets, wins, losses, pushes, profit, roi, max_drawdown,
                   created_at, results_json
            FROM backtest_runs
            WHERE id = %s
        ''', (backtest_id,))

        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

        return {
            "status": "success",
            "backtest": {
                "id": row[0],
                "model_version": row[1],
                "start_date": str(row[2]),
                "end_date": str(row[3]),
                "min_value_score": row[4],
                "total_bets": row[5],
                "wins": row[6],
                "losses": row[7],
                "pushes": row[8],
                "profit": float(row[9]) if row[9] else 0,
                "roi": float(row[10]) if row[10] else 0,
                "max_drawdown": float(row[11]) if row[11] else 0,
                "created_at": row[12].isoformat() if row[12] else None,
                "details": row[13] if row[13] else {}
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quick")
async def quick_backtest(
    days: int = Query(14, ge=1, le=90, description="Number of days to analyze"),
    min_value: int = Query(65, ge=0, le=100, description="Minimum value score")
) -> dict:
    """
    Quick backtest endpoint for simple analysis.

    Returns performance summary for the specified time period.
    """
    try:
        end_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')

        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            min_value_score=min_value,
            model_version="spread_v2"
        )

        result = run_backtest(config)

        return {
            "status": "success",
            "period": {
                "start": start_date,
                "end": end_date,
                "days": days
            },
            "min_value_score": min_value,
            "performance": {
                "total_bets": result.overall.total_bets,
                "record": f"{result.overall.wins}-{result.overall.losses}-{result.overall.pushes}",
                "win_rate": result.overall.win_rate,
                "profit": result.overall.profit,
                "roi": result.overall.roi,
                "max_drawdown": result.overall.max_drawdown,
                "avg_value_score": result.overall.avg_value_score
            },
            "by_bucket": {
                b.bucket: {
                    "bets": b.bets,
                    "win_rate": b.win_rate,
                    "roi": b.roi
                }
                for b in result.by_bucket if b.bets > 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
