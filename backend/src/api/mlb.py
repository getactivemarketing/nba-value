"""MLB API endpoints for games, picks, and evaluation."""

from datetime import datetime, timezone, timedelta, date
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, Query, HTTPException, Path
from pydantic import BaseModel
from sqlalchemy import select, and_, desc, func

from src.database import async_session
from src.models import (
    MLBGame, MLBTeam, MLBPitcher, MLBPitcherStats,
    MLBTeamStats, MLBMarket, MLBPrediction,
    MLBPredictionSnapshot, MLBGameContext,
)

router = APIRouter(prefix="/mlb", tags=["MLB"])


# Response Models

class PitcherInfo(BaseModel):
    """Pitcher information for matchup display."""
    pitcher_id: int
    name: str
    team: str | None
    throws: str | None
    era: float | None
    whip: float | None
    k_per_9: float | None
    quality_score: float | None

    class Config:
        from_attributes = True


class GameContextInfo(BaseModel):
    """Game context (venue, weather)."""
    venue_name: str | None
    park_factor: float | None
    temperature: int | None
    wind_speed: int | None
    is_dome: bool
    weather_factor: float | None


class MarketInfo(BaseModel):
    """Market odds information."""
    market_type: str
    line: float | None
    home_odds: float | None
    away_odds: float | None
    over_odds: float | None
    under_odds: float | None
    book: str | None


class ValueBetInfo(BaseModel):
    """Value bet recommendation."""
    market_type: str
    bet_type: str
    team: str | None
    line: float | None
    odds_decimal: float
    odds_american: int
    model_prob: float
    market_prob: float
    edge: float
    value_score: float
    confidence: str


class MLBGameResponse(BaseModel):
    """Complete game information with predictions."""
    game_id: str
    game_date: str
    game_time: datetime | None
    home_team: str
    away_team: str
    status: str

    # Starters
    home_starter: PitcherInfo | None
    away_starter: PitcherInfo | None

    # Context
    context: GameContextInfo | None

    # Markets
    markets: list[MarketInfo]

    # Predictions
    predicted_run_diff: float | None
    predicted_total: float | None
    p_home_win: float | None
    p_away_win: float | None

    # Value bets
    best_ml: ValueBetInfo | None
    best_rl: ValueBetInfo | None
    best_total: ValueBetInfo | None
    best_bet: ValueBetInfo | None

    # Final scores (if completed)
    home_score: int | None
    away_score: int | None

    class Config:
        from_attributes = True


class MLBGamesResponse(BaseModel):
    """Response for games list."""
    games: list[MLBGameResponse]
    total: int
    date: str


class TopPickResponse(BaseModel):
    """Top value pick."""
    game_id: str
    game_date: str
    game_time: datetime | None
    home_team: str
    away_team: str
    home_starter: str | None
    away_starter: str | None
    bet_type: str
    team: str | None
    line: float | None
    odds_decimal: float
    odds_american: int
    value_score: float
    edge: float
    confidence: str
    predicted_run_diff: float | None


class TopPicksResponse(BaseModel):
    """Response for top picks."""
    picks: list[TopPickResponse]
    total: int
    min_value_score: float


class DailyPerformance(BaseModel):
    """Daily performance stats."""
    date: str
    predictions: int
    wins: int
    losses: int
    pushes: int
    win_rate: float | None
    profit: float


class EvaluationSummary(BaseModel):
    """Overall evaluation summary."""
    total_predictions: int
    graded_predictions: int
    wins: int
    losses: int
    pushes: int
    overall_win_rate: float | None
    total_profit: float
    by_value_tier: dict


# Endpoints

@router.get("/games", response_model=MLBGamesResponse)
async def get_games(
    game_date: str | None = Query(None, description="Date in YYYY-MM-DD format"),
) -> MLBGamesResponse:
    """
    Get MLB games for a specific date with predictions and value analysis.

    Defaults to today's games.
    """
    if game_date:
        try:
            target_date = datetime.strptime(game_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        target_date = date.today()

    async with async_session() as session:
        # Get games for the date
        stmt = select(MLBGame).where(
            MLBGame.game_date == target_date
        ).order_by(MLBGame.game_time)

        result = await session.execute(stmt)
        games = result.scalars().all()

        response_games = []
        for game in games:
            game_response = await _build_game_response(session, game)
            response_games.append(game_response)

        return MLBGamesResponse(
            games=response_games,
            total=len(response_games),
            date=target_date.isoformat(),
        )


@router.get("/games/{game_id}", response_model=MLBGameResponse)
async def get_game(
    game_id: str = Path(..., description="Game ID"),
) -> MLBGameResponse:
    """Get detailed information for a specific game."""
    async with async_session() as session:
        stmt = select(MLBGame).where(MLBGame.game_id == game_id)
        result = await session.execute(stmt)
        game = result.scalar_one_or_none()

        if not game:
            raise HTTPException(status_code=404, detail="Game not found")

        return await _build_game_response(session, game)


@router.get("/picks/top", response_model=TopPicksResponse)
async def get_top_picks(
    min_value_score: float = Query(65, ge=0, le=100, description="Minimum value score"),
    limit: int = Query(20, ge=1, le=50, description="Maximum picks to return"),
    game_date: str | None = Query(None, description="Date in YYYY-MM-DD format"),
) -> TopPicksResponse:
    """
    Get top value picks based on value score threshold.

    Returns picks with value_score >= min_value_score, sorted by value.
    """
    if game_date:
        try:
            target_date = datetime.strptime(game_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
    else:
        target_date = date.today()

    async with async_session() as session:
        # Get snapshots with high value scores for the date
        stmt = select(MLBPredictionSnapshot).where(
            and_(
                MLBPredictionSnapshot.game_date == target_date,
                MLBPredictionSnapshot.best_bet_value_score >= min_value_score,
            )
        ).order_by(desc(MLBPredictionSnapshot.best_bet_value_score)).limit(limit)

        result = await session.execute(stmt)
        snapshots = result.scalars().all()

        picks = []
        for snap in snapshots:
            if snap.best_bet_type and snap.best_bet_value_score:
                picks.append(TopPickResponse(
                    game_id=snap.game_id,
                    game_date=snap.game_date.isoformat() if snap.game_date else "",
                    game_time=snap.game_time,
                    home_team=snap.home_team,
                    away_team=snap.away_team,
                    home_starter=snap.home_starter_name,
                    away_starter=snap.away_starter_name,
                    bet_type=snap.best_bet_type,
                    team=snap.best_bet_team,
                    line=float(snap.best_bet_line) if snap.best_bet_line else None,
                    odds_decimal=float(snap.best_bet_odds) if snap.best_bet_odds else 2.0,
                    odds_american=_decimal_to_american(float(snap.best_bet_odds) if snap.best_bet_odds else 2.0),
                    value_score=float(snap.best_bet_value_score),
                    edge=float(snap.best_bet_edge) if snap.best_bet_edge else 0,
                    confidence="high" if snap.best_bet_value_score >= 70 else "medium",
                    predicted_run_diff=float(snap.predicted_run_diff) if snap.predicted_run_diff else None,
                ))

        return TopPicksResponse(
            picks=picks,
            total=len(picks),
            min_value_score=min_value_score,
        )


@router.get("/pitchers/{name}", response_model=PitcherInfo)
async def get_pitcher(
    name: str = Path(..., description="Pitcher name (partial match)"),
) -> PitcherInfo:
    """Get pitcher stats by name."""
    async with async_session() as session:
        stmt = select(MLBPitcher).where(
            MLBPitcher.player_name.ilike(f"%{name}%")
        ).limit(1)

        result = await session.execute(stmt)
        pitcher = result.scalar_one_or_none()

        if not pitcher:
            raise HTTPException(status_code=404, detail="Pitcher not found")

        # Get latest stats
        stats_stmt = select(MLBPitcherStats).where(
            MLBPitcherStats.pitcher_id == pitcher.pitcher_id
        ).order_by(desc(MLBPitcherStats.stat_date)).limit(1)

        stats_result = await session.execute(stats_stmt)
        stats = stats_result.scalar_one_or_none()

        return PitcherInfo(
            pitcher_id=pitcher.pitcher_id,
            name=pitcher.player_name,
            team=pitcher.team_abbr,
            throws=pitcher.throws,
            era=float(stats.era) if stats and stats.era else None,
            whip=float(stats.whip) if stats and stats.whip else None,
            k_per_9=float(stats.k_per_9) if stats and stats.k_per_9 else None,
            quality_score=float(stats.quality_score) if stats and stats.quality_score else None,
        )


@router.get("/evaluation/daily", response_model=list[DailyPerformance])
async def get_daily_evaluation(
    days: int = Query(7, ge=1, le=30, description="Number of days to include"),
) -> list[DailyPerformance]:
    """Get daily prediction performance."""
    async with async_session() as session:
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        # Get graded snapshots
        stmt = select(MLBPredictionSnapshot).where(
            and_(
                MLBPredictionSnapshot.game_date >= start_date,
                MLBPredictionSnapshot.game_date < end_date,
                MLBPredictionSnapshot.best_bet_result.isnot(None),
            )
        ).order_by(MLBPredictionSnapshot.game_date)

        result = await session.execute(stmt)
        snapshots = result.scalars().all()

        # Group by date
        by_date = {}
        for snap in snapshots:
            d = snap.game_date.isoformat()
            if d not in by_date:
                by_date[d] = {"predictions": 0, "wins": 0, "losses": 0, "pushes": 0, "profit": 0.0}

            by_date[d]["predictions"] += 1
            if snap.best_bet_result == "win":
                by_date[d]["wins"] += 1
            elif snap.best_bet_result == "loss":
                by_date[d]["losses"] += 1
            else:
                by_date[d]["pushes"] += 1

            if snap.best_bet_profit:
                by_date[d]["profit"] += float(snap.best_bet_profit)

        # Build response
        daily = []
        for d, stats in sorted(by_date.items()):
            decided = stats["wins"] + stats["losses"]
            win_rate = stats["wins"] / decided if decided > 0 else None

            daily.append(DailyPerformance(
                date=d,
                predictions=stats["predictions"],
                wins=stats["wins"],
                losses=stats["losses"],
                pushes=stats["pushes"],
                win_rate=round(win_rate, 3) if win_rate else None,
                profit=round(stats["profit"], 2),
            ))

        return daily


@router.get("/evaluation/summary", response_model=EvaluationSummary)
async def get_evaluation_summary() -> EvaluationSummary:
    """Get overall prediction performance summary."""
    async with async_session() as session:
        # Get all graded snapshots
        stmt = select(MLBPredictionSnapshot).where(
            MLBPredictionSnapshot.best_bet_result.isnot(None)
        )

        result = await session.execute(stmt)
        snapshots = result.scalars().all()

        total = len(snapshots)
        wins = sum(1 for s in snapshots if s.best_bet_result == "win")
        losses = sum(1 for s in snapshots if s.best_bet_result == "loss")
        pushes = sum(1 for s in snapshots if s.best_bet_result == "push")
        profit = sum(float(s.best_bet_profit or 0) for s in snapshots)

        decided = wins + losses
        win_rate = wins / decided if decided > 0 else None

        # Group by value tier
        tiers = {
            "80+": {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0},
            "70-79": {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0},
            "65-69": {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0},
            "60-64": {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0},
            "<60": {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0},
        }

        for snap in snapshots:
            vs = snap.best_bet_value_score or 0
            if vs >= 80:
                tier = "80+"
            elif vs >= 70:
                tier = "70-79"
            elif vs >= 65:
                tier = "65-69"
            elif vs >= 60:
                tier = "60-64"
            else:
                tier = "<60"

            if snap.best_bet_result == "win":
                tiers[tier]["wins"] += 1
            elif snap.best_bet_result == "loss":
                tiers[tier]["losses"] += 1
            else:
                tiers[tier]["pushes"] += 1

            tiers[tier]["profit"] += float(snap.best_bet_profit or 0)

        # Add win rates to tiers
        for tier_name, tier_stats in tiers.items():
            tier_decided = tier_stats["wins"] + tier_stats["losses"]
            tier_stats["win_rate"] = round(tier_stats["wins"] / tier_decided, 3) if tier_decided > 0 else None
            tier_stats["count"] = tier_stats["wins"] + tier_stats["losses"] + tier_stats["pushes"]

        return EvaluationSummary(
            total_predictions=total,
            graded_predictions=total,
            wins=wins,
            losses=losses,
            pushes=pushes,
            overall_win_rate=round(win_rate, 3) if win_rate else None,
            total_profit=round(profit, 2),
            by_value_tier=tiers,
        )


# Helper functions

async def _build_game_response(session, game: MLBGame) -> MLBGameResponse:
    """Build complete game response with all related data."""
    # Get starters
    home_starter = None
    away_starter = None

    if game.home_starter_id:
        pitcher_result = await session.execute(
            select(MLBPitcher).where(MLBPitcher.pitcher_id == game.home_starter_id)
        )
        pitcher = pitcher_result.scalar_one_or_none()
        if pitcher:
            home_starter = await _get_pitcher_info(session, pitcher)

    if game.away_starter_id:
        pitcher_result = await session.execute(
            select(MLBPitcher).where(MLBPitcher.pitcher_id == game.away_starter_id)
        )
        pitcher = pitcher_result.scalar_one_or_none()
        if pitcher:
            away_starter = await _get_pitcher_info(session, pitcher)

    # Get context
    context = None
    context_result = await session.execute(
        select(MLBGameContext).where(MLBGameContext.game_id == game.game_id)
    )
    ctx = context_result.scalar_one_or_none()
    if ctx:
        context = GameContextInfo(
            venue_name=ctx.venue_name,
            park_factor=float(ctx.park_factor) if ctx.park_factor else None,
            temperature=ctx.temperature,
            wind_speed=ctx.wind_speed,
            is_dome=ctx.is_dome,
            weather_factor=float(ctx.weather_factor) if ctx.weather_factor else None,
        )

    # Get markets
    markets_result = await session.execute(
        select(MLBMarket).where(MLBMarket.game_id == game.game_id)
    )
    markets = [
        MarketInfo(
            market_type=m.market_type,
            line=float(m.line) if m.line else None,
            home_odds=float(m.home_odds) if m.home_odds else None,
            away_odds=float(m.away_odds) if m.away_odds else None,
            over_odds=float(m.over_odds) if m.over_odds else None,
            under_odds=float(m.under_odds) if m.under_odds else None,
            book=m.book,
        )
        for m in markets_result.scalars().all()
    ]

    # Get prediction snapshot
    snapshot_result = await session.execute(
        select(MLBPredictionSnapshot).where(
            MLBPredictionSnapshot.game_id == game.game_id
        ).order_by(desc(MLBPredictionSnapshot.snapshot_time)).limit(1)
    )
    snapshot = snapshot_result.scalar_one_or_none()

    # Build value bet info from snapshot
    best_ml = None
    best_rl = None
    best_total = None
    best_bet = None
    predicted_run_diff = None
    predicted_total = None
    p_home_win = None
    p_away_win = None

    if snapshot:
        predicted_run_diff = float(snapshot.predicted_run_diff) if snapshot.predicted_run_diff else None
        predicted_total = float(snapshot.predicted_total) if snapshot.predicted_total else None
        p_home_win = float(snapshot.winner_probability) if snapshot.predicted_winner == game.home_team else None
        p_away_win = float(snapshot.winner_probability) if snapshot.predicted_winner == game.away_team else None

        if snapshot.best_ml_team and snapshot.best_ml_value_score:
            best_ml = ValueBetInfo(
                market_type="moneyline",
                bet_type="home_ml" if snapshot.best_ml_team == game.home_team else "away_ml",
                team=snapshot.best_ml_team,
                line=None,
                odds_decimal=float(snapshot.best_ml_odds) if snapshot.best_ml_odds else 2.0,
                odds_american=_decimal_to_american(float(snapshot.best_ml_odds) if snapshot.best_ml_odds else 2.0),
                model_prob=0.5,  # Simplified
                market_prob=0.5,
                edge=float(snapshot.best_ml_edge) if snapshot.best_ml_edge else 0,
                value_score=float(snapshot.best_ml_value_score),
                confidence="high" if snapshot.best_ml_value_score >= 70 else "medium",
            )

        if snapshot.best_rl_team and snapshot.best_rl_value_score:
            best_rl = ValueBetInfo(
                market_type="runline",
                bet_type="home_rl" if snapshot.best_rl_team == game.home_team else "away_rl",
                team=snapshot.best_rl_team,
                line=float(snapshot.best_rl_line) if snapshot.best_rl_line else None,
                odds_decimal=float(snapshot.best_rl_odds) if snapshot.best_rl_odds else 2.0,
                odds_american=_decimal_to_american(float(snapshot.best_rl_odds) if snapshot.best_rl_odds else 2.0),
                model_prob=0.5,
                market_prob=0.5,
                edge=float(snapshot.best_rl_edge) if snapshot.best_rl_edge else 0,
                value_score=float(snapshot.best_rl_value_score),
                confidence="high" if snapshot.best_rl_value_score >= 70 else "medium",
            )

        if snapshot.best_total_direction and snapshot.best_total_value_score:
            best_total = ValueBetInfo(
                market_type="total",
                bet_type=snapshot.best_total_direction,
                team=None,
                line=float(snapshot.best_total_line) if snapshot.best_total_line else None,
                odds_decimal=float(snapshot.best_total_odds) if snapshot.best_total_odds else 2.0,
                odds_american=_decimal_to_american(float(snapshot.best_total_odds) if snapshot.best_total_odds else 2.0),
                model_prob=0.5,
                market_prob=0.5,
                edge=float(snapshot.best_total_edge) if snapshot.best_total_edge else 0,
                value_score=float(snapshot.best_total_value_score),
                confidence="high" if snapshot.best_total_value_score >= 70 else "medium",
            )

        if snapshot.best_bet_type and snapshot.best_bet_value_score:
            best_bet = ValueBetInfo(
                market_type=snapshot.best_bet_type,
                bet_type=snapshot.best_bet_type,
                team=snapshot.best_bet_team,
                line=float(snapshot.best_bet_line) if snapshot.best_bet_line else None,
                odds_decimal=float(snapshot.best_bet_odds) if snapshot.best_bet_odds else 2.0,
                odds_american=_decimal_to_american(float(snapshot.best_bet_odds) if snapshot.best_bet_odds else 2.0),
                model_prob=0.5,
                market_prob=0.5,
                edge=float(snapshot.best_bet_edge) if snapshot.best_bet_edge else 0,
                value_score=float(snapshot.best_bet_value_score),
                confidence="high" if snapshot.best_bet_value_score >= 70 else "medium",
            )

    return MLBGameResponse(
        game_id=game.game_id,
        game_date=game.game_date.isoformat(),
        game_time=game.game_time,
        home_team=game.home_team,
        away_team=game.away_team,
        status=game.status,
        home_starter=home_starter,
        away_starter=away_starter,
        context=context,
        markets=markets,
        predicted_run_diff=predicted_run_diff,
        predicted_total=predicted_total,
        p_home_win=p_home_win,
        p_away_win=p_away_win,
        best_ml=best_ml,
        best_rl=best_rl,
        best_total=best_total,
        best_bet=best_bet,
        home_score=game.home_score,
        away_score=game.away_score,
    )


async def _get_pitcher_info(session, pitcher: MLBPitcher) -> PitcherInfo:
    """Build pitcher info with latest stats."""
    stats_result = await session.execute(
        select(MLBPitcherStats).where(
            MLBPitcherStats.pitcher_id == pitcher.pitcher_id
        ).order_by(desc(MLBPitcherStats.stat_date)).limit(1)
    )
    stats = stats_result.scalar_one_or_none()

    return PitcherInfo(
        pitcher_id=pitcher.pitcher_id,
        name=pitcher.player_name,
        team=pitcher.team_abbr,
        throws=pitcher.throws,
        era=float(stats.era) if stats and stats.era else None,
        whip=float(stats.whip) if stats and stats.whip else None,
        k_per_9=float(stats.k_per_9) if stats and stats.k_per_9 else None,
        quality_score=float(stats.quality_score) if stats and stats.quality_score else None,
    )


def _decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American format."""
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))
