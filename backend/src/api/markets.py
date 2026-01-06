"""Markets API for the Market Board."""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload

from src.database import async_session
from src.models import Game, Market, ValueScore, ModelPrediction, Team, TeamStats

router = APIRouter()


class MarketResponse(BaseModel):
    """Market with Value Score for the Market Board."""

    market_id: str
    game_id: str

    # Game info
    home_team: str
    away_team: str
    tip_time: datetime
    time_to_tip_minutes: int

    # Market info
    market_type: str
    outcome_label: str
    line: float | None
    odds_decimal: float
    odds_american: int
    book: str | None

    # Probabilities
    p_true: float
    p_market: float
    raw_edge: float
    edge_band: str | None

    # Algorithm A
    algo_a_value_score: float
    algo_a_confidence: float
    algo_a_edge_score: float

    # Algorithm B
    algo_b_value_score: float
    algo_b_confidence: float
    algo_b_combined_edge: float

    # Market quality
    market_quality: float

    # Scoring metadata
    calc_time: datetime
    active_algorithm: str

    class Config:
        from_attributes = True


class MarketBoardResponse(BaseModel):
    """Response for Market Board endpoint."""

    markets: list[MarketResponse]
    total: int
    algorithm: str
    filters_applied: dict


@router.get("/markets", response_model=MarketBoardResponse)
async def get_markets(
    algorithm: Literal["a", "b"] = Query("a", description="Which algorithm's score to sort by"),
    market_type: str | None = Query(None, description="Filter by market type (spread, moneyline, total)"),
    min_value_score: float = Query(0, ge=0, le=100, description="Minimum Value Score"),
    min_confidence: float = Query(0, ge=0, le=2, description="Minimum confidence"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> MarketBoardResponse:
    """
    Get the Market Board - scored betting opportunities ranked by Value Score.

    Returns markets with their Value Scores from both algorithms, sorted by
    the selected algorithm's score.
    """
    now = datetime.now(timezone.utc)

    async with async_session() as session:
        # Build query for markets with value scores
        # Get the most recent value score for each market
        subquery = (
            select(
                ValueScore.market_id,
                ValueScore.value_id,
            )
            .distinct(ValueScore.market_id)
            .order_by(ValueScore.market_id, desc(ValueScore.calc_time))
            .subquery()
        )

        query = (
            select(ValueScore)
            .join(subquery, ValueScore.value_id == subquery.c.value_id)
            .join(ValueScore.market)
            .join(Market.game)
            .join(ValueScore.prediction)
            .where(Game.tip_time_utc > now)  # Only upcoming games
            .where(Game.status == "scheduled")
            .options(
                selectinload(ValueScore.market).selectinload(Market.game),
                selectinload(ValueScore.prediction),
            )
        )

        # Apply filters
        if market_type:
            query = query.where(Market.market_type == market_type)

        if algorithm == "a":
            query = query.where(ValueScore.algo_a_value_score >= Decimal(str(min_value_score)))
            if min_confidence > 0:
                query = query.where(ValueScore.algo_a_confidence >= Decimal(str(min_confidence)))
            query = query.order_by(desc(ValueScore.algo_a_value_score))
        else:
            query = query.where(ValueScore.algo_b_value_score >= Decimal(str(min_value_score)))
            if min_confidence > 0:
                query = query.where(ValueScore.algo_b_confidence >= Decimal(str(min_confidence)))
            query = query.order_by(desc(ValueScore.algo_b_value_score))

        # Apply pagination
        query = query.offset(offset).limit(limit)

        result = await session.execute(query)
        value_scores = result.scalars().all()

        # Get team names for response
        team_ids = set()
        for vs in value_scores:
            team_ids.add(vs.market.game.home_team_id)
            team_ids.add(vs.market.game.away_team_id)

        teams = {}
        if team_ids:
            teams_query = select(Team).where(Team.team_id.in_(team_ids))
            teams_result = await session.execute(teams_query)
            teams = {t.team_id: t for t in teams_result.scalars().all()}

        # Build response
        markets = []
        for vs in value_scores:
            market = vs.market
            game = market.game
            prediction = vs.prediction

            home_team = teams.get(game.home_team_id)
            away_team = teams.get(game.away_team_id)

            time_to_tip = int((game.tip_time_utc - now).total_seconds() / 60)

            markets.append(MarketResponse(
                market_id=market.market_id,
                game_id=game.game_id,
                home_team=home_team.abbreviation if home_team else game.home_team_id,
                away_team=away_team.abbreviation if away_team else game.away_team_id,
                tip_time=game.tip_time_utc,
                time_to_tip_minutes=time_to_tip,
                market_type=market.market_type,
                outcome_label=market.outcome_label,
                line=float(market.line) if market.line else None,
                odds_decimal=float(market.odds_decimal),
                odds_american=market.odds_american,
                book=market.book,
                p_true=float(prediction.p_true),
                p_market=float(prediction.p_market),
                raw_edge=float(prediction.raw_edge),
                edge_band=prediction.edge_band,
                algo_a_value_score=float(vs.algo_a_value_score or 0),
                algo_a_confidence=float(vs.algo_a_confidence or 0),
                algo_a_edge_score=float(vs.algo_a_edge_score or 0),
                algo_b_value_score=float(vs.algo_b_value_score or 0),
                algo_b_confidence=float(vs.algo_b_confidence or 0),
                algo_b_combined_edge=float(vs.algo_b_combined_edge or 0),
                market_quality=float(vs.algo_a_market_quality or 0),
                calc_time=vs.calc_time,
                active_algorithm=vs.active_algorithm,
            ))

    return MarketBoardResponse(
        markets=markets,
        total=len(markets),
        algorithm=algorithm,
        filters_applied={
            "market_type": market_type,
            "min_value_score": min_value_score,
            "min_confidence": min_confidence,
        },
    )


@router.get("/markets/live", response_model=MarketBoardResponse)
async def get_live_markets(
    algorithm: Literal["a", "b"] = Query("a", description="Which algorithm's score to sort by"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
) -> MarketBoardResponse:
    """
    Get markets for games starting within the next 24 hours.

    Optimized endpoint for the main Market Board view.
    """
    return await get_markets(
        algorithm=algorithm,
        market_type=None,
        min_value_score=0,
        min_confidence=0,
        limit=limit,
        offset=0,
    )


class MarketDetailResponse(BaseModel):
    """Detailed view of a single market with full scoring breakdown."""

    market_id: str
    game_id: str

    # Game info
    home_team: str
    away_team: str
    tip_time: datetime
    time_to_tip_minutes: int

    # Market info
    market_type: str
    outcome_label: str
    line: float | None
    odds_decimal: float
    odds_american: int
    book: str | None

    # Probabilities
    p_ensemble_mean: float
    p_ensemble_std: float | None
    p_true: float
    p_market: float
    raw_edge: float
    edge_band: str | None

    # Algorithm A breakdown
    algo_a: dict

    # Algorithm B breakdown
    algo_b: dict

    # Market quality breakdown
    market_quality: dict

    # Scoring metadata
    calc_time: datetime
    active_algorithm: str

    class Config:
        from_attributes = True


@router.get("/markets/{market_id}", response_model=MarketDetailResponse)
async def get_market_detail(market_id: str) -> MarketDetailResponse:
    """
    Get detailed breakdown of a single market's Value Score.

    Returns full scoring breakdown including both algorithms,
    confidence components, and market quality factors.
    """
    now = datetime.now(timezone.utc)

    async with async_session() as session:
        # Get the most recent value score for this market
        query = (
            select(ValueScore)
            .where(ValueScore.market_id == market_id)
            .order_by(desc(ValueScore.calc_time))
            .limit(1)
            .options(
                selectinload(ValueScore.market).selectinload(Market.game),
                selectinload(ValueScore.prediction),
            )
        )

        result = await session.execute(query)
        vs = result.scalar_one_or_none()

        if not vs:
            raise HTTPException(status_code=404, detail="Market not found or not scored")

        market = vs.market
        game = market.game
        prediction = vs.prediction

        # Get team names
        teams_query = select(Team).where(
            Team.team_id.in_([game.home_team_id, game.away_team_id])
        )
        teams_result = await session.execute(teams_query)
        teams = {t.team_id: t for t in teams_result.scalars().all()}

        home_team = teams.get(game.home_team_id)
        away_team = teams.get(game.away_team_id)

        time_to_tip = int((game.tip_time_utc - now).total_seconds() / 60)

        return MarketDetailResponse(
            market_id=market.market_id,
            game_id=game.game_id,
            home_team=home_team.abbreviation if home_team else game.home_team_id,
            away_team=away_team.abbreviation if away_team else game.away_team_id,
            tip_time=game.tip_time_utc,
            time_to_tip_minutes=time_to_tip,
            market_type=market.market_type,
            outcome_label=market.outcome_label,
            line=float(market.line) if market.line else None,
            odds_decimal=float(market.odds_decimal),
            odds_american=market.odds_american,
            book=market.book,
            p_ensemble_mean=float(prediction.p_ensemble_mean),
            p_ensemble_std=float(prediction.p_ensemble_std) if prediction.p_ensemble_std else None,
            p_true=float(prediction.p_true),
            p_market=float(prediction.p_market),
            raw_edge=float(prediction.raw_edge),
            edge_band=prediction.edge_band,
            algo_a={
                "value_score": float(vs.algo_a_value_score or 0),
                "edge_score": float(vs.algo_a_edge_score or 0),
                "confidence": float(vs.algo_a_confidence or 0),
                "market_quality": float(vs.algo_a_market_quality or 0),
                "formula": "tanh(raw_edge/scale) * confidence * market_quality * 100",
            },
            algo_b={
                "value_score": float(vs.algo_b_value_score or 0),
                "combined_edge": float(vs.algo_b_combined_edge or 0),
                "confidence": float(vs.algo_b_confidence or 0),
                "market_quality": float(vs.algo_b_market_quality or 0),
                "formula": "100 * tanh((raw_edge * confidence * market_quality) / scale)",
            },
            market_quality={
                "final_score": float(vs.algo_a_market_quality or 0),
                "time_to_tip_minutes": vs.time_to_tip_minutes,
            },
            calc_time=vs.calc_time,
            active_algorithm=vs.active_algorithm,
        )


@router.get("/markets/by-game/{game_id}")
async def get_markets_by_game(
    game_id: str,
    algorithm: Literal["a", "b"] = Query("a", description="Algorithm to sort by"),
) -> list[MarketResponse]:
    """Get all scored markets for a specific game."""
    now = datetime.now(timezone.utc)

    async with async_session() as session:
        # Get all value scores for this game's markets
        query = (
            select(ValueScore)
            .join(ValueScore.market)
            .where(Market.game_id == game_id)
            .order_by(desc(ValueScore.calc_time))
            .options(
                selectinload(ValueScore.market).selectinload(Market.game),
                selectinload(ValueScore.prediction),
            )
        )

        if algorithm == "a":
            query = query.order_by(desc(ValueScore.algo_a_value_score))
        else:
            query = query.order_by(desc(ValueScore.algo_b_value_score))

        result = await session.execute(query)
        value_scores = result.scalars().all()

        if not value_scores:
            return []

        # Get unique markets (most recent score for each)
        seen_markets = set()
        unique_scores = []
        for vs in value_scores:
            if vs.market_id not in seen_markets:
                seen_markets.add(vs.market_id)
                unique_scores.append(vs)

        # Get team names
        game = unique_scores[0].market.game
        teams_query = select(Team).where(
            Team.team_id.in_([game.home_team_id, game.away_team_id])
        )
        teams_result = await session.execute(teams_query)
        teams = {t.team_id: t for t in teams_result.scalars().all()}

        home_team = teams.get(game.home_team_id)
        away_team = teams.get(game.away_team_id)

        # Build response
        markets = []
        for vs in unique_scores:
            market = vs.market
            prediction = vs.prediction
            time_to_tip = int((game.tip_time_utc - now).total_seconds() / 60)

            markets.append(MarketResponse(
                market_id=market.market_id,
                game_id=game.game_id,
                home_team=home_team.abbreviation if home_team else game.home_team_id,
                away_team=away_team.abbreviation if away_team else game.away_team_id,
                tip_time=game.tip_time_utc,
                time_to_tip_minutes=time_to_tip,
                market_type=market.market_type,
                outcome_label=market.outcome_label,
                line=float(market.line) if market.line else None,
                odds_decimal=float(market.odds_decimal),
                odds_american=market.odds_american,
                book=market.book,
                p_true=float(prediction.p_true),
                p_market=float(prediction.p_market),
                raw_edge=float(prediction.raw_edge),
                edge_band=prediction.edge_band,
                algo_a_value_score=float(vs.algo_a_value_score or 0),
                algo_a_confidence=float(vs.algo_a_confidence or 0),
                algo_a_edge_score=float(vs.algo_a_edge_score or 0),
                algo_b_value_score=float(vs.algo_b_value_score or 0),
                algo_b_confidence=float(vs.algo_b_confidence or 0),
                algo_b_combined_edge=float(vs.algo_b_combined_edge or 0),
                market_quality=float(vs.algo_a_market_quality or 0),
                calc_time=vs.calc_time,
                active_algorithm=vs.active_algorithm,
            ))

        return markets


@router.get("/games/upcoming")
async def get_upcoming_games(
    hours: int = Query(24, ge=1, le=168, description="Hours ahead to look"),
) -> list[dict]:
    """Get list of upcoming games with their market counts and team trends."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours)
    today = now.date()

    async with async_session() as session:
        query = (
            select(Game)
            .where(Game.tip_time_utc > now)
            .where(Game.tip_time_utc < cutoff)
            .where(Game.status == "scheduled")
            .options(selectinload(Game.markets))
            .order_by(Game.tip_time_utc)
        )

        result = await session.execute(query)
        games = result.scalars().all()

        # Get team names
        team_ids = set()
        for g in games:
            team_ids.add(g.home_team_id)
            team_ids.add(g.away_team_id)

        teams = {}
        team_stats_map = {}
        if team_ids:
            # Get team info
            teams_query = select(Team).where(Team.team_id.in_(team_ids))
            teams_result = await session.execute(teams_query)
            teams = {t.team_id: t for t in teams_result.scalars().all()}

            # Get latest team stats for each team
            for team_id in team_ids:
                stats_query = (
                    select(TeamStats)
                    .where(TeamStats.team_id == team_id)
                    .where(TeamStats.stat_date <= today)
                    .order_by(desc(TeamStats.stat_date))
                    .limit(1)
                )
                stats_result = await session.execute(stats_query)
                stats = stats_result.scalar_one_or_none()
                if stats:
                    team_stats_map[team_id] = stats

        def build_team_trends(team_id: str, is_home: bool) -> dict:
            """Build trends dict for a team."""
            stats = team_stats_map.get(team_id)
            if not stats:
                return {
                    "record": "0-0",
                    "home_record": "0-0",
                    "away_record": "0-0",
                    "l10_record": "0-0",
                    "net_rtg_l10": None,
                    "rest_days": None,
                    "is_b2b": False,
                }

            record = f"{stats.wins}-{stats.losses}"
            home_record = f"{stats.home_wins or 0}-{stats.home_losses or 0}"
            away_record = f"{stats.away_wins or 0}-{stats.away_losses or 0}"
            l10_record = f"{stats.wins_l10 or 0}-{stats.losses_l10 or 0}"
            net_rtg = float(stats.net_rtg_10) if stats.net_rtg_10 else None
            rest = stats.days_rest
            b2b = stats.is_back_to_back or False

            return {
                "record": record,
                "home_record": home_record,
                "away_record": away_record,
                "l10_record": l10_record,
                "net_rtg_l10": net_rtg,
                "rest_days": rest,
                "is_b2b": b2b,
            }

        response = []
        for game in games:
            home = teams.get(game.home_team_id)
            away = teams.get(game.away_team_id)

            response.append({
                "game_id": game.game_id,
                "home_team": home.abbreviation if home else game.home_team_id,
                "away_team": away.abbreviation if away else game.away_team_id,
                "home_team_full": home.full_name if home else game.home_team_id,
                "away_team_full": away.full_name if away else game.away_team_id,
                "tip_time": game.tip_time_utc.isoformat(),
                "time_to_tip_minutes": int((game.tip_time_utc - now).total_seconds() / 60),
                "markets_count": len(game.markets),
                "status": game.status,
                "home_trends": build_team_trends(game.home_team_id, is_home=True),
                "away_trends": build_team_trends(game.away_team_id, is_home=False),
            })

        return response
