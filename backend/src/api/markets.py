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
            # Use getattr for new columns that might not exist during migration
            home_wins = getattr(stats, 'home_wins', None) or 0
            home_losses = getattr(stats, 'home_losses', None) or 0
            away_wins = getattr(stats, 'away_wins', None) or 0
            away_losses = getattr(stats, 'away_losses', None) or 0
            wins_l10 = getattr(stats, 'wins_l10', None) or 0
            losses_l10 = getattr(stats, 'losses_l10', None) or 0

            home_record = f"{home_wins}-{home_losses}"
            away_record = f"{away_wins}-{away_losses}"
            l10_record = f"{wins_l10}-{losses_l10}"
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

        # Get value scores for all games' markets
        game_ids = [g.game_id for g in games]
        value_scores_map = {}
        if game_ids:
            # Get latest value scores for each market
            vs_query = (
                select(ValueScore)
                .join(ValueScore.market)
                .join(ValueScore.prediction)
                .where(Market.game_id.in_(game_ids))
                .options(
                    selectinload(ValueScore.market),
                    selectinload(ValueScore.prediction),
                )
                .order_by(desc(ValueScore.calc_time))
            )
            vs_result = await session.execute(vs_query)
            all_scores = vs_result.scalars().all()

            # Group by game_id, keep only latest score per market
            seen_markets = set()
            for vs in all_scores:
                if vs.market_id not in seen_markets:
                    seen_markets.add(vs.market_id)
                    game_id = vs.market.game_id
                    if game_id not in value_scores_map:
                        value_scores_map[game_id] = []
                    value_scores_map[game_id].append(vs)

        def build_prediction(game, home_abbr: str, away_abbr: str, home_trends: dict, away_trends: dict) -> dict | None:
            """Build prediction dict for a game."""
            scores = value_scores_map.get(game.game_id, [])
            if not scores:
                return None

            # Find moneyline markets to determine winner
            home_ml = None
            away_ml = None
            best_bet = None
            best_score = 0

            for vs in scores:
                market = vs.market
                prediction = vs.prediction
                score = float(vs.algo_b_value_score or 0)

                # Track best value bet
                if score > best_score:
                    best_score = score
                    best_bet = {
                        "type": market.market_type,
                        "team": home_abbr if "home" in market.outcome_label else away_abbr,
                        "line": float(market.line) if market.line else None,
                        "value_score": round(score),
                        "edge": round(float(prediction.raw_edge) * 100, 1) if prediction.raw_edge else 0,
                        "p_true": round(float(prediction.p_true) * 100, 1) if prediction.p_true else 0,
                        "p_market": round(float(prediction.p_market) * 100, 1) if prediction.p_market else 0,
                    }

                # Track moneyline for winner prediction
                if market.market_type == "moneyline":
                    if "home" in market.outcome_label:
                        home_ml = {
                            "p_true": float(prediction.p_true) if prediction.p_true else 0.5,
                            "p_market": float(prediction.p_market) if prediction.p_market else 0.5,
                        }
                    else:
                        away_ml = {
                            "p_true": float(prediction.p_true) if prediction.p_true else 0.5,
                            "p_market": float(prediction.p_market) if prediction.p_market else 0.5,
                        }

            # Determine winner from moneyline probabilities
            if home_ml and away_ml:
                home_prob = home_ml["p_true"]
                away_prob = away_ml["p_true"]
            else:
                # Fallback: use net rating differential to estimate
                home_net = home_trends.get("net_rtg_l10") or 0
                away_net = away_trends.get("net_rtg_l10") or 0
                # Simple logistic based on net rating diff + home court
                diff = (home_net - away_net) * 0.03 + 0.03  # ~3% home court
                home_prob = 0.5 + diff
                away_prob = 1 - home_prob

            winner = home_abbr if home_prob >= away_prob else away_abbr
            winner_prob = max(home_prob, away_prob)

            # Confidence level based on probability edge
            if winner_prob >= 0.65:
                confidence = "high"
            elif winner_prob >= 0.55:
                confidence = "medium"
            else:
                confidence = "low"

            # Build explanation factors
            factors = []

            # 1. Net rating comparison
            home_net = home_trends.get("net_rtg_l10")
            away_net = away_trends.get("net_rtg_l10")
            if home_net is not None and away_net is not None:
                diff = home_net - away_net
                if abs(diff) >= 1.0:
                    better = home_abbr if diff > 0 else away_abbr
                    factors.append(f"{better} +{abs(diff):.1f} Net Rating (L10)")

            # 2. Rest/B2B advantage
            home_rest = home_trends.get("rest_days") or 0
            away_rest = away_trends.get("rest_days") or 0
            home_b2b = home_trends.get("is_b2b", False)
            away_b2b = away_trends.get("is_b2b", False)

            if home_b2b and not away_b2b:
                factors.append(f"{away_abbr} rest advantage (vs B2B)")
            elif away_b2b and not home_b2b:
                factors.append(f"{home_abbr} rest advantage (vs B2B)")
            elif abs(home_rest - away_rest) >= 2:
                better = home_abbr if home_rest > away_rest else away_abbr
                factors.append(f"{better} +{abs(home_rest - away_rest)} days rest")

            # 3. Model edge on best bet
            if best_bet and best_bet["edge"] > 0:
                factors.append(f"Model: {best_bet['p_true']:.0f}% vs Market: {best_bet['p_market']:.0f}% (+{best_bet['edge']:.1f}% edge)")

            # 4. Record comparison if significant
            home_l10 = home_trends.get("l10_record", "0-0")
            away_l10 = away_trends.get("l10_record", "0-0")
            try:
                home_l10_wins = int(home_l10.split("-")[0])
                away_l10_wins = int(away_l10.split("-")[0])
                if abs(home_l10_wins - away_l10_wins) >= 3:
                    better = home_abbr if home_l10_wins > away_l10_wins else away_abbr
                    better_record = home_l10 if home_l10_wins > away_l10_wins else away_l10
                    factors.append(f"{better} is {better_record} in L10")
            except (ValueError, IndexError):
                pass

            return {
                "winner": winner,
                "winner_prob": round(winner_prob * 100),
                "confidence": confidence,
                "best_bet": best_bet,
                "factors": factors[:4],  # Limit to 4 factors
            }

        response = []
        for game in games:
            home = teams.get(game.home_team_id)
            away = teams.get(game.away_team_id)

            home_abbr = home.abbreviation if home else game.home_team_id
            away_abbr = away.abbreviation if away else game.away_team_id
            home_trends = build_team_trends(game.home_team_id, is_home=True)
            away_trends = build_team_trends(game.away_team_id, is_home=False)

            response.append({
                "game_id": game.game_id,
                "home_team": home_abbr,
                "away_team": away_abbr,
                "home_team_full": home.full_name if home else game.home_team_id,
                "away_team_full": away.full_name if away else game.away_team_id,
                "tip_time": game.tip_time_utc.isoformat(),
                "time_to_tip_minutes": int((game.tip_time_utc - now).total_seconds() / 60),
                "markets_count": len(game.markets),
                "status": game.status,
                "home_trends": home_trends,
                "away_trends": away_trends,
                "prediction": build_prediction(game, home_abbr, away_abbr, home_trends, away_trends),
            })

        return response


class BacktestResult(BaseModel):
    """Results from backtesting predictions against actual outcomes."""

    # Summary stats
    total_games: int
    total_predictions: int
    days_analyzed: int

    # Overall performance
    overall_win_rate: float
    overall_roi: float
    total_profit_loss: float  # Assuming $100 per bet

    # By market type
    spread_stats: dict
    moneyline_stats: dict
    total_stats: dict

    # By value score bucket
    by_value_bucket: list[dict]

    # Individual bet results for analysis
    sample_bets: list[dict]

    # Issues detected
    issues: list[str]


@router.get("/backtest")
async def run_backtest(
    days: int = Query(14, ge=1, le=90, description="Number of days to backtest"),
    min_value_score: float = Query(50, ge=0, le=100, description="Min value score to consider a bet"),
    algorithm: Literal["a", "b"] = Query("b", description="Algorithm to evaluate"),
) -> BacktestResult:
    """
    Backtest the model against completed games.

    Evaluates all predictions made for games that have final scores,
    calculating win rate and ROI.
    """
    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=days)

    async with async_session() as session:
        # Get completed games with final scores
        games_query = (
            select(Game)
            .where(Game.game_date >= start_date.date())
            .where(Game.home_score.isnot(None))
            .where(Game.away_score.isnot(None))
            .options(selectinload(Game.markets))
            .order_by(desc(Game.game_date))
        )

        result = await session.execute(games_query)
        games = result.scalars().all()

        if not games:
            return BacktestResult(
                total_games=0,
                total_predictions=0,
                days_analyzed=days,
                overall_win_rate=0.0,
                overall_roi=0.0,
                total_profit_loss=0.0,
                spread_stats={},
                moneyline_stats={},
                total_stats={},
                by_value_bucket=[],
                sample_bets=[],
                issues=["No completed games found in the specified date range"],
            )

        # Get all value scores for these games
        game_ids = [g.game_id for g in games]

        vs_query = (
            select(ValueScore)
            .join(ValueScore.market)
            .join(ValueScore.prediction)
            .where(Market.game_id.in_(game_ids))
            .options(
                selectinload(ValueScore.market).selectinload(Market.game),
                selectinload(ValueScore.prediction),
            )
        )

        vs_result = await session.execute(vs_query)
        all_scores = vs_result.scalars().all()

        # Get unique most recent score per market
        seen_markets = {}
        for vs in all_scores:
            if vs.market_id not in seen_markets or vs.calc_time > seen_markets[vs.market_id].calc_time:
                seen_markets[vs.market_id] = vs

        # Get team names for display
        team_ids = set()
        for g in games:
            team_ids.add(g.home_team_id)
            team_ids.add(g.away_team_id)

        teams = {}
        if team_ids:
            teams_query = select(Team).where(Team.team_id.in_(team_ids))
            teams_result = await session.execute(teams_query)
            teams = {t.team_id: t for t in teams_result.scalars().all()}

        # Evaluate each prediction
        results = []
        for market_id, vs in seen_markets.items():
            market = vs.market
            game = market.game
            prediction = vs.prediction

            # Get value score based on algorithm
            if algorithm == "a":
                value_score = float(vs.algo_a_value_score or 0)
            else:
                value_score = float(vs.algo_b_value_score or 0)

            # Skip if below threshold
            if value_score < min_value_score:
                continue

            # Calculate actual outcome
            home_score = game.home_score
            away_score = game.away_score
            home_margin = home_score - away_score
            total_points = home_score + away_score

            # Determine if bet won
            bet_won = False
            bet_pushed = False
            is_home = "home" in market.outcome_label.lower()

            if market.market_type == "spread":
                line = float(market.line) if market.line else 0
                if is_home:
                    # Home spread: home margin + line > 0 means home covered
                    covered_margin = home_margin + line
                else:
                    # Away spread: -home_margin + line > 0 means away covered
                    covered_margin = -home_margin + line

                if covered_margin > 0:
                    bet_won = True
                elif covered_margin == 0:
                    bet_pushed = True

            elif market.market_type == "moneyline":
                if is_home:
                    bet_won = home_margin > 0
                else:
                    bet_won = home_margin < 0

            elif market.market_type == "total":
                line = float(market.line) if market.line else 0
                is_over = "over" in market.outcome_label.lower()

                if is_over:
                    if total_points > line:
                        bet_won = True
                    elif total_points == line:
                        bet_pushed = True
                else:
                    if total_points < line:
                        bet_won = True
                    elif total_points == line:
                        bet_pushed = True

            # Calculate profit/loss (assuming $100 bet)
            odds_decimal = float(market.odds_decimal)
            if bet_pushed:
                profit = 0
            elif bet_won:
                profit = 100 * (odds_decimal - 1)
            else:
                profit = -100

            # Get team names
            home_team = teams.get(game.home_team_id)
            away_team = teams.get(game.away_team_id)
            home_abbr = home_team.abbreviation if home_team else game.home_team_id
            away_abbr = away_team.abbreviation if away_team else game.away_team_id

            # Determine bet team/side
            if market.market_type == "total":
                bet_side = "Over" if "over" in market.outcome_label.lower() else "Under"
                bet_desc = f"{bet_side} {market.line}"
            else:
                bet_side = home_abbr if is_home else away_abbr
                if market.market_type == "spread":
                    bet_desc = f"{bet_side} {'+' if market.line and market.line > 0 else ''}{market.line}"
                else:
                    bet_desc = f"{bet_side} ML"

            results.append({
                "game_id": game.game_id,
                "game_date": game.game_date.isoformat(),
                "matchup": f"{away_abbr} @ {home_abbr}",
                "final_score": f"{away_score}-{home_score}",
                "market_type": market.market_type,
                "bet": bet_desc,
                "line": float(market.line) if market.line else None,
                "odds_decimal": odds_decimal,
                "value_score": value_score,
                "p_true": float(prediction.p_true) * 100 if prediction.p_true else 0,
                "p_market": float(prediction.p_market) * 100 if prediction.p_market else 0,
                "edge": float(prediction.raw_edge) * 100 if prediction.raw_edge else 0,
                "won": bet_won,
                "pushed": bet_pushed,
                "profit": profit,
                "is_underdog": not is_home if market.market_type == "moneyline" else None,
            })

        if not results:
            return BacktestResult(
                total_games=len(games),
                total_predictions=0,
                days_analyzed=days,
                overall_win_rate=0.0,
                overall_roi=0.0,
                total_profit_loss=0.0,
                spread_stats={},
                moneyline_stats={},
                total_stats={},
                by_value_bucket=[],
                sample_bets=[],
                issues=[f"No predictions above {min_value_score}% value score found"],
            )

        # Calculate stats
        total_bets = len([r for r in results if not r["pushed"]])
        wins = len([r for r in results if r["won"]])
        total_profit = sum(r["profit"] for r in results)

        overall_win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
        overall_roi = (total_profit / (total_bets * 100) * 100) if total_bets > 0 else 0

        # By market type
        def calc_type_stats(market_type: str) -> dict:
            type_results = [r for r in results if r["market_type"] == market_type and not r["pushed"]]
            if not type_results:
                return {"bets": 0, "wins": 0, "win_rate": 0, "roi": 0, "profit": 0}

            type_wins = len([r for r in type_results if r["won"]])
            type_profit = sum(r["profit"] for r in type_results)
            type_bets = len(type_results)

            return {
                "bets": type_bets,
                "wins": type_wins,
                "win_rate": round(type_wins / type_bets * 100, 1) if type_bets > 0 else 0,
                "roi": round(type_profit / (type_bets * 100) * 100, 1) if type_bets > 0 else 0,
                "profit": round(type_profit, 2),
            }

        # By value score bucket
        buckets = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
        bucket_stats = []
        for low, high in buckets:
            bucket_results = [r for r in results if low <= r["value_score"] < high and not r["pushed"]]
            if bucket_results:
                b_wins = len([r for r in bucket_results if r["won"]])
                b_profit = sum(r["profit"] for r in bucket_results)
                b_bets = len(bucket_results)
                bucket_stats.append({
                    "bucket": f"{low}-{high}%",
                    "bets": b_bets,
                    "wins": b_wins,
                    "win_rate": round(b_wins / b_bets * 100, 1) if b_bets > 0 else 0,
                    "roi": round(b_profit / (b_bets * 100) * 100, 1) if b_bets > 0 else 0,
                    "profit": round(b_profit, 2),
                })

        # Detect issues
        issues = []

        # Check underdog bias
        ml_results = [r for r in results if r["market_type"] == "moneyline" and r["is_underdog"] is not None]
        underdog_bets = [r for r in ml_results if r["is_underdog"]]
        fav_bets = [r for r in ml_results if not r["is_underdog"]]

        if len(underdog_bets) > len(fav_bets) * 2 and len(ml_results) >= 10:
            issues.append(f"Underdog bias detected: {len(underdog_bets)} underdog bets vs {len(fav_bets)} favorite bets")

        if overall_roi < -10:
            issues.append(f"Significant negative ROI: {overall_roi:.1f}%")

        # Check if high value scores perform worse than low
        high_bucket = next((b for b in bucket_stats if b["bucket"] == "80-90%" or b["bucket"] == "90-100%"), None)
        low_bucket = next((b for b in bucket_stats if b["bucket"] == "50-60%"), None)
        if high_bucket and low_bucket and high_bucket["win_rate"] < low_bucket["win_rate"]:
            issues.append(f"Calibration issue: High value bets ({high_bucket['win_rate']}% win rate) underperform low value bets ({low_bucket['win_rate']}%)")

        # Sample recent bets for detailed review
        sample_bets = sorted(results, key=lambda x: x["game_date"], reverse=True)[:20]

        return BacktestResult(
            total_games=len(games),
            total_predictions=len(results),
            days_analyzed=days,
            overall_win_rate=round(overall_win_rate, 1),
            overall_roi=round(overall_roi, 1),
            total_profit_loss=round(total_profit, 2),
            spread_stats=calc_type_stats("spread"),
            moneyline_stats=calc_type_stats("moneyline"),
            total_stats=calc_type_stats("total"),
            by_value_bucket=bucket_stats,
            sample_bets=sample_bets,
            issues=issues,
        )
