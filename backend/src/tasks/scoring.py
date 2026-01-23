"""Scoring tasks for pre-game Value Score calculation."""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import structlog
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload

# Optional celery import - allows running without celery installed
try:
    from src.celery_app import celery_app
except ImportError:
    celery_app = None

from src.database import async_session
from src.models import Game, Market, TeamStats, ModelPrediction, ValueScore
from src.services.scoring.scorer import ScoringService, ScoringInput, get_scoring_service
from src.services.injuries import get_all_team_injury_reports

logger = structlog.get_logger()


async def _run_pre_game_scoring_async() -> dict:
    """
    Async implementation of pre-game scoring.

    Flow:
    1. Fetch injury reports for all teams
    2. Query games starting in next 24 hours
    3. Get markets for those games
    4. Get team stats for both teams
    5. Run ScoringService for each market (with injury data)
    6. Store ModelPrediction and ValueScore records
    """
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=24)
    today = now.date()

    markets_scored = 0
    errors = 0

    # Fetch injury reports for all teams
    try:
        injury_reports = await get_all_team_injury_reports()
        # Count teams with significant injuries
        injured_teams = sum(1 for r in injury_reports.values() if r.injury_score > 0.1)
        logger.info(
            "Fetched injury reports",
            total_teams=len(injury_reports),
            teams_with_injuries=injured_teams,
        )
    except Exception as e:
        logger.error(
            "CRITICAL: Failed to fetch injury reports - scoring without injury adjustments!",
            error=str(e),
        )
        injury_reports = {}

    scoring_service = get_scoring_service()

    async with async_session() as session:
        # Get upcoming games with their markets
        games_query = (
            select(Game)
            .where(Game.tip_time_utc > now)
            .where(Game.tip_time_utc < cutoff)
            .where(Game.status == "scheduled")
            .options(selectinload(Game.markets))
        )

        result = await session.execute(games_query)
        games = result.scalars().all()

        logger.info("Found games to score", count=len(games))

        for game in games:
            try:
                # Get team stats for home team
                home_stats_query = (
                    select(TeamStats)
                    .where(TeamStats.team_id == game.home_team_id)
                    .where(TeamStats.stat_date <= today)
                    .order_by(TeamStats.stat_date.desc())
                    .limit(1)
                )
                home_result = await session.execute(home_stats_query)
                home_stats = home_result.scalar_one_or_none()

                # Get team stats for away team
                away_stats_query = (
                    select(TeamStats)
                    .where(TeamStats.team_id == game.away_team_id)
                    .where(TeamStats.stat_date <= today)
                    .order_by(TeamStats.stat_date.desc())
                    .limit(1)
                )
                away_result = await session.execute(away_stats_query)
                away_stats = away_result.scalar_one_or_none()

                # Build feature dicts from stats
                home_features = _stats_to_features(home_stats, "home")
                away_features = _stats_to_features(away_stats, "away")

                # Delete old predictions and scores for this game's markets
                market_ids = [m.market_id for m in game.markets if m.is_active]
                if market_ids:
                    # Delete old value scores first (foreign key constraint)
                    await session.execute(
                        delete(ValueScore).where(ValueScore.market_id.in_(market_ids))
                    )
                    # Delete old predictions
                    await session.execute(
                        delete(ModelPrediction).where(ModelPrediction.market_id.in_(market_ids))
                    )

                # Get injury scores for both teams (general and market-specific)
                home_injury_report = injury_reports.get(game.home_team_id)
                away_injury_report = injury_reports.get(game.away_team_id)

                # General injury scores (for spread/ML)
                home_injury_score = home_injury_report.spread_injury_score if home_injury_report else 0.0
                away_injury_score = away_injury_report.spread_injury_score if away_injury_report else 0.0

                # Totals-specific injury scores (weighted toward centers/rebounding)
                home_totals_injury = home_injury_report.totals_injury_score if home_injury_report else 0.0
                away_totals_injury = away_injury_report.totals_injury_score if away_injury_report else 0.0

                logger.debug(
                    "Injury data for game",
                    game_id=game.game_id,
                    home_team=game.home_team_id,
                    away_team=game.away_team_id,
                    home_spread_injury=home_injury_score,
                    away_spread_injury=away_injury_score,
                    home_totals_injury=home_totals_injury,
                    away_totals_injury=away_totals_injury,
                    home_ppg_lost=home_injury_report.total_ppg_lost if home_injury_report else 0,
                    away_ppg_lost=away_injury_report.total_ppg_lost if away_injury_report else 0,
                    home_players_out=home_injury_report.players_out[:3] if home_injury_report else [],
                    away_players_out=away_injury_report.players_out[:3] if away_injury_report else [],
                )

                # Score each market for this game
                for market in game.markets:
                    if not market.is_active:
                        continue

                    try:
                        # Find opposite market for de-vigging
                        opposite_odds = _find_opposite_odds(market, game.markets)

                        # Create scoring input with position-aware injury data
                        scoring_input = ScoringInput(
                            game_id=game.game_id,
                            market_type=market.market_type,
                            outcome_label=market.outcome_label,
                            line=float(market.line) if market.line else None,
                            odds_decimal=float(market.odds_decimal),
                            opposite_odds=opposite_odds,
                            home_features=home_features,
                            away_features=away_features,
                            tip_time=game.tip_time_utc,
                            book=market.book,
                            home_injury_score=home_injury_score,
                            away_injury_score=away_injury_score,
                            home_totals_injury=home_totals_injury,
                            away_totals_injury=away_totals_injury,
                        )

                        # Run scoring
                        score_result = scoring_service.score_market(scoring_input)

                        # Create ModelPrediction record
                        prediction = ModelPrediction(
                            market_id=market.market_id,
                            prediction_time=score_result.calc_time,
                            p_ensemble_mean=Decimal(str(score_result.p_raw)),
                            p_ensemble_std=Decimal("0.05"),  # Placeholder
                            p_true=Decimal(str(score_result.p_calibrated)),
                            p_market=Decimal(str(score_result.p_market)),
                            raw_edge=Decimal(str(score_result.raw_edge)),
                            edge_band=_get_edge_band(score_result.raw_edge),
                        )
                        session.add(prediction)
                        await session.flush()  # Get prediction_id

                        # Create ValueScore record
                        value_score = ValueScore(
                            prediction_id=prediction.prediction_id,
                            market_id=market.market_id,
                            calc_time=score_result.calc_time,
                            algo_a_edge_score=Decimal(str(score_result.algo_a.edge_score)),
                            algo_a_confidence=Decimal(str(score_result.algo_a_confidence.final_multiplier)),
                            algo_a_market_quality=Decimal(str(score_result.market_quality.final_score)),
                            algo_a_value_score=Decimal(str(score_result.algo_a.value_score)),
                            algo_b_combined_edge=Decimal(str(score_result.algo_b.combined_edge)),
                            algo_b_confidence=Decimal(str(score_result.algo_b_confidence.final_multiplier)),
                            algo_b_market_quality=Decimal(str(score_result.market_quality.final_score)),
                            algo_b_value_score=Decimal(str(score_result.algo_b.value_score)),
                            active_algorithm="A",
                            time_to_tip_minutes=score_result.time_to_tip_minutes,
                        )
                        session.add(value_score)

                        markets_scored += 1

                        logger.debug(
                            "Scored market",
                            market_id=market.market_id,
                            algo_a_score=score_result.algo_a.value_score,
                            algo_b_score=score_result.algo_b.value_score,
                        )

                    except Exception as e:
                        logger.error(
                            "Failed to score market",
                            market_id=market.market_id,
                            error=str(e),
                        )
                        errors += 1

            except Exception as e:
                logger.error(
                    "Failed to process game",
                    game_id=game.game_id,
                    error=str(e),
                )
                errors += 1

        # Commit all changes
        await session.commit()

    return {
        "markets_scored": markets_scored,
        "errors": errors,
        "games_processed": len(games),
        "status": "completed",
    }


def _stats_to_features(stats: TeamStats | None, prefix: str) -> dict[str, float]:
    """Convert TeamStats model to feature dict.

    Includes v2 spread model features: ATS tendencies, home/away splits, schedule density.
    """
    if stats is None:
        # Return reasonable defaults
        return {
            f"{prefix}_ortg_10": None,  # None so model knows data is missing
            f"{prefix}_ortg_season": None,
            f"{prefix}_drtg_10": None,
            f"{prefix}_drtg_season": None,
            f"{prefix}_net_rtg_10": 0.0,  # Neutral
            f"{prefix}_net_rtg_season": 0.0,
            f"{prefix}_pace_10": 100.0,
            f"{prefix}_rest_days": 1,
            f"{prefix}_b2b": 0,
            f"{prefix}_win_pct_10": 0.5,
            # V2 features - defaults
            f"{prefix}_ppg": 110.0,
            f"{prefix}_opp_ppg": 110.0,
            f"{prefix}_net_ppg": 0.0,
            f"{prefix}_scoring_std": 10.0,  # Default variance
            f"{prefix}_ats_pct_l10": 0.5,
            f"{prefix}_home_win_pct": 0.5,
            f"{prefix}_away_win_pct": 0.5,
            f"{prefix}_games_last_7": 3,
        }

    # Compute ATS percentage from L10 record
    ats_wins = stats.ats_wins_l10 or 0
    ats_losses = stats.ats_losses_l10 or 0
    ats_total = ats_wins + ats_losses
    ats_pct = ats_wins / ats_total if ats_total > 0 else 0.5

    # Compute home/away win percentages
    home_wins = stats.home_wins or 0
    home_losses = stats.home_losses or 0
    home_total = home_wins + home_losses
    home_win_pct = home_wins / home_total if home_total > 0 else 0.5

    away_wins = stats.away_wins or 0
    away_losses = stats.away_losses or 0
    away_total = away_wins + away_losses
    away_win_pct = away_wins / away_total if away_total > 0 else 0.5

    # PPG for net calculation
    ppg = float(stats.ppg_10) if stats.ppg_10 else 110.0
    opp_ppg = float(stats.opp_ppg_10) if stats.opp_ppg_10 else 110.0
    net_ppg = ppg - opp_ppg

    return {
        # Original features
        f"{prefix}_ortg_5": float(stats.ortg_5) if stats.ortg_5 else None,
        f"{prefix}_ortg_10": float(stats.ortg_10) if stats.ortg_10 else None,
        f"{prefix}_ortg_20": float(stats.ortg_20) if stats.ortg_20 else None,
        f"{prefix}_ortg_season": float(stats.ortg_season) if stats.ortg_season else None,
        f"{prefix}_drtg_5": float(stats.drtg_5) if stats.drtg_5 else None,
        f"{prefix}_drtg_10": float(stats.drtg_10) if stats.drtg_10 else None,
        f"{prefix}_drtg_20": float(stats.drtg_20) if stats.drtg_20 else None,
        f"{prefix}_drtg_season": float(stats.drtg_season) if stats.drtg_season else None,
        f"{prefix}_net_rtg_10": float(stats.net_rtg_10) if stats.net_rtg_10 else 0.0,
        f"{prefix}_net_rtg_season": float(stats.net_rtg_season) if stats.net_rtg_season else 0.0,
        f"{prefix}_pace_10": float(stats.pace_10) if stats.pace_10 else 100.0,
        f"{prefix}_pace_season": float(stats.pace_season) if stats.pace_season else 100.0,
        f"{prefix}_ppg_10": ppg,
        f"{prefix}_ppg_season": float(stats.ppg_season) if stats.ppg_season else 110.0,
        f"{prefix}_opp_ppg_10": opp_ppg,
        f"{prefix}_opp_ppg_season": float(stats.opp_ppg_season) if stats.opp_ppg_season else 110.0,
        f"{prefix}_rest_days": stats.days_rest if stats.days_rest else 1,
        f"{prefix}_b2b": 1 if stats.is_back_to_back else 0,
        f"{prefix}_win_pct_10": float(stats.win_pct_10) if stats.win_pct_10 else 0.5,
        # V2 spread model features
        f"{prefix}_ppg": ppg,
        f"{prefix}_opp_ppg": opp_ppg,
        f"{prefix}_net_ppg": net_ppg,
        f"{prefix}_scoring_std": 10.0,  # Estimated default - would need historical data to compute
        f"{prefix}_ats_pct_l10": ats_pct,
        f"{prefix}_home_win_pct": home_win_pct,
        f"{prefix}_away_win_pct": away_win_pct,
        f"{prefix}_games_last_7": stats.games_last_7_days if stats.games_last_7_days else 3,
    }


def _find_opposite_odds(market: Market, all_markets: list[Market]) -> float:
    """Find the opposite side odds for de-vigging."""
    # For spread/ML markets, find the opposing side
    opposite_label = None

    if "home" in market.outcome_label.lower():
        opposite_label = market.outcome_label.lower().replace("home", "away")
    elif "away" in market.outcome_label.lower():
        opposite_label = market.outcome_label.lower().replace("away", "home")
    elif "over" in market.outcome_label.lower():
        opposite_label = market.outcome_label.lower().replace("over", "under")
    elif "under" in market.outcome_label.lower():
        opposite_label = market.outcome_label.lower().replace("under", "over")

    if opposite_label:
        for m in all_markets:
            if (m.market_type == market.market_type and
                m.outcome_label.lower() == opposite_label and
                m.book == market.book):
                return float(m.odds_decimal)

    # Default to same odds if not found (assumes even odds)
    return float(market.odds_decimal)


def _get_edge_band(raw_edge: float) -> str:
    """Categorize raw edge into bands for analysis."""
    if raw_edge < 0:
        return "negative"
    elif raw_edge < 0.02:
        return "0-2%"
    elif raw_edge < 0.05:
        return "2-5%"
    elif raw_edge < 0.10:
        return "5-10%"
    else:
        return "10%+"


def run_pre_game_scoring() -> dict:
    """
    Calculate Value Scores for all active markets.

    This task runs every 10 minutes during game hours and:
    1. Fetches all markets with games starting in next 24 hours
    2. Gets latest model predictions
    3. Calculates confidence and market quality factors
    4. Computes Value Scores for both algorithms
    5. Stores results in value_scores table
    """
    logger.info("Starting pre-game scoring task")

    result = asyncio.run(_run_pre_game_scoring_async())

    logger.info("Completed pre-game scoring", **result)
    return result

# Register as celery task if celery is available
if celery_app:
    run_pre_game_scoring = celery_app.task(name="src.tasks.scoring.run_pre_game_scoring")(run_pre_game_scoring)


async def _score_single_market_async(market_id: str) -> dict:
    """Score a single market by ID."""
    scoring_service = get_scoring_service()

    async with async_session() as session:
        # Get the market with its game
        market_query = (
            select(Market)
            .where(Market.market_id == market_id)
            .options(selectinload(Market.game))
        )
        result = await session.execute(market_query)
        market = result.scalar_one_or_none()

        if not market:
            return {"market_id": market_id, "status": "not_found", "error": "Market not found"}

        game = market.game
        today = datetime.now(timezone.utc).date()

        # Get team stats
        home_stats_query = (
            select(TeamStats)
            .where(TeamStats.team_id == game.home_team_id)
            .where(TeamStats.stat_date <= today)
            .order_by(TeamStats.stat_date.desc())
            .limit(1)
        )
        home_result = await session.execute(home_stats_query)
        home_stats = home_result.scalar_one_or_none()

        away_stats_query = (
            select(TeamStats)
            .where(TeamStats.team_id == game.away_team_id)
            .where(TeamStats.stat_date <= today)
            .order_by(TeamStats.stat_date.desc())
            .limit(1)
        )
        away_result = await session.execute(away_stats_query)
        away_stats = away_result.scalar_one_or_none()

        # Build features
        home_features = _stats_to_features(home_stats, "home")
        away_features = _stats_to_features(away_stats, "away")

        # Get all markets for opposite odds
        markets_query = select(Market).where(Market.game_id == game.game_id)
        markets_result = await session.execute(markets_query)
        all_markets = list(markets_result.scalars().all())

        opposite_odds = _find_opposite_odds(market, all_markets)

        # Fetch injury reports for this game's teams (position-aware)
        try:
            injury_reports = await get_all_team_injury_reports()
            home_injury_report = injury_reports.get(game.home_team_id)
            away_injury_report = injury_reports.get(game.away_team_id)

            # Spread/ML injury scores
            home_injury_score = home_injury_report.spread_injury_score if home_injury_report else 0.0
            away_injury_score = away_injury_report.spread_injury_score if away_injury_report else 0.0

            # Totals-specific injury scores (weighted toward centers)
            home_totals_injury = home_injury_report.totals_injury_score if home_injury_report else 0.0
            away_totals_injury = away_injury_report.totals_injury_score if away_injury_report else 0.0
        except Exception:
            home_injury_score = 0.0
            away_injury_score = 0.0
            home_totals_injury = 0.0
            away_totals_injury = 0.0

        # Create scoring input with position-aware injury data
        scoring_input = ScoringInput(
            game_id=game.game_id,
            market_type=market.market_type,
            outcome_label=market.outcome_label,
            line=float(market.line) if market.line else None,
            odds_decimal=float(market.odds_decimal),
            opposite_odds=opposite_odds,
            home_features=home_features,
            away_features=away_features,
            tip_time=game.tip_time_utc,
            book=market.book,
            home_injury_score=home_injury_score,
            away_injury_score=away_injury_score,
            home_totals_injury=home_totals_injury,
            away_totals_injury=away_totals_injury,
        )

        # Run scoring
        score_result = scoring_service.score_market(scoring_input)

        return {
            "market_id": market_id,
            "status": "scored",
            "algo_a_value_score": score_result.algo_a.value_score,
            "algo_b_value_score": score_result.algo_b.value_score,
            "p_true": score_result.p_calibrated,
            "p_market": score_result.p_market,
            "raw_edge": score_result.raw_edge,
        }


def score_single_market(market_id: str) -> dict:
    """
    Calculate Value Score for a single market.

    Useful for on-demand scoring when odds change significantly.
    """
    logger.info("Scoring single market", market_id=market_id)

    result = asyncio.run(_score_single_market_async(market_id))

    return result

# Register as celery task if celery is available
if celery_app:
    score_single_market = celery_app.task(name="src.tasks.scoring.score_single_market")(score_single_market)
