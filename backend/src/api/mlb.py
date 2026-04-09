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

    # First inning data (if completed)
    home_first_inning_runs: int | None = None
    away_first_inning_runs: int | None = None

    class Config:
        from_attributes = True


class FirstInningTeamStats(BaseModel):
    """First inning scoring stats for a team (both sides of the ball)."""
    team: str
    games: int
    # Offensive: how often team scores in 1st as batters
    scored: int
    scoreless: int
    score_pct: float
    avg_runs: float
    # Defensive: how often opponents score against them in 1st
    runs_allowed: int
    no_runs_allowed: int
    opp_score_pct: float
    avg_runs_allowed: float


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

        if not games:
            return MLBGamesResponse(games=[], total=0, date=target_date.isoformat())

        # Batch-fetch all related data to avoid N+1 queries
        game_ids = [g.game_id for g in games]
        pitcher_ids = set()
        for g in games:
            if g.home_starter_id:
                pitcher_ids.add(g.home_starter_id)
            if g.away_starter_id:
                pitcher_ids.add(g.away_starter_id)

        # Fetch all pitchers, contexts, markets, and snapshots in bulk
        pitchers_map = {}
        if pitcher_ids:
            p_result = await session.execute(
                select(MLBPitcher).where(MLBPitcher.pitcher_id.in_(pitcher_ids))
            )
            pitchers_map = {p.pitcher_id: p for p in p_result.scalars().all()}

        ctx_result = await session.execute(
            select(MLBGameContext).where(MLBGameContext.game_id.in_(game_ids))
        )
        contexts_map = {c.game_id: c for c in ctx_result.scalars().all()}

        mkt_result = await session.execute(
            select(MLBMarket).where(MLBMarket.game_id.in_(game_ids))
        )
        markets_map: dict[str, list] = {gid: [] for gid in game_ids}
        for m in mkt_result.scalars().all():
            markets_map[m.game_id].append(m)

        snap_result = await session.execute(
            select(MLBPredictionSnapshot).where(
                MLBPredictionSnapshot.game_id.in_(game_ids)
            ).order_by(desc(MLBPredictionSnapshot.snapshot_time))
        )
        snapshots_map: dict[str, MLBPredictionSnapshot] = {}
        for s in snap_result.scalars().all():
            if s.game_id not in snapshots_map:  # Keep latest per game
                snapshots_map[s.game_id] = s

        # Fetch predictions (fallback when no snapshot exists yet)
        pred_result = await session.execute(
            select(MLBPrediction).where(
                and_(
                    MLBPrediction.game_id.in_(game_ids),
                    MLBPrediction.market_type == "moneyline",
                )
            )
        )
        predictions_map = {p.game_id: p for p in pred_result.scalars().all()}

        # Fetch pitcher stats in bulk
        pitcher_stats_map = {}
        if pitcher_ids:
            from sqlalchemy import tuple_
            ps_result = await session.execute(
                select(MLBPitcherStats).where(
                    MLBPitcherStats.pitcher_id.in_(pitcher_ids)
                ).order_by(desc(MLBPitcherStats.stat_date))
            )
            for ps in ps_result.scalars().all():
                if ps.pitcher_id not in pitcher_stats_map:
                    pitcher_stats_map[ps.pitcher_id] = ps

        # Build responses using pre-fetched data
        response_games = []
        for game in games:
            game_response = _build_game_response_fast(
                game, pitchers_map, pitcher_stats_map,
                contexts_map, markets_map, snapshots_map,
                predictions_map,
            )
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


@router.get("/stats/first-inning")
async def get_first_inning_stats() -> list[FirstInningTeamStats]:
    """
    Get first inning scoring stats for all teams.

    Returns teams sorted by first inning scoring percentage (descending).
    """
    from sqlalchemy import text

    async with async_session() as session:
        # Use raw SQL to aggregate without loading all game objects
        result = await session.execute(text("""
            SELECT
                team,
                COUNT(*) AS games,
                SUM(CASE WHEN runs_for > 0 THEN 1 ELSE 0 END) AS scored,
                SUM(CASE WHEN runs_for = 0 THEN 1 ELSE 0 END) AS scoreless,
                SUM(runs_for) AS total_runs_for,
                SUM(CASE WHEN runs_against > 0 THEN 1 ELSE 0 END) AS runs_allowed,
                SUM(CASE WHEN runs_against = 0 THEN 1 ELSE 0 END) AS no_runs_allowed,
                SUM(runs_against) AS total_runs_against
            FROM (
                SELECT
                    home_team AS team,
                    COALESCE(home_first_inning_runs, 0) AS runs_for,
                    COALESCE(away_first_inning_runs, 0) AS runs_against
                FROM mlb_games
                WHERE status = 'final' AND game_type = 'R'
                  AND home_first_inning_runs IS NOT NULL
                UNION ALL
                SELECT
                    away_team AS team,
                    COALESCE(away_first_inning_runs, 0) AS runs_for,
                    COALESCE(home_first_inning_runs, 0) AS runs_against
                FROM mlb_games
                WHERE status = 'final' AND game_type = 'R'
                  AND away_first_inning_runs IS NOT NULL
            ) t
            GROUP BY team
            ORDER BY SUM(CASE WHEN runs_against > 0 THEN 1 ELSE 0 END)::float / COUNT(*) ASC
        """))

        rows = result.fetchall()
        if not rows:
            return []

        response = []
        for row in rows:
            (team, games, scored, scoreless, total_runs_for,
             runs_allowed, no_runs_allowed, total_runs_against) = row
            games = int(games)
            scored = int(scored)
            scoreless = int(scoreless)
            total_runs_for = int(total_runs_for)
            runs_allowed = int(runs_allowed)
            no_runs_allowed = int(no_runs_allowed)
            total_runs_against = int(total_runs_against)

            response.append(FirstInningTeamStats(
                team=team,
                games=games,
                scored=scored,
                scoreless=scoreless,
                score_pct=round(scored / games, 3) if games > 0 else 0.0,
                avg_runs=round(total_runs_for / games, 2) if games > 0 else 0.0,
                runs_allowed=runs_allowed,
                no_runs_allowed=no_runs_allowed,
                opp_score_pct=round(runs_allowed / games, 3) if games > 0 else 0.0,
                avg_runs_allowed=round(total_runs_against / games, 2) if games > 0 else 0.0,
            ))

        return response


def _build_game_response_fast(
    game: MLBGame,
    pitchers_map: dict,
    pitcher_stats_map: dict,
    contexts_map: dict,
    markets_map: dict,
    snapshots_map: dict,
    predictions_map: dict | None = None,
) -> MLBGameResponse:
    """Build game response using pre-fetched data (no DB queries)."""
    # Starters
    home_starter = None
    away_starter = None

    if game.home_starter_id and game.home_starter_id in pitchers_map:
        p = pitchers_map[game.home_starter_id]
        stats = pitcher_stats_map.get(p.pitcher_id)
        home_starter = PitcherInfo(
            pitcher_id=p.pitcher_id, name=p.player_name, team=p.team_abbr,
            throws=p.throws,
            era=float(stats.era) if stats and stats.era else None,
            whip=float(stats.whip) if stats and stats.whip else None,
            k_per_9=float(stats.k_per_9) if stats and stats.k_per_9 else None,
            quality_score=float(stats.quality_score) if stats and stats.quality_score else None,
        )

    if game.away_starter_id and game.away_starter_id in pitchers_map:
        p = pitchers_map[game.away_starter_id]
        stats = pitcher_stats_map.get(p.pitcher_id)
        away_starter = PitcherInfo(
            pitcher_id=p.pitcher_id, name=p.player_name, team=p.team_abbr,
            throws=p.throws,
            era=float(stats.era) if stats and stats.era else None,
            whip=float(stats.whip) if stats and stats.whip else None,
            k_per_9=float(stats.k_per_9) if stats and stats.k_per_9 else None,
            quality_score=float(stats.quality_score) if stats and stats.quality_score else None,
        )

    # Context
    context = None
    ctx = contexts_map.get(game.game_id)
    if ctx:
        context = GameContextInfo(
            venue_name=ctx.venue_name,
            park_factor=float(ctx.park_factor) if ctx.park_factor else None,
            temperature=ctx.temperature, wind_speed=ctx.wind_speed,
            is_dome=ctx.is_dome,
            weather_factor=float(ctx.weather_factor) if ctx.weather_factor else None,
        )

    # Markets
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
        for m in markets_map.get(game.game_id, [])
    ]

    # Snapshot / predictions
    snapshot = snapshots_map.get(game.game_id)
    best_ml = best_rl = best_total = best_bet = None
    predicted_run_diff = predicted_total = p_home_win = p_away_win = None

    # Fallback to MLBPrediction if no snapshot exists yet
    if not snapshot and predictions_map:
        pred = predictions_map.get(game.game_id)
        if pred:
            predicted_run_diff = float(pred.predicted_run_diff) if pred.predicted_run_diff else None
            predicted_total = float(pred.predicted_total) if pred.predicted_total else None
            if pred.p_home_win is not None:
                p_home_win = float(pred.p_home_win)
                p_away_win = float(pred.p_away_win) if pred.p_away_win else None

    if snapshot:
        predicted_run_diff = float(snapshot.predicted_run_diff) if snapshot.predicted_run_diff else None
        predicted_total = float(snapshot.predicted_total) if snapshot.predicted_total else None
        p_home_win = float(snapshot.winner_probability) if snapshot.predicted_winner == game.home_team else None
        p_away_win = float(snapshot.winner_probability) if snapshot.predicted_winner == game.away_team else None

        if snapshot.best_ml_team and snapshot.best_ml_value_score:
            best_ml = ValueBetInfo(
                market_type="moneyline",
                bet_type="home_ml" if snapshot.best_ml_team == game.home_team else "away_ml",
                team=snapshot.best_ml_team, line=None,
                odds_decimal=float(snapshot.best_ml_odds) if snapshot.best_ml_odds else 2.0,
                odds_american=_decimal_to_american(float(snapshot.best_ml_odds) if snapshot.best_ml_odds else 2.0),
                model_prob=0.5, market_prob=0.5,
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
                model_prob=0.5, market_prob=0.5,
                edge=float(snapshot.best_rl_edge) if snapshot.best_rl_edge else 0,
                value_score=float(snapshot.best_rl_value_score),
                confidence="high" if snapshot.best_rl_value_score >= 70 else "medium",
            )

        if snapshot.best_total_direction and snapshot.best_total_value_score:
            best_total = ValueBetInfo(
                market_type="total", bet_type=snapshot.best_total_direction, team=None,
                line=float(snapshot.best_total_line) if snapshot.best_total_line else None,
                odds_decimal=float(snapshot.best_total_odds) if snapshot.best_total_odds else 2.0,
                odds_american=_decimal_to_american(float(snapshot.best_total_odds) if snapshot.best_total_odds else 2.0),
                model_prob=0.5, market_prob=0.5,
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
                model_prob=0.5, market_prob=0.5,
                edge=float(snapshot.best_bet_edge) if snapshot.best_bet_edge else 0,
                value_score=float(snapshot.best_bet_value_score),
                confidence="high" if snapshot.best_bet_value_score >= 70 else "medium",
            )

    return MLBGameResponse(
        game_id=game.game_id,
        game_date=game.game_date.isoformat(),
        game_time=game.game_time,
        home_team=game.home_team, away_team=game.away_team,
        status=game.status,
        home_starter=home_starter, away_starter=away_starter,
        context=context, markets=markets,
        predicted_run_diff=predicted_run_diff, predicted_total=predicted_total,
        p_home_win=p_home_win, p_away_win=p_away_win,
        best_ml=best_ml, best_rl=best_rl, best_total=best_total, best_bet=best_bet,
        home_score=game.home_score, away_score=game.away_score,
        home_first_inning_runs=game.home_first_inning_runs,
        away_first_inning_runs=game.away_first_inning_runs,
    )


# Helper functions (kept for single-game endpoint)

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
        home_first_inning_runs=game.home_first_inning_runs,
        away_first_inning_runs=game.away_first_inning_runs,
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


@router.get("/debug/odds")
async def debug_odds() -> dict:
    """Debug endpoint: check odds ingestion status without calling external APIs."""
    from sqlalchemy import text
    from src.config import settings

    results = {}

    # Check API key
    results["odds_api_key_set"] = bool(settings.odds_api_key)
    results["odds_api_key_prefix"] = settings.odds_api_key[:8] + "..." if settings.odds_api_key else None

    # Check existing data
    async with async_session() as session:
        row = await session.execute(text("SELECT COUNT(*) FROM mlb_markets"))
        results["existing_markets"] = row.scalar()

        row = await session.execute(text("SELECT COUNT(*) FROM mlb_games"))
        results["existing_games"] = row.scalar()

        row = await session.execute(text("SELECT COUNT(*) FROM mlb_predictions"))
        results["mlb_predictions_count"] = row.scalar()

        row = await session.execute(text("SELECT COUNT(*) FROM mlb_prediction_snapshots"))
        results["mlb_snapshots_count"] = row.scalar()

        # Most recent prediction
        row = await session.execute(text(
            "SELECT game_id, market_type, predicted_run_diff, created_at FROM mlb_predictions ORDER BY created_at DESC LIMIT 3"
        ))
        results["recent_predictions"] = [
            {"game_id": r[0], "market": r[1], "run_diff": float(r[2]) if r[2] else None, "created_at": str(r[3])}
            for r in row.fetchall()
        ]

        # Check game dates
        row = await session.execute(text(
            "SELECT game_date, home_team, away_team, status FROM mlb_games ORDER BY game_date DESC LIMIT 5"
        ))
        results["recent_games"] = [
            {"date": str(r[0]), "home": r[1], "away": r[2], "status": r[3]}
            for r in row.fetchall()
        ]

    return results


@router.get("/debug/odds/ingest")
async def debug_odds_ingest() -> dict:
    """Debug endpoint: actually try odds ingestion and report results."""
    from src.services.mlb.ingest import MLBDataIngestor, MLBOddsClient

    results = {}

    # Try fetching odds
    try:
        client = MLBOddsClient()
        odds_data = await client.get_mlb_odds()
        results["odds_api_games"] = len(odds_data)
        if odds_data:
            results["odds_sample"] = {
                "home": odds_data[0].get("home_team"),
                "away": odds_data[0].get("away_team"),
                "commence_time": odds_data[0].get("commence_time"),
                "bookmakers_count": len(odds_data[0].get("bookmakers", [])),
            }
    except Exception as e:
        results["odds_api_error"] = str(e)

    # Try full ingestion
    try:
        async with async_session() as session:
            ingestor = MLBDataIngestor(session)
            count = await ingestor.ingest_odds()
            results["markets_ingested"] = count
    except Exception as e:
        results["ingestion_error"] = str(e)
        import traceback
        results["ingestion_traceback"] = traceback.format_exc()

    return results


@router.get("/debug/social-config")
async def debug_social_config() -> dict:
    """Show which social posting services are configured."""
    from src.config import settings
    return {
        "twitter_posting_enabled": settings.twitter_posting_enabled,
        "blotato_api_key_set": bool(settings.blotato_api_key),
        "blotato_api_key_prefix": (settings.blotato_api_key[:8] + "...") if settings.blotato_api_key else None,
        "blotato_twitter_account_id": settings.blotato_twitter_account_id or None,
        "typefully_api_key_set": bool(settings.typefully_api_key),
    }


@router.get("/debug/tweet-test")
async def debug_tweet_test() -> dict:
    """Send a test intro tweet via Blotato."""
    from src.services.social.blotato import post_tweet

    intro = (
        "Introducing TruLine\n\n"
        "AI-powered MLB & NBA value bets.\n"
        "Every pick scored, tracked, and graded.\n\n"
        "- 28+ features per game\n"
        "- 11 sportsbooks compared\n"
        "- NRFI specialist\n\n"
        "Free daily picks at truline.app\n\n"
        "#MLB #NBA #NRFI #GamblingX"
    )

    result = post_tweet(intro)
    return {
        "tweet_text": intro,
        "tweet_length": len(intro),
        "result": result,
        "posted": result is not None,
    }


@router.get("/debug/trigger-posts")
async def debug_trigger_posts(task: str = "all") -> dict:
    """Manually trigger social scheduler tasks for testing.

    Each task generates and posts content directly (bypassing the
    scheduler's dedicated event loop to avoid cross-loop issues).

    Query params:
        task: "all", "picks", "nrfi", "results", "nrfi_results", "pregame"
    """
    from datetime import timedelta
    from sqlalchemy import text
    from src.services.social.content import (
        generate_results_tweet,
        generate_nrfi_results_tweet,
        generate_daily_picks_thread,
        generate_nrfi_tweet,
        generate_pregame_nrfi_tweet,
        _get_team_first_inning_pct,
        _get_pitcher_era,
        TEAM_NAMES,
        TEAM_HANDLES,
    )
    from src.services.social.blotato import post_tweet, post_thread, upload_media
    from src.services.social.image_generator import generate_nrfi_card

    eastern = timedelta(hours=-5)
    today = (datetime.now(timezone.utc) + eastern).date()
    yesterday = today - timedelta(days=1)

    results = {}

    async with async_session() as session:
        # Results recap (yesterday)
        if task in ("all", "results"):
            try:
                tweet = await generate_results_tweet(session, yesterday)
                if tweet:
                    r = post_tweet(tweet, schedule_at="next-free-slot")
                    results["results"] = {"posted": r is not None, "text": tweet[:100]}
                else:
                    results["results"] = {"posted": False, "reason": "no_data"}
            except Exception as e:
                results["results_error"] = str(e)[:200]

        # NRFI results recap (yesterday)
        if task in ("all", "nrfi_results"):
            try:
                tweet = await generate_nrfi_results_tweet(session, yesterday)
                if tweet:
                    r = post_tweet(tweet, schedule_at="next-free-slot")
                    results["nrfi_results"] = {"posted": r is not None, "text": tweet[:100]}
                else:
                    results["nrfi_results"] = {"posted": False, "reason": "no_data"}
            except Exception as e:
                results["nrfi_results_error"] = str(e)[:200]

        # Daily picks thread (today)
        if task in ("all", "picks"):
            try:
                tweets = await generate_daily_picks_thread(session, today)
                if tweets:
                    r = post_thread(tweets, schedule_at="next-free-slot")
                    results["picks"] = {"posted": r is not None, "count": len(tweets)}
                else:
                    results["picks"] = {"posted": False, "reason": "no_data"}
            except Exception as e:
                results["picks_error"] = str(e)[:200]

        # NRFI plays (today)
        if task in ("all", "nrfi"):
            try:
                tweet = await generate_nrfi_tweet(session, today)
                if tweet:
                    r = post_tweet(tweet, schedule_at="next-free-slot")
                    results["nrfi"] = {"posted": r is not None, "text": tweet[:100]}
                else:
                    results["nrfi"] = {"posted": False, "reason": "no_data"}
            except Exception as e:
                results["nrfi_error"] = str(e)[:200]

        # NBA daily picks
        if task in ("all", "nba_picks"):
            try:
                from src.services.social.content import generate_nba_picks_thread
                tweets = await generate_nba_picks_thread(session, today)
                if tweets and len(tweets) > 1:
                    r = post_thread(tweets, schedule_at="next-free-slot")
                    results["nba_picks"] = {
                        "posted": r is not None,
                        "count": len(tweets),
                        "submission_id": r.get("postSubmissionId") if isinstance(r, dict) else None,
                        "first_tweet": tweets[0][:200],
                    }
                else:
                    results["nba_picks"] = {"posted": False, "reason": "no_data", "tweet_count": len(tweets) if tweets else 0}
            except Exception as e:
                import traceback
                results["nba_picks_error"] = str(e)[:200]
                results["nba_picks_trace"] = traceback.format_exc()[-800:]

        # NBA results (yesterday)
        if task in ("all", "nba_results"):
            try:
                from src.services.social.content import generate_nba_results_tweet
                tweet = await generate_nba_results_tweet(session, yesterday)
                if tweet:
                    r = post_tweet(tweet, schedule_at="next-free-slot")
                    results["nba_results"] = {"posted": r is not None, "text": tweet[:100]}
                else:
                    results["nba_results"] = {"posted": False, "reason": "no_data"}
            except Exception as e:
                results["nba_results_error"] = str(e)[:200]

        # First inning recaps (games whose 1st inning just ended)
        if task in ("all", "first_inning"):
            from sqlalchemy import or_
            from src.services.social.content import generate_first_inning_recap_tweet
            from src.services.social.image_generator import generate_recap_card

            stmt = select(MLBGame).where(
                and_(
                    MLBGame.game_date == today,
                    MLBGame.first_inning_tweet_posted == False,  # noqa: E712
                    MLBGame.home_first_inning_runs.isnot(None),
                    MLBGame.away_first_inning_runs.isnot(None),
                    or_(
                        and_(MLBGame.status == "in_progress", MLBGame.inning >= 2),
                        MLBGame.status == "final",
                    ),
                )
            )
            games = list((await session.execute(stmt)).scalars().all())
            posted = 0
            for game in games:
                try:
                    tweet = generate_first_inning_recap_tweet(game)
                    if not tweet:
                        continue

                    media_urls = None
                    try:
                        away_fi = game.away_first_inning_runs or 0
                        home_fi = game.home_first_inning_runs or 0
                        is_nrfi = (away_fi + home_fi) == 0
                        home_off, home_def = await _get_team_first_inning_pct(session, game.home_team)
                        away_off, away_def = await _get_team_first_inning_pct(session, game.away_team)
                        predicted_nrfi_pct = None
                        if all(x is not None for x in [home_off, home_def, away_off, away_def]):
                            p_away_scores = (away_off + home_def) / 2.0
                            p_home_scores = (home_off + away_def) / 2.0
                            predicted_nrfi_pct = (1.0 - p_away_scores) * (1.0 - p_home_scores) * 100
                        png = generate_recap_card(
                            away_team=game.away_team,
                            home_team=game.home_team,
                            away_name=TEAM_NAMES.get(game.away_team, game.away_team),
                            home_name=TEAM_NAMES.get(game.home_team, game.home_team),
                            away_first=away_fi,
                            home_first=home_fi,
                            is_nrfi=is_nrfi,
                            predicted_nrfi_pct=predicted_nrfi_pct,
                        )
                        public_url = upload_media(png, filename=f"recap_{game.game_id}.png")
                        if public_url:
                            media_urls = [public_url]
                    except Exception as img_err:
                        results.setdefault("image_errors", []).append(str(img_err)[:150])

                    r = post_tweet(tweet, schedule_at="next-free-slot", media_urls=media_urls)
                    if r:
                        await session.execute(
                            text("UPDATE mlb_games SET first_inning_tweet_posted = TRUE WHERE game_id = :gid"),
                            {"gid": game.game_id},
                        )
                        posted += 1
                except Exception as e:
                    results.setdefault("first_inning_errors", []).append(str(e)[:100])
            await session.commit()
            results["first_inning"] = {"posted": posted, "checked": len(games)}

        # Final recaps (games just finished)
        if task in ("all", "final"):
            from src.services.social.content import generate_final_recap_tweet

            stmt = select(MLBGame).where(
                and_(
                    MLBGame.game_date == today,
                    MLBGame.status == "final",
                    MLBGame.final_tweet_posted == False,  # noqa: E712
                    MLBGame.home_score.isnot(None),
                    MLBGame.away_score.isnot(None),
                )
            )
            games = list((await session.execute(stmt)).scalars().all())
            posted = 0
            for game in games:
                try:
                    tweet = generate_final_recap_tweet(game)
                    if not tweet:
                        continue
                    r = post_tweet(tweet, schedule_at="next-free-slot")
                    if r:
                        await session.execute(
                            text("UPDATE mlb_games SET final_tweet_posted = TRUE WHERE game_id = :gid"),
                            {"gid": game.game_id},
                        )
                        posted += 1
                except Exception as e:
                    results.setdefault("final_errors", []).append(str(e)[:100])
            await session.commit()
            results["final"] = {"posted": posted, "checked": len(games)}

    return results


@router.get("/debug/version")
async def debug_version() -> dict:
    """Returns a unique version string to verify deployments."""
    return {"version": "2026-04-07-v4-add-constraint", "commit": "latest"}


@router.get("/debug/backfill-predictions")
async def debug_backfill_predictions(days: int = 14) -> dict:
    """Score completed games retroactively to build historical track record."""
    from datetime import timedelta
    from src.services.mlb.scorer import MLBScorer

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    results = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "total_games": 0,
        "scored": 0,
        "errors": [],
    }

    async with async_session() as session:
        stmt = select(MLBGame).where(
            and_(
                MLBGame.game_date >= start_date,
                MLBGame.game_date <= end_date,
                MLBGame.status == "final",
                MLBGame.home_score.isnot(None),
            )
        ).order_by(MLBGame.game_date)

        result = await session.execute(stmt)
        games = list(result.scalars().all())
        results["total_games"] = len(games)

        scorer = MLBScorer(session)
        predictions = []

        for game in games:
            try:
                prediction = await scorer.score_game(game)
                predictions.append(prediction)
                results["scored"] += 1
            except Exception as e:
                results["errors"].append(f"{game.game_id}: {str(e)[:100]}")

        if predictions:
            try:
                saved = await scorer.save_predictions(predictions)
                results["saved_rows"] = saved
            except Exception as e:
                results["save_error"] = str(e)[:200]

        # Keep first 5 errors only
        if len(results["errors"]) > 5:
            results["total_errors"] = len(results["errors"])
            results["errors"] = results["errors"][:5]

    return results


@router.get("/debug/pregame-tweets")
async def debug_pregame_tweets(dry_run: bool = True) -> dict:
    """Generate pregame tweets for top NRFI picks (dry-run by default)."""
    from src.services.social.content import generate_pregame_nrfi_tweet, _get_team_first_inning_pct
    from datetime import timedelta

    eastern = timedelta(hours=-5)
    today = (datetime.now(timezone.utc) + eastern).date()

    results = {"today_et": str(today), "tweets": []}

    async with async_session() as session:
        stmt = select(MLBGame).where(
            and_(
                MLBGame.game_date == today,
                MLBGame.status == "scheduled",
            )
        ).order_by(MLBGame.game_time)
        games = (await session.execute(stmt)).scalars().all()

        for game in games:
            tweet = await generate_pregame_nrfi_tweet(session, game)
            if tweet:
                results["tweets"].append({
                    "game": f"{game.away_team} @ {game.home_team}",
                    "length": len(tweet),
                    "text": tweet,
                })

    return results


def _decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American format."""
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))
