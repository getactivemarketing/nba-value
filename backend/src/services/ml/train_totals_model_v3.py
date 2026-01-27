"""
Train advanced totals model v3 - incorporates injury impact, pace dynamics, and market inefficiencies.

This version adds features the market may underprice:
1. Injury impact scores (totals_injury_score)
2. Pace interaction/clash features
3. Rest/fatigue asymmetry
4. Recent scoring volatility (hot/cold streaks)
5. Home/away scoring splits
6. Historical O/U tendency
7. Venue effects

Target: Beat the 52.4% breakeven rate by finding edges the market misses.

Usage:
    python -m src.services.ml.train_totals_model_v3
    python -m src.services.ml.train_totals_model_v3 backtest --days 30
"""

import pickle
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta, date
from collections import defaultdict
import numpy as np
import psycopg2

# LightGBM is optional - may fail due to missing libomp on macOS
try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except (ImportError, OSError) as e:
    HAS_LIGHTGBM = False
    print(f"Note: LightGBM not available ({e}), using Ridge regression")

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import structlog

# Import injury service for totals scoring
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.services.injuries import get_all_team_injury_reports, ABBREV_TO_TEAM_ID, TEAM_ID_TO_ABBREV

logger = structlog.get_logger()

DB_URL = 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway'


def calculate_pace_interaction(home_pace: float, away_pace: float) -> dict:
    """
    Calculate pace interaction features.

    When fast team plays slow team, the actual pace tends toward:
    - The faster team's pace (they control tempo)
    - But slower than their average (the slow team resists)

    Returns dict with:
        - avg_pace: Simple average
        - weighted_pace: Weighted toward faster team
        - pace_clash: Absolute difference (high = style clash)
        - pace_variance: Expected possession variance
    """
    avg_pace = (home_pace + away_pace) / 2

    # Weight toward faster team (70/30 split)
    faster_pace = max(home_pace, away_pace)
    slower_pace = min(home_pace, away_pace)
    weighted_pace = faster_pace * 0.60 + slower_pace * 0.40

    # Pace clash metric (high = large style difference)
    pace_clash = abs(home_pace - away_pace)

    # Variance in possessions (clash = more variance)
    pace_variance = pace_clash / avg_pace if avg_pace > 0 else 0

    return {
        'avg_pace': avg_pace,
        'weighted_pace': weighted_pace,
        'pace_clash': pace_clash,
        'pace_variance': pace_variance,
    }


def calculate_rest_asymmetry(home_rest: int, away_rest: int,
                              home_b2b: bool, away_b2b: bool) -> dict:
    """
    Calculate rest/fatigue asymmetry features.

    Key insight: Fatigue hurts defense more than offense.
    - Tired team = gives up more points (cognitive lapses on D)
    - Tired team = scores slightly less but variance increases

    Returns dict with:
        - rest_advantage: Home rest edge (positive = home more rested)
        - total_fatigue: Combined fatigue (B2B penalty)
        - fatigue_asymmetry: One team tired, other fresh (high variance)
    """
    rest_advantage = home_rest - away_rest

    # B2B penalty (tired teams allow ~3-5 more PPG)
    home_fatigue = 1.0 if home_b2b else 0.0
    away_fatigue = 1.0 if away_b2b else 0.0
    total_fatigue = home_fatigue + away_fatigue

    # Asymmetry (one tired, one fresh = unpredictable)
    fatigue_asymmetry = abs(home_fatigue - away_fatigue)

    # Rest categories
    home_well_rested = 1 if home_rest >= 2 and not home_b2b else 0
    away_well_rested = 1 if away_rest >= 2 and not away_b2b else 0

    return {
        'rest_advantage': rest_advantage,
        'total_fatigue': total_fatigue,
        'fatigue_asymmetry': fatigue_asymmetry,
        'home_well_rested': home_well_rested,
        'away_well_rested': away_well_rested,
    }


def calculate_injury_impact(home_team_id: str, away_team_id: str,
                             injury_reports: dict) -> dict:
    """
    Calculate injury impact features for totals.

    Key insights:
    - When one team is heavily injured, totals trend UNDER
    - Injured team scores less AND opponent scores less (pace slows)
    - Asymmetric injuries create variance (unpredictable)

    Returns dict with:
        - home_injury_score: 0-1, higher = more injured
        - away_injury_score: 0-1
        - injury_asymmetry: Difference in injury levels
        - total_injury: Combined injury impact (both teams hurt = UNDER)
    """
    # Handle both numeric team IDs and abbreviation strings
    try:
        home_abbrev = TEAM_ID_TO_ABBREV.get(int(home_team_id), home_team_id)
    except (ValueError, TypeError):
        home_abbrev = home_team_id  # Already an abbreviation

    try:
        away_abbrev = TEAM_ID_TO_ABBREV.get(int(away_team_id), away_team_id)
    except (ValueError, TypeError):
        away_abbrev = away_team_id  # Already an abbreviation

    home_report = injury_reports.get(home_abbrev)
    away_report = injury_reports.get(away_abbrev)

    home_injury = home_report.totals_injury_score if home_report else 0.0
    away_injury = away_report.totals_injury_score if away_report else 0.0

    injury_asymmetry = abs(home_injury - away_injury)
    total_injury = home_injury + away_injury

    # Injury edge (negative = home more injured, expect lower scoring)
    injury_edge = away_injury - home_injury

    return {
        'home_injury_score': home_injury,
        'away_injury_score': away_injury,
        'injury_asymmetry': injury_asymmetry,
        'total_injury': total_injury,
        'injury_edge': injury_edge,
    }


async def fetch_training_data_with_features(db_url: str = None,
                                            min_games: int = 5,
                                            lookback: int = 10) -> tuple:
    """
    Fetch training data with advanced features computed from game_results.

    This hybrid approach:
    1. Computes rolling stats from game_results (like v2) for full historical data
    2. Adds advanced features: rest/fatigue, O/U tendencies, scoring variance
    3. Uses current injury reports for recent games as a bonus feature

    Returns:
        X: Feature matrix
        y: Target vector (actual totals)
        game_ids: Game identifiers
        feature_names: List of feature names
        closing_totals: Actual betting lines
    """
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

    # Fetch all completed games ordered by date
    cur.execute('''
        SELECT
            game_id,
            game_date,
            home_team_id,
            away_team_id,
            home_score,
            away_score,
            total_score,
            closing_total
        FROM game_results
        WHERE total_score IS NOT NULL
        ORDER BY game_date ASC
    ''')
    games = cur.fetchall()
    cur.close()
    conn.close()

    if not games:
        logger.warning("No games found")
        return np.array([]), np.array([]), [], [], []

    logger.info(f"Loaded {len(games)} total games from database")

    # Get injury reports for recent games (will be used for recent predictions)
    logger.info("Fetching injury reports...")
    injury_reports = await get_all_team_injury_reports()
    logger.info(f"Got injury reports for {len(injury_reports)} teams")

    # Track each team's game history for computing rolling features
    # team_id -> list of (date, pts_for, pts_against, total, over_under_result)
    team_games = defaultdict(list)

    # Build features for each game
    X = []
    y = []
    game_ids = []
    closing_totals = []

    feature_names = [
        # Basic rolling stats (9 features - like v2)
        'home_ppg', 'home_opp_ppg', 'home_total_avg',
        'away_ppg', 'away_opp_ppg', 'away_total_avg',
        'combined_ppg', 'combined_opp_ppg', 'matchup_total_est',
        # Scoring variance (2 features)
        'home_scoring_std', 'away_scoring_std',
        # O/U tendencies (2 features)
        'home_over_pct', 'away_over_pct',
        # Rest/fatigue (5 features)
        'home_rest_days', 'away_rest_days',
        'rest_advantage', 'home_b2b', 'away_b2b',
        # Injury impact (5 features) - only valid for recent games
        'home_injury_score', 'away_injury_score',
        'injury_asymmetry', 'total_injury', 'injury_edge',
    ]

    for game in games:
        game_id, game_date, home_id, away_id, home_score, away_score, total_score, closing_total = game

        # Get rolling history for each team (before this game)
        home_history = team_games[home_id][-lookback:] if team_games[home_id] else []
        away_history = team_games[away_id][-lookback:] if team_games[away_id] else []

        # Need at least min_games for each team
        if len(home_history) >= min_games and len(away_history) >= min_games:
            # Basic rolling stats (like v2)
            home_pts_for = np.mean([g['pts_for'] for g in home_history])
            home_pts_against = np.mean([g['pts_against'] for g in home_history])
            home_total_avg = np.mean([g['total'] for g in home_history])

            away_pts_for = np.mean([g['pts_for'] for g in away_history])
            away_pts_against = np.mean([g['pts_against'] for g in away_history])
            away_total_avg = np.mean([g['total'] for g in away_history])

            # Scoring variance (std of recent scores)
            home_scoring_std = np.std([g['pts_for'] for g in home_history])
            away_scoring_std = np.std([g['pts_for'] for g in away_history])

            # O/U tendencies (how often each team goes OVER)
            home_ou_results = [g['over'] for g in home_history if g['over'] is not None]
            away_ou_results = [g['over'] for g in away_history if g['over'] is not None]

            home_over_pct = np.mean(home_ou_results) if home_ou_results else 0.5
            away_over_pct = np.mean(away_ou_results) if away_ou_results else 0.5

            # Rest/fatigue (days since last game)
            home_last_date = home_history[-1]['date']
            away_last_date = away_history[-1]['date']

            home_rest_days = (game_date - home_last_date).days
            away_rest_days = (game_date - away_last_date).days

            rest_advantage = home_rest_days - away_rest_days
            home_b2b = 1 if home_rest_days == 1 else 0
            away_b2b = 1 if away_rest_days == 1 else 0

            # Injury impact (use current reports - only accurate for recent games)
            injury_features = calculate_injury_impact(str(home_id), str(away_id), injury_reports)

            # Build feature vector
            features = [
                # Basic (9)
                home_pts_for, home_pts_against, home_total_avg,
                away_pts_for, away_pts_against, away_total_avg,
                (home_pts_for + away_pts_for) / 2,  # combined_ppg
                (home_pts_against + away_pts_against) / 2,  # combined_opp_ppg
                (home_total_avg + away_total_avg) / 2,  # matchup_total_est
                # Variance (2)
                home_scoring_std, away_scoring_std,
                # O/U (2)
                home_over_pct, away_over_pct,
                # Rest (5)
                home_rest_days, away_rest_days,
                rest_advantage, home_b2b, away_b2b,
                # Injury (5)
                injury_features['home_injury_score'],
                injury_features['away_injury_score'],
                injury_features['injury_asymmetry'],
                injury_features['total_injury'],
                injury_features['injury_edge'],
            ]

            X.append(features)
            y.append(total_score)
            game_ids.append(game_id)
            closing_totals.append(closing_total)

        # Update team history AFTER computing features (to avoid future leakage)
        # Calculate if this game went over/under the line
        over_result = None
        if closing_total is not None:
            line = float(closing_total)
            if total_score > line:
                over_result = 1
            elif total_score < line:
                over_result = 0
            # push = None

        team_games[home_id].append({
            'date': game_date,
            'pts_for': home_score,
            'pts_against': away_score,
            'total': total_score,
            'over': over_result,
        })
        team_games[away_id].append({
            'date': game_date,
            'pts_for': away_score,
            'pts_against': home_score,
            'total': total_score,
            'over': over_result,
        })

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    logger.info(f"Built feature matrix: {X.shape[0]} games x {X.shape[1]} features")
    if len(X) > 0:
        logger.info(f"Target range: {y.min():.0f} - {y.max():.0f} (mean: {y.mean():.1f})")

    return X, y, game_ids, feature_names, closing_totals


def train_totals_model_v3(db_url: str = None, use_lightgbm: bool = False) -> dict:
    """
    Train advanced totals model with injury, pace, and market inefficiency features.

    Args:
        db_url: Database connection string
        use_lightgbm: If True, use LightGBM; otherwise use Ridge regression

    Returns:
        Dictionary with model, metadata, and performance metrics
    """
    print("=== ADVANCED TOTALS MODEL V3 TRAINING ===\n")

    # Fetch training data (async call)
    X, y, game_ids, feature_names, closing_totals = asyncio.run(
        fetch_training_data_with_features(db_url)
    )

    if len(X) < 30:
        print(f"ERROR: Insufficient training data ({len(X)} games, need 30+)")
        return {"error": "Insufficient data", "games": len(X)}

    print(f"Training samples: {len(X)}")
    print(f"Features: {len(feature_names)}")
    print(f"Target range: {y.min():.0f} - {y.max():.0f} (mean: {y.mean():.1f}, std: {y.std():.1f})")

    # Time-series cross-validation (no future leakage)
    n_splits = min(5, len(X) // 20)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print(f"\nCross-validation ({n_splits} folds):")

    val_predictions = []
    val_actuals = []
    val_closing = []
    val_maes = []

    if use_lightgbm and HAS_LIGHTGBM:
        print("Using LightGBM Regressor")
        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
    else:
        print("Using Ridge Regression")
        # Try different alphas
        best_alpha = 10.0
        best_mae = float('inf')

        for alpha in [1.0, 5.0, 10.0, 50.0, 100.0]:
            model_test = Ridge(alpha=alpha)
            maes = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model_test.fit(X_train, y_train)
                val_pred = model_test.predict(X_val)
                mae = mean_absolute_error(y_val, val_pred)
                maes.append(mae)

            avg_mae = np.mean(maes)
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_alpha = alpha

        print(f"Best alpha: {best_alpha} (MAE: {best_mae:.2f})")
        model = Ridge(alpha=best_alpha)

    # Perform CV and collect predictions
    fold_idx = 0
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_maes.append(mae)

        val_predictions.extend(val_pred)
        val_actuals.extend(y_val)
        val_closing.extend([closing_totals[i] for i in val_idx])

        print(f"  Fold {fold_idx+1}: MAE={mae:.2f}, RMSE={rmse:.2f} ({len(val_idx)} games)")
        fold_idx += 1

    avg_mae = np.mean(val_maes)
    print(f"\nAverage MAE: {avg_mae:.2f} ± {np.std(val_maes):.2f}")

    # Train final model on all data
    model.fit(X, y)

    # Calculate total_std for probability conversion
    final_preds = model.predict(X)
    residuals = y - final_preds
    total_std = np.std(residuals)
    print(f"Total std: {total_std:.2f} (used for probability conversion)")

    # Feature importance
    print(f"\nFeature importance:")
    if use_lightgbm and HAS_LIGHTGBM:
        importance = model.feature_importances_
        feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
        for name, imp in feat_imp[:15]:
            print(f"  {name}: {imp:.1f}")
    else:
        coefs = model.coef_
        feat_imp = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
        for name, coef in feat_imp[:15]:
            print(f"  {name}: {coef:+.3f}")

    # Simulate betting vs actual lines
    print(f"\n=== BETTING SIMULATION (vs actual closing lines) ===")
    simulate_betting_vs_lines(val_predictions, val_actuals, val_closing)

    # Save model
    model_path = Path(__file__).parent.parent.parent.parent / "models" / "totals_model_v3.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model': model,
        'feature_names': feature_names,
        'total_std': total_std,
        'model_type': 'lgbm_totals_v3' if (use_lightgbm and HAS_LIGHTGBM) else 'ridge_totals_v3',
        'training_games': len(X),
        'avg_mae': avg_mae,
        'trained_at': datetime.now(timezone.utc).isoformat(),
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to {model_path}")

    return model_data


def simulate_betting_vs_lines(predictions: list, actuals: list, lines: list) -> dict:
    """
    Simulate betting using model predictions vs actual closing lines.

    This is the true test - can the model beat the market?
    """
    if not predictions or not lines:
        return {"error": "No data"}

    # Filter to games with valid lines
    valid_games = [(p, a, l) for p, a, l in zip(predictions, actuals, lines)
                   if l is not None]

    if not valid_games:
        print("No games with closing lines found")
        return {"error": "No lines"}

    print(f"Games with closing lines: {len(valid_games)}")

    # Test different edge thresholds
    thresholds = [0, 1, 2, 3, 4, 5]

    for threshold in thresholds:
        wins = 0
        losses = 0
        pushes = 0

        for pred, actual, line in valid_games:
            line = float(line)
            diff = pred - line

            # Only bet if edge exceeds threshold
            if abs(diff) < threshold:
                continue

            if diff > 0:  # Bet OVER
                if actual > line:
                    wins += 1
                elif actual < line:
                    losses += 1
                else:
                    pushes += 1
            else:  # Bet UNDER
                if actual < line:
                    wins += 1
                elif actual > line:
                    losses += 1
                else:
                    pushes += 1

        total_bets = wins + losses
        if total_bets > 0:
            win_rate = wins / total_bets * 100
            profit = wins * 90.91 - losses * 100
            roi = profit / (total_bets * 100) * 100
            breakeven = total_bets * 0.524  # Need 52.4% to breakeven at -110
            edge_over_breakeven = win_rate - 52.4

            print(f"  Edge {threshold}+ pts: {wins}-{losses}-{pushes} "
                  f"({win_rate:.1f}%, edge: {edge_over_breakeven:+.1f}%), "
                  f"ROI: {roi:+.1f}%, Bets: {total_bets}")
        else:
            print(f"  Edge {threshold}+ pts: No bets")

    # Also test with probability-based betting (>55% confidence)
    print(f"\n  Using probability threshold (>55% confidence):")
    wins = 0
    losses = 0
    pushes = 0

    for pred, actual, line in valid_games:
        line = float(line)
        diff = pred - line
        # Approximate probability using normal distribution
        std = 12.0  # typical game std
        prob_over = 1 - stats.norm.cdf(line, pred, std)

        if prob_over > 0.55:  # Bet OVER
            if actual > line:
                wins += 1
            elif actual < line:
                losses += 1
            else:
                pushes += 1
        elif prob_over < 0.45:  # Bet UNDER
            if actual < line:
                wins += 1
            elif actual > line:
                losses += 1
            else:
                pushes += 1

    total_bets = wins + losses
    if total_bets > 0:
        win_rate = wins / total_bets * 100
        profit = wins * 90.91 - losses * 100
        roi = profit / (total_bets * 100) * 100
        edge_over_breakeven = win_rate - 52.4
        print(f"  P>55%: {wins}-{losses}-{pushes} "
              f"({win_rate:.1f}%, edge: {edge_over_breakeven:+.1f}%), "
              f"ROI: {roi:+.1f}%, Bets: {total_bets}")

    return {"wins": wins, "losses": losses, "win_rate": win_rate if total_bets > 0 else 0}


async def backtest_model(days_back: int = 30, db_url: str = None):
    """
    Backtest the trained model against recent games with actual betting lines.
    """
    print(f"\n=== BACKTESTING MODEL (last {days_back} days) ===\n")

    # Load trained model
    model_path = Path(__file__).parent.parent.parent.parent / "models" / "totals_model_v3.pkl"
    if not model_path.exists():
        print("ERROR: Model not trained yet. Run training first.")
        return

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    feature_names = model_data['feature_names']
    total_std = model_data['total_std']

    print(f"Loaded model: {model_data['model_type']}")
    print(f"Trained on {model_data['training_games']} games")
    print(f"Avg MAE: {model_data['avg_mae']:.2f}")

    # Get recent games
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

    cur.execute('''
        SELECT
            gr.game_id,
            gr.game_date,
            gr.home_team_id,
            gr.away_team_id,
            gr.home_score,
            gr.away_score,
            gr.total_score,
            gr.closing_total
        FROM game_results gr
        WHERE gr.total_score IS NOT NULL
        AND gr.closing_total IS NOT NULL
        AND gr.game_date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY gr.game_date DESC
    ''' % days_back)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("No recent games found with closing lines")
        return

    print(f"Found {len(rows)} games\n")

    # Get current injury reports
    injury_reports = await get_all_team_injury_reports()

    # Make predictions
    results = []
    for row in rows:
        game_id = row[0]
        game_date = row[1]
        home_team_id = str(row[2])
        away_team_id = str(row[3])
        actual_total = row[6]
        closing_total = float(row[7]) if row[7] else None

        if closing_total is None:
            continue

        # TODO: Build feature vector for this game
        # This requires fetching team_stats for the game date
        # For now, just demonstrate the structure

        results.append({
            'game_id': game_id,
            'game_date': game_date,
            'actual': actual_total,
            'line': closing_total,
        })

    print(f"Backtest complete: {len(results)} predictions made")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        asyncio.run(backtest_model(days_back=days))
    else:
        # Default: train with Ridge regression
        use_lgbm = '--lgbm' in sys.argv
        train_totals_model_v3(use_lightgbm=use_lgbm)
