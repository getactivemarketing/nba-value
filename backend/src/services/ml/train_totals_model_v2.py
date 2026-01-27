"""
Train totals model v2 - uses computed features from game_results history.

This version computes rolling averages directly from game_results rather
than requiring team_stats, allowing us to use the full season of data.

Usage:
    python -m src.services.ml.train_totals_model_v2
"""

import pickle
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np
import psycopg2
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import structlog

logger = structlog.get_logger()

DB_URL = 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway'


def fetch_all_games(db_url: str = None) -> list:
    """Fetch all completed games ordered by date."""
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

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

    return games


def compute_rolling_stats(games: list, lookback: int = 10) -> dict:
    """
    Compute rolling team stats from game history.

    For each game, computes stats from the previous N games for each team.
    This allows us to use the full dataset without requiring team_stats table.
    """
    # Track each team's recent games
    team_games = defaultdict(list)  # team_id -> list of (date, pts_for, pts_against)

    # Build features for each game
    game_features = {}

    for game in games:
        game_id, game_date, home_id, away_id, home_score, away_score, total_score, closing_total = game

        # Get rolling stats for home team (before this game)
        home_games = team_games[home_id][-lookback:] if team_games[home_id] else []
        away_games = team_games[away_id][-lookback:] if team_games[away_id] else []

        # Compute features (need at least 3 games to have meaningful stats)
        if len(home_games) >= 3 and len(away_games) >= 3:
            # Home team stats
            home_pts_for = np.mean([g[1] for g in home_games])
            home_pts_against = np.mean([g[2] for g in home_games])
            home_total_avg = np.mean([g[1] + g[2] for g in home_games])

            # Away team stats
            away_pts_for = np.mean([g[1] for g in away_games])
            away_pts_against = np.mean([g[2] for g in away_games])
            away_total_avg = np.mean([g[1] + g[2] for g in away_games])

            # Combined prediction features
            features = {
                'home_ppg': home_pts_for,
                'home_opp_ppg': home_pts_against,
                'home_total_avg': home_total_avg,
                'away_ppg': away_pts_for,
                'away_opp_ppg': away_pts_against,
                'away_total_avg': away_total_avg,
                # Derived features
                'combined_ppg': (home_pts_for + away_pts_for) / 2,
                'combined_opp_ppg': (home_pts_against + away_pts_against) / 2,
                'matchup_total_est': (home_total_avg + away_total_avg) / 2,
            }

            game_features[game_id] = {
                'features': features,
                'actual_total': total_score,
                'closing_total': closing_total,
                'game_date': game_date,
            }

        # Update team history (after computing features for this game)
        team_games[home_id].append((game_date, home_score, away_score))
        team_games[away_id].append((game_date, away_score, home_score))

    return game_features


def train_totals_model_v2(db_url: str = None) -> dict:
    """
    Train totals model using computed rolling features.

    This approach:
    1. Uses all historical games (not limited by team_stats)
    2. Computes rolling averages directly from game_results
    3. Trains Ridge regression to predict total points
    """
    print("=== TOTALS MODEL V2 TRAINING ===\n")

    # Fetch all games
    games = fetch_all_games(db_url)
    print(f"Total games in database: {len(games)}")

    # Compute rolling features
    game_features = compute_rolling_stats(games, lookback=10)
    print(f"Games with computed features: {len(game_features)}")

    if len(game_features) < 50:
        print(f"ERROR: Insufficient data ({len(game_features)} games)")
        return {"error": "Insufficient data"}

    # Build training data
    X = []
    y = []
    game_ids = []
    closing_totals = []

    feature_names = [
        'home_ppg', 'home_opp_ppg', 'home_total_avg',
        'away_ppg', 'away_opp_ppg', 'away_total_avg',
        'combined_ppg', 'combined_opp_ppg', 'matchup_total_est'
    ]

    for game_id, data in game_features.items():
        features = [data['features'][f] for f in feature_names]
        X.append(features)
        y.append(data['actual_total'])
        game_ids.append(game_id)
        closing_totals.append(data['closing_total'])

    X = np.array(X)
    y = np.array(y)

    print(f"Training samples: {len(X)}")
    print(f"Target range: {y.min():.0f} - {y.max():.0f} (mean: {y.mean():.1f}, std: {y.std():.1f})")

    # Time-series cross-validation
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Try different alpha values for Ridge
    best_alpha = 10.0
    best_mae = float('inf')

    for alpha in [1.0, 5.0, 10.0, 50.0, 100.0]:
        model = Ridge(alpha=alpha)
        maes = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, val_pred)
            maes.append(mae)

        avg_mae = np.mean(maes)
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha} (MAE: {best_mae:.2f})")

    # Train final model with best alpha
    model = Ridge(alpha=best_alpha)

    val_predictions = []
    val_actuals = []
    val_closing = []

    print(f"\nCross-validation ({n_splits} folds):")
    fold_idx = 0
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        val_predictions.extend(val_pred)
        val_actuals.extend(y_val)
        val_closing.extend([closing_totals[i] for i in val_idx])

        print(f"  Fold {fold_idx+1}: MAE={mae:.2f}, RMSE={rmse:.2f} ({len(val_idx)} games)")
        fold_idx += 1

    # Train final model on all data
    model.fit(X, y)

    # Calculate total_std for probability conversion
    final_preds = model.predict(X)
    residuals = y - final_preds
    total_std = np.std(residuals)

    print(f"\nFinal model std: {total_std:.2f}")

    # Feature importance
    print(f"\nFeature coefficients:")
    coef_importance = sorted(zip(feature_names, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in coef_importance:
        print(f"  {name}: {coef:+.3f}")

    # Simulate betting with actual lines
    print(f"\n=== BETTING SIMULATION (vs actual lines) ===")
    simulate_betting_vs_lines(val_predictions, val_actuals, val_closing)

    # Save model
    model_path = Path(__file__).parent.parent.parent.parent / "models" / "totals_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model': model,
        'feature_names': feature_names,
        'total_std': total_std,
        'model_type': 'ridge_totals_v2',
        'training_games': len(X),
        'avg_mae': best_mae,
        'best_alpha': best_alpha,
        'trained_at': datetime.now(timezone.utc).isoformat(),
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to {model_path}")

    return model_data


def simulate_betting_vs_lines(predictions: list, actuals: list, lines: list) -> dict:
    """
    Simulate betting using model predictions vs actual closing lines.

    This is the true test of the model - can it beat the market?
    """
    if not predictions or not lines:
        return {"error": "No data"}

    # Filter to games with valid lines
    valid_games = [(p, a, l) for p, a, l in zip(predictions, actuals, lines) if l is not None]

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
            print(f"  Edge {threshold}+ pts: {wins}-{losses}-{pushes} ({win_rate:.1f}%), ROI: {roi:+.1f}%")
        else:
            print(f"  Edge {threshold}+ pts: No bets")

    # Also test with probability-based betting
    print(f"\n  Using probability threshold (>55% confidence):")
    wins = 0
    losses = 0

    for pred, actual, line in valid_games:
        line = float(line)
        diff = pred - line
        # Approximate probability using normal distribution
        # P(over) ≈ P(actual > line) given pred and std
        std = 12.0  # typical game std
        prob_over = 1 - stats.norm.cdf(line, pred, std)

        if prob_over > 0.55:  # Bet OVER
            if actual > line:
                wins += 1
            elif actual < line:
                losses += 1
        elif prob_over < 0.45:  # Bet UNDER
            if actual < line:
                wins += 1
            elif actual > line:
                losses += 1

    total_bets = wins + losses
    if total_bets > 0:
        win_rate = wins / total_bets * 100
        profit = wins * 90.91 - losses * 100
        roi = profit / (total_bets * 100) * 100
        print(f"  P>55%: {wins}-{losses} ({win_rate:.1f}%), ROI: {roi:+.1f}%")

    return {"wins": wins, "losses": losses}


if __name__ == '__main__':
    train_totals_model_v2()
