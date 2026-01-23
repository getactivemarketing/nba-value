"""
Train spread model v2 - uses our database with ATS tendencies and situational features.

Key improvements over v1:
1. Uses game_results + team_stats from our database (not NBA API)
2. Adds ATS tendency features (do teams cover spreads?)
3. Adds home/away performance splits
4. Adds schedule fatigue features
5. Trains against actual closing lines

Usage:
    python -m src.services.ml.train_spread_model_v2
"""

import pickle
from pathlib import Path
from datetime import datetime, timezone
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


def fetch_games_with_team_stats(db_url: str = None, lookback: int = 10) -> tuple:
    """
    Fetch games with pre-computed team stats from our database.

    Computes rolling features from game_results history, similar to totals v3.

    Returns:
        X: Feature matrix
        y: Target vector (home margin)
        game_ids: List of game IDs
        feature_names: List of feature names
        closing_spreads: List of closing spreads
    """
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

    # Fetch all completed games
    cur.execute('''
        SELECT
            game_id,
            game_date,
            home_team_id,
            away_team_id,
            home_score,
            away_score,
            closing_spread,
            spread_result
        FROM game_results
        WHERE home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY game_date ASC
    ''')
    games = cur.fetchall()
    cur.close()
    conn.close()

    if not games:
        logger.warning("No games found")
        return np.array([]), np.array([]), [], [], []

    logger.info(f"Loaded {len(games)} total games from database")

    # Track team history for computing rolling features
    # team_id -> list of game stats
    team_games = defaultdict(list)

    # Build features
    X = []
    y = []  # home margin (positive = home win)
    game_ids = []
    closing_spreads = []

    feature_names = [
        # Basic rolling stats (6 features)
        'home_ppg', 'home_opp_ppg', 'home_net_ppg',
        'away_ppg', 'away_opp_ppg', 'away_net_ppg',
        # Scoring variance (2 features)
        'home_scoring_std', 'away_scoring_std',
        # Win rate L10 - CENTERED around 0.5 (2 features)
        'home_win_pct_l10_centered', 'away_win_pct_l10_centered',
        # ATS tendencies - CENTERED around 0.5 (2 features)
        'home_ats_pct_centered', 'away_ats_pct_centered',
        # Home/away performance - CENTERED (4 features)
        'home_home_win_pct_centered', 'home_away_win_pct_centered',
        'away_home_win_pct_centered', 'away_away_win_pct_centered',
        # Rest/fatigue (5 features)
        'home_rest_days', 'away_rest_days',
        'rest_advantage', 'home_b2b', 'away_b2b',
        # Schedule density (2 features)
        'home_games_last_7', 'away_games_last_7',
    ]

    for game in games:
        game_id, game_date, home_id, away_id, home_score, away_score, closing_spread, spread_result = game

        margin = home_score - away_score

        # Get rolling history for each team
        home_history = team_games[home_id][-lookback:] if team_games[home_id] else []
        away_history = team_games[away_id][-lookback:] if team_games[away_id] else []

        # Need at least 5 games of history for each team
        if len(home_history) >= 5 and len(away_history) >= 5:
            # Basic rolling stats
            home_ppg = np.mean([g['pts_for'] for g in home_history])
            home_opp_ppg = np.mean([g['pts_against'] for g in home_history])
            home_net_ppg = home_ppg - home_opp_ppg

            away_ppg = np.mean([g['pts_for'] for g in away_history])
            away_opp_ppg = np.mean([g['pts_against'] for g in away_history])
            away_net_ppg = away_ppg - away_opp_ppg

            # Scoring variance
            home_scoring_std = np.std([g['pts_for'] for g in home_history])
            away_scoring_std = np.std([g['pts_for'] for g in away_history])

            # Win rate L10
            home_win_pct = np.mean([g['win'] for g in home_history])
            away_win_pct = np.mean([g['win'] for g in away_history])

            # ATS tendencies - KEY FEATURE
            home_ats_results = [g['covered'] for g in home_history if g['covered'] is not None]
            away_ats_results = [g['covered'] for g in away_history if g['covered'] is not None]

            home_ats_pct = np.mean(home_ats_results) if home_ats_results else 0.5
            away_ats_pct = np.mean(away_ats_results) if away_ats_results else 0.5

            # Home/away performance splits
            home_home_games = [g for g in home_history if g['is_home']]
            home_away_games = [g for g in home_history if not g['is_home']]
            away_home_games = [g for g in away_history if g['is_home']]
            away_away_games = [g for g in away_history if not g['is_home']]

            home_home_win_pct = np.mean([g['win'] for g in home_home_games]) if home_home_games else 0.5
            home_away_win_pct = np.mean([g['win'] for g in home_away_games]) if home_away_games else 0.5
            away_home_win_pct = np.mean([g['win'] for g in away_home_games]) if away_home_games else 0.5
            away_away_win_pct = np.mean([g['win'] for g in away_away_games]) if away_away_games else 0.5

            # Rest/fatigue
            home_last_date = home_history[-1]['date']
            away_last_date = away_history[-1]['date']

            home_rest_days = (game_date - home_last_date).days
            away_rest_days = (game_date - away_last_date).days
            rest_advantage = home_rest_days - away_rest_days

            home_b2b = 1 if home_rest_days == 1 else 0
            away_b2b = 1 if away_rest_days == 1 else 0

            # Schedule density (games in last 7 days)
            home_games_last_7 = sum(1 for g in home_history if (game_date - g['date']).days <= 7)
            away_games_last_7 = sum(1 for g in away_history if (game_date - g['date']).days <= 7)

            features = [
                # Basic (6)
                home_ppg, home_opp_ppg, home_net_ppg,
                away_ppg, away_opp_ppg, away_net_ppg,
                # Variance (2)
                home_scoring_std, away_scoring_std,
                # Win rate - CENTERED (2) - 0 = average, positive = above average
                home_win_pct - 0.5, away_win_pct - 0.5,
                # ATS - CENTERED (2) - 0 = average, positive = covering well
                home_ats_pct - 0.5, away_ats_pct - 0.5,
                # Home/away splits - CENTERED (4)
                home_home_win_pct - 0.5, home_away_win_pct - 0.5,
                away_home_win_pct - 0.5, away_away_win_pct - 0.5,
                # Rest (5)
                home_rest_days, away_rest_days,
                rest_advantage, home_b2b, away_b2b,
                # Schedule (2)
                home_games_last_7, away_games_last_7,
            ]

            X.append(features)
            y.append(margin)
            game_ids.append(game_id)
            closing_spreads.append(closing_spread)

        # Update team history AFTER building features (avoid future leakage)
        # Determine if team covered the spread
        covered = None
        if closing_spread is not None:
            # closing_spread is from home perspective (negative = home favored)
            home_adjusted = home_score + float(closing_spread)
            if home_adjusted > away_score:
                covered_home = 1  # home covered
            elif home_adjusted < away_score:
                covered_home = 0  # home didn't cover
            else:
                covered_home = None  # push
        else:
            covered_home = None

        team_games[home_id].append({
            'date': game_date,
            'pts_for': home_score,
            'pts_against': away_score,
            'win': 1 if margin > 0 else 0,
            'covered': covered_home,
            'is_home': True,
        })

        team_games[away_id].append({
            'date': game_date,
            'pts_for': away_score,
            'pts_against': home_score,
            'win': 1 if margin < 0 else 0,
            'covered': 1 - covered_home if covered_home is not None else None,  # Away covers when home doesn't
            'is_home': False,
        })

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Built feature matrix: {X.shape[0]} games x {X.shape[1]} features")
    if len(X) > 0:
        logger.info(f"Margin range: {y.min():.0f} to {y.max():.0f} (mean: {y.mean():.1f})")

    return X, y, game_ids, feature_names, closing_spreads


def train_spread_model_v2(db_url: str = None) -> dict:
    """
    Train spread model with ATS tendencies and situational features.

    Returns:
        Dictionary with model, metadata, and performance metrics
    """
    print("=== SPREAD MODEL V2 TRAINING ===\n")

    # Fetch training data
    X, y, game_ids, feature_names, closing_spreads = fetch_games_with_team_stats(db_url)

    if len(X) < 50:
        print(f"ERROR: Insufficient training data ({len(X)} games, need 50+)")
        return {"error": "Insufficient data", "games": len(X)}

    print(f"Training samples: {len(X)}")
    print(f"Features: {len(feature_names)}")
    print(f"Margin range: {y.min():.0f} to {y.max():.0f} (mean: {y.mean():.1f}, std: {y.std():.1f})")

    # Time-series cross-validation
    n_splits = min(5, len(X) // 50)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Find best alpha for Ridge
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

    print(f"\nBest alpha: {best_alpha} (MAE: {best_mae:.2f})")

    # Train final model
    model = Ridge(alpha=best_alpha)

    print(f"\nCross-validation ({n_splits} folds):")

    val_predictions = []
    val_actuals = []
    val_closing_spreads = []
    val_maes = []

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
        val_closing_spreads.extend([closing_spreads[i] for i in val_idx])

        print(f"  Fold {fold_idx+1}: MAE={mae:.2f}, RMSE={rmse:.2f} ({len(val_idx)} games)")
        fold_idx += 1

    avg_mae = np.mean(val_maes)
    print(f"\nAverage MAE: {avg_mae:.2f} ± {np.std(val_maes):.2f}")

    # Train final model on all data
    model.fit(X, y)

    # Calculate MOV std for probability conversion
    final_preds = model.predict(X)
    residuals = y - final_preds
    mov_std = np.std(residuals)
    print(f"MOV std: {mov_std:.2f}")

    # Feature importance
    print(f"\nFeature importance (coefficients):")
    coefs = model.coef_
    feat_imp = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in feat_imp[:15]:
        print(f"  {name}: {coef:+.3f}")

    # Simulate spread betting vs actual closing lines
    print(f"\n=== SPREAD BETTING SIMULATION (vs closing lines) ===")
    simulate_spread_betting(val_predictions, val_actuals, val_closing_spreads)

    # Save model
    model_path = Path(__file__).parent.parent.parent.parent / "models" / "spread_model_v2.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model': model,
        'feature_names': feature_names,
        'mov_std': mov_std,
        'model_type': 'ridge_spread_v2',
        'training_games': len(X),
        'avg_mae': avg_mae,
        'best_alpha': best_alpha,
        'trained_at': datetime.now(timezone.utc).isoformat(),
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to {model_path}")

    return model_data


def simulate_spread_betting(predictions: list, actuals: list, closing_spreads: list) -> dict:
    """
    Simulate spread betting: predict which side covers.

    Model predicts home margin. Compare to closing spread:
    - If predicted_margin > -closing_spread: bet HOME to cover
    - If predicted_margin < -closing_spread: bet AWAY to cover

    Note: closing_spread is from home perspective (negative = home favored)
    """
    if not predictions or not closing_spreads:
        return {"error": "No data"}

    # Filter to games with valid spreads
    valid_games = [(p, a, s) for p, a, s in zip(predictions, actuals, closing_spreads)
                   if s is not None]

    if not valid_games:
        print("No games with closing spreads found")
        return {"error": "No spreads"}

    print(f"Games with closing spreads: {len(valid_games)}")

    # Test different edge thresholds
    thresholds = [0, 1, 2, 3, 4, 5]

    for threshold in thresholds:
        wins = 0
        losses = 0
        pushes = 0

        for pred_margin, actual_margin, spread in valid_games:
            spread = float(spread)

            # Our edge: difference between predicted margin and what spread implies
            # spread is negative if home is favored (e.g., -5.5 means home favored by 5.5)
            # We predict home margin directly
            # If pred_margin > -spread, we think home will cover
            edge = pred_margin - (-spread)  # = pred_margin + spread

            if abs(edge) < threshold:
                continue

            # Determine actual result
            home_adjusted = actual_margin + spread  # Did home cover?

            if edge > 0:  # Bet HOME to cover
                if home_adjusted > 0:
                    wins += 1
                elif home_adjusted < 0:
                    losses += 1
                else:
                    pushes += 1
            else:  # Bet AWAY to cover
                if home_adjusted < 0:
                    wins += 1
                elif home_adjusted > 0:
                    losses += 1
                else:
                    pushes += 1

        total_bets = wins + losses
        if total_bets > 0:
            win_rate = wins / total_bets * 100
            profit = wins * 90.91 - losses * 100
            roi = profit / (total_bets * 100) * 100
            edge_vs_breakeven = win_rate - 52.4

            print(f"  Edge {threshold}+ pts: {wins}-{losses}-{pushes} "
                  f"({win_rate:.1f}%, edge: {edge_vs_breakeven:+.1f}%), "
                  f"ROI: {roi:+.1f}%, Bets: {total_bets}")
        else:
            print(f"  Edge {threshold}+ pts: No bets")

    return {"wins": wins, "losses": losses}


if __name__ == '__main__':
    train_spread_model_v2()
