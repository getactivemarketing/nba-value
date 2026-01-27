"""
Train dedicated totals (over/under) prediction model.

The current totals model is a naive pace/PPG estimator with 41% win rate.
This script trains a proper ML model using:
- Pace factors (most important for totals)
- Offensive/defensive efficiency ratings
- PPG and opponent PPG
- Rest and fatigue factors
- Historical O/U tendencies

Target: Predict total points (home_score + away_score)
Goal: Achieve >52.4% win rate at -110 odds (breakeven)

Usage:
    python -m src.services.ml.train_totals_model
"""

import pickle
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import psycopg2
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import structlog

logger = structlog.get_logger()

DB_URL = 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway'

# Features for totals prediction (order matters for feature vector)
TOTALS_FEATURES = [
    # Pace factors - most important for totals
    "home_pace_10",
    "away_pace_10",
    # Offensive efficiency
    "home_ortg_10",
    "away_ortg_10",
    # Defensive efficiency (lower = better defense = lower scoring)
    "home_drtg_10",
    "away_drtg_10",
    # Direct scoring metrics
    "home_ppg_10",
    "away_ppg_10",
    "home_opp_ppg_10",
    "away_opp_ppg_10",
    # Rest factors (fatigue = lower scoring)
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    # O/U tendencies (teams that consistently go over/under)
    "home_ou_over_pct",
    "away_ou_over_pct",
]


def fetch_training_data(db_url: str = None) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Fetch historical games with team stats and actual totals.

    Returns:
        X: Feature matrix
        y: Target vector (actual totals)
        game_ids: List of game identifiers for debugging
    """
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

    # Get games with team stats available
    # Use stats from the day BEFORE the game (what we'd have at prediction time)
    cur.execute('''
        SELECT
            gr.game_id,
            gr.game_date,
            gr.home_team_id,
            gr.away_team_id,
            gr.total_score,
            -- Home team features
            ts_h.pace_10 as home_pace_10,
            ts_h.ortg_10 as home_ortg_10,
            ts_h.drtg_10 as home_drtg_10,
            ts_h.ppg_10 as home_ppg_10,
            ts_h.opp_ppg_10 as home_opp_ppg_10,
            ts_h.days_rest as home_rest_days,
            ts_h.is_back_to_back as home_b2b,
            CASE WHEN (ts_h.ou_overs_l10 + ts_h.ou_unders_l10) > 0
                 THEN ts_h.ou_overs_l10::float / (ts_h.ou_overs_l10 + ts_h.ou_unders_l10)
                 ELSE 0.5 END as home_ou_over_pct,
            -- Away team features
            ts_a.pace_10 as away_pace_10,
            ts_a.ortg_10 as away_ortg_10,
            ts_a.drtg_10 as away_drtg_10,
            ts_a.ppg_10 as away_ppg_10,
            ts_a.opp_ppg_10 as away_opp_ppg_10,
            ts_a.days_rest as away_rest_days,
            ts_a.is_back_to_back as away_b2b,
            CASE WHEN (ts_a.ou_overs_l10 + ts_a.ou_unders_l10) > 0
                 THEN ts_a.ou_overs_l10::float / (ts_a.ou_overs_l10 + ts_a.ou_unders_l10)
                 ELSE 0.5 END as away_ou_over_pct
        FROM game_results gr
        LEFT JOIN team_stats ts_h ON gr.home_team_id = ts_h.team_id
            AND ts_h.stat_date = (
                SELECT MAX(stat_date) FROM team_stats
                WHERE team_id = gr.home_team_id AND stat_date < gr.game_date
            )
        LEFT JOIN team_stats ts_a ON gr.away_team_id = ts_a.team_id
            AND ts_a.stat_date = (
                SELECT MAX(stat_date) FROM team_stats
                WHERE team_id = gr.away_team_id AND stat_date < gr.game_date
            )
        WHERE gr.total_score IS NOT NULL
        AND ts_h.pace_10 IS NOT NULL  -- Ensure we have home stats
        AND ts_a.pace_10 IS NOT NULL  -- Ensure we have away stats
        ORDER BY gr.game_date ASC
    ''')

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        logger.warning("No training data found with complete features")
        return np.array([]), np.array([]), []

    X = []
    y = []
    game_ids = []

    for row in rows:
        game_id = row[0]
        total_score = row[4]

        # Extract features in order: home_pace, away_pace, home_ortg, away_ortg, ...
        # Columns 5-12 are home features, 13-20 are away features
        home_features = [
            float(row[5]) if row[5] else 100.0,   # home_pace_10
            float(row[6]) if row[6] else 110.0,   # home_ortg_10
            float(row[7]) if row[7] else 110.0,   # home_drtg_10
            float(row[8]) if row[8] else 110.0,   # home_ppg_10
            float(row[9]) if row[9] else 110.0,   # home_opp_ppg_10
            float(row[10]) if row[10] else 1,     # home_rest_days
            1 if row[11] else 0,                   # home_b2b
            float(row[12]) if row[12] else 0.5,   # home_ou_over_pct
        ]

        away_features = [
            float(row[13]) if row[13] else 100.0,  # away_pace_10
            float(row[14]) if row[14] else 110.0,  # away_ortg_10
            float(row[15]) if row[15] else 110.0,  # away_drtg_10
            float(row[16]) if row[16] else 110.0,  # away_ppg_10
            float(row[17]) if row[17] else 110.0,  # away_opp_ppg_10
            float(row[18]) if row[18] else 1,      # away_rest_days
            1 if row[19] else 0,                    # away_b2b
            float(row[20]) if row[20] else 0.5,    # away_ou_over_pct
        ]

        # Interleave home/away features: home_pace, away_pace, home_ortg, away_ortg, ...
        features = []
        for h, a in zip(home_features, away_features):
            features.append(h)
            features.append(a)

        X.append(features)
        y.append(total_score)
        game_ids.append(game_id)

    logger.info(f"Loaded {len(X)} games with complete features")
    return np.array(X), np.array(y), game_ids


def train_totals_model(db_url: str = None) -> dict:
    """
    Train Ridge regression model for totals prediction.

    Uses Ridge regression because:
    - More stable than OLS with correlated features
    - Less prone to overfitting on small datasets
    - Provides good baseline performance

    Returns:
        Dictionary with model, metadata, and performance metrics
    """
    X, y, game_ids = fetch_training_data(db_url)

    if len(X) < 10:
        logger.error(f"Insufficient training data: {len(X)} games (need at least 10)")
        return {"error": "Insufficient training data", "games": len(X)}

    if len(X) < 30:
        logger.warning(f"Limited training data: {len(X)} games (recommend 30+)")

    print(f"\n=== TOTALS MODEL TRAINING ===")
    print(f"Training data: {len(X)} games")
    print(f"Target range: {y.min():.0f} - {y.max():.0f} (mean: {y.mean():.1f})")

    # Time-series cross-validation (don't leak future data)
    n_splits = min(5, len(X) // 10)  # At least 10 samples per fold
    if n_splits < 2:
        n_splits = 2

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Ridge regression with regularization
    model = Ridge(alpha=10.0)

    val_maes = []
    val_rmses = []
    val_predictions = []
    val_actuals = []

    print(f"\nCross-validation ({n_splits} folds):")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_maes.append(mae)
        val_rmses.append(rmse)

        val_predictions.extend(val_pred)
        val_actuals.extend(y_val)

        print(f"  Fold {fold+1}: MAE={mae:.2f}, RMSE={rmse:.2f} ({len(val_idx)} games)")

    avg_mae = np.mean(val_maes)
    avg_rmse = np.mean(val_rmses)
    print(f"\nAverage MAE: {avg_mae:.2f} ± {np.std(val_maes):.2f}")
    print(f"Average RMSE: {avg_rmse:.2f} ± {np.std(val_rmses):.2f}")

    # Train final model on all data
    model.fit(X, y)

    # Calculate total_std for probability conversion
    final_preds = model.predict(X)
    residuals = y - final_preds
    total_std = np.std(residuals)
    print(f"Total std: {total_std:.2f} (used for probability conversion)")

    # Feature importance (Ridge coefficients)
    feature_names = []
    for i in range(8):
        feature_names.append(f"home_{['pace', 'ortg', 'drtg', 'ppg', 'opp_ppg', 'rest', 'b2b', 'ou_pct'][i]}")
        feature_names.append(f"away_{['pace', 'ortg', 'drtg', 'ppg', 'opp_ppg', 'rest', 'b2b', 'ou_pct'][i]}")

    print(f"\nFeature coefficients:")
    coef_importance = sorted(zip(feature_names, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in coef_importance[:10]:
        print(f"  {name}: {coef:.3f}")

    # Simulate betting performance
    print(f"\n=== SIMULATED BETTING PERFORMANCE ===")
    simulate_betting(val_predictions, val_actuals)

    # Save model
    model_path = Path(__file__).parent.parent.parent.parent / "models" / "totals_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model': model,
        'feature_names': feature_names,
        'total_std': total_std,
        'model_type': 'ridge_totals',
        'training_games': len(X),
        'avg_mae': avg_mae,
        'avg_rmse': avg_rmse,
        'trained_at': datetime.now(timezone.utc).isoformat(),
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to {model_path}")

    return model_data


def simulate_betting(predictions: list, actuals: list, typical_line_offset: float = 0) -> dict:
    """
    Simulate betting performance using cross-validation predictions.

    Assumes:
    - Typical total lines are around actual total (market is efficient)
    - We bet OVER when prediction > line, UNDER when prediction < line
    - Standard -110 odds (risk $110 to win $100)
    """
    if not predictions or not actuals:
        return {"error": "No predictions to simulate"}

    # Simulate with various edge thresholds
    thresholds = [0, 1, 2, 3, 5]

    for threshold in thresholds:
        wins = 0
        losses = 0
        pushes = 0

        for pred, actual in zip(predictions, actuals):
            # Assume line is at the actual total (worst case - market is perfect)
            # In reality, we'd compare against actual betting lines
            line = actual  # Simulated line at actual result

            # Only bet if our prediction differs from line by threshold
            diff = pred - line
            if abs(diff) < threshold:
                continue  # Skip - not enough edge

            if diff > 0:  # We predict OVER
                if actual > line:
                    wins += 1
                elif actual < line:
                    losses += 1
                else:
                    pushes += 1
            else:  # We predict UNDER
                if actual < line:
                    wins += 1
                elif actual > line:
                    losses += 1
                else:
                    pushes += 1

        total_bets = wins + losses
        if total_bets > 0:
            win_rate = wins / total_bets * 100
            # ROI at -110 odds: wins * 0.909 - losses
            profit = wins * 90.91 - losses * 100
            roi = profit / (total_bets * 100) * 100
            print(f"  Threshold {threshold}+ pts: {wins}-{losses} ({win_rate:.1f}%), ROI: {roi:.1f}%, Bets: {total_bets}")


def backtest_against_lines(db_url: str = None, days_back: int = 14) -> dict:
    """
    Backtest the trained model against actual betting lines.

    This is the true test - comparing model predictions against
    actual market lines from the closing_total field.
    """
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

    # Load trained model
    model_path = Path(__file__).parent.parent.parent.parent / "models" / "totals_model.pkl"
    if not model_path.exists():
        return {"error": "Model not trained yet"}

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    total_std = model_data['total_std']

    # Get recent games with betting lines
    cur.execute('''
        SELECT
            gr.game_id,
            gr.game_date,
            gr.total_score,
            gr.closing_total,
            gr.total_result,
            -- Home team features
            ts_h.pace_10, ts_h.ortg_10, ts_h.drtg_10, ts_h.ppg_10, ts_h.opp_ppg_10,
            ts_h.days_rest, ts_h.is_back_to_back,
            CASE WHEN (ts_h.ou_overs_l10 + ts_h.ou_unders_l10) > 0
                 THEN ts_h.ou_overs_l10::float / (ts_h.ou_overs_l10 + ts_h.ou_unders_l10)
                 ELSE 0.5 END,
            -- Away team features
            ts_a.pace_10, ts_a.ortg_10, ts_a.drtg_10, ts_a.ppg_10, ts_a.opp_ppg_10,
            ts_a.days_rest, ts_a.is_back_to_back,
            CASE WHEN (ts_a.ou_overs_l10 + ts_a.ou_unders_l10) > 0
                 THEN ts_a.ou_overs_l10::float / (ts_a.ou_overs_l10 + ts_a.ou_unders_l10)
                 ELSE 0.5 END
        FROM game_results gr
        LEFT JOIN team_stats ts_h ON gr.home_team_id = ts_h.team_id
            AND ts_h.stat_date = (
                SELECT MAX(stat_date) FROM team_stats
                WHERE team_id = gr.home_team_id AND stat_date < gr.game_date
            )
        LEFT JOIN team_stats ts_a ON gr.away_team_id = ts_a.team_id
            AND ts_a.stat_date = (
                SELECT MAX(stat_date) FROM team_stats
                WHERE team_id = gr.away_team_id AND stat_date < gr.game_date
            )
        WHERE gr.total_score IS NOT NULL
        AND gr.closing_total IS NOT NULL
        AND ts_h.pace_10 IS NOT NULL
        AND ts_a.pace_10 IS NOT NULL
        AND gr.game_date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY gr.game_date
    ''' % days_back)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return {"error": "No games with betting lines found"}

    print(f"\n=== BACKTEST AGAINST ACTUAL LINES ({days_back} days) ===")
    print(f"Games with lines: {len(rows)}")

    wins = 0
    losses = 0
    pushes = 0
    results = []

    for row in rows:
        game_id = row[0]
        total_score = row[2]
        closing_total = float(row[3])
        total_result = row[4]

        # Build feature vector
        features = []
        for i in range(8):
            h_idx = 5 + i
            a_idx = 13 + i
            h_val = float(row[h_idx]) if row[h_idx] is not None else [100, 110, 110, 110, 110, 1, 0, 0.5][i]
            a_val = float(row[a_idx]) if row[a_idx] is not None else [100, 110, 110, 110, 110, 1, 0, 0.5][i]
            if i == 6:  # b2b is boolean
                h_val = 1 if row[h_idx] else 0
                a_val = 1 if row[a_idx] else 0
            features.append(h_val)
            features.append(a_val)

        # Predict total
        pred_total = model.predict([features])[0]

        # Determine bet direction
        diff = pred_total - closing_total
        if diff > 1:  # Predict OVER (with 1pt buffer)
            bet_direction = 'over'
        elif diff < -1:  # Predict UNDER
            bet_direction = 'under'
        else:
            bet_direction = 'pass'  # Too close to call

        # Check result
        if bet_direction == 'pass':
            continue

        if total_result == 'push':
            pushes += 1
        elif bet_direction == total_result:
            wins += 1
        else:
            losses += 1

        results.append({
            'pred': pred_total,
            'line': closing_total,
            'actual': total_score,
            'bet': bet_direction,
            'result': total_result,
            'hit': bet_direction == total_result,
        })

    total_bets = wins + losses
    if total_bets > 0:
        win_rate = wins / total_bets * 100
        profit = wins * 90.91 - losses * 100
        roi = profit / (total_bets * 100) * 100

        print(f"\nResults: {wins}-{losses}-{pushes}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"ROI: {roi:.1f}%")
        print(f"Profit: ${profit:.2f} (on ${total_bets * 100} wagered)")

        return {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": win_rate,
            "roi": roi,
            "profit": profit,
            "total_bets": total_bets,
        }
    else:
        print("No bets placed (all predictions too close to line)")
        return {"error": "No bets placed"}


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
        backtest_against_lines(days_back=days)
    else:
        train_totals_model()
