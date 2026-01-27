"""
Test script for totals model v3 - validates feature engineering and model predictions.

Usage:
    python -m src.services.ml.test_totals_v3
"""

import asyncio
import pickle
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import structlog

from src.services.ml.totals_features import (
    build_prediction_features,
    validate_features,
    get_feature_dict,
    FEATURE_NAMES,
)

logger = structlog.get_logger()


async def test_feature_builder():
    """Test feature builder on recent games."""
    print("=== Testing Feature Builder ===\n")

    # Test games (adjust team IDs as needed)
    test_games = [
        ('LAL', 'GSW', date.today() - timedelta(days=1)),
        ('BOS', 'MIA', date.today() - timedelta(days=1)),
        ('DEN', 'PHX', date.today() - timedelta(days=2)),
    ]

    for home_id, away_id, game_date in test_games:
        print(f"\n{home_id} vs {away_id} on {game_date}")

        try:
            features = await build_prediction_features(home_id, away_id, game_date)

            if features is None:
                print("  ERROR: Could not build features (missing data)")
                continue

            if not validate_features(features):
                print("  ERROR: Feature validation failed")
                continue

            print(f"  ✓ Built {len(features)} features")

            # Show top 10 features
            feature_dict = get_feature_dict(features)
            print("  Top features:")
            for i, name in enumerate(FEATURE_NAMES[:10]):
                print(f"    {name}: {features[i]:.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")


async def test_model_prediction():
    """Test model predictions on recent games."""
    print("\n=== Testing Model Predictions ===\n")

    # Load trained model
    model_path = Path(__file__).parent.parent.parent.parent / "models" / "totals_model_v3.pkl"

    if not model_path.exists():
        print("ERROR: Model not found. Train model first:")
        print("  python -m src.services.ml.train_totals_model_v3")
        return

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    total_std = model_data['total_std']

    print(f"Loaded model: {model_data['model_type']}")
    print(f"Trained on {model_data['training_games']} games")
    print(f"Avg MAE: {model_data['avg_mae']:.2f}")
    print(f"Total std: {total_std:.2f}\n")

    # Test predictions
    test_games = [
        ('LAL', 'GSW', date.today() - timedelta(days=1)),
        ('BOS', 'MIA', date.today() - timedelta(days=1)),
    ]

    for home_id, away_id, game_date in test_games:
        print(f"\n{home_id} vs {away_id} on {game_date}")

        try:
            features = await build_prediction_features(home_id, away_id, game_date)

            if features is None:
                print("  ERROR: Could not build features")
                continue

            # Make prediction
            predicted_total = model.predict([features])[0]

            print(f"  Predicted total: {predicted_total:.1f}")
            print(f"  Prediction range (±1 std): {predicted_total - total_std:.1f} - {predicted_total + total_std:.1f}")

            # Simulate probability vs typical line
            from scipy.stats import norm

            # Assume typical lines around prediction
            for line in [predicted_total - 3, predicted_total, predicted_total + 3]:
                prob_over = 1 - norm.cdf(line, predicted_total, total_std)
                prob_under = norm.cdf(line, predicted_total, total_std)

                print(f"  vs line {line:.1f}: Over {prob_over*100:.1f}%, Under {prob_under*100:.1f}%")

        except Exception as e:
            print(f"  ERROR: {e}")


def test_feature_coverage():
    """Test that all features are properly named and counted."""
    print("\n=== Feature Coverage Test ===\n")

    print(f"Total features: {len(FEATURE_NAMES)}")
    print("\nFeature categories:")

    categories = {
        'Basic stats': FEATURE_NAMES[0:8],
        'Pace interactions': FEATURE_NAMES[8:12],
        'Rest/fatigue': FEATURE_NAMES[12:17],
        'Injury impact': FEATURE_NAMES[17:22],
        'Scoring variance': FEATURE_NAMES[22:24],
        'Home/away splits': FEATURE_NAMES[24:28],
        'O/U tendencies': FEATURE_NAMES[28:30],
    }

    for category, features in categories.items():
        print(f"\n{category} ({len(features)} features):")
        for f in features:
            print(f"  - {f}")

    total_counted = sum(len(v) for v in categories.values())
    print(f"\nTotal counted: {total_counted}")

    if total_counted == len(FEATURE_NAMES):
        print("✓ All features accounted for")
    else:
        print(f"ERROR: Missing {len(FEATURE_NAMES) - total_counted} features")


async def main():
    """Run all tests."""
    test_feature_coverage()
    await test_feature_builder()
    await test_model_prediction()


if __name__ == '__main__':
    asyncio.run(main())
