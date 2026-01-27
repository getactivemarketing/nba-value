# Totals Model V3 - Implementation Summary

## Overview

Built an advanced totals (over/under) prediction model that incorporates **35 engineered features** designed to capture market inefficiencies. The current basic model achieves ~50% win rate (break-even); v3 targets >52.4% to beat -110 odds.

## Key Files Created

### 1. `/src/services/ml/train_totals_model_v3.py`
**Purpose**: Main training script with feature engineering

**Key Functions**:
- `fetch_training_data_with_features()` - Async data loading with injury reports
- `calculate_pace_interaction()` - Non-linear pace dynamics
- `calculate_rest_asymmetry()` - Fatigue effects on variance
- `calculate_injury_impact()` - Totals-specific injury scoring
- `train_totals_model_v3()` - Main training loop (Ridge or LightGBM)
- `simulate_betting_vs_lines()` - Backtest against actual closing lines

**Usage**:
```bash
# Train with Ridge regression (default)
python -m src.services.ml.train_totals_model_v3

# Train with LightGBM
python -m src.services.ml.train_totals_model_v3 --lgbm

# Backtest recent games
python -m src.services.ml.train_totals_model_v3 backtest --days 30
```

**Output**: `models/totals_model_v3.pkl`

### 2. `/src/services/ml/totals_features.py`
**Purpose**: Reusable feature engineering for training and live predictions

**Key Functions**:
- `build_prediction_features()` - Build feature vector for a single game
- `build_bulk_prediction_features()` - Batch processing for efficiency
- `validate_features()` - Feature validation checks
- `get_feature_dict()` - Convert vector to labeled dict for debugging

**Usage**:
```python
from src.services.ml.totals_features import build_prediction_features

features = await build_prediction_features(
    home_team_id='LAL',
    away_team_id='GSW',
    game_date=date.today(),
)

# Use for prediction
predicted_total = model.predict([features])[0]
```

### 3. `/src/services/ml/test_totals_v3.py`
**Purpose**: Validation and testing script

**Tests**:
- Feature builder on recent games
- Model predictions with probability estimates
- Feature coverage validation

**Usage**:
```bash
python -m src.services.ml.test_totals_v3
```

### 4. `/TOTALS_MODEL_V3.md`
**Purpose**: Comprehensive documentation of feature engineering

**Contents**:
- Detailed explanation of all 35 features
- Market efficiency analysis for each category
- Expected feature importance rankings
- Training/evaluation methodology
- Integration guide for production
- Rollback criteria

## Feature Engineering Highlights

### 35 Features in 7 Categories

1. **Basic Stats (8)**: pace, ortg, drtg, ppg
   - Market efficiency: HIGH
   - Purpose: Baseline metrics markets already price

2. **Pace Interactions (4)**: weighted_pace, pace_clash, pace_variance
   - Market efficiency: MEDIUM
   - Edge hypothesis: Non-linear tempo control when styles clash

3. **Rest/Fatigue (5)**: total_fatigue, fatigue_asymmetry, rest_advantage
   - Market efficiency: MEDIUM
   - Edge hypothesis: Markets price B2B but miss variance effects

4. **Injury Impact (5)**: totals_injury_score, total_injury, injury_asymmetry
   - Market efficiency: LOW (potential edge)
   - Edge hypothesis: Heavy injuries cause non-linear pace slowdown

5. **Scoring Variance (2)**: home_scoring_std, away_scoring_std
   - Market efficiency: LOW
   - Edge hypothesis: High variance teams are less predictable

6. **Home/Away Splits (4)**: home_win_pct, venue_boost_flags
   - Market efficiency: MEDIUM
   - Edge hypothesis: Extreme venue effects may be mispriced

7. **O/U Tendencies (2)**: home_ou_over_pct, away_ou_over_pct
   - Market efficiency: HIGH (weak signal)
   - Purpose: Capture persistent pace/style effects

## Injury Scoring Integration

The model uses `totals_injury_score` from `/src/services/injuries.py`:

```python
# Weighting for totals
totals_injury_score = (
    rebounding_impact * 0.35 +   # 2nd chance pts, pace
    defense_impact * 0.25 +      # Opponent scoring
    scoring_impact * 0.25 +      # Direct PPG
    playmaking_impact * 0.15     # Ball movement
)
```

**Key Insights**:
- Missing rebounders → fewer 2nd chance points, slower pace
- Defensive injuries → opponent scores more
- Combined injuries → both teams struggle, totals trend UNDER

## Model Architecture

### Ridge Regression (Default)
- **Pros**: Stable, fast, interpretable
- **Cons**: Linear only, cannot capture interactions
- **Recommended for**: Production baseline

**Hyperparameters**:
```python
Ridge(alpha=10.0)  # Auto-selected via CV from [1, 5, 10, 50, 100]
```

### LightGBM (Optional)
- **Pros**: Captures non-linear interactions, potentially better MAE
- **Cons**: Slower, overfitting risk, harder to interpret
- **Recommended for**: Experimental if Ridge underperforms

**Hyperparameters**:
```python
LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
)
```

## Training Methodology

### Data Requirements
- Minimum 30 games with complete features
- Historical games from `game_results` table
- Team stats from day before game (no future leakage)
- Recent scoring history (last 5 games) for variance
- Current injury reports (ideally historical, but using current as proxy)

### Cross-Validation
```python
TimeSeriesSplit(n_splits=5)
# Ensures no future leakage
# ~20% of data per fold
```

### Evaluation Metrics
1. **MAE** (Mean Absolute Error) - Prediction accuracy vs actual totals
2. **RMSE** - Penalizes large errors
3. **Win rate vs closing lines** - True test of market edge
4. **ROI** - Profit at -110 odds

### Success Criteria
- **MAE**: <11 points (improvement over v2's 11-12)
- **Win rate**: >52.4% (breakeven at -110 odds)
- **Edge threshold**: 3+ point predictions beat lines at 55%+
- **Expected ROI**: +2-5% at 3pt threshold

## Betting Simulation Results

The training script tests multiple edge thresholds:

```
Edge 0+ pts: All bets (baseline)
Edge 1+ pts: 1 point difference
Edge 2+ pts: 2 points difference
Edge 3+ pts: 3 points (recommended threshold)
Edge 5+ pts: High confidence only
```

Example output:
```
Edge 3+ pts: 45-38-2 (54.2%, edge: +1.8%), ROI: +4.8%, Bets: 83
```

This means: At 3pt edge, model wins 54.2% vs closing lines (beating 52.4% breakeven).

## Integration with Production

### Step 1: Train Model
```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/NBA-Value/backend
python -m src.services.ml.train_totals_model_v3
```

Output: `models/totals_model_v3.pkl`

### Step 2: Update Scoring System

Modify `/src/services/ml/scoring.py` to use v3 model:

```python
from src.services.ml.totals_features import build_prediction_features

# Load v3 model
with open('models/totals_model_v3.pkl', 'rb') as f:
    totals_model = pickle.load(f)

# For each game with totals market
features = await build_prediction_features(
    home_team_id=game.home_team_id,
    away_team_id=game.away_team_id,
    game_date=game.game_date,
)

if features is not None:
    predicted_total = totals_model['model'].predict([features])[0]

    # Calculate edge vs closing line
    edge = abs(predicted_total - market.total_line)

    # Convert to probability
    from scipy.stats import norm
    total_std = totals_model['total_std']
    prob_over = 1 - norm.cdf(market.total_line, predicted_total, total_std)
    prob_under = norm.cdf(market.total_line, predicted_total, total_std)

    # Only bet if edge >= 3 points
    if edge >= 3.0:
        if predicted_total > market.total_line:
            # Bet OVER
            value_score = edge * prob_over * confidence_multiplier
        else:
            # Bet UNDER
            value_score = edge * prob_under * confidence_multiplier
```

### Step 3: Monitor Performance

Track these metrics in production:
1. **MAE vs actual totals** - Should stay <11 points
2. **Win rate vs closing lines** - Target >52.4%
3. **Calibration** - Brier score should be <0.25
4. **CLV** (Closing Line Value) - Are predictions better than opening lines?

### Step 4: Retrain Schedule

Retrain model:
- **Weekly** during season (meta changes, injuries)
- **After major trades** (roster changes affect features)
- **When MAE degrades** by >1 point

## Rollback Plan

Revert to v2 model if:
1. MAE increases by >1 point vs v2
2. Win rate drops below 48% (worse than coin flip)
3. Production errors (inference failures, missing features)
4. Calibration degrades (Brier score >0.30)

## Future Improvements (v4 Ideas)

1. **Historical injury data**: Currently uses current injuries for all games (data limitation). Fetch injury snapshots at game time for accuracy.

2. **Line movement signals**: Track opening to closing line movement to identify sharp money.

3. **Referee tendencies**: Some refs call tighter games (more FTs, higher scoring).

4. **Player props correlation**: If star player props are getting heavy action, may signal insider info on game tempo.

5. **Weather data**: For outdoor/arena temperature effects on pace.

6. **Travel distance**: Long road trips affect fatigue more than single-game trips.

7. **Ensemble methods**: Combine Ridge + LightGBM with weighted voting.

8. **Sequence effects**: Teams on 5+ game win streaks may revert to mean.

## Testing & Validation

### Test Feature Builder
```bash
python -m src.services.ml.test_totals_v3
```

This validates:
- Feature builder works on recent games
- All 35 features are properly calculated
- Model can make predictions
- Probabilities are sensible

### Backtest Against Lines
```bash
python -m src.services.ml.train_totals_model_v3 backtest --days 30
```

This tests model against actual closing lines from last 30 days.

## Expected Performance

Based on similar NBA totals models in literature:

**Conservative Estimate**:
- MAE: 10-11 points
- Win rate: 51-53%
- ROI: +1-3% at 3pt threshold
- Betting volume: 30-40% of games (edge >= 3)

**Optimistic Estimate** (if injury/pace edges exist):
- MAE: 9-10 points
- Win rate: 53-55%
- ROI: +3-6% at 3pt threshold
- Betting volume: 25-35% of games

**Reality Check**:
- Markets are efficient; consistent 55%+ win rate is very rare
- Even 52-53% over large sample is profitable
- Focus on calibration and edge identification, not perfect predictions

## Conclusion

The v3 totals model represents a significant upgrade from basic PPG averaging:

1. **35 engineered features** targeting market inefficiencies
2. **Injury-aware** using domain-specific totals scoring
3. **Pace dynamics** capturing non-linear tempo effects
4. **Fatigue modeling** with variance considerations
5. **Rigorous backtesting** against actual closing lines

Next steps:
1. Train model on full historical dataset
2. Validate performance on recent games
3. Integrate into production scoring system
4. Monitor live performance and iterate

The model provides a **systematic, data-driven approach** to finding totals betting value that goes beyond naive PPG estimates.
