# Advanced Totals Model V3 - Feature Engineering Documentation

## Overview

The v3 totals model incorporates **35 features** designed to capture market inefficiencies that basic PPG/pace models miss. The hypothesis is that betting markets efficiently price obvious factors (team PPG, pace) but may underweight:

1. **Injury impact asymmetry**
2. **Non-linear pace dynamics**
3. **Fatigue effects on variance**
4. **Recent form volatility**
5. **Venue-specific scoring patterns**

## Feature Categories (35 features total)

### 1. Basic Stats (8 features)
Standard team metrics that markets price efficiently:

```
- home_pace, away_pace         # Possessions per 48 min
- home_ortg, away_ortg         # Offensive rating (pts/100 poss)
- home_drtg, away_drtg         # Defensive rating (opp pts/100 poss)
- home_ppg, away_ppg           # Points per game (10-game avg)
```

**Market Efficiency**: HIGH - These are publicly available and heavily weighted by oddsmakers.

### 2. Pace Interaction Features (4 features)

**Hypothesis**: When fast and slow teams meet, the resulting pace is non-linear and affects total variance.

```python
def calculate_pace_interaction(home_pace, away_pace):
    # Simple average (what basic models use)
    avg_pace = (home_pace + away_pace) / 2

    # Weighted toward faster team (they control tempo)
    faster = max(home_pace, away_pace)
    slower = min(home_pace, away_pace)
    weighted_pace = faster * 0.60 + slower * 0.40

    # Pace clash (style difference creates variance)
    pace_clash = abs(home_pace - away_pace)

    # Variance metric (higher = less predictable)
    pace_variance = pace_clash / avg_pace
```

**Features**:
- `avg_pace` - Standard average
- `weighted_pace` - Tempo control factor
- `pace_clash` - Style mismatch magnitude
- `pace_variance` - Predictability metric

**Market Edge**: When pace_clash > 5 possessions, totals have higher variance. Markets may underprice this uncertainty.

### 3. Rest & Fatigue Asymmetry (5 features)

**Hypothesis**: Fatigue hurts defense more than offense, creating asymmetric scoring effects.

**Key Insights**:
- B2B teams allow ~3-5 more PPG (defense suffers)
- B2B teams score ~1-2 less PPG (offense slightly worse)
- Net effect: B2B games trend slightly OVER
- **BUT**: When BOTH teams are on B2B, variance increases (unpredictable)

```python
def calculate_rest_asymmetry(home_rest, away_rest, home_b2b, away_b2b):
    # Rest advantage (positive = home more rested)
    rest_advantage = home_rest - away_rest

    # Total fatigue score
    total_fatigue = (1 if home_b2b else 0) + (1 if away_b2b else 0)

    # Asymmetry (one tired, one fresh = variance)
    fatigue_asymmetry = abs(total_fatigue)

    # Well-rested flags (2+ days rest, not B2B)
    home_well_rested = 1 if home_rest >= 2 and not home_b2b else 0
    away_well_rested = 1 if away_rest >= 2 and not away_b2b else 0
```

**Features**:
- `rest_advantage` - Home rest edge
- `total_fatigue` - Combined B2B penalty
- `fatigue_asymmetry` - Variance from mismatch
- `home_well_rested`, `away_well_rested` - Recovery flags

**Market Edge**: Markets price B2B but may miss the **variance** component when only one team is fatigued.

### 4. Injury Impact (5 features)

**Hypothesis**: Heavy injuries cause non-linear effects on totals.

**Key Insights** (from `injuries.py`):
- Injured team scores less (obvious)
- Opponent ALSO scores less (pace slows, fewer possessions)
- Net effect: When one team is severely injured, totals trend UNDER
- Asymmetric injuries create variance

```python
# Uses totals_injury_score from injuries.py
# Weighting: Rebounding 35%, Defense 25%, Scoring 25%, Playmaking 15%
def calculate_injury_impact(home_team_id, away_team_id, injury_reports):
    home_injury = injury_reports[home_team].totals_injury_score  # 0-1 scale
    away_injury = injury_reports[away_team].totals_injury_score

    injury_asymmetry = abs(home_injury - away_injury)
    total_injury = home_injury + away_injury  # Both injured = UNDER
    injury_edge = away_injury - home_injury   # Spread effect
```

**Features**:
- `home_injury_score`, `away_injury_score` - 0-1 injury severity
- `injury_asymmetry` - Mismatch creates variance
- `total_injury` - Combined injury (high = UNDER)
- `injury_edge` - Directional spread effect

**Market Edge**: Markets adjust lines for star players but may miss:
1. **Pace impact** of missing rebounders (fewer 2nd chance points)
2. **Systemic effects** when multiple role players are out
3. **Recency weighting** (recent injury = team not adjusted yet)

### 5. Scoring Variance (2 features)

**Hypothesis**: Teams on hot/cold streaks with high variance are less predictable.

```python
# Standard deviation of last 5 games' scores
home_scoring_std = np.std([prev1_pts, prev2_pts, prev3_pts, prev4_pts, prev5_pts])
away_scoring_std = np.std([same for away])
```

**Features**:
- `home_scoring_std` - Recent scoring volatility
- `away_scoring_std` - Away scoring volatility

**Market Edge**: High variance teams (std > 12 pts) are less predictable. Markets may overprice their recent average.

### 6. Home/Away Splits (4 features)

**Hypothesis**: Some teams have extreme venue effects on offense.

**Key Insights**:
- Average home court advantage: ~2-3 PPG
- But some teams: +5-8 PPG at home (elite home court)
- Road teams in hostile environments: -5-8 PPG

```python
# Calculate win% at home (for home team) and away (for away team)
home_home_win_pct = home_wins_at_home / (home_wins + home_losses)
away_away_win_pct = away_wins_on_road / (away_wins + away_losses)

# Flags for extreme venue effects
home_ortg_home_boost = 1 if home_home_win_pct > 0.55 else 0  # Elite home court
away_ortg_away_penalty = 1 if away_away_win_pct < 0.45 else 0  # Struggles on road
```

**Features**:
- `home_home_win_pct` - Home team's home record
- `away_away_win_pct` - Away team's road record
- `home_ortg_home_boost` - Elite home offense flag
- `away_ortg_away_penalty` - Poor road offense flag

**Market Edge**: Markets use average home court (2.5 pts). Teams with extreme splits may be mispriced.

### 7. O/U Tendency (2 features)

**Hypothesis**: Some teams consistently go over/under due to pace or style.

```python
# From team_stats: ou_overs_l10, ou_unders_l10
home_ou_over_pct = home_overs / (home_overs + home_unders)  # Last 10 games
away_ou_over_pct = away_overs / (away_overs + away_unders)
```

**Features**:
- `home_ou_over_pct` - Home team's O/U tendency
- `away_ou_over_pct` - Away team's O/U tendency

**Market Edge**: Weak signal (mean reversion is strong), but teams with extreme tendencies (>65% or <35%) may have persistent pace effects.

## Feature Importance Expectations

**High Importance** (markets already price these):
1. `home_ppg`, `away_ppg` - Direct scoring
2. `home_ortg`, `away_ortg` - Efficiency
3. `avg_pace` - Possessions

**Medium Importance** (potential edge):
4. `weighted_pace` - Tempo control
5. `total_injury` - Combined injury impact
6. `home_scoring_std`, `away_scoring_std` - Volatility
7. `total_fatigue` - B2B effects

**Low Importance** (weak signals):
8. `home_ou_over_pct`, `away_ou_over_pct` - Mean reversion
9. `pace_variance` - Second-order effect
10. `home_ortg_home_boost` - Niche edge

## Model Architecture

### Ridge Regression (default)
```python
Ridge(alpha=10.0)
```
- **Pros**: Stable, interpretable, fast training
- **Cons**: Cannot capture non-linear interactions
- **Use case**: Baseline, production inference

### LightGBM (optional, use `--lgbm` flag)
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
- **Pros**: Captures feature interactions, better MAE
- **Cons**: Slower, overfitting risk, less interpretable
- **Use case**: Experimental, if Ridge underperforms

## Training & Evaluation

### Cross-Validation Strategy
```python
TimeSeriesSplit(n_splits=5)
```
- **No future leakage**: Train on past, validate on future
- **Fold size**: ~20% of data per fold
- **Metrics**: MAE, RMSE, win rate vs closing lines

### Success Criteria

**Baseline (v2 model)**:
- MAE: ~11-12 points
- Win rate vs closing lines: ~48-50%

**V3 Target**:
- MAE: <11 points (improvement)
- Win rate: >52.4% (breakeven at -110)
- **Edge threshold**: 3+ pts prediction vs line
- **Expected ROI**: +2-5% at 3pt threshold

### Evaluation Against Market

```python
# Bet OVER if prediction > line + threshold
# Bet UNDER if prediction < line - threshold
# Pass if abs(diff) < threshold

thresholds = [0, 1, 2, 3, 4, 5]

# Success = win_rate > 52.4% at any threshold with reasonable bet volume
```

## Usage

### Training
```bash
# Train with Ridge (default)
python -m src.services.ml.train_totals_model_v3

# Train with LightGBM
python -m src.services.ml.train_totals_model_v3 --lgbm
```

### Backtesting
```bash
# Backtest last 30 days
python -m src.services.ml.train_totals_model_v3 backtest --days 30
```

### Model Output
```
models/totals_model_v3.pkl
{
    'model': Ridge/LGBMRegressor instance,
    'feature_names': List[str] (35 features),
    'total_std': float (for probability conversion),
    'model_type': 'ridge_totals_v3' or 'lgbm_totals_v3',
    'training_games': int,
    'avg_mae': float,
    'trained_at': ISO timestamp,
}
```

## Integration with Scoring System

### Prediction Time
```python
# Load model
with open('models/totals_model_v3.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_names = model_data['feature_names']
total_std = model_data['total_std']

# Build feature vector (must match training order)
features = build_feature_vector(game, team_stats, injury_reports)

# Predict total
predicted_total = model.predict([features])[0]

# Convert to probability vs line
from scipy.stats import norm
prob_over = 1 - norm.cdf(closing_total, predicted_total, total_std)
prob_under = norm.cdf(closing_total, predicted_total, total_std)
```

### Value Scoring
```python
# Calculate edge
edge = abs(predicted_total - closing_total)

# Edge threshold for betting (3+ pts = 55%+ win rate)
if edge >= 3.0:
    if predicted_total > closing_total:
        # Bet OVER
        value_score = edge * prob_over * confidence_multiplier
    else:
        # Bet UNDER
        value_score = edge * prob_under * confidence_multiplier
```

## Rollback Criteria

Revert to v2 model if:
1. **MAE increases** by >1 point vs v2
2. **Win rate vs lines** < 48% (worse than coin flip)
3. **Calibration issues**: Brier score > 0.30
4. **Production issues**: Inference latency > 100ms

## Future Improvements (v4)

1. **Historical injury data**: Currently uses current injuries for all games (data limitation). Fetch injury reports at game time for accuracy.

2. **Weather data**: Outdoor courts (if applicable) or arena temperature for pace effects.

3. **Referee tendencies**: Some refs call tighter games (fewer FTs, lower scoring).

4. **Line movement signals**: Sharp money indicators from opening to closing line.

5. **Player prop correlation**: If star player O/U props are getting heavy action, it may signal insider info.

6. **Ensemble methods**: Combine Ridge + LightGBM predictions with weighted voting.

7. **Recalibration**: Monthly retraining to adapt to meta changes (league pace trends).

## References

- Injury scoring: `/backend/src/services/injuries.py`
- Team stats schema: `/backend/src/models/team_stats.py`
- Game results: `/backend/src/models/game_result.py`
- Previous models: `train_totals_model.py`, `train_totals_model_v2.py`
