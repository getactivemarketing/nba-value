# Model Changes - January 13, 2026

## Overview
Major improvements to the NBA Value Betting model to address overconfidence issues and add new tracking capabilities.

---

## Problem Identified

### 70-79 Value Score Bucket Underperforming
- **Expected**: ~55% win rate
- **Actual**: 40% win rate (6-9 record)
- **Root Cause**: Model's MOV (Margin of Victory) predictions were too extreme
  - Example: Model predicted SAC to lose by 15 points when market line was only 9.5
  - This caused ALL spread picks to favor away teams

### Evidence
```
Before fix: 27/27 spread bets were on AWAY teams
Trained Ridge model: MOV = -15.27 (extreme)
Baseline model: MOV = -7.54 (reasonable)
```

---

## Changes Implemented

### 1. Market Regression (scorer.py)
**File**: `backend/src/services/scoring/scorer.py`

Blends model's MOV prediction with market-implied MOV to reduce overconfidence.

```python
# New parameter
market_regression_weight: float = 0.50  # 50% model, 50% market

# Logic
market_implied_mov = -spread_line  # If line is -5, market expects home to win by 5
blended_mov = (1 - weight) * model_mov + weight * market_implied_mov
```

**Impact**: Model now picks BOTH home and away spreads based on where it finds edge, not just away teams.

---

### 2. Totals Betting Tracking (prediction_tracker.py)
**File**: `backend/src/tasks/prediction_tracker.py`

Added separate tracking for over/under picks.

**New columns in `prediction_snapshots`**:
- `best_total_direction` (over/under)
- `best_total_line`
- `best_total_value_score`
- `best_total_edge`
- `best_total_odds`
- `best_total_result`
- `best_total_profit`

---

### 3. Line Movement Tracking (prediction_tracker.py)
**File**: `backend/src/tasks/prediction_tracker.py`

Tracks opening vs closing lines to identify sharp money movement.

**New columns in `prediction_snapshots`**:
- `opening_spread` - First spread line seen
- `current_spread` - Spread at snapshot time
- `spread_movement` - Difference (positive = moved toward away)
- `opening_total` - First total line seen
- `current_total` - Total at snapshot time
- `total_movement` - Difference (positive = raised)
- `line_movement_direction` - Sharp money indicator

**Direction Labels**:
- `sharp_home` - Line moved toward home team (money on home)
- `sharp_away` - Line moved toward away team (money on away)
- `steam_over` - Total raised significantly
- `steam_under` - Total dropped significantly
- Combined: `sharp_home_over`, `sharp_away_under`, etc.

**New function**: `analyze_line_movement_performance()` - Tracks bet results relative to line movement.

---

### 4. Star Player Injury Weighting (injuries.py)
**File**: `backend/src/services/injuries.py`

Star players now have multiplied impact beyond their raw stats.

**Rationale**: Stars have outsized impact due to:
1. Usage in crunch time
2. Defensive attention (creates open shots for teammates)
3. Gravity/spacing effects
4. Leadership in pressure situations

**Multipliers**:
| Player Type | PPG | APG | Multiplier |
|------------|-----|-----|------------|
| Superstar | 25+ | any | 1.40x |
| Star | 20-25 | any | 1.25x |
| Elite Playmaker | any | 8+ | +0.15x |
| Quality Starter | 15-20 | any | 1.10x |
| Guard Playmaker | any | 5+ | +0.05x |

**Cap**: Maximum multiplier is 1.60x

**Updated Status Weights**:
- Out: 1.0 (unchanged)
- Doubtful: 0.80 (was 0.75)
- Questionable/GTD: 0.50 (was 0.40)
- Probable: 0.15 (was 0.10)

---

### 5. Historical Backtesting Module (backtester.py)
**File**: `backend/src/tasks/backtester.py`

New module for analyzing historical performance.

**Commands**:
```bash
python -m src.tasks.backtester backtest [days]  # Historical spread/total analysis
python -m src.tasks.backtester compare [days]   # Model vs baseline strategies
```

**Baseline Strategies Compared**:
- Blind favorite betting
- Blind underdog betting
- Blind home team betting

**Key Insight from Backtest (60 days)**:
- Underdogs covering at 55.8%
- Big underdogs (7+) covering at 58.2%
- Blind favorite betting: -12.3% ROI (terrible)
- Blind underdog betting: +3.2% ROI (profitable)

---

## Database Migrations

### prediction_snapshots table
```sql
-- Totals tracking
ALTER TABLE prediction_snapshots ADD COLUMN best_total_direction VARCHAR(10);
ALTER TABLE prediction_snapshots ADD COLUMN best_total_line NUMERIC(5,1);
ALTER TABLE prediction_snapshots ADD COLUMN best_total_value_score INTEGER;
ALTER TABLE prediction_snapshots ADD COLUMN best_total_edge NUMERIC(5,2);
ALTER TABLE prediction_snapshots ADD COLUMN best_total_odds NUMERIC(6,3);
ALTER TABLE prediction_snapshots ADD COLUMN best_total_result VARCHAR(20);
ALTER TABLE prediction_snapshots ADD COLUMN best_total_profit NUMERIC(8,2);

-- Line movement tracking
ALTER TABLE prediction_snapshots ADD COLUMN opening_spread NUMERIC(5,1);
ALTER TABLE prediction_snapshots ADD COLUMN current_spread NUMERIC(5,1);
ALTER TABLE prediction_snapshots ADD COLUMN spread_movement NUMERIC(5,1);
ALTER TABLE prediction_snapshots ADD COLUMN opening_total NUMERIC(5,1);
ALTER TABLE prediction_snapshots ADD COLUMN current_total NUMERIC(5,1);
ALTER TABLE prediction_snapshots ADD COLUMN total_movement NUMERIC(5,1);
ALTER TABLE prediction_snapshots ADD COLUMN line_movement_direction VARCHAR(20);
```

---

## Expected Improvements

### Before Changes
- 70-79 bucket: 40% win rate, -$352
- All spread picks on away teams
- Overconfident MOV predictions

### After Changes (Expected)
- 70-79 bucket: ~52-55% win rate
- Balanced home/away spread picks
- More conservative, market-aligned predictions
- Better tracking of totals and line movement
- Proper weighting of star player injuries

---

## Monitoring Plan

### Metrics to Track (Next 7 Days)
1. **Win rate by value score bucket** - Is 70-79 improving?
2. **Home vs away spread distribution** - Are we picking both sides now?
3. **High-value picks (80+)** - Are we maintaining edge?
4. **Line movement correlation** - Does betting with/against sharp money matter?
5. **Star injury impact** - Are games with star injuries grading correctly?

### Commands for Monitoring
```bash
# Daily performance summary
python -m src.tasks.prediction_tracker summary 7

# Line movement analysis
python -m src.tasks.prediction_tracker line-movement 14

# Model vs baseline comparison
python -m src.tasks.backtester compare 7

# Run injury report
python -m src.services.injuries
```

---

## Git Commits

1. `4338a02` - Add market regression and totals betting support
2. `1de94cd` - Add line movement tracking to predictions
3. `25b68fe` - Add star player multiplier to injury impact calculation
4. `3fda150` - Add historical backtesting module

---

## Rollback Plan

If model performance degrades significantly:

1. **Market Regression**: Change `market_regression_weight` from 0.50 to 0.0 in `scorer.py`
2. **Star Multiplier**: Remove `* self.star_multiplier` from impact calculations in `injuries.py`
3. **Status Weights**: Revert to original values (Questionable: 0.4, Doubtful: 0.75)

---

## Notes

- Changes deployed to Railway on Jan 13, 2026
- First picks using new model will be for Jan 14 games
- Jan 13 results (2-4) used OLD model before deployment
- Need 50+ graded picks to evaluate statistical significance of changes
