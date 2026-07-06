# MLB Model Features v2 — Real Stats + First Inning Data

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded defaults in the MLB scorer with real batting splits (BA, OBP, SLG), team ERA/WHIP, starter IP from the database, and add first inning scoring features to the model.

**Architecture:** The trained LightGBM model expects 28 features in `MODEL_FEATURE_NAMES` order. The scorer's `_build_model_feature_vector()` maps `MLBGameFeatures` to that vector but hardcodes 8 values (BA, OBP, SLG, team WHIP, starter IP) because `MLBGameFeatures` and `MLBTeamStats` don't carry them. We add those fields to `MLBGameFeatures`, populate them from `MLBTeamStats`/`MLBPitcherStats` in the feature calculator, and wire them into the scorer. Separately, we add first inning scoring features (score_pct, avg_runs for each team) as new model inputs for a future v2 model retrain, while keeping the existing v1 model working via a clean fallback.

**Tech Stack:** Python, SQLAlchemy, LightGBM, PostgreSQL

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `backend/src/services/mlb/features.py` | Modify | Add BA/OBP/SLG, team ERA/WHIP, starter IP, first inning fields to `MLBGameFeatures` + populate in calculator |
| `backend/src/services/mlb/scorer.py` | Modify | Replace hardcoded defaults with real feature values in `_build_model_feature_vector()`, add first inning features to heuristic fallback |
| `backend/src/database.py` | Modify | Add `innings_pitched` column migration for `mlb_pitcher_stats` (already exists in model) |

---

### Task 1: Add real batting splits and team pitching fields to MLBGameFeatures

**Files:**
- Modify: `backend/src/services/mlb/features.py:19-89` (MLBGameFeatures dataclass)

- [ ] **Step 1: Add new fields to the MLBGameFeatures dataclass**

Add these fields after the existing team offensive/defensive features in `backend/src/services/mlb/features.py`:

```python
# In MLBGameFeatures dataclass, after away_ops (line ~49):

    # Batting splits (BA, OBP, SLG)
    home_batting_avg: float | None = None
    away_batting_avg: float | None = None
    home_obp: float | None = None
    away_obp: float | None = None
    home_slg: float | None = None
    away_slg: float | None = None

    # Team pitching (full staff)
    home_team_era: float | None = None
    away_team_era: float | None = None
    home_team_whip: float | None = None
    away_team_whip: float | None = None

    # Starter workload
    home_starter_ip: float | None = None
    away_starter_ip: float | None = None

    # First inning scoring tendencies
    home_first_inning_score_pct: float | None = None
    away_first_inning_score_pct: float | None = None
    home_first_inning_runs_avg: float | None = None
    away_first_inning_runs_avg: float | None = None
```

- [ ] **Step 2: Update get_feature_vector() to include the new fields**

In `get_feature_vector()` (line ~101), add the new features at the end of the list, after `total_run_environment`:

```python
            # New v2 features
            self.home_batting_avg or 0.250,
            self.away_batting_avg or 0.250,
            self.home_obp or 0.320,
            self.away_obp or 0.320,
            self.home_slg or 0.400,
            self.away_slg or 0.400,
            self.home_team_era or 4.00,
            self.away_team_era or 4.00,
            self.home_team_whip or 1.30,
            self.away_team_whip or 1.30,
            self.home_starter_ip or 100.0,
            self.away_starter_ip or 100.0,
            self.home_first_inning_score_pct or 0.30,
            self.away_first_inning_score_pct or 0.30,
            self.home_first_inning_runs_avg or 0.50,
            self.away_first_inning_runs_avg or 0.50,
```

- [ ] **Step 3: Update get_feature_names() to match**

In `get_feature_names()` (line ~141), add matching names at the end:

```python
            # New v2 features
            "home_batting_avg",
            "away_batting_avg",
            "home_obp",
            "away_obp",
            "home_slg",
            "away_slg",
            "home_team_era",
            "away_team_era",
            "home_team_whip",
            "away_team_whip",
            "home_starter_ip",
            "away_starter_ip",
            "home_first_inning_score_pct",
            "away_first_inning_score_pct",
            "home_first_inning_runs_avg",
            "away_first_inning_runs_avg",
```

- [ ] **Step 4: Commit**

```bash
git add backend/src/services/mlb/features.py
git commit -m "feat: add batting splits, team pitching, starter IP, and first inning fields to MLBGameFeatures"
```

---

### Task 2: Populate the new features from the database

**Files:**
- Modify: `backend/src/services/mlb/features.py:240-300` (MLBFeatureCalculator.calculate_game_features)

- [ ] **Step 1: Add batting splits and team pitching to the home_stats block**

In `calculate_game_features()`, inside the `if home_stats:` block (around line 243), add after the existing assignments:

```python
            features.home_batting_avg = float(home_stats.batting_avg) if home_stats.batting_avg else None
            features.home_obp = float(home_stats.obp) if home_stats.obp else None
            features.home_slg = float(home_stats.slg) if home_stats.slg else None
            features.home_team_era = float(home_stats.era) if home_stats.era else None
            # first inning stats
            features.home_first_inning_score_pct = float(home_stats.first_inning_score_pct) if home_stats.first_inning_score_pct else None
            features.home_first_inning_runs_avg = float(home_stats.first_inning_runs_avg) if home_stats.first_inning_runs_avg else None
```

- [ ] **Step 2: Add batting splits and team pitching to the away_stats block**

In the `if away_stats:` block (around line 261), add after the existing assignments:

```python
            features.away_batting_avg = float(away_stats.batting_avg) if away_stats.batting_avg else None
            features.away_obp = float(away_stats.obp) if away_stats.obp else None
            features.away_slg = float(away_stats.slg) if away_stats.slg else None
            features.away_team_era = float(away_stats.era) if away_stats.era else None
            features.away_first_inning_score_pct = float(away_stats.first_inning_score_pct) if away_stats.first_inning_score_pct else None
            features.away_first_inning_runs_avg = float(away_stats.first_inning_runs_avg) if away_stats.first_inning_runs_avg else None
```

- [ ] **Step 3: Add starter IP from pitcher stats**

In the pitcher stats blocks (around lines 206-227), add innings_pitched after the existing assignments. Inside the `if home_pitcher_stats:` block:

```python
                features.home_starter_ip = float(home_pitcher_stats.innings_pitched) if home_pitcher_stats.innings_pitched else None
```

And inside the `if away_pitcher_stats:` block:

```python
                features.away_starter_ip = float(away_pitcher_stats.innings_pitched) if away_pitcher_stats.innings_pitched else None
```

- [ ] **Step 4: Commit**

```bash
git add backend/src/services/mlb/features.py
git commit -m "feat: populate batting splits, team ERA, starter IP, and first inning features from database"
```

---

### Task 3: Wire real values into the scorer's model feature vector

**Files:**
- Modify: `backend/src/services/mlb/scorer.py:130-168` (_build_model_feature_vector)

- [ ] **Step 1: Replace hardcoded defaults with real feature values**

Replace the `_build_model_feature_vector` method (lines ~130-168) with:

```python
    def _build_model_feature_vector(self, features: MLBGameFeatures) -> np.ndarray:
        """
        Build feature vector matching the trained model's expected features.

        The model was trained on specific features in a specific order.
        This method maps MLBGameFeatures to that expected format.
        """
        vector = [
            features.home_runs_per_game or self.AVG_RUNS_PER_TEAM,
            features.away_runs_per_game or self.AVG_RUNS_PER_TEAM,
            features.home_ops or 0.720,
            features.away_ops or 0.720,
            features.home_batting_avg or 0.250,
            features.away_batting_avg or 0.250,
            features.home_obp or 0.320,
            features.away_obp or 0.320,
            features.home_slg or 0.400,
            features.away_slg or 0.400,
            features.home_team_era or 4.00,
            features.away_team_era or 4.00,
            features.home_team_whip or 1.30,
            features.away_team_whip or 1.30,
            features.home_starter_era or 4.00,
            features.away_starter_era or 4.00,
            features.home_starter_whip or 1.25,
            features.away_starter_whip or 1.25,
            features.home_starter_k_rate or 8.5,
            features.away_starter_k_rate or 8.5,
            features.home_starter_bb_rate or 3.0,
            features.away_starter_bb_rate or 3.0,
            features.home_starter_ip or 100.0,
            features.away_starter_ip or 100.0,
            features.park_factor,
            features.offense_matchup_edge or 0.0,
            features.starter_era_diff or 0.0,
            (features.away_team_era or 4.0) - (features.home_team_era or 4.0),  # team_era_diff
        ]
        return np.array([vector])
```

This replaces the 8 hardcoded values (`0.250`, `0.320`, `0.400`, `1.30`, `100.0`) with real values from `MLBGameFeatures`, falling back to the same defaults when data is missing.

- [ ] **Step 2: Add first inning features to the heuristic fallback**

In `_estimate_run_diff()` (find it in scorer.py), add first inning weighting. Find the method and add this adjustment before the return:

```python
        # First inning scoring tendency adjustment
        # Teams that score more in the 1st tend to jump ahead, slight advantage
        if features.home_first_inning_score_pct and features.away_first_inning_score_pct:
            fi_diff = features.home_first_inning_score_pct - features.away_first_inning_score_pct
            run_diff += fi_diff * 0.3  # Small weight
```

- [ ] **Step 3: Commit**

```bash
git add backend/src/services/mlb/scorer.py
git commit -m "feat: use real batting splits, team ERA/WHIP, starter IP in model feature vector"
```

---

### Task 4: Add team WHIP column migration and populate it

**Files:**
- Modify: `backend/src/models/mlb_team_stats.py` (add team_whip column)
- Modify: `backend/src/database.py` (add column migration)
- Modify: `backend/src/services/mlb/ingest.py` (populate team_whip in update_team_stats)

- [ ] **Step 1: Add team_whip field to MLBTeamStats model**

In `backend/src/models/mlb_team_stats.py`, after `bullpen_era` (line ~38):

```python
    team_whip: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)
```

- [ ] **Step 2: Add column migration in database.py**

In `backend/src/database.py`, add to the `column_migrations` list:

```python
        ("mlb_team_stats", "team_whip", "NUMERIC(5,3)"),
```

- [ ] **Step 3: Populate team_whip in the ingest pipeline**

In `backend/src/services/mlb/ingest.py`, in the `update_team_stats()` method, the standings data from `get_team_standings()` needs to be checked for WHIP availability. If the standings API doesn't provide WHIP directly, derive it from team pitching stats, or set it to None for now so the feature falls back to the default.

In the `insert(MLBTeamStats).values(...)` call inside `update_team_stats()`, add:

```python
                team_whip=None,  # Will be populated when pitching stats ingestion adds WHIP
```

And in the `.on_conflict_do_update` set_ dict, add:

```python
                    "team_whip": None,
```

- [ ] **Step 4: Wire team_whip into the feature calculator**

In `backend/src/services/mlb/features.py`, in the `if home_stats:` block, add:

```python
            features.home_team_whip = float(home_stats.team_whip) if home_stats.team_whip else None
```

And in the `if away_stats:` block:

```python
            features.away_team_whip = float(away_stats.team_whip) if away_stats.team_whip else None
```

- [ ] **Step 5: Commit**

```bash
git add backend/src/models/mlb_team_stats.py backend/src/database.py backend/src/services/mlb/ingest.py backend/src/services/mlb/features.py
git commit -m "feat: add team_whip column and wire into feature pipeline"
```

---

### Task 5: Update training data builder to include new features

**Files:**
- Modify: `backend/src/services/mlb/features.py:360-416` (build_training_data)

- [ ] **Step 1: Ensure build_training_data captures new fields**

The `build_training_data()` function at line 360 already calls `features.to_dict()` which serializes all fields. Since we added the new fields to the dataclass, they'll be automatically included in training data export. No code change needed here — just verify by reading the function.

- [ ] **Step 2: Update MODEL_FEATURE_NAMES for future v2 model training**

The v1 model uses 28 features defined in `MODEL_FEATURE_NAMES`. Adding features there would break the existing model. Instead, add a `V2_FEATURE_NAMES` constant to `scorer.py` that includes the new features, for use when retraining:

In `backend/src/services/mlb/scorer.py`, after `MODEL_FEATURE_NAMES` (line ~98):

```python
    # V2 features — includes first inning data, used when retraining
    V2_FEATURE_NAMES = MODEL_FEATURE_NAMES + [
        "home_first_inning_score_pct",
        "away_first_inning_score_pct",
        "home_first_inning_runs_avg",
        "away_first_inning_runs_avg",
    ]
```

- [ ] **Step 3: Commit**

```bash
git add backend/src/services/mlb/scorer.py
git commit -m "feat: add V2_FEATURE_NAMES for future model retrain with first inning data"
```

---

### Task 6: Final integration commit

- [ ] **Step 1: Verify no import errors**

```bash
cd backend && python3 -c "
from src.services.mlb.features import MLBGameFeatures, MLBFeatureCalculator
from src.services.mlb.scorer import MLBScorer
f = MLBGameFeatures(game_id='test', game_date=__import__('datetime').date.today(), home_team='NYY', away_team='BOS')
print(f'Feature vector length: {len(f.get_feature_vector())}')
print(f'Feature names length: {len(f.get_feature_names())}')
print(f'v1 model features: {len(MLBScorer.MODEL_FEATURE_NAMES)}')
print(f'v2 model features: {len(MLBScorer.V2_FEATURE_NAMES)}')
assert len(f.get_feature_vector()) == len(f.get_feature_names()), 'Feature vector/names mismatch!'
print('All checks passed')
"
```

Expected: Feature vector and names lengths match, v1=28, v2=32, no import errors.

- [ ] **Step 2: Push to deploy**

```bash
git push origin main
```
