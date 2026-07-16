# NFL Model — Phase 2: MOV + Totals Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train two LightGBM regressors — a margin-of-victory (MOV) model driving spread + moneyline probabilities and a totals model driving over/under — on the Phase 1 `nfl_*` data, validated walk-forward with a written backtest (ATS%, units, calibration, saturation) before any of it is wired to scoring.

**Architecture:** A training-data builder joins each game to its point-in-time team features (`nfl_team_stats` through week `w-1`) plus targets/lines, producing a leakage-free feature matrix. Two LightGBM regressors predict `margin` (home−away) and `total` points; their validation-residual standard deviations convert point predictions into probabilities via the **existing** `services/ml/probability.py` helpers (`mov_to_spread_prob`, `mov_to_moneyline_prob`, `mov_to_total_prob`). A walk-forward backtest grades picks against nflverse historical closing lines (`spread_line`, `total_line`, moneylines).

**Tech Stack:** Python 3.11+, LightGBM, scikit-learn, scipy, pandas, joblib, SQLAlchemy 2.0 async, structlog, pytest.

## Global Constraints

- **Naming:** NFL artifacts/files/config prefixed `nfl_` / `NFL`.
- **Point-in-time correctness:** a game in week `w` may only use features from `nfl_team_stats.through_week = w-1` (same season). No feature may reference week `w` or later. Week-1 games (no `through_week=0` row) are excluded from modeling — document, don't hack a cold-start.
- **Reuse, don't reinvent:** probability conversion uses `src/services/ml/probability.py` verbatim; calibration uses `src/services/ml/calibration.py::CalibrationLayer`; the residual-std pattern follows `src/services/ml/mov_model.py`. Do not write new norm.cdf math.
- **No leakage in backtest:** walk-forward only — to predict season `S`, train exclusively on seasons `< S`. Never fit on the test season.
- **Spread sign convention (fixed for the whole plan):** nflverse `spread_line` is **points the home team is favored by** (positive ⇒ home favored; the sample game `2023_01_DET_KC` has home KC `spread_line=4.0`, `home_moneyline=-198`). Home covers iff `margin > spread_line`. Totals: over hits iff `home_score+away_score > total_line`.
- **Git staging:** stage specific files only. Never `git add -A` / `git add .`.
- **Push:** commit locally; do not push unless explicitly asked.
- **Commit trailer:** end messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **DB:** no local Postgres — training/backtest read the prod DB via `export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)`. Reads only; Phase 2 writes no DB rows (only model artifacts on disk).
- **Model artifacts:** `models/nfl_mov_v1.joblib`, `models/nfl_totals_v1.joblib`; each bundle stores `{model, feature_cols, resid_std, trained_seasons}`.

## File Structure

**Create:**
- `src/services/nfl/training_data.py` — build the leakage-free feature matrix (DB features + nflverse targets/lines).
- `src/services/nfl/model_training.py` — train MOV + totals LightGBM regressors; compute residual std; save/load bundles.
- `src/services/nfl/backtest.py` — walk-forward backtest + metrics (ATS%, units, calibration, saturation).
- `src/tasks/nfl_train.py` — CLI: build data → train both → save artifacts → print metrics.
- `tests/unit/test_nfl_training_data.py`, `test_nfl_model_training.py`, `test_nfl_backtest.py`.

**Modify:**
- `src/config.py` — add `nfl_mov_model_path`, `nfl_totals_model_path`.

**Reuse (do not modify):** `src/services/ml/probability.py`, `src/services/ml/calibration.py`.

**Out of scope (later phases):** scorer/value_calculator (P3), `nfl_prediction_snapshots` + live scoring (P3), scheduler/API (P4), QB-injury adjustment (P2.5).

---

### Task 1: Training-data builder (leakage-free feature matrix)

**Files:**
- Create: `src/services/nfl/training_data.py`
- Test: `tests/unit/test_nfl_training_data.py`

**Interfaces:**
- Produces:
  - `build_feature_frame(games_df, team_stats_df, context_df, lines_df) -> pd.DataFrame` — **pure** join over already-loaded frames. One row per modelable game (week ≥ 2, both teams have a `through_week = week-1` row). Columns: identity (`game_id, season, week, home_team, away_team`), features (see below), targets (`margin`, `total`), lines (`spread_line`, `total_line`, `home_moneyline`, `away_moneyline`).
  - `async load_training_frames(session, seasons) -> tuple[pd.DataFrame, ...]` — loads `nfl_games`, `nfl_team_stats`, `nfl_game_context` for the seasons from the DB, and `lines` from nflverse (`load_schedules` → `game_id, spread_line, total_line, home_moneyline, away_moneyline`). Thin I/O wrapper; `build_feature_frame` gets the real test.
- **Features produced** (all point-in-time from `through_week = week-1`):
  - Differentials: `off_epa_diff, def_epa_diff, pass_epa_diff, rush_epa_diff, success_rate_diff, pace_diff, power_diff` (each = home_stat − away_stat).
  - Rest: `rest_diff` (= home_rest_days − away_rest_days).
  - Flags: `is_divisional` (int 0/1), `is_primetime` (int 0/1).
  - Totals-specific (kept in the frame for the totals model): `pace_sum` (home+away pace), `off_epa_sum`, `is_dome` (int), `wind_mph`, `temp_f`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_training_data.py
import pandas as pd
from src.services.nfl.training_data import build_feature_frame


def _frames():
    # Two teams, season 2023. Team stats exist through_week 1 (entering wk2).
    games = pd.DataFrame([
        {"game_id": "G2", "season": 2023, "week": 2, "home_team": "KC", "away_team": "DET",
         "home_score": 24, "away_score": 17, "is_divisional": False, "is_primetime": True},
        {"game_id": "G1", "season": 2023, "week": 1, "home_team": "KC", "away_team": "DET",
         "home_score": 20, "away_score": 21, "is_divisional": False, "is_primetime": True},
    ])
    team_stats = pd.DataFrame([
        {"team": "KC", "season": 2023, "through_week": 1, "off_epa_play": 0.1, "def_epa_play": -0.05,
         "pass_epa": 0.2, "rush_epa": 0.0, "success_rate": 0.48, "pace": 62.0, "power_rating": 0.15},
        {"team": "DET", "season": 2023, "through_week": 1, "off_epa_play": 0.0, "def_epa_play": 0.05,
         "pass_epa": 0.1, "rush_epa": -0.1, "success_rate": 0.45, "pace": 64.0, "power_rating": -0.05},
    ])
    context = pd.DataFrame([
        {"game_id": "G2", "home_rest_days": 7, "away_rest_days": 7, "is_dome": False,
         "wind_mph": 5.0, "temp_f": 60.0},
        {"game_id": "G1", "home_rest_days": 7, "away_rest_days": 7, "is_dome": False,
         "wind_mph": None, "temp_f": None},
    ])
    lines = pd.DataFrame([
        {"game_id": "G2", "spread_line": 3.0, "total_line": 47.0,
         "home_moneyline": -160, "away_moneyline": 140},
        {"game_id": "G1", "spread_line": 4.0, "total_line": 53.0,
         "home_moneyline": -198, "away_moneyline": 164},
    ])
    return games, team_stats, context, lines


def test_week1_excluded_and_features_point_in_time():
    frame = build_feature_frame(*_frames())
    # Only week-2 game G2 is modelable (week 1 has no through_week=0 stats)
    assert list(frame["game_id"]) == ["G2"]
    row = frame.iloc[0]
    # diffs = home(KC) - away(DET)
    assert round(row["off_epa_diff"], 3) == 0.1        # 0.1 - 0.0
    assert round(row["power_diff"], 3) == 0.2          # 0.15 - (-0.05)
    assert round(row["pace_sum"], 1) == 126.0          # 62 + 64
    assert row["is_primetime"] == 1
    # targets + lines
    assert row["margin"] == 7                           # 24 - 17
    assert row["total"] == 41                            # 24 + 17
    assert row["spread_line"] == 3.0
    assert row["total_line"] == 47.0


def test_missing_team_stats_row_drops_game():
    games, team_stats, context, lines = _frames()
    # Remove DET's stats -> G2 can't be built
    team_stats = team_stats[team_stats["team"] != "DET"]
    frame = build_feature_frame(games, team_stats, context, lines)
    assert len(frame) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_nfl_training_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.nfl.training_data'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/nfl/training_data.py
"""Build the leakage-free NFL modeling feature matrix.

Features come from nfl_team_stats at through_week = game.week - 1 (point-in-time);
targets (margin, total) from nfl_games scores; betting lines from nflverse
schedules. Week-1 games have no through_week=0 stats row and are excluded.
"""
import pandas as pd
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import NFLGame, NFLTeamStats, NFLGameContext
from src.services.nfl.nfl_data import load_schedules

logger = structlog.get_logger()

_DIFF_COLS = ["off_epa_play", "def_epa_play", "pass_epa", "rush_epa",
              "success_rate", "pace", "power_rating"]


def build_feature_frame(
    games: pd.DataFrame, team_stats: pd.DataFrame,
    context: pd.DataFrame, lines: pd.DataFrame,
) -> pd.DataFrame:
    ts = team_stats.set_index(["team", "season", "through_week"])
    ctx = context.set_index("game_id")
    ln = lines.set_index("game_id")
    rows = []
    for _, g in games.iterrows():
        w = int(g["week"])
        if w < 2:
            continue  # no through_week = w-1 row exists
        key_h = (g["home_team"], int(g["season"]), w - 1)
        key_a = (g["away_team"], int(g["season"]), w - 1)
        if key_h not in ts.index or key_a not in ts.index:
            continue
        h, a = ts.loc[key_h], ts.loc[key_a]
        c = ctx.loc[g["game_id"]] if g["game_id"] in ctx.index else None
        li = ln.loc[g["game_id"]] if g["game_id"] in ln.index else None
        if li is None:
            continue  # no line -> not gradable
        row = {
            "game_id": g["game_id"], "season": int(g["season"]), "week": w,
            "home_team": g["home_team"], "away_team": g["away_team"],
            "is_divisional": int(bool(g["is_divisional"])),
            "is_primetime": int(bool(g["is_primetime"])),
            "margin": int(g["home_score"]) - int(g["away_score"]),
            "total": int(g["home_score"]) + int(g["away_score"]),
            "spread_line": float(li["spread_line"]),
            "total_line": float(li["total_line"]),
            "home_moneyline": None if pd.isna(li["home_moneyline"]) else float(li["home_moneyline"]),
            "away_moneyline": None if pd.isna(li["away_moneyline"]) else float(li["away_moneyline"]),
            "pace_sum": float(h["pace"]) + float(a["pace"]),
            "off_epa_sum": float(h["off_epa_play"]) + float(a["off_epa_play"]),
            "rest_diff": (0 if c is None or pd.isna(c["home_rest_days"]) else int(c["home_rest_days"]))
                         - (0 if c is None or pd.isna(c["away_rest_days"]) else int(c["away_rest_days"])),
            "is_dome": 0 if c is None else int(bool(c["is_dome"])),
            "wind_mph": None if c is None or pd.isna(c["wind_mph"]) else float(c["wind_mph"]),
            "temp_f": None if c is None or pd.isna(c["temp_f"]) else float(c["temp_f"]),
        }
        for col in _DIFF_COLS:
            row[f"{col.replace('_play','')}_diff" if col.endswith("_play") else f"{col}_diff"] = \
                float(h[col]) - float(a[col])
        rows.append(row)
    return pd.DataFrame(rows)


async def load_training_frames(session: AsyncSession, seasons: list[int]):
    games = pd.DataFrame([r._mapping for r in (await session.execute(
        select(NFLGame.game_id, NFLGame.season, NFLGame.week, NFLGame.home_team,
               NFLGame.away_team, NFLGame.home_score, NFLGame.away_score,
               NFLGame.is_divisional, NFLGame.is_primetime)
        .where(NFLGame.season.in_(seasons), NFLGame.home_score.isnot(None)))).all()])
    team_stats = pd.DataFrame([dict(
        team=r.team, season=r.season, through_week=r.through_week,
        off_epa_play=r.off_epa_play, def_epa_play=r.def_epa_play, pass_epa=r.pass_epa,
        rush_epa=r.rush_epa, success_rate=r.success_rate, pace=r.pace,
        power_rating=r.power_rating)
        for r in (await session.execute(
            select(NFLTeamStats).where(NFLTeamStats.season.in_(seasons)))).scalars().all()])
    context = pd.DataFrame([dict(
        game_id=r.game_id, home_rest_days=r.home_rest_days, away_rest_days=r.away_rest_days,
        is_dome=r.is_dome, wind_mph=r.wind_mph, temp_f=r.temp_f)
        for r in (await session.execute(select(NFLGameContext))).scalars().all()])
    sched = load_schedules(seasons)
    lines = sched[["game_id", "spread_line", "total_line", "home_moneyline", "away_moneyline"]].copy()
    return games, team_stats, context, lines


# Feature columns each model consumes (kept here as the single source of truth).
MOV_FEATURES = ["off_epa_diff", "def_epa_diff", "pass_epa_diff", "rush_epa_diff",
                "success_rate_diff", "pace_diff", "power_diff", "rest_diff",
                "is_divisional", "is_primetime"]
TOTALS_FEATURES = ["off_epa_sum", "pace_sum", "pass_epa_diff", "is_dome",
                   "wind_mph", "temp_f", "total_line"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_nfl_training_data.py -v`
Expected: PASS (2 tests). Note the diff-column naming: `off_epa_play`→`off_epa_diff`, `def_epa_play`→`def_epa_diff`, others append `_diff` (`pass_epa`→`pass_epa_diff`, `power_rating`→`power_rating_diff`). Verify the test's expected keys (`off_epa_diff`, `power_diff`) match — **adjust the `_DIFF_COLS` renaming so `power_rating`→`power_diff` and `success_rate`→`success_rate_diff`**; make the mapping explicit rather than clever:

```python
_DIFF_MAP = {"off_epa_play": "off_epa_diff", "def_epa_play": "def_epa_diff",
             "pass_epa": "pass_epa_diff", "rush_epa": "rush_epa_diff",
             "success_rate": "success_rate_diff", "pace": "pace_diff",
             "power_rating": "power_diff"}
# in the loop:
for col, name in _DIFF_MAP.items():
    row[name] = float(h[col]) - float(a[col])
```
Replace the clever `.replace` line with this explicit map. Re-run: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/nfl/training_data.py tests/unit/test_nfl_training_data.py
git commit -m "feat(nfl): leakage-free training-data builder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: MOV model training + probability conversion

**Files:**
- Create: `src/services/nfl/model_training.py`
- Test: `tests/unit/test_nfl_model_training.py`

**Interfaces:**
- Consumes: `training_data.MOV_FEATURES`; `services/ml/probability.py`.
- Produces:
  - `train_regressor(frame, feature_cols, target_col, seed=42) -> tuple[booster, float]` — trains a LightGBM regressor on `frame[feature_cols]` → `frame[target_col]` with an internal train/val split; returns `(model, resid_std)` where `resid_std = std(y_val - pred_val)`.
  - `save_bundle(path, model, feature_cols, resid_std, trained_seasons)` / `load_bundle(path) -> dict` — joblib bundle `{model, feature_cols, resid_std, trained_seasons}`.
  - `predict_mov(bundle, frame) -> np.ndarray` — point predictions for the bundle's feature_cols (fillna(0)).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_model_training.py
import numpy as np
import pandas as pd
from src.services.nfl.model_training import train_regressor, save_bundle, load_bundle, predict_mov
from src.services.ml.probability import mov_to_spread_prob, mov_to_moneyline_prob


def _synth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    power_diff = rng.normal(0, 0.15, n)
    off = rng.normal(0, 0.1, n)
    noise = rng.normal(0, 10, n)
    # margin driven by power_diff (~50 pts per unit) + noise
    margin = 50 * power_diff + noise
    return pd.DataFrame({
        "off_epa_diff": off, "def_epa_diff": rng.normal(0, 0.1, n),
        "pass_epa_diff": rng.normal(0, 0.1, n), "rush_epa_diff": rng.normal(0, 0.1, n),
        "success_rate_diff": rng.normal(0, 0.03, n), "pace_diff": rng.normal(0, 3, n),
        "power_diff": power_diff, "rest_diff": rng.integers(-3, 4, n),
        "is_divisional": rng.integers(0, 2, n), "is_primetime": rng.integers(0, 2, n),
        "margin": margin,
    })


def test_train_returns_positive_resid_std_and_signal(tmp_path):
    from src.services.nfl.training_data import MOV_FEATURES
    frame = _synth()
    model, resid_std = train_regressor(frame, MOV_FEATURES, "margin")
    assert resid_std > 0
    # NFL MOV residual std is realistically ~9-15; synthetic noise=10 -> in range
    assert 5 < resid_std < 20
    # Higher power_diff -> higher predicted margin (learned signal)
    hi = frame.assign(power_diff=0.4).iloc[[0]]
    lo = frame.assign(power_diff=-0.4).iloc[[0]]
    bundle = {"model": model, "feature_cols": MOV_FEATURES, "resid_std": resid_std}
    assert predict_mov(bundle, hi)[0] > predict_mov(bundle, lo)[0]


def test_bundle_roundtrip_and_prob_monotonic(tmp_path):
    from src.services.nfl.training_data import MOV_FEATURES
    frame = _synth()
    model, resid_std = train_regressor(frame, MOV_FEATURES, "margin")
    p = tmp_path / "nfl_mov_test.joblib"
    save_bundle(str(p), model, MOV_FEATURES, resid_std, [2020, 2021])
    b = load_bundle(str(p))
    assert b["feature_cols"] == MOV_FEATURES
    assert b["trained_seasons"] == [2020, 2021]
    # A team predicted to win by 7, favored by 3 -> covers with prob > 0.5
    assert mov_to_spread_prob(7.0, -3.0, b["resid_std"]) > 0.5
    # Predicted to win by 7 -> ML win prob > 0.5
    assert mov_to_moneyline_prob(7.0, b["resid_std"]) > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_nfl_model_training.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.nfl.model_training'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/nfl/model_training.py
"""Train NFL MOV + totals LightGBM regressors and convert predictions to
probabilities via the shared services/ml/probability.py helpers."""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

logger = structlog.get_logger()

_PARAMS = {
    "objective": "regression", "metric": "rmse", "num_leaves": 31,
    "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 5, "min_data_in_leaf": 30, "verbose": -1,
}


def train_regressor(frame: pd.DataFrame, feature_cols: list[str],
                    target_col: str, seed: int = 42):
    X = frame[feature_cols].fillna(0.0)
    y = frame[target_col].astype(float)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(_PARAMS, dtrain, num_boost_round=2000,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    resid = y_val.values - model.predict(X_val, num_iteration=model.best_iteration)
    return model, float(np.std(resid))


def save_bundle(path: str, model, feature_cols: list[str], resid_std: float,
                trained_seasons: list[int]) -> None:
    joblib.dump({"model": model, "feature_cols": feature_cols,
                 "resid_std": resid_std, "trained_seasons": list(trained_seasons)}, path)
    logger.info("nfl_bundle_saved", path=path, resid_std=round(resid_std, 3))


def load_bundle(path: str) -> dict:
    return joblib.load(path)


def predict_mov(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    X = frame[bundle["feature_cols"]].fillna(0.0)
    model = bundle["model"]
    it = getattr(model, "best_iteration", None)
    return model.predict(X, num_iteration=it)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_nfl_model_training.py -v`
Expected: PASS (2 tests). If early_stopping warns about no validation improvement on tiny synthetic data, that's acceptable noise; the asserts still hold.

- [ ] **Step 5: Commit**

```bash
git add src/services/nfl/model_training.py tests/unit/test_nfl_model_training.py
git commit -m "feat(nfl): MOV/totals regressor training + joblib bundles

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Walk-forward backtest + metrics

**Files:**
- Create: `src/services/nfl/backtest.py`
- Test: `tests/unit/test_nfl_backtest.py`

**Interfaces:**
- Consumes: `model_training.train_regressor/predict_mov`; `training_data.{MOV_FEATURES, TOTALS_FEATURES}`; `services/ml/probability.py` (`mov_to_spread_prob`, `mov_to_moneyline_prob`, `mov_to_total_prob`); `services/ml/probability.py::devig_two_way_odds` for market-implied edge.
- Produces:
  - `grade_spread_pick(pred_mov, resid_std, spread_line, actual_margin, threshold) -> dict|None` — returns `{side, edge, won, profit}` for a flat $100 pick when model edge vs the line exceeds `threshold` (in probability points, e.g. 0.05), else None. Home covers iff `actual_margin > spread_line`. Profit assumes -110 pricing (win +90.9, loss -100).
  - `walk_forward(frame, test_seasons, threshold) -> dict` — for each test season, train MOV + totals on all earlier seasons in `frame`, predict the test season, grade spread + total picks, and return aggregate metrics: `{spread: {n, wins, ats_pct, units}, totals: {...}, mov_resid_std, saturation_max_prob}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_backtest.py
from src.services.nfl.backtest import grade_spread_pick


def test_home_cover_win_and_loss():
    # Model predicts home by 7, line home -3 (spread_line=3). Big edge -> bet home.
    # Actual margin 10 -> home covers -3 -> win.
    r = grade_spread_pick(pred_mov=7.0, resid_std=13.0, spread_line=3.0,
                          actual_margin=10, threshold=0.03)
    assert r is not None and r["side"] == "home" and r["won"] is True
    assert round(r["profit"], 1) == 90.9

    # Actual margin 1 -> home does NOT cover -3 (1 < 3) -> loss.
    r2 = grade_spread_pick(pred_mov=7.0, resid_std=13.0, spread_line=3.0,
                           actual_margin=1, threshold=0.03)
    assert r2["won"] is False and r2["profit"] == -100


def test_no_edge_returns_none():
    # Model predicts home by 3, line home -3 -> fair, no edge.
    r = grade_spread_pick(pred_mov=3.0, resid_std=13.0, spread_line=3.0,
                          actual_margin=5, threshold=0.05)
    assert r is None


def test_push_returns_none_or_zero():
    # Predicted home by 7 (bet home), line 3, actual margin exactly 3 -> push.
    r = grade_spread_pick(pred_mov=7.0, resid_std=13.0, spread_line=3.0,
                          actual_margin=3, threshold=0.03)
    assert r is None or r["profit"] == 0


def test_reliability_buckets_by_edge():
    from src.services.nfl.backtest import _reliability
    picks = [
        {"side": "home", "edge": 0.04, "won": True, "profit": 90.9},
        {"side": "home", "edge": 0.04, "won": False, "profit": -100},
        {"side": "home", "edge": 0.12, "won": True, "profit": 90.9},
        None,
    ]
    rel = _reliability(picks)
    band_004 = next(b for b in rel if b["edge_band"] == "0.03-0.06")
    band_010 = next(b for b in rel if b["edge_band"] == "0.10-0.15")
    assert band_004["n"] == 2 and band_004["win_pct"] == 50.0
    assert band_010["n"] == 1 and band_010["win_pct"] == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_nfl_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.nfl.backtest'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/nfl/backtest.py
"""Walk-forward NFL backtest: train on prior seasons, grade the test season's
spread + total picks against nflverse closing lines at flat -110 pricing."""
import numpy as np
import pandas as pd
import structlog

from src.services.nfl.model_training import train_regressor, predict_mov
from src.services.nfl.training_data import MOV_FEATURES, TOTALS_FEATURES
from src.services.ml.probability import mov_to_spread_prob, mov_to_total_prob

logger = structlog.get_logger()

_WIN_110 = 100 * (100 / 110)  # +90.909... profit on a winning -110 bet


def grade_spread_pick(pred_mov, resid_std, spread_line, actual_margin, threshold):
    # Model P(home covers -spread_line) vs implied 0.5 at a pick'em -110 market.
    p_home = mov_to_spread_prob(pred_mov, -spread_line, resid_std)
    edge = p_home - 0.5
    if abs(edge) < threshold:
        return None
    side = "home" if edge > 0 else "away"
    home_covers = actual_margin > spread_line
    if actual_margin == spread_line:
        return None  # push
    won = home_covers if side == "home" else not home_covers
    return {"side": side, "edge": float(edge), "won": bool(won),
            "profit": _WIN_110 if won else -100.0}


def grade_total_pick(pred_total, total_std, total_line, actual_total, threshold):
    p_over = mov_to_total_prob(pred_total, total_line, total_std)
    edge = p_over - 0.5
    if abs(edge) < threshold:
        return None
    side = "over" if edge > 0 else "under"
    if actual_total == total_line:
        return None
    over_hit = actual_total > total_line
    won = over_hit if side == "over" else not over_hit
    return {"side": side, "edge": float(edge), "won": bool(won),
            "profit": _WIN_110 if won else -100.0}


def _aggregate(picks):
    graded = [p for p in picks if p is not None]
    wins = sum(1 for p in graded if p["won"])
    units = sum(p["profit"] for p in graded) / 100.0
    n = len(graded)
    return {"n": n, "wins": wins, "ats_pct": round(100 * wins / n, 1) if n else 0.0,
            "units": round(units, 2)}


def _reliability(picks):
    """Calibration check: bucket graded picks by |edge| and report realized win
    rate per band. A calibrated model wins MORE often in higher-edge bands."""
    graded = [p for p in picks if p is not None]
    bands = [(0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 1.01)]
    out = []
    for lo, hi in bands:
        b = [p for p in graded if lo <= abs(p["edge"]) < hi]
        n = len(b)
        wr = round(100 * sum(1 for p in b if p["won"]) / n, 1) if n else None
        out.append({"edge_band": f"{lo:.2f}-{hi:.2f}", "n": n, "win_pct": wr})
    return out


def walk_forward(frame: pd.DataFrame, test_seasons: list[int], threshold: float = 0.05) -> dict:
    spread_picks, total_picks, sat = [], [], 0.0
    for s in test_seasons:
        train = frame[frame["season"] < s]
        test = frame[frame["season"] == s]
        if train.empty or test.empty:
            continue
        mov, mov_std = train_regressor(train, MOV_FEATURES, "margin")
        tot, tot_std = train_regressor(train, TOTALS_FEATURES, "total")
        mov_pred = predict_mov({"model": mov, "feature_cols": MOV_FEATURES}, test)
        tot_pred = predict_mov({"model": tot, "feature_cols": TOTALS_FEATURES}, test)
        for i, (_, g) in enumerate(test.iterrows()):
            sp = grade_spread_pick(mov_pred[i], mov_std, g["spread_line"], g["margin"], threshold)
            tp = grade_total_pick(tot_pred[i], tot_std, g["total_line"], g["total"], threshold)
            spread_picks.append(sp)
            total_picks.append(tp)
            sat = max(sat, abs(mov_to_spread_prob(mov_pred[i], -g["spread_line"], mov_std) - 0.5) + 0.5)
    return {"spread": _aggregate(spread_picks), "totals": _aggregate(total_picks),
            "spread_reliability": _reliability(spread_picks),
            "totals_reliability": _reliability(total_picks),
            "saturation_max_prob": round(sat, 3)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_nfl_backtest.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/services/nfl/backtest.py tests/unit/test_nfl_backtest.py
git commit -m "feat(nfl): walk-forward backtest and pick grading

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Train CLI, config, and the written backtest (exit gate)

**Files:**
- Create: `src/tasks/nfl_train.py`
- Modify: `src/config.py`
- Test: manual run documented below (the phase's acceptance gate).

**Interfaces:**
- Consumes: `training_data.load_training_frames/build_feature_frame`, `model_training.*`, `backtest.walk_forward`.
- Produces: `python3 -m src.tasks.nfl_train` — loads 2010-2024, builds the frame, runs the walk-forward backtest (test seasons 2019-2024), trains final MOV + totals models on 2010-2023, saves `models/nfl_mov_v1.joblib` + `models/nfl_totals_v1.joblib`, prints the backtest report.

- [ ] **Step 1: Add config keys**

In `src/config.py`, alongside the `mlb_*` model-path settings, add:

```python
    nfl_mov_model_path: str = "models/nfl_mov_v1.joblib"
    nfl_totals_model_path: str = "models/nfl_totals_v1.joblib"
```

- [ ] **Step 2: Write the train CLI**

```python
# src/tasks/nfl_train.py
"""Build NFL features, run the walk-forward backtest, train + save final models."""
import asyncio

import structlog

from src.config import settings
from src.database import async_session_maker
from src.services.nfl.training_data import (
    load_training_frames, build_feature_frame, MOV_FEATURES, TOTALS_FEATURES)
from src.services.nfl.model_training import train_regressor, save_bundle
from src.services.nfl.backtest import walk_forward

logger = structlog.get_logger()
ALL_SEASONS = list(range(2010, 2025))
TEST_SEASONS = list(range(2019, 2025))
FINAL_TRAIN = list(range(2010, 2024))  # hold out 2024 as the headline walk-forward year


async def main() -> None:
    async with async_session_maker() as session:
        frames = await load_training_frames(session, ALL_SEASONS)
    frame = build_feature_frame(*frames)
    print(f"modelable games: {len(frame)} ({frame['season'].min()}-{frame['season'].max()})")

    report = walk_forward(frame, TEST_SEASONS, threshold=0.05)
    print("\n=== WALK-FORWARD BACKTEST (2019-2024, flat -110 units) ===")
    for mkt in ("spread", "totals"):
        m = report[mkt]
        print(f"  {mkt:7} {m['wins']}/{m['n']}  ATS={m['ats_pct']}%  units={m['units']:+.2f}")
    print("  reliability (spread, |edge| band -> realized win%):")
    for b in report["spread_reliability"]:
        print(f"    {b['edge_band']}  n={b['n']}  win%={b['win_pct']}")
    print(f"  saturation_max_prob={report['saturation_max_prob']} (should be < 1.0)")

    final = frame[frame["season"].isin(FINAL_TRAIN)]
    mov, mov_std = train_regressor(final, MOV_FEATURES, "margin")
    tot, tot_std = train_regressor(final, TOTALS_FEATURES, "total")
    save_bundle(settings.nfl_mov_model_path, mov, MOV_FEATURES, mov_std, FINAL_TRAIN)
    save_bundle(settings.nfl_totals_model_path, tot, TOTALS_FEATURES, tot_std, FINAL_TRAIN)
    print(f"\nsaved models: mov_std={mov_std:.2f} total_std={tot_std:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Run the train CLI against prod data (ACCEPTANCE GATE)**

Run:

```bash
export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)
export DEBUG=false
python3 -m src.tasks.nfl_train
```

Expected: prints ~3,600 modelable games (≈3,900 − ~240 week-1), a walk-forward report with spread/totals ATS% and units, `saturation_max_prob < 1.0`, and saves both joblib bundles with realistic residual stds (MOV ≈ 12-14, totals ≈ 9-11). **Review the ATS%/units/saturation with the user — this is the Phase 2 exit gate.** A sound MOV model should land spread ATS% in the low 50s% at threshold 0.05 (beating the ~52.4% break-even is the bar; do NOT fudge the threshold to manufacture a number — report the honest result).

- [ ] **Step 4: Commit**

```bash
git add src/tasks/nfl_train.py src/config.py
git commit -m "feat(nfl): train CLI + model-path config + walk-forward report

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 5: Commit the model artifacts**

```bash
git add models/nfl_mov_v1.joblib models/nfl_totals_v1.joblib
git commit -m "feat(nfl): trained MOV + totals model artifacts (v1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 2 Exit Criteria (review with user before Phase 3)

1. `pytest tests/unit -k nfl` fully green (Phase 1 + Phase 2 tests).
2. `python3 -m src.tasks.nfl_train` completes: ~3,600 modelable games, both bundles saved with realistic residual stds.
3. The walk-forward backtest (2019-2024) is reported honestly: spread ATS%, units, totals ATS%/units, the reliability-by-edge-band table (higher edge should win more — the calibration check), and `saturation_max_prob < 1.0` — reviewed with the user.
4. No leakage: the backtest trains only on `season < test_season`; the feature builder only uses `through_week = week-1`.

Only after the backtest is reviewed do we proceed to Phase 3 (scorer + value_calculator + snapshots), where these models score live markets from The Odds API.

## Notes / deferred

- **Calibration:** v1 uses the residual-normal probability directly. If the backtest's realized win rate diverges from predicted probability (reliability check), wrap with `services/ml/calibration.py::CalibrationLayer` in a follow-up — not required to pass the gate.
- **QB adjustment (P2.5):** stored `nfl_games.home_qb`/`away_qb` enable a starting-QB feature later.
- **Totals features** include `total_line` as an input (the market is a strong prior for totals) — this is standard and not leakage (the line is known pre-game).
