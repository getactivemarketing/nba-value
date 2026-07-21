# NFL QB-Adjustment (P2.5a) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `qb_delta` feature (projected-starter vs trailing-form QB, empirical-Bayes shrunk EPA) to the NFL MOV model, retrain a candidate `nfl_mov_v2`, and backtest through the real gate to decide GO/NO-GO on whether spread clears break-even.

**Architecture:** A new pure `qb_ratings.py` builds per-QB career-to-date dropback-EPA ratings (shrunk to replacement) from nflverse pbp, then a `qb_delta` per game = (home starter rating − home trailing-form QB rating) − (away …). It's added to `MOV_FEATURES` only (totals untouched). A self-contained gate script trains MOV with vs without `qb_delta` across a walk-forward and compares spread ATS/units — overall and on the non-zero-delta subset the feature is meant to help.

**Tech Stack:** Python 3.11, pandas, LightGBM, nflverse (`nfl_data_py`), pytest. Runs against prod Railway DB for the gate.

**Spec:** `backend/docs/superpowers/specs/2026-07-20-nfl-qb-adjustment-design.md`.

## Global Constraints

- **Naming:** NFL code prefixed `nfl_`/`NFL`; work from `backend/`.
- **`qb_delta` goes in `MOV_FEATURES` ONLY** — `TOTALS_FEATURES` and the totals bundle stay byte-identical. The live totals product must be unaffected.
- **Point-in-time / no leakage:** a QB's rating entering a game uses only his pbp strictly BEFORE that game (career-to-date through the prior week). Walk-forward trains on seasons `< S`.
- **Do NOT overwrite `models/nfl_mov_v1.joblib`.** Any candidate is `models/nfl_mov_v2.joblib`.
- **Do NOT flip the gate flag** (`nfl_spread_in_best_bet` stays `False`) and **do NOT touch the scheduler / live scoring** — that's P2.5b, only if GO.
- **DB (gate only):** prod via `export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)`. Mask passwords in any output.
- **Git:** stage specific files only (never `git add -A`/`.`). Commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Do NOT push.
- **Constants:** `REPLACEMENT_EPA = -0.10`, `PRIOR_DROPBACKS = 200` (module-level; a coarse sane-range check is allowed at the gate, no fine grid).

## File Structure

**Create:**
- `src/services/nfl/qb_ratings.py` — pure QB rating + `qb_delta` builder.
- `src/tasks/nfl_qb_backtest.py` — the GO/NO-GO gate (train MOV ±qb_delta, walk-forward compare).
- `tests/unit/test_nfl_qb_ratings.py`.

**Modify:**
- `src/services/nfl/training_data.py` — add `home_qb_id`/`away_qb_id` to the games query, load pbp + compute `qb_deltas` in `load_training_frames`, add optional `qb_deltas` arg to `build_feature_frame`, append `qb_delta` to `MOV_FEATURES`.
- `src/tasks/nfl_train.py` — unpack the new 5-tuple from `load_training_frames` and pass `qb_deltas` into `build_feature_frame` (still writes v1 paths; v2 is produced by the gate script, not here).
- `tests/unit/test_nfl_training_data.py` — add a `qb_delta` column assertion; confirm existing tests still pass with the default (no `qb_deltas`).

---

### Task 1: `qb_ratings.py` — shrink + as-of career rating

**Files:**
- Create: `src/services/nfl/qb_ratings.py`
- Test: `tests/unit/test_nfl_qb_ratings.py`

**Interfaces:**
- Produces:
  - `REPLACEMENT_EPA: float = -0.10`, `PRIOR_DROPBACKS: int = 200`
  - `shrink(epa_sum, dropbacks, replacement=REPLACEMENT_EPA, k=PRIOR_DROPBACKS) -> float`
  - `qb_game_dropback_epa(pbp: pd.DataFrame) -> pd.DataFrame` → columns `passer_player_id, season, week, team, dropbacks, epa_sum`
  - `build_qb_timelines(qb_game_epa: pd.DataFrame) -> dict` → `{passer_id: {"ord": [int], "cdb": [float], "cepa": [float]}}` where `ord = season*100 + week` sorted ascending and `cdb`/`cepa` are cumulative-INCLUSIVE dropbacks/EPA.
  - `rating_as_of(timelines, qb, season, week, replacement=REPLACEMENT_EPA, k=PRIOR_DROPBACKS) -> float` → shrunk career rating using only games STRICTLY before `(season, week)`. Works for any `(qb, season, week)`, including weeks the QB didn't play. Unknown QB or no prior games → `replacement`.

  The single `rating_as_of` lookup is used for BOTH the starter and the trailing-form QB (Task 2), which is what makes a stable QB's delta exactly 0.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_qb_ratings.py
import pandas as pd
from src.services.nfl.qb_ratings import (
    shrink, qb_game_dropback_epa, build_qb_timelines, rating_as_of,
    REPLACEMENT_EPA, PRIOR_DROPBACKS,
)


def test_shrink_pulls_low_sample_to_replacement_and_high_sample_to_observed():
    assert shrink(0.0, 0) == REPLACEMENT_EPA                 # 0 dropbacks -> replacement
    assert abs(shrink(2.0, 5) - REPLACEMENT_EPA) < 0.02      # tiny sample -> near replacement
    assert abs(shrink(0.20 * 5000, 5000) - 0.20) < 0.01      # huge sample -> observed
    assert shrink(30.0, 300, k=100) > shrink(30.0, 300, k=500)  # more prior -> pulled harder


def _pbp():
    # 2 dropbacks QB A (epa +1,+2), 1 non-dropback (ignored), 1 dropback QB B (epa -1)
    return pd.DataFrame([
        {"passer_player_id": "A", "season": 2022, "week": 1, "posteam": "KC", "qb_dropback": 1, "qb_epa": 1.0},
        {"passer_player_id": "A", "season": 2022, "week": 1, "posteam": "KC", "qb_dropback": 1, "qb_epa": 2.0},
        {"passer_player_id": None, "season": 2022, "week": 1, "posteam": "KC", "qb_dropback": 0, "qb_epa": 5.0},
        {"passer_player_id": "B", "season": 2022, "week": 1, "posteam": "CIN", "qb_dropback": 1, "qb_epa": -1.0},
    ])


def test_qb_game_dropback_epa_aggregates_dropbacks_only():
    g = qb_game_dropback_epa(_pbp()).set_index("passer_player_id")
    assert g.loc["A", "dropbacks"] == 2 and g.loc["A", "epa_sum"] == 3.0
    assert g.loc["A", "team"] == "KC"
    assert g.loc["B", "dropbacks"] == 1 and g.loc["B", "epa_sum"] == -1.0
    assert g.loc["B", "team"] == "CIN"


def test_rating_as_of_is_point_in_time_and_career_cumulative():
    # QB A: (2022,wk1) 100 db @ +0.30 (30 epa); (2022,wk3) 100 db @ +0.99.
    qge = pd.DataFrame([
        {"passer_player_id": "A", "season": 2022, "week": 1, "team": "KC", "dropbacks": 100, "epa_sum": 30.0},
        {"passer_player_id": "A", "season": 2022, "week": 3, "team": "KC", "dropbacks": 100, "epa_sum": 99.0},
    ])
    tl = build_qb_timelines(qge)
    # entering wk1: zero prior -> replacement (no self-leak)
    assert rating_as_of(tl, "A", 2022, 1) == REPLACEMENT_EPA
    # entering wk2 (A didn't play wk2): only wk1 counts
    assert abs(rating_as_of(tl, "A", 2022, 2) - shrink(30.0, 100)) < 1e-9
    # entering wk3: strictly-before excludes wk3's own 99 -> still only wk1
    assert abs(rating_as_of(tl, "A", 2022, 3) - shrink(30.0, 100)) < 1e-9
    # next season: BOTH games now in the past
    assert abs(rating_as_of(tl, "A", 2023, 1) - shrink(30.0 + 99.0, 200)) < 1e-9
    # unknown QB -> replacement
    assert rating_as_of(tl, "NOBODY", 2022, 5) == REPLACEMENT_EPA
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_nfl_qb_ratings.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.nfl.qb_ratings'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/nfl/qb_ratings.py
"""Per-QB career dropback-EPA ratings, empirical-Bayes shrunk to replacement.

Point-in-time: a QB's rating AS OF (season, week) uses only his dropbacks
strictly BEFORE that game (crosses seasons). Feeds the qb_delta MOV feature
(see training_data). Self-contained from nflverse pbp.
"""
import bisect

import pandas as pd

REPLACEMENT_EPA: float = -0.10   # replacement-level QB dropback EPA/play
PRIOR_DROPBACKS: int = 200       # empirical-Bayes prior strength (K)


def shrink(epa_sum: float, dropbacks: float,
           replacement: float = REPLACEMENT_EPA, k: float = PRIOR_DROPBACKS) -> float:
    """(epa_sum + k*replacement) / (dropbacks + k). 0 dropbacks -> replacement."""
    return (epa_sum + k * replacement) / (dropbacks + k)


def qb_game_dropback_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pbp to per-(passer, season, week, team) dropback count + EPA sum.

    Dropback = nflverse `qb_dropback == 1` (pass attempts + sacks + scrambles);
    EPA from `qb_epa`. Rows with no passer id or NaN epa are dropped.
    """
    db = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna()].copy()
    db["qb_epa"] = pd.to_numeric(db["qb_epa"], errors="coerce")
    db = db.dropna(subset=["qb_epa"])
    g = db.groupby(["passer_player_id", "season", "week", "posteam"], as_index=False).agg(
        dropbacks=("qb_epa", "size"), epa_sum=("qb_epa", "sum"))
    return g.rename(columns={"posteam": "team"})


def build_qb_timelines(qb_game_epa: pd.DataFrame) -> dict:
    """Per QB: sorted game ordinals + cumulative-INCLUSIVE dropbacks/EPA.

    ord = season*100 + week. Used by rating_as_of for a strictly-before lookup.
    """
    tl: dict = {}
    if qb_game_epa.empty:
        return tl
    df = qb_game_epa.copy()
    df["ord"] = df["season"].astype(int) * 100 + df["week"].astype(int)
    for qb, sub in df.sort_values("ord").groupby("passer_player_id"):
        tl[qb] = {"ord": sub["ord"].tolist(),
                  "cdb": sub["dropbacks"].cumsum().tolist(),
                  "cepa": sub["epa_sum"].cumsum().tolist()}
    return tl


def rating_as_of(timelines: dict, qb, season: int, week: int,
                 replacement: float = REPLACEMENT_EPA, k: float = PRIOR_DROPBACKS) -> float:
    """Shrunk career rating for `qb` using only games STRICTLY before (season, week)."""
    t = timelines.get(qb)
    if not t:
        return replacement
    q = int(season) * 100 + int(week)
    i = bisect.bisect_left(t["ord"], q)   # first game with ord >= q; strictly-before = index i-1
    if i == 0:
        return replacement
    return shrink(t["cepa"][i - 1], t["cdb"][i - 1], replacement, k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_nfl_qb_ratings.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify nflverse pbp columns exist (integration, quick)**

Add to the test file:
```python
import pytest

@pytest.mark.integration
def test_nflverse_pbp_has_qb_columns():
    # Confirms the real column names this module assumes actually exist.
    from src.services.nfl.nfl_data import load_pbp
    pbp = load_pbp([2023])
    for col in ("passer_player_id", "qb_dropback", "qb_epa", "posteam", "season", "week"):
        assert col in pbp.columns, col
    g = qb_game_dropback_epa(pbp)
    assert len(g) > 0 and g["dropbacks"].sum() > 1000
```
Run: `export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1) && python3 -m pytest tests/unit/test_nfl_qb_ratings.py -q -m integration`
Expected: PASS. If a column name differs, fix `qb_game_dropback_epa` to the real name and re-run Step 4's unit tests. Report the column-check result.

- [ ] **Step 6: Commit**

```bash
git add src/services/nfl/qb_ratings.py tests/unit/test_nfl_qb_ratings.py
git commit -m "feat(nfl): per-QB as-of career dropback-EPA ratings (shrunk to replacement)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `compute_qb_deltas` — per-game QB delta

**Files:**
- Modify: `src/services/nfl/qb_ratings.py`
- Test: `tests/unit/test_nfl_qb_ratings.py`

**Interfaces:**
- Consumes: `qb_game_dropback_epa`, `build_qb_timelines`, `rating_as_of` (Task 1).
- Produces:
  - `compute_qb_deltas(pbp, games, form_window=8, replacement=REPLACEMENT_EPA, k=PRIOR_DROPBACKS) -> pd.DataFrame` with columns `game_id, qb_delta`. `games` must have `game_id, season, week, home_team, away_team, home_qb_id, away_qb_id`.
  - Definition, per team entering week `w` season `s`:
    - `starter_rating = rating_as_of(starter_qb_id, s, w)`.
    - `form_qb` = the passer with the **most dropbacks** for that team over its games in weeks `[w-form_window, w-1]` of season `s` (the QB the trailing-form EPA features chiefly reflect); `None` if the team has no trailing dropbacks.
    - `form_rating = rating_as_of(form_qb, s, w)` (or `replacement` if `form_qb is None`).
    - `qb_delta = (home_starter − home_form) − (away_starter − away_form)`.
  - Because starter and form use the SAME `rating_as_of`, a stable QB (starter == form_qb) gives each side exactly 0 → `qb_delta == 0`. It swings only on a genuine QB change.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_nfl_qb_ratings.py
from src.services.nfl.qb_ratings import compute_qb_deltas


def _career_pbp(rows):
    # rows: list of (qb, season, week, team, n_dropbacks, epa_each)
    out = []
    for qb, s, w, t, n, e in rows:
        for _ in range(n):
            out.append({"passer_player_id": qb, "season": s, "week": w,
                        "posteam": t, "qb_dropback": 1, "qb_epa": e})
    return pd.DataFrame(out)


def test_qb_delta_exactly_zero_when_same_starter_and_form_qb():
    # STAR is KC's only QB weeks 1-3; COOL is CIN's only QB weeks 1-3.
    pbp = _career_pbp([
        ("STAR", 2022, 1, "KC", 40, 0.3), ("STAR", 2022, 2, "KC", 40, 0.3),
        ("COOL", 2022, 1, "CIN", 40, 0.1), ("COOL", 2022, 2, "CIN", 40, 0.1),
        ("STAR", 2022, 3, "KC", 40, 0.3), ("COOL", 2022, 3, "CIN", 40, 0.1),
    ])
    games = pd.DataFrame([{"game_id": "g3", "season": 2022, "week": 3,
                           "home_team": "KC", "away_team": "CIN",
                           "home_qb_id": "STAR", "away_qb_id": "COOL"}])
    d = compute_qb_deltas(pbp, games).set_index("game_id")
    # each side: starter == form_qb -> (starter-form)=0 exactly -> delta == 0
    assert abs(d.loc["g3", "qb_delta"]) < 1e-12


def test_qb_delta_positive_when_home_upgrades_from_backup_to_star():
    pbp = _career_pbp([
        # STAR: long strong history in 2021 -> high career rating by 2022
        ("STAR", 2021, 1, "KC", 500, 0.30),
        # BACKUP starts KC weeks 1-2 (weak); STAR returns wk3
        ("BACKUP", 2022, 1, "KC", 40, -0.15), ("BACKUP", 2022, 2, "KC", 40, -0.15),
        ("COOL", 2022, 1, "CIN", 40, 0.05), ("COOL", 2022, 2, "CIN", 40, 0.05),
        ("STAR", 2022, 3, "KC", 40, 0.30), ("COOL", 2022, 3, "CIN", 40, 0.05),
    ])
    games = pd.DataFrame([{"game_id": "g3", "season": 2022, "week": 3,
                           "home_team": "KC", "away_team": "CIN",
                           "home_qb_id": "STAR", "away_qb_id": "COOL"}])
    d = compute_qb_deltas(pbp, games).set_index("game_id")
    # KC: starter STAR (high) - form_qb BACKUP (low) >> 0; CIN stable ~0 -> delta > 0
    assert d.loc["g3", "qb_delta"] > 0.15
    # symmetry: swap home/away -> sign flips
    d2 = compute_qb_deltas(pbp, games.assign(
        home_team="CIN", away_team="KC", home_qb_id="COOL", away_qb_id="STAR")
    ).set_index("game_id")
    assert d2.loc["g3", "qb_delta"] < -0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_nfl_qb_ratings.py -q`
Expected: FAIL with `ImportError: cannot import name 'compute_qb_deltas'`.

- [ ] **Step 3: Write minimal implementation** (append to `qb_ratings.py`)

```python
def _form_qb(team_games: pd.DataFrame, season: int, week: int, form_window: int):
    """Passer with the most dropbacks for a team over weeks [w-window, w-1] of the season."""
    win = team_games[(team_games["season"] == season)
                     & (team_games["week"] >= week - form_window)
                     & (team_games["week"] <= week - 1)]
    if win.empty:
        return None
    return win.groupby("passer_player_id")["dropbacks"].sum().idxmax()


def compute_qb_deltas(pbp: pd.DataFrame, games: pd.DataFrame, form_window: int = 8,
                      replacement: float = REPLACEMENT_EPA,
                      k: float = PRIOR_DROPBACKS) -> pd.DataFrame:
    """Per-game qb_delta = (home starter - home form_qb) - (away starter - away form_qb).

    Both the starter and the plurality trailing-form QB are rated through the same
    rating_as_of lookup, so a stable QB (starter == form_qb) yields exactly 0.
    """
    if games.empty:
        return pd.DataFrame({"game_id": [], "qb_delta": []})
    qge = qb_game_dropback_epa(pbp)
    tl = build_qb_timelines(qge)

    def component(qb_id, team, s, w):
        starter = rating_as_of(tl, qb_id, s, w, replacement, k)
        fq = _form_qb(qge[qge["team"] == team], s, w, form_window)
        form = replacement if fq is None else rating_as_of(tl, fq, s, w, replacement, k)
        return starter - form

    rows = []
    for _, g in games.iterrows():
        s, w = int(g["season"]), int(g["week"])
        h = component(g["home_qb_id"], g["home_team"], s, w)
        a = component(g["away_qb_id"], g["away_team"], s, w)
        rows.append({"game_id": g["game_id"], "qb_delta": h - a})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_nfl_qb_ratings.py -q`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/services/nfl/qb_ratings.py tests/unit/test_nfl_qb_ratings.py
git commit -m "feat(nfl): compute_qb_deltas (starter vs plurality trailing-form QB)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Integrate `qb_delta` into the feature frame

**Files:**
- Modify: `src/services/nfl/training_data.py`
- Modify: `src/tasks/nfl_train.py`
- Test: `tests/unit/test_nfl_training_data.py`

**Interfaces:**
- Consumes: `compute_qb_deltas` (Task 2).
- Produces:
  - `build_feature_frame(games, team_stats, context, lines, qb_deltas=None) -> pd.DataFrame` — now emits a `qb_delta` column (0.0 when `qb_deltas` is None or the game is absent).
  - `MOV_FEATURES` now ends with `"qb_delta"`. `TOTALS_FEATURES` unchanged.
  - `load_training_frames(session, seasons) -> (games, team_stats, context, lines, qb_deltas)` — games query gains `home_qb_id`/`away_qb_id`; loads pbp via `nfl_data.load_pbp(seasons)` and returns `qb_deltas = compute_qb_deltas(pbp, games)`.

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_nfl_training_data.py`)

```python
def test_build_feature_frame_adds_qb_delta_column():
    import pandas as pd
    from src.services.nfl.training_data import build_feature_frame, MOV_FEATURES
    # qb_delta must be a declared MOV feature
    assert MOV_FEATURES[-1] == "qb_delta"
    # minimal 1-game frame (reuse the module's existing fixture builders if present;
    # otherwise construct inline as below)
    games = pd.DataFrame([{
        "game_id": "g", "season": 2022, "week": 2, "home_team": "KC", "away_team": "CIN",
        "home_score": 27, "away_score": 20, "is_divisional": False, "is_primetime": True,
    }])
    ts = pd.DataFrame([
        {"team": "KC", "season": 2022, "through_week": 1, "off_epa_play": 0.1, "def_epa_play": -0.05,
         "pass_epa": 0.1, "rush_epa": 0.0, "success_rate": 0.47, "pace": 62.0, "power_rating": 0.2},
        {"team": "CIN", "season": 2022, "through_week": 1, "off_epa_play": 0.0, "def_epa_play": 0.05,
         "pass_epa": 0.1, "rush_epa": 0.0, "success_rate": 0.47, "pace": 64.0, "power_rating": -0.05},
    ])
    ctx = pd.DataFrame([{"game_id": "g", "home_rest_days": 7, "away_rest_days": 7,
                         "is_dome": False, "wind_mph": 6.0, "temp_f": 70.0}])
    lines = pd.DataFrame([{"game_id": "g", "spread_line": 3.0, "total_line": 44.0,
                           "home_moneyline": -160, "away_moneyline": 140}])
    qbd = pd.DataFrame([{"game_id": "g", "qb_delta": 0.12}])
    out = build_feature_frame(games, ts, ctx, lines, qbd)
    assert "qb_delta" in out.columns and abs(out.iloc[0]["qb_delta"] - 0.12) < 1e-9
    # default (no qb_deltas) -> column present, 0.0 (keeps back-compat for callers that omit it)
    out0 = build_feature_frame(games, ts, ctx, lines)
    assert out0.iloc[0]["qb_delta"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_nfl_training_data.py -q`
Expected: FAIL (`MOV_FEATURES[-1] != "qb_delta"` / `build_feature_frame` has no `qb_deltas` arg).

- [ ] **Step 3: Implement**

In `training_data.py`:

(a) Append the feature:
```python
MOV_FEATURES = ["off_epa_diff", "def_epa_diff", "pass_epa_diff", "rush_epa_diff",
                "success_rate_diff", "pace_diff", "power_diff", "rest_diff",
                "is_divisional", "is_primetime", "spread_line", "qb_delta"]
```

(b) Add the param + column to `build_feature_frame` — change the signature and set the value from a `game_id -> qb_delta` map (default 0.0):
```python
def build_feature_frame(
    games: pd.DataFrame, team_stats: pd.DataFrame,
    context: pd.DataFrame, lines: pd.DataFrame,
    qb_deltas: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ts = team_stats.set_index(["team", "season", "through_week"])
    ctx = context.set_index("game_id")
    ln = lines.set_index("game_id")
    qbd = ({} if qb_deltas is None or qb_deltas.empty
           else dict(zip(qb_deltas["game_id"], qb_deltas["qb_delta"])))
    rows = []
    for _, g in games.iterrows():
        # ... existing body unchanged, then inside the row dict add:
        #     "qb_delta": float(qbd.get(g["game_id"], 0.0)),
```
Add `"qb_delta": float(qbd.get(g["game_id"], 0.0)),` to the `row = {...}` dict (alongside `spread_line`).

(c) In `load_training_frames`, add the QB ids to the games query and return `qb_deltas`:
```python
from src.services.nfl.nfl_data import load_schedules, load_pbp
from src.services.nfl.qb_ratings import compute_qb_deltas
# ...
        select(NFLGame.game_id, NFLGame.season, NFLGame.week, NFLGame.home_team,
               NFLGame.away_team, NFLGame.home_score, NFLGame.away_score,
               NFLGame.is_divisional, NFLGame.is_primetime,
               NFLGame.home_qb_id, NFLGame.away_qb_id)
# ... after `lines = ...`:
    pbp = load_pbp(seasons)
    qb_deltas = compute_qb_deltas(pbp, games)
    return games, team_stats, context, lines, qb_deltas
```

In `nfl_train.py`, unpack 5 and pass through:
```python
    games, team_stats, context, lines, qb_deltas = await load_training_frames(session, seasons)
    final = build_feature_frame(games, team_stats, context, lines, qb_deltas)
```
(Find the existing `load_training_frames(...)` unpack + `build_feature_frame(...)` call and update both. `nfl_train` still saves to the v1 paths — do NOT change the output paths here.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_nfl_training_data.py tests/unit/test_nfl_scorer.py -q`
Expected: PASS. (The scorer test exercises the feature dict; confirm nothing broke.)

- [ ] **Step 5: Full NFL slice green**

Run: `python3 -m pytest tests/unit/ -k nfl -q`
Expected: PASS (all NFL tests; the live-scoring path still works because `build_live_feature_row` supplies `qb_delta`? — NO: it does not yet. If any live-features/scorer test now fails on a missing `qb_delta` key, that is expected and handled in Task 4's note; for THIS task, `predict_mov` does `frame[feature_cols].fillna(0.0)`, so a missing column would KeyError. Confirm: `build_live_feature_row` is NOT used in these unit tests' MOV path. If `tests/unit/test_nfl_live_features.py` asserts the row covers `MOV_FEATURES`, update that test's expectation OR add `qb_delta` to the live row as 0.0 — see note below.)

> **Note (read before Step 5):** `test_nfl_live_features.py::test_live_row_matches_model_feature_columns` asserts the live row contains every `MOV_FEATURES` column. Adding `qb_delta` to `MOV_FEATURES` will break it. Since P2.5a does NOT build the live QB feed, set `qb_delta` to a neutral `0.0` in `build_live_feature_row` (so live scoring stays runnable and identical to v1 behavior for now) and keep that test green. Add this one line to `live_features.build_live_feature_row`'s returned dict: `"qb_delta": 0.0,` with a comment `# P2.5b will populate from the live starter projection; 0.0 = no adjustment`. This keeps live scoring on v1 semantics and does not change any live prediction.

- [ ] **Step 6: Commit**

```bash
git add src/services/nfl/training_data.py src/tasks/nfl_train.py \
        src/services/nfl/live_features.py tests/unit/test_nfl_training_data.py
git commit -m "feat(nfl): qb_delta feature into MOV_FEATURES + training frame

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: GO/NO-GO gate — retrain MOV ±qb_delta, walk-forward compare

**Files:**
- Create: `src/tasks/nfl_qb_backtest.py`
- (Conditionally) Create: `models/nfl_mov_v2.joblib` (only if GO)

**Interfaces:**
- Consumes: `load_training_frames`, `build_feature_frame`, `MOV_FEATURES` (Task 3); `backtest.walk_forward(frame, test_seasons, threshold=0.05)`; `model_training.train_regressor`, `save_bundle`.
- Produces: a printed comparison report + a GO/NO-GO recommendation. `walk_forward` internally trains per season and grades spread + totals; it reads the module-level `MOV_FEATURES`, so the v2 run (qb_delta live) vs a v1-equivalent run (qb_delta zeroed) isolates the feature.

- [ ] **Step 1: Write the gate script**

```python
# src/tasks/nfl_qb_backtest.py
"""P2.5a GO/NO-GO: does qb_delta push NFL spread through the real gate?

Walk-forward 2019-24. Runs the model WITH qb_delta (v2) vs a v1-equivalent
(qb_delta zeroed so the feature is inert), and reports spread ATS% / units:
overall AND on the subset of games where qb_delta != 0 (the games the feature
is meant to help). Totals are untouched (sanity: should be identical run-to-run).

Run: export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)
     python3 -m src.tasks.nfl_qb_backtest
"""
import asyncio
import structlog

from src.database import async_session_maker
from src.services.nfl.training_data import (
    load_training_frames, build_feature_frame, MOV_FEATURES)
from src.services.nfl.backtest import walk_forward

logger = structlog.get_logger()
TEST_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]


async def main() -> None:
    async with async_session_maker() as session:
        seasons = list(range(2010, 2025))
        games, ts, ctx, lines, qb_deltas = await load_training_frames(session, seasons)

    frame_v2 = build_feature_frame(games, ts, ctx, lines, qb_deltas)      # qb_delta live
    frame_v1 = frame_v2.copy(); frame_v1["qb_delta"] = 0.0                # feature inert
    nonzero = frame_v2["qb_delta"].abs() > 1e-9

    print(f"modelable games={len(frame_v2)}  non-zero qb_delta games={int(nonzero.sum())} "
          f"({100*nonzero.mean():.1f}%)")

    res_v1 = walk_forward(frame_v1, TEST_SEASONS)
    res_v2 = walk_forward(frame_v2, TEST_SEASONS)
    # walk_forward returns spread/total aggregates; print the spread block for each.
    # (Confirm the exact return keys by reading backtest.walk_forward; print spread
    #  win%/units/n for v1 vs v2, and totals for v2 as an unchanged-sanity check.)
    print("V1 (no qb_delta):", res_v1)
    print("V2 (qb_delta)   :", res_v2)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Read `backtest.walk_forward` and finalize the report**

Read `src/services/nfl/backtest.py::walk_forward` to confirm its exact return shape (it uses `_aggregate`/`_reliability`). Update the two `print` lines to extract and label **spread ATS%, units, and n** for v1 vs v2. Keep totals as an unchanged-sanity line (v2 totals must equal v1 totals — totals model is untouched).

**Non-zero-delta subset (train on ALL, grade the subset — do NOT retrain on the subset):** the feature is inert on ~85% of games, so the overall ATS barely moves even if the feature helps a lot on the games it touches. Evaluate the subset correctly by keeping the full-data training and filtering the *graded picks* to non-zero-delta games:
- Read whether `walk_forward`'s per-pick records carry `game_id`. If they do, filter those picks to the `nonzero` game_ids and re-aggregate spread ATS%/units.
- If they do NOT, make the one minimal, safe addition to `backtest.py`: include `game_id` on each pick record `walk_forward` produces (reporting-only; changes no grading math), then filter. Do this rather than retraining on a subset — retraining on the non-zero games alone would train the model on ~15% of the data and is methodologically wrong.

Report the subset spread ATS%/units for v2 alongside the overall numbers.

- [ ] **Step 3: Run the gate against prod**

Run:
```bash
export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)
python3 -m src.tasks.nfl_qb_backtest
```
Expected: prints modelable-game count, non-zero-delta share, and spread ATS%/units for v1 vs v2 (overall + non-zero subset). Mask any DB password in reported output.

- [ ] **Step 4: Decide GO / NO-GO**

- **GO** if V2 spread clears **≥ 52.4% ATS with positive units** over 2019-24 (and the non-zero-delta subset shows the lift is coming from QB-change games, not noise). Then: retrain the full-data MOV with qb_delta and save the candidate:
  ```python
  # one-off (add to the script under a `--save` guard or run inline):
  from src.services.nfl.model_training import train_regressor, save_bundle
  mov, mov_std = train_regressor(frame_v2, MOV_FEATURES, "margin")
  save_bundle("models/nfl_mov_v2.joblib", mov, MOV_FEATURES, mov_std, seasons)
  ```
  Do NOT overwrite v1, do NOT flip `nfl_spread_in_best_bet`, do NOT wire live — record "GO → P2.5b (live starter feed + gate flip)".
- **NO-GO** if V2 spread stays below break-even → keep v1, spread stays SHADOW, record the negative result (numbers + non-zero-delta subset). Do NOT save v2.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/nfl_qb_backtest.py            # + models/nfl_mov_v2.joblib ONLY if GO
git commit -m "feat(nfl): P2.5a qb_delta GO/NO-GO gate (walk-forward spread compare)

<one line with the actual result: GO/NO-GO + spread ATS%/units v1 vs v2>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase P2.5a Exit Criteria

1. `pytest tests/unit/ -k nfl` fully green (qb_ratings + training_data + unchanged live/scorer/scheduler/api).
2. `qb_delta` is in `MOV_FEATURES`; `TOTALS_FEATURES` and the totals bundle are byte-identical (totals untouched).
3. The gate ran on prod and produced a spread ATS%/units comparison (v1 vs v2, overall + non-zero-delta subset) and a clear **GO/NO-GO**.
4. `nfl_mov_v1.joblib` is unchanged. `nfl_mov_v2.joblib` exists **only if GO**. `nfl_spread_in_best_bet` stays `False`. No scheduler/live change.
5. Result recorded (memory + ledger). **If GO:** next is P2.5b (live starter projection + flip spread live). **If NO-GO:** spread stays SHADOW; document that QB-EPA delta was insufficient.
