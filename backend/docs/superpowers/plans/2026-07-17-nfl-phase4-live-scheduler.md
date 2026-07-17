# NFL Model — Phase 4: Live Odds + Weekly Scheduler + API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the NFL model run live — ingest current odds from The Odds API into `nfl_markets`, assemble live feature rows (team form + the current market spread/total), score upcoming games into graded `nfl_prediction_snapshots` on the NFL weekly rhythm, and expose picks via `api/nfl.py`.

**Architecture:** An `NFLOddsClient` (subclass of the shared `OddsAPIClient`, `SPORT="americanfootball_nfl"`) writes current odds to `nfl_markets`. A live feature-row builder joins `nfl_team_stats` (through the prior week) + `nfl_game_context` + the current market `spread_line`/`total_line` into the exact feature vector the Phase-2/3 models expect. `tasks/nfl_scheduler.py` orchestrates the weekly cycle on a `schedule` loop (mirroring `mlb_scheduler`): refresh schedule + team stats, refresh odds, snapshot each game ~90 min pre-kick via the Phase-3 scorer, and grade after finals. `api/nfl.py` serves `/picks` and `/games`.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2.0 async, `nfl_data_py`, The Odds API, `schedule`, structlog, pytest.

## Global Constraints

- **Naming:** NFL tables/services/tasks/files/config prefixed `nfl_` / `NFL`.
- **`spread_line` / `total_line` are LIVE model inputs:** the scorer needs the *current market* spread (home-favored positive) as the `spread_line` feature and the current total as `total_line`. The live feature-row builder MUST source these from the ingested `nfl_markets`, not from nflverse. A game with no current spread/total line cannot be scored (skip it, log).
- **Reuse:** `OddsAPIClient` (`services/data/odds_api.py`), the Phase-3 `scorer.score_game`, `snapshot.build_snapshot/grade_snapshot`, the Phase-2 `training_data` feature conventions (`MOV_FEATURES`/`TOTALS_FEATURES`, `build_feature_frame` diff logic), and the calibrated bundles. Do not reimplement scoring/grading.
- **Gating unchanged:** totals live (`nfl_totals_in_best_bet=True`), spread+ML shadow. Snapshots freeze all three; `best_bet` is totals-only.
- **Scheduler must be season-safe:** all NFL scheduler tasks no-op cleanly out of season (no upcoming games → 0 work, no errors). It must NOT disrupt the existing MLB/NBA schedulers (separate registration, separate engine like `mlb_scheduler._init_engine`).
- **Odds API budget:** The Odds API bills per request. Reuse the shared key (`settings.odds_api_key`). Do not poll more than the scheduler cadence requires (odds refresh ≤ a few times/day out of the pre-kick window).
- **Git staging:** stage specific files only. Never `git add -A`/`git add .`.
- **Push:** commit locally; do not push unless explicitly asked. (Pushing `main` deploys to Railway and would START the NFL scheduler in prod — only push when intentionally going live.)
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **DB:** prod via `export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)`.

## Season-timing reality (read before executing)

It is mid-July; NFL preseason starts early August, regular season ~September. **The Odds API returns odds only for upcoming games**, so the full live pipeline cannot be graded end-to-end until the season. This plan therefore: (a) builds + unit-tests every component, (b) integration-tests the `NFLOddsClient` against the *live API response shape* (which may be empty or preseason now — that's an acceptable "0 games" result, verified by hitting the endpoint), and (c) leaves the full weekly snapshot→grade validation as a **season-time acceptance step** (documented, run in September). Do not fabricate live games to "prove" grading — unit tests + the historical Phase-3 dry-run already cover scoring/grading logic.

## File Structure

**Create:**
- `src/services/nfl/odds_client.py` — `NFLOddsClient` + `parse_nfl_odds_to_markets`.
- `src/services/nfl/live_features.py` — assemble a live scorer feature row for an upcoming game.
- `src/services/nfl/season_update.py` — ingest upcoming schedule + recompute current-season `nfl_team_stats`.
- `src/tasks/nfl_scheduler.py` — the weekly orchestration loop.
- `src/api/nfl.py` — `/picks`, `/games`, `/debug` router.
- `tests/unit/test_nfl_odds_client.py`, `test_nfl_live_features.py`, `test_nfl_scheduler_tasks.py`.
- `tests/integration/test_nfl_odds_live.py` — hits the real Odds API (marked integration).

**Modify:**
- `src/main.py` — register `api/nfl.py` router; optionally start the NFL scheduler (guarded).
- `src/config.py` — `nfl_scheduler_enabled` (default False), `nfl_snapshot_minutes_before` (default 90).

**Reuse (do not modify):** `services/data/odds_api.py`, `services/nfl/{scorer,snapshot,training_data,model_training,calibration_fit}.py`.

---

### Task 1: `NFLOddsClient` + parse to `nfl_markets`

**Files:**
- Create: `src/services/nfl/odds_client.py`
- Test: `tests/unit/test_nfl_odds_client.py`, `tests/integration/test_nfl_odds_live.py`

**Interfaces:**
- Produces:
  - `NFLOddsClient(OddsAPIClient)` with `SPORT = "americanfootball_nfl"` and `async get_nfl_odds(markets=["h2h","spreads","totals"], bookmakers=None) -> list[dict]` (mirrors `MLBOddsClient.get_mlb_odds` in `services/mlb/ingest.py`).
  - `parse_nfl_odds_to_markets(odds_events, team_name_to_abbr) -> list[dict]` — **pure**: maps The Odds API event JSON to `NFLMarket` kwargs dicts. One event yields up to 3 market rows (spread, moneyline, total). Extracts: `game_id` (matched to `nfl_games` by teams+date — see note), `market_type`, `line` (home spread as home-favored positive / total points), `home_odds`/`away_odds` (spreads+h2h, decimal), `over_odds`/`under_odds` (totals), `book`.
  - `NFL_TEAM_NAME_TO_ABBR: dict[str,str]` — full team name → nflverse abbr (32 teams), like MLB's map in `services/mlb/ingest.py`.

- [ ] **Step 1: Write the failing test** (pure parser against a recorded Odds API event)

```python
# tests/unit/test_nfl_odds_client.py
from src.services.nfl.odds_client import parse_nfl_odds_to_markets, NFL_TEAM_NAME_TO_ABBR


def _event():
    # Minimal Odds API v4 shape: one game, one book, three markets.
    return [{
        "id": "evt1", "commence_time": "2026-09-13T17:00:00Z",
        "home_team": "Kansas City Chiefs", "away_team": "Cincinnati Bengals",
        "bookmakers": [{"key": "draftkings", "markets": [
            {"key": "h2h", "outcomes": [
                {"name": "Kansas City Chiefs", "price": 1.62},
                {"name": "Cincinnati Bengals", "price": 2.40}]},
            {"key": "spreads", "outcomes": [
                {"name": "Kansas City Chiefs", "price": 1.91, "point": -3.5},
                {"name": "Cincinnati Bengals", "price": 1.91, "point": 3.5}]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": 1.91, "point": 48.5},
                {"name": "Under", "price": 1.91, "point": 48.5}]},
        ]}]}]


def test_all_32_teams_mapped():
    assert len(NFL_TEAM_NAME_TO_ABBR) == 32
    assert NFL_TEAM_NAME_TO_ABBR["Kansas City Chiefs"] == "KC"
    assert NFL_TEAM_NAME_TO_ABBR["Las Vegas Raiders"] == "LV"


def test_parse_yields_three_markets_with_correct_convention():
    rows = parse_nfl_odds_to_markets(_event(), NFL_TEAM_NAME_TO_ABBR)
    by_type = {r["market_type"]: r for r in rows}
    assert set(by_type) == {"spread", "moneyline", "total"}
    # home spread stored as HOME-FAVORED POSITIVE: KC is -3.5, so home favored by 3.5 -> line = 3.5
    assert by_type["spread"]["line"] == 3.5
    assert round(by_type["spread"]["home_odds"], 2) == 1.91
    assert by_type["total"]["line"] == 48.5
    assert round(by_type["total"]["over_odds"], 2) == 1.91
    assert round(by_type["moneyline"]["home_odds"], 2) == 1.62
    assert by_type["spread"]["home_team_abbr"] == "KC"
    assert by_type["spread"]["away_team_abbr"] == "CIN"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_nfl_odds_client.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Write `odds_client.py`: `NFL_TEAM_NAME_TO_ABBR` (32 full names → nflverse abbrs, matching `constants.NFL_DIVISIONS` keys), the `NFLOddsClient` subclass (copy `MLBOddsClient.get_mlb_odds`'s httpx call, `SPORT="americanfootball_nfl"`), and the pure `parse_nfl_odds_to_markets`. Convention: nflverse/our `spread_line` is **home-favored positive**, but the Odds API `point` for the home team is **negative when favored** — so `line = -home_point` (KC point -3.5 → line +3.5). Use the first bookmaker (or a book preference); one market row per (spread/moneyline/total). Include `home_team_abbr`/`away_team_abbr` in the dict for the ingest layer to match `game_id`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_nfl_odds_client.py -v`
Expected: PASS.

- [ ] **Step 5: Live API shape check (integration)**

```python
# tests/integration/test_nfl_odds_live.py
import pytest
from src.services.nfl.odds_client import NFLOddsClient, parse_nfl_odds_to_markets, NFL_TEAM_NAME_TO_ABBR


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_nfl_odds_shape():
    events = await NFLOddsClient().get_nfl_odds()
    # Out of season this is legitimately empty — assert it parses without error either way.
    rows = parse_nfl_odds_to_markets(events, NFL_TEAM_NAME_TO_ABBR)
    for r in rows:
        assert r["market_type"] in {"spread", "moneyline", "total"}
        assert "home_team_abbr" in r
```

Run: `export DATABASE_URL=... && python3 -m pytest tests/integration/test_nfl_odds_live.py -v -m integration`
Expected: PASS (0 rows out of season is fine — the point is the call + parse don't error and the shape holds). Report how many events came back.

- [ ] **Step 6: Commit**

```bash
git add src/services/nfl/odds_client.py tests/unit/test_nfl_odds_client.py tests/integration/test_nfl_odds_live.py
git commit -m "feat(nfl): Odds API client + parse to nfl_markets

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Live feature-row builder

**Files:**
- Create: `src/services/nfl/live_features.py`
- Test: `tests/unit/test_nfl_live_features.py`

**Interfaces:**
- Produces:
  - `build_live_feature_row(game, home_stats, away_stats, context, spread_line, total_line) -> dict | None` — **pure**: assembles the exact feature dict `scorer.score_game` expects (the same columns `training_data.build_feature_frame` produces: `off_epa_diff`…`power_diff`, `rest_diff`, `is_divisional`, `is_primetime`, `spread_line`, `off_epa_sum`, `pace_sum`, `is_dome`, `wind_mph`, `temp_f`, `total_line`). `home_stats`/`away_stats` are the `nfl_team_stats` rows at `through_week = game.week - 1`. `spread_line`/`total_line` are the CURRENT market lines (from `nfl_markets`). Returns None if either team's prior-week stats are missing or the lines are None (can't score).
  - This deliberately reuses the diff/sum logic from `training_data` — factor the shared `_DIFF_MAP` + diff/sum computation into a helper both call, so live and training features can never drift.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_live_features.py
from src.services.nfl.live_features import build_live_feature_row
from src.services.nfl.training_data import MOV_FEATURES, TOTALS_FEATURES


def _stats(off, deff, pw, pace):
    return {"off_epa_play": off, "def_epa_play": deff, "pass_epa": 0.1, "rush_epa": 0.0,
            "success_rate": 0.47, "pace": pace, "power_rating": pw}


def test_live_row_matches_model_feature_columns():
    game = {"game_id": "2026_02_CIN_KC", "week": 2, "home_team": "KC", "away_team": "CIN",
            "is_divisional": False, "is_primetime": True}
    ctx = {"home_rest_days": 7, "away_rest_days": 7, "is_dome": False, "wind_mph": 6.0, "temp_f": 70.0}
    row = build_live_feature_row(game, _stats(0.15, -0.05, 0.20, 62.0),
                                 _stats(0.0, 0.05, -0.05, 64.0), ctx,
                                 spread_line=3.0, total_line=47.5)
    # every model feature is present (so scorer won't KeyError)
    for col in set(MOV_FEATURES) | set(TOTALS_FEATURES):
        assert col in row, col
    assert round(row["off_epa_diff"], 3) == 0.15
    assert round(row["power_diff"], 3) == 0.25
    assert row["spread_line"] == 3.0 and row["total_line"] == 47.5
    assert row["is_primetime"] == 1


def test_missing_stats_returns_none():
    game = {"game_id": "g", "week": 2, "home_team": "KC", "away_team": "CIN",
            "is_divisional": False, "is_primetime": False}
    assert build_live_feature_row(game, None, _stats(0, 0, 0, 60), {}, 3.0, 47.5) is None
    assert build_live_feature_row(game, _stats(0, 0, 0, 60), _stats(0, 0, 0, 60), {}, None, 47.5) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_nfl_live_features.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation** — build the row using the same `_DIFF_MAP` diffs and `pace_sum`/`off_epa_sum` as `training_data.build_feature_frame`; set `spread_line`/`total_line` from the passed market lines; `is_divisional`/`is_primetime` int-cast from the game; rest_diff/is_dome/wind/temp from context. Return None on missing stats or None lines. (Refactor the diff computation into a shared helper imported by both modules to prevent drift.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_nfl_live_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/nfl/live_features.py tests/unit/test_nfl_live_features.py
git commit -m "feat(nfl): live feature-row builder (market line as model input)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Season update — upcoming schedule + current-season team stats

**Files:**
- Create: `src/services/nfl/season_update.py`
- Test: `tests/unit/test_nfl_season_update.py` (pure parts) + a controlled prod run

**Interfaces:**
- Produces (async):
  - `refresh_schedule(session, season) -> int` — pull the current season's nflverse schedule (played + upcoming), upsert `nfl_games` + `nfl_game_context` (reuse Phase-1 `ingest.upsert_games`/`upsert_game_context` + `nfl_data.schedule_to_game_rows`). Returns games upserted.
  - `recompute_team_stats(session, season) -> int` — recompute `nfl_team_stats` for the season from played games' pbp (reuse `features.team_game_epa` + `rolling_team_stats` + `ingest.upsert_team_stats`). Idempotent. Returns rows.
  - `odds_to_markets(session, season) -> int` — fetch `NFLOddsClient` odds, match events to `nfl_games` by team abbrs + date, upsert `NFLMarket` rows. Returns markets written.
- These are thin orchestrations of already-built + reviewed Phase-1 functions; the test covers the event→game_id matching logic (pure) and idempotency.

- [ ] **Step 1: Write the failing test** — a pure `match_event_to_game(event_row, games)` that resolves an Odds API event to a `nfl_games.game_id` by (home_abbr, away_abbr, date); asserts a correct match and a None when no game matches.

```python
# tests/unit/test_nfl_season_update.py
from src.services.nfl.season_update import match_event_to_game


def test_match_event_to_game_by_teams_and_date():
    games = [{"game_id": "2026_02_CIN_KC", "home_team": "KC", "away_team": "CIN",
              "kickoff_date": "2026-09-13"}]
    ev = {"home_team_abbr": "KC", "away_team_abbr": "CIN", "commence_date": "2026-09-13"}
    assert match_event_to_game(ev, games) == "2026_02_CIN_KC"
    ev2 = {"home_team_abbr": "KC", "away_team_abbr": "BUF", "commence_date": "2026-09-13"}
    assert match_event_to_game(ev2, games) is None
```

- [ ] **Step 2: Run to fail; Step 3: implement; Step 4: pass** (standard TDD; the async orchestrations wrap Phase-1 functions and are exercised in Step 5).

- [ ] **Step 5: Controlled prod smoke** — run `refresh_schedule` + `recompute_team_stats` for 2026 against prod; confirm the 2026 schedule rows land (even if scores are NULL for upcoming games) and team-stats recompute idempotently. `odds_to_markets` will write 0 rows out of season (fine). Report counts.

- [ ] **Step 6: Commit** (`season_update.py` + test).

---

### Task 4: `nfl_scheduler.py` — weekly orchestration

**Files:**
- Create: `src/tasks/nfl_scheduler.py`
- Modify: `src/config.py` (`nfl_scheduler_enabled=False`, `nfl_snapshot_minutes_before=90`)
- Test: `tests/unit/test_nfl_scheduler_tasks.py`

**Interfaces:**
- Produces async task functions (mirroring `mlb_scheduler`), each returning a summary dict and safe to run out of season (0 work, no error):
  - `snapshot_due_games(session, minutes_before) -> dict` — for each `nfl_games` game kicking off within the window that isn't already snapshotted: load bundles, fetch its `nfl_markets` rows, build the live feature row (`live_features` + current spread/total from markets), `scorer.score_game`, `snapshot.build_snapshot`, upsert `NFLPredictionSnapshot`. Returns `{snapshotted}`.
  - `grade_finals(session) -> dict` — for completed games with an ungraded snapshot: `snapshot.grade_snapshot` vs final score, update. Returns `{graded}`.
  - `weekly_refresh(session) -> dict` — `season_update.refresh_schedule` + `recompute_team_stats`.
  - `start_scheduler()` — registers the `schedule` cadence (weekly_refresh Tue, odds refresh Wed-Sun daily, snapshot check hourly, grade hourly) on a dedicated engine like `mlb_scheduler._init_engine`. Only starts when `settings.nfl_scheduler_enabled`.
- **Season-safety:** every task early-returns `{...: 0}` when there are no due/completed games. Unit tests assert the no-op path returns cleanly with an empty DB fixture / mocked empty queries.

- [ ] Standard TDD steps: unit-test `snapshot_due_games`/`grade_finals` with mocked sessions returning no due games (assert 0-work no-op) and with one due game (assert it calls score+build+upsert). Do NOT run the live scheduler loop. Commit.

- [ ] **Step (config):** add `nfl_scheduler_enabled: bool = False` and `nfl_snapshot_minutes_before: int = 90` to `config.py`. The scheduler stays OFF until intentionally enabled for the season.

---

### Task 5: `api/nfl.py` — picks + games

**Files:**
- Create: `src/api/nfl.py`
- Modify: `src/main.py` (register router)
- Test: `tests/unit/test_nfl_api.py`

**Interfaces:**
- Produces `APIRouter(prefix="/nfl", tags=["NFL"])` with:
  - `GET /nfl/picks` — reads `nfl_prediction_snapshots` where `best_bet_value_score >= min_value_score` (Query default 40, like MLB), ordered by kickoff; returns the totals best_bet picks.
  - `GET /nfl/games` — upcoming `nfl_games` for the current/next week with their snapshot (if any).
  - `GET /nfl/debug/odds` — like `api/mlb.py`'s debug: fetch `NFLOddsClient` odds count + key prefix (no secrets).
- Register in `main.py`: `app.include_router(nfl.router, prefix=settings.api_v1_prefix, tags=["NFL"])`.

- [ ] Standard TDD: unit-test the router with FastAPI `TestClient` against a mocked DB session returning a snapshot row → `/nfl/picks` returns it; `min_value_score` filter works. Register in main.py; assert the app imports and `/openapi.json` lists `/api/v1/nfl/picks`. Commit.

---

## Phase 4 Exit Criteria

1. `pytest tests/unit -k nfl` fully green; `api/nfl.py` registered and in the OpenAPI schema.
2. `NFLOddsClient` live-shape integration test passes (0 events out of season is acceptable — the call + parse must not error).
3. Task-3 prod smoke: the 2026 NFL schedule is in `nfl_games`; team stats recompute idempotently.
4. The scheduler is implemented, unit-tested for the no-op (out-of-season) and one-due-game paths, and ships **disabled** (`nfl_scheduler_enabled=False`).
5. **Season-time acceptance (documented, run ~Sept 2026, NOT part of this build's merge gate):** enable the scheduler, confirm a real week snapshots totals pre-kick and grades after finals; wire the nightly performance tracker (like the MLB one in [[truline]]).

## Notes / deferred

- **Go-live is a deliberate switch:** flipping `nfl_scheduler_enabled=True` + pushing `main` starts live scoring on Railway. Do that intentionally near Week 1, not as part of merging this phase.
- **Odds book selection:** v1 uses the first/consensus book; a book-preference list can come later.
- **P2.5 QB adjustment** and spread-model improvement remain open (spread stays shadow).
- Backfill the nightly NFL tracker once snapshots accumulate.
