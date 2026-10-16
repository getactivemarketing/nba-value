# TruLine Edge v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a market-consensus +EV finder, odds screen, player hit-rate pages and an honest CLV record for the NBA, live for the opener on Tue 2026-10-20.

**Architecture:**
- A new `backend/src/edge/` package.
- Pure-function core: `quotes`, `engine`, `schedule`, `players` (pure parts) and `grading` (pure parts).
- Thin psycopg2 storage layer: `store`.
- An ingest/pricing/grading loop (`worker`) running as its own Railway service.
- A read-only FastAPI router at `/api/v1/edge`.
- Four new React pages on truline.app.
- Legacy NBA model picks are hidden behind a flag but keep running.

**Tech Stack:**
- Backend: Python 3.11, FastAPI, SQLAlchemy 2 (table definitions and `create_all`), psycopg2 (worker and queries), httpx, pytest.
- Frontend: React 18 + TypeScript, React Query 5, Tailwind 3, recharts, vitest.
- Data: The Odds API v4 and BallDontLie v1.

**Spec:** `backend/docs/superpowers/specs/2026-10-15-edge-ev-finder-design.md`

## Global Constraints

- **Secrets:** never put `ODDS_API_KEY`, `BALLDONTLIE_API_KEY` or any upstream URL containing `apiKey=` in an API response, log line or DB row. Pass every error string through `src.utils.redact.redact()`.
- **Books (16):** the `bookmakers=` value is exactly `pinnacle,novig,prophetx,draftkings,fanduel,betmgm,williamhill_us,espnbet,fanatics,betrivers,hardrockbet,betonlineag,bovada,lowvig,prizepicks,underdog`.
- **Book roles:**
  - Sharp books are `pinnacle`, `novig`, `prophetx`.
  - DFS books are `prizepicks`, `underdog`. They are never consensus inputs and never priced for EV.
- **Weights:** Pinnacle 3.0, Novig 2.0, ProphetX 2.0, DraftKings 1.5, FanDuel 1.5, all others 1.0.
- **Prop markets:** `player_points`, `player_rebounds`, `player_assists`, `player_threes`, `player_points_rebounds_assists`, `player_points_rebounds`, `player_points_assists`, `player_rebounds_assists`.
- **Game markets:** `h2h`, `spreads`, `totals`.
- **Request format:** `oddsFormat=american`, `includeLinks=true`.
- **Devig:** power method. `devig_method` is stored as `"power"`.
- **Consensus rule:** at least 3 books quoting both sides at the same line, including at least one sharp book. Fair price is leave-one-out at bet time. At close, all books are included.
- **Thresholds:**
  - `EDGE_MIN_EV` = 0.02.
  - `EDGE_VERIFY_EV` = 0.15.
  - `EDGE_CREDIT_CAP` = 3,500,000 credits a month.
  - Guard factor is 1, 2, 4 or 8.
- **Polling cadence (minutes):**

  | Time to tip | Game lines | Props |
  |---|---|---|
  | Over 6 h (up to 36 h) | 10 | 15 |
  | 1–6 h | 3 | 5 |
  | Under 1 h | 2 | 2 |

  Polling for a game stops at tip.
- **Staleness:** a quote is stale when `now − observed_at > 2 × its current interval`. Stale quotes are never +EV.
- **CLV:** `clv = closing_fair_prob − 1/decimal_price`.
- **Line convention:**
  - `spreads`: `line` is the **home-perspective** point. The away outcome's point is negated.
  - `totals` and props: `line` = point.
  - `h2h`: `line` = 0.0.
- **Season label:** the NBA season label is the calendar year it starts in. A new season starts on Oct 1.
- **Game dates:** stored as `America/New_York` dates of `commence_time`.
- **Commits:** end every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push only in Task 15.
- **Commands:** backend commands run from `backend/` and frontend commands from `frontend/`.

## Review Focus

1. **Player not on tonight's roster / DNP.** A prop on a player with no box-score row must grade `void` once his team's box scores for that date exist. Before that it must stay ungraded, never `loss`. Covered by tests in Task 11.
2. **A book posts only one side, or a different line.** A one-sided quote can be a target, but never a consensus input. A book at 24.5 must not price a 25.5 market. Covered in Task 5.
3. **Upstream error strings carrying the key.** httpx `raise_for_status()` messages include the full request URL with `apiKey=`. Every path that stores or returns an error must redact it. Covered in Tasks 1 and 6.
4. **Accented or suffixed player names** (Jokić, Jaren Jackson Jr., two players with the same surname on different teams). These must match correctly or stay `unmatched`, never match the wrong player. Covered in Task 10.
5. **Spread side and sign.** An away +5.5 bet stored as home line −5.5 must grade correctly, and display as "+5.5" for the away team. Covered in Tasks 11 and 12.

---

## File Map

**Backend: create**
- `src/utils/redact.py`: secret scrubbing.
- `src/api/deps.py`: `require_admin`, `require_nba_picks_public`.
- `src/utils/season.py`: `current_nba_season`, `nba_season_start`.
- `src/models/edge.py`: SQLAlchemy tables for `edge_*`.
- `src/edge/__init__.py`, `src/edge/config.py`: constants.
- `src/edge/teams.py`: NBA full name ↔ abbreviation.
- `src/edge/quotes.py`: `Quote` and side helpers.
- `src/edge/engine.py`: odds math, devig, fair price, EV, CLV.
- `src/edge/schedule.py`: cadence, due/stale checks, credit guard, poll plan.
- `src/edge/client.py`: The Odds API client.
- `src/edge/ingest.py`: response → `Quote` parsing.
- `src/edge/store.py`: psycopg2 reads and writes.
- `src/edge/opportunities.py`: record +EV.
- `src/edge/players.py`: names, BallDontLie box scores, hit rates.
- `src/edge/grading.py`: closing, CLV and results.
- `src/edge/views.py`: response builders for the API.
- `src/edge/api.py`: the FastAPI router.
- `src/edge/worker.py`: the loop and the `python -m src.edge.worker` entry point.
- **Tests:** `tests/unit/test_redact.py`, `test_season.py`, `test_nba_picks_gate.py`, `test_edge_models.py`, `test_edge_engine.py`, `test_edge_ingest.py`, `test_edge_schedule.py`, `test_edge_store.py`, `test_edge_players.py`, `test_edge_grading.py`, `test_edge_api.py`.
- **Fixtures:** `tests/fixtures/edge/lines.json`, `tests/fixtures/edge/props.json`.

**Backend: modify**
- `src/config.py`: new settings.
- `src/api/nfl.py`, `src/api/mlb.py`, `src/api/health.py`: redaction and admin gating.
- `tests/unit/test_nfl_api.py`: debug test.
- `src/main.py`: router gating; mount edge.
- `src/api/trends.py`: gate `/trends/model`.
- `src/tasks/social_scheduler.py`: gate NBA posts.
- Season rollover: `src/tasks/scheduler.py`, `src/services/injuries.py`, `src/tasks/ingestion.py`, `src/api/admin.py`, `src/services/scoring/prop_scorer.py`.
- `src/models/__init__.py`: register edge models.

**Frontend: create**
- `src/types/edge.ts`.
- `src/lib/edgeApi.ts`.
- `src/lib/edgeFormat.ts` and `src/lib/edgeFormat.test.ts`.
- `src/hooks/useEdge.ts`.
- `src/components/edge/EdgeStatusBanner.tsx`, `EdgeFooter.tsx`, `EvTable.tsx`.
- `src/pages/EdgeHome.tsx`, `EdgeFinder.tsx`, `EdgeOdds.tsx`, `EdgePlayer.tsx`, `EdgeResults.tsx`.

**Frontend: modify**
- `src/App.tsx`.
- `src/components/Layout.tsx`.

---

### Task 1: Redact secrets and lock the debug endpoints

**Files:**
- Create: `backend/src/utils/redact.py`, `backend/src/api/deps.py`, `backend/tests/unit/test_redact.py`
- Modify:
  - `backend/src/api/nfl.py:207-252` (`debug_odds`)
  - `backend/src/api/mlb.py` (every `@router.get("/debug/...")`; lines 1176, 1223, 1259, 1272, 1297, 1512, 1518, 1575; key prefixes at 1186 and 1266; `str(e)` uses)
  - `backend/src/api/health.py:33,49,165`
  - `backend/tests/unit/test_nfl_api.py:179-195`

**Interfaces:**
- Produces:
  - `redact(text: object) -> str`
  - `require_admin(x_cron_key: str | None = Header(None)) -> None`. It raises 404 when the key is invalid.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/unit/test_redact.py
from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from src.utils.redact import redact


def test_query_param_keys_are_masked():
    msg = "Client error '401' for url 'https://api.the-odds-api.com/v4/sports/x/odds?apiKey=b1f7f077c2189762&regions=us'"
    out = redact(msg)
    assert "b1f7f077c2189762" not in out
    assert "apiKey=***" in out
    assert "regions=us" in out


def test_configured_secret_is_masked_anywhere(monkeypatch):
    monkeypatch.setattr(settings, "odds_api_key", "abcd1234efgh5678", raising=False)
    assert "abcd1234efgh5678" not in redact("boom abcd1234efgh5678 boom")


def test_short_or_empty_secrets_do_not_mangle_text(monkeypatch):
    monkeypatch.setattr(settings, "odds_api_key", "", raising=False)
    assert redact("plain message") == "plain message"


def test_nfl_debug_requires_admin_key():
    client = TestClient(app)
    r = client.get(f"{settings.api_v1_prefix}/nfl/debug/odds", params={"live": "false"})
    assert r.status_code == 404


def test_mlb_debug_requires_admin_key():
    client = TestClient(app)
    assert client.get(f"{settings.api_v1_prefix}/mlb/debug/odds").status_code == 404
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `python3 -m pytest tests/unit/test_redact.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.redact'`

- [ ] **Step 3: Implement `redact` and `require_admin`**

```python
# backend/src/utils/redact.py
"""Strip secrets from text before it reaches a response, a log line or the DB.

httpx's raise_for_status() message embeds the full request URL, and The Odds
API takes its key as a query parameter, so any `str(e)` of an Odds API error
contains the key. Everything that surfaces an error goes through redact().
"""
import re

from src.config import settings

_QUERY_SECRET = re.compile(r"(?i)\b(apiKey|api_key|token)=([^&\s'\"]+)")


def redact(text: object) -> str:
    s = _QUERY_SECRET.sub(lambda m: f"{m.group(1)}=***", str(text))
    for secret in (
        settings.odds_api_key,
        settings.balldontlie_api_key,
        settings.blotato_api_key,
        settings.typefully_api_key,
        settings.jwt_secret,
    ):
        if secret and len(secret) >= 8:
            s = s.replace(secret, "***")
    return s
```

```python
# backend/src/api/deps.py
"""Shared FastAPI dependencies."""
from fastapi import Header, HTTPException

from src.api.admin import _verify_cron_key


def require_admin(x_cron_key: str | None = Header(None)) -> None:
    """404 (not 401) so probes cannot tell an admin route exists."""
    if not _verify_cron_key(x_cron_key):
        raise HTTPException(status_code=404, detail="Not Found")
```

- [ ] **Step 4: Apply it to the endpoints**

1. In `src/api/nfl.py`:
   - Add `from fastapi import Depends`, `from src.api.deps import require_admin` and `from src.utils.redact import redact`.
   - Change the decorator to `@router.get("/debug/odds", dependencies=[Depends(require_admin)])`.
   - Delete the `"odds_api_key_prefix"` entry.
   - Change `detail=f"NFL data probe failed: {e}"` to `detail=f"NFL data probe failed: {redact(e)}"`.
   - Change `results["live_odds_error"] = str(e)` to `results["live_odds_error"] = redact(e)`.
2. In `src/api/mlb.py`:
   - Add the same imports.
   - Add `dependencies=[Depends(require_admin)]` to all eight `@router.get("/debug/...")` decorators.
   - Delete both `*_key_prefix` lines (1186, 1266).
   - Replace every `str(e)` in the debug handlers with `redact(e)`, keeping any existing `[:200]` / `[:100]` slice: `redact(e)[:200]`.
3. In `src/api/health.py`: import `redact` and replace the three `str(e)` uses with `redact(e)`.
4. Update `tests/unit/test_nfl_api.py::test_debug_odds_masks_key_and_skips_network_when_live_false`:
   - Pass the header: `client.get(..., params={"live": "false"}, headers={"X-Cron-Key": admin_api.CRON_API_KEY})`, with `from src.api import admin as admin_api`.
   - Replace the prefix assertion with `assert "odds_api_key_prefix" not in body`.

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `python3 -m pytest tests/unit/test_redact.py tests/unit/test_nfl_api.py -v`
Expected: all PASS

- [ ] **Step 6: Run the full unit suite**

Run: `python3 -m pytest -q tests/unit`
Expected: `597 + 5 passed` (no regressions)

- [ ] **Step 7: Commit**

```bash
git add src/utils/redact.py src/api/deps.py src/api/nfl.py src/api/mlb.py src/api/health.py tests/unit/test_redact.py tests/unit/test_nfl_api.py
git commit -m "fix(security): stop debug endpoints leaking the Odds API key

The NFL/MLB debug probes returned raw httpx errors, whose message embeds the
request URL including apiKey=. Redact all error text, drop key prefixes, and
put every /debug route behind the cron admin key (404 otherwise).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Derive the NBA season from the date

**Files:**
- Create: `backend/src/utils/season.py`, `backend/tests/unit/test_season.py`
- Modify:
  - `backend/src/tasks/scheduler.py:86,95,358,937,1079,1093`
  - `backend/src/services/injuries.py:267-269`
  - `backend/src/tasks/ingestion.py:134`
  - `backend/src/api/admin.py:181`
  - `backend/src/services/scoring/prop_scorer.py:192`

**Interfaces:**
- Produces:
  - `current_nba_season(today: date | None = None) -> int`
  - `nba_season_start(season: int) -> date` (returns Oct 1 of that year, the lower bound for "this season's" games)

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/test_season.py
from datetime import date

from src.utils.season import current_nba_season, nba_season_start


def test_october_starts_the_new_season():
    assert current_nba_season(date(2026, 10, 1)) == 2026
    assert current_nba_season(date(2026, 10, 20)) == 2026


def test_spring_and_summer_belong_to_the_previous_season():
    assert current_nba_season(date(2027, 4, 15)) == 2026
    assert current_nba_season(date(2026, 9, 30)) == 2025


def test_season_start_is_october_first():
    assert nba_season_start(2026) == date(2026, 10, 1)
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `python3 -m pytest tests/unit/test_season.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# backend/src/utils/season.py
"""NBA season label = the calendar year the season starts in (2026-27 -> 2026).

BallDontLie uses the same convention for `seasons[]`. Oct 1 is the rollover:
preseason begins early October and nothing in Jul-Sep belongs to a new season.
"""
from datetime import date


def current_nba_season(today: date | None = None) -> int:
    d = today or date.today()
    return d.year if d.month >= 10 else d.year - 1


def nba_season_start(season: int) -> date:
    return date(season, 10, 1)
```

- [ ] **Step 4: Replace the hardcoded values**

Add `from src.utils.season import current_nba_season, nba_season_start` to each file, then make these replacements:

| File:line | Before | After |
|---|---|---|
| `scheduler.py:86` and `:1079` | `season_start = date(2025, 10, 22)  # 2025-26 NBA season` | `season_start = nba_season_start(current_nba_season())` |
| `scheduler.py:95`, `:937`, `:1093` | `seasons=[2025]` | `seasons=[current_nba_season()]` |
| `scheduler.py:358` | the literal `2025` in the INSERT parameter tuple | `current_nba_season(game_date_et)` (`game_date_et` is a `date` there) |
| `injuries.py:267-269` | comment + `season=2025` | delete the comment; `season=current_nba_season()` |
| `ingestion.py:134` | `season=2025,  # Current season` | `season=current_nba_season(),` |
| `admin.py:181` | `season=2025,` | `season=current_nba_season(),` |
| `prop_scorer.py:192` | `season=2025` | `season=current_nba_season()` |

Then run: `grep -rn "2025" src/tasks/scheduler.py src/services/injuries.py src/tasks/ingestion.py src/api/admin.py src/services/scoring/prop_scorer.py`
Expected: only comments and dates in log strings remain, with no season arguments.

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest -q tests/unit`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/utils/season.py tests/unit/test_season.py src/tasks/scheduler.py src/services/injuries.py src/tasks/ingestion.py src/api/admin.py src/services/scoring/prop_scorer.py
git commit -m "fix(nba): derive the season from the date instead of hardcoding 2025

Without this the 2026-27 opener is scored on 2025-26 team stats with ~180
days of rest.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Hide legacy NBA picks (tracking-only)

**Files:**
- Modify:
  - `backend/src/config.py`: add settings after `runline_in_best_bet`.
  - `backend/src/api/deps.py`: add `require_nba_picks_public`.
  - `backend/src/main.py:224-230`: router includes.
  - `backend/src/api/trends.py:331`: the `/trends/model` decorator.
  - `backend/src/tasks/social_scheduler.py:782,793`: the NBA post calls.
- Create: `backend/tests/unit/test_nba_picks_gate.py`

**Interfaces:**
- Produces:
  - Settings fields `nba_picks_public: bool = False`, `edge_min_ev: float = 0.02`, `edge_verify_ev: float = 0.15`, `edge_credit_cap: int = 3_500_000`.
  - `require_nba_picks_public(x_cron_key: str | None = Header(None)) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/test_nba_picks_gate.py
from fastapi.testclient import TestClient

from src.api import admin as admin_api
from src.config import settings
from src.main import app

P = settings.api_v1_prefix
LEGACY = [f"{P}/markets", f"{P}/markets/live", f"{P}/evaluation/summary",
          f"{P}/props/top", f"{P}/games/upcoming", f"{P}/trends/model", f"{P}/backtest/results"]


def test_legacy_nba_pick_routes_404_when_not_public(monkeypatch):
    monkeypatch.setattr(settings, "nba_picks_public", False, raising=False)
    client = TestClient(app)
    for path in LEGACY:
        assert client.get(path).status_code == 404, path


def test_admin_key_still_reaches_them(monkeypatch):
    monkeypatch.setattr(settings, "nba_picks_public", False, raising=False)
    client = TestClient(app, raise_server_exceptions=False)  # no local DB: handler may 500, but not 404
    r = client.get(f"{P}/evaluation/summary", headers={"X-Cron-Key": admin_api.CRON_API_KEY})
    assert r.status_code != 404


def test_mlb_and_nfl_are_untouched(monkeypatch):
    monkeypatch.setattr(settings, "nba_picks_public", False, raising=False)
    schema = TestClient(app).get("/openapi.json").json()
    assert f"{P}/nfl/games" in schema["paths"]
    assert any(p.startswith(f"{P}/mlb/") for p in schema["paths"])


def test_social_skips_nba_when_not_public(monkeypatch):
    from src.tasks import social_scheduler as ss
    monkeypatch.setattr(settings, "nba_picks_public", False, raising=False)
    assert ss.nba_posting_allowed() is False
    monkeypatch.setattr(settings, "nba_picks_public", True, raising=False)
    assert ss.nba_posting_allowed() is True
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `python3 -m pytest tests/unit/test_nba_picks_gate.py -v`
Expected: FAIL (routes return non-404; `nba_posting_allowed` not defined)

- [ ] **Step 3: Implement**

Add to `Settings` in `src/config.py`:

```python
    # Legacy NBA model picks are TRACKING-ONLY from 2026-10-15: spread v2 was
    # fed the wrong feature names and predicted a constant -9.86 MOV all of
    # 2025-26. Snapshots and grading keep running; nothing reaches the public
    # until a pre-registered CLV gate passes. See the Edge spec.
    nba_picks_public: bool = False

    # TruLine Edge (docs/superpowers/specs/2026-10-15-edge-ev-finder-design.md)
    edge_min_ev: float = 0.02
    edge_verify_ev: float = 0.15
    edge_credit_cap: int = 3_500_000
```

Add `from src.config import settings` to the imports at the top of `src/api/deps.py`, then append:

```python
def require_nba_picks_public(x_cron_key: str | None = Header(None)) -> None:
    if settings.nba_picks_public or _verify_cron_key(x_cron_key):
        return
    raise HTTPException(status_code=404, detail="Not Found")
```

In `src/main.py`:
- Add `from fastapi import Depends` and `from src.api.deps import require_nba_picks_public`.
- Define `_nba_gate = [Depends(require_nba_picks_public)]`.
- Pass `dependencies=_nba_gate` to the `include_router` calls for `markets.router`, `bets.router`, `evaluation.router` and `backtest.router`.

In `src/api/trends.py`:
- Change `@router.get("/trends/model")` to `@router.get("/trends/model", dependencies=[Depends(require_nba_picks_public)])`, adding the imports.

In `src/tasks/social_scheduler.py`, add near the top:

```python
from src.config import settings


def nba_posting_allowed() -> bool:
    """NBA model picks are tracking-only; never tweet them while hidden."""
    return settings.nba_picks_public
```

Then wrap both NBA calls (lines 782 and 793) in `if nba_posting_allowed():`, with an `else:` branch that logs via the module's existing `log_task(...)`. For example:

```python
        if nba_posting_allowed():
            result = _run_async(_post_nba_daily_picks_async())
        else:
            log_task("NBA picks are tracking-only; skipping post")
            result = {"posted": False, "type": "nba_picks", "skipped": "tracking_only"}
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/unit/test_nba_picks_gate.py -v && python3 -m pytest -q tests/unit`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/config.py src/api/deps.py src/main.py src/api/trends.py src/tasks/social_scheduler.py tests/unit/test_nba_picks_gate.py
git commit -m "feat(nba): hide legacy model picks behind NBA_PICKS_PUBLIC (default off)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Edge tables

**Files:**
- Create: `backend/src/models/edge.py`, `backend/tests/unit/test_edge_models.py`
- Modify: `backend/src/models/__init__.py`: import the new models.

**Interfaces:**
- Produces these tables, created by the web service's `init_db()` `create_all`:
  - `edge_events`
  - `edge_quotes`
  - `edge_quote_history`
  - `edge_players`
  - `edge_player_games`
  - `edge_opportunities`
  - `edge_api_calls`
  - `edge_worker_state`
- The column names below are the contract that `store.py` uses.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/test_edge_models.py
from src.database import Base
import src.models  # noqa: F401  registers all tables


def _uniques(table):
    return {tuple(c.name for c in u.columns) for u in table.constraints
            if u.__class__.__name__ == "UniqueConstraint"}


def test_edge_tables_registered():
    for name in ("edge_events", "edge_quotes", "edge_quote_history", "edge_players",
                 "edge_player_games", "edge_opportunities", "edge_api_calls", "edge_worker_state"):
        assert name in Base.metadata.tables, name


def test_quote_and_opportunity_keys():
    t = Base.metadata.tables
    assert ("event_id", "market", "player", "side", "line", "book") in _uniques(t["edge_quotes"])
    assert ("event_id", "market", "player", "side", "line", "book", "price") in _uniques(t["edge_opportunities"])
    assert ("bdl_player_id", "game_date") in _uniques(t["edge_player_games"])
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `python3 -m pytest tests/unit/test_edge_models.py -v`
Expected: FAIL (`edge_events` not in metadata)

- [ ] **Step 3: Implement the models**

```python
# backend/src/models/edge.py
"""TruLine Edge tables. See docs/superpowers/specs/2026-10-15-edge-ev-finder-design.md §5.

edge_quotes is MUTABLE-CURRENT (one row per price key); edge_quote_history is
its append-only counterpart, written only when a key's price changes — it is
what makes line movement and closing-line value possible.
"""
from datetime import date, datetime

from sqlalchemy import (BigInteger, Boolean, Date, DateTime, Float, Index, Integer,
                        Numeric, String, Text, UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class EdgeEvent(Base):
    __tablename__ = "edge_events"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sport: Mapped[str] = mapped_column(String(40), nullable=False)
    home_team: Mapped[str] = mapped_column(String(60), nullable=False)
    away_team: Mapped[str] = mapped_column(String(60), nullable=False)
    commence_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(12), nullable=False, default="scheduled")
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    last_lines_poll: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_props_poll: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EdgeQuote(Base):
    __tablename__ = "edge_quotes"
    __table_args__ = (UniqueConstraint("event_id", "market", "player", "side", "line", "book",
                                       name="uq_edge_quotes_key"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    market: Mapped[str] = mapped_column(String(48), nullable=False)
    player: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    line: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    book: Mapped[str] = mapped_column(String(32), nullable=False)
    price: Mapped[int] = mapped_column(Integer, nullable=False)
    link: Mapped[str | None] = mapped_column(Text)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EdgeQuoteHistory(Base):
    __tablename__ = "edge_quote_history"
    __table_args__ = (Index("ix_edge_hist_market", "event_id", "market", "player", "line", "observed_at"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False)
    market: Mapped[str] = mapped_column(String(48), nullable=False)
    player: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    line: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    book: Mapped[str] = mapped_column(String(32), nullable=False)
    price: Mapped[int] = mapped_column(Integer, nullable=False)
    link: Mapped[str | None] = mapped_column(Text)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EdgePlayer(Base):
    __tablename__ = "edge_players"
    odds_name: Mapped[str] = mapped_column(String(80), primary_key=True)
    bdl_player_id: Mapped[int | None] = mapped_column(Integer, index=True)
    team: Mapped[str | None] = mapped_column(String(4))
    match_method: Mapped[str] = mapped_column(String(12), nullable=False)  # exact|normalized|fuzzy|manual|unmatched
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EdgePlayerGame(Base):
    __tablename__ = "edge_player_games"
    __table_args__ = (UniqueConstraint("bdl_player_id", "game_date", name="uq_edge_player_games"),
                      Index("ix_edge_pg_team_date", "team", "game_date"))
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    bdl_player_id: Mapped[int] = mapped_column(Integer, nullable=False)
    bdl_game_id: Mapped[int] = mapped_column(Integer, nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    team: Mapped[str] = mapped_column(String(4), nullable=False)
    opponent: Mapped[str] = mapped_column(String(4), nullable=False)
    is_home: Mapped[bool] = mapped_column(Boolean, nullable=False)
    minutes: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reb: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ast: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fg3m: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stl: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    blk: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tov: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class EdgeOpportunity(Base):
    __tablename__ = "edge_opportunities"
    __table_args__ = (UniqueConstraint("event_id", "market", "player", "side", "line", "book", "price",
                                       name="uq_edge_opportunity"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    market: Mapped[str] = mapped_column(String(48), nullable=False)
    player: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    line: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    book: Mapped[str] = mapped_column(String(32), nullable=False)
    price: Mapped[int] = mapped_column(Integer, nullable=False)
    link: Mapped[str | None] = mapped_column(Text)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fair_prob: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    ev: Mapped[float] = mapped_column(Numeric(7, 5), nullable=False)
    max_ev: Mapped[float] = mapped_column(Numeric(7, 5), nullable=False)
    n_books: Mapped[int] = mapped_column(Integer, nullable=False)
    devig_method: Mapped[str] = mapped_column(String(12), nullable=False, default="power")
    closing_done: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    closing_fair_prob: Mapped[float | None] = mapped_column(Numeric(6, 5))
    clv: Mapped[float | None] = mapped_column(Numeric(7, 5))
    clv_note: Mapped[str | None] = mapped_column(String(80))
    result: Mapped[str | None] = mapped_column(String(8))  # win|loss|push|void
    profit_units: Mapped[float | None] = mapped_column(Numeric(7, 4))


class EdgeApiCall(Base):
    __tablename__ = "edge_api_calls"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    called_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    endpoint: Mapped[str] = mapped_column(String(32), nullable=False)
    cost: Mapped[int] = mapped_column(Integer, nullable=False)
    remaining: Mapped[int | None] = mapped_column(Integer)


class EdgeWorkerState(Base):
    __tablename__ = "edge_worker_state"
    key: Mapped[str] = mapped_column(String(40), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
```

Append to `src/models/__init__.py`, following the style of its existing NFL block:

```python
# TruLine Edge
from src.models.edge import (EdgeApiCall, EdgeEvent, EdgeOpportunity, EdgePlayer,
                             EdgePlayerGame, EdgeQuote, EdgeQuoteHistory, EdgeWorkerState)
```

If the file defines `__all__`, add the eight names to it.

- [ ] **Step 4: Run the test**

Run: `python3 -m pytest tests/unit/test_edge_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/edge.py src/models/__init__.py tests/unit/test_edge_models.py
git commit -m "feat(edge): tables for quotes, history, players, opportunities and worker state

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Quotes and the EV engine (pure)

**Files:**
- Create:
  - `backend/src/edge/__init__.py` (empty)
  - `backend/src/edge/config.py`
  - `backend/src/edge/quotes.py`
  - `backend/src/edge/engine.py`
  - `backend/tests/unit/test_edge_engine.py`

**Interfaces:**
- Produces:
  - `quotes.Quote(event_id, market, player, side, line, book, price, link, observed_at)`, frozen, with `.market_key` and `.key`.
  - `quotes.sides_for(market) -> tuple[str, str]`
  - `engine.american_to_decimal(price: int) -> float`
  - `engine.implied_prob(price: int) -> float`
  - `engine.prob_to_american(p: float) -> int`
  - `engine.devig_power(p1: float, p2: float) -> tuple[float, float]`
  - `engine.Fair(prob: float | None, n_books: int, has_sharp: bool)`, with `.thin`
  - `engine.group(quotes) -> dict[market_key, dict[book, dict[side, Quote]]]`
  - `engine.fair_prob(book_sides, market, side, exclude_book) -> Fair`
  - `engine.Priced(quote, fair, ev, is_ev, verify)`
  - `engine.price_quotes(quotes, min_ev, verify_ev) -> list[Priced]`
  - `engine.latest_before(quotes, cutoff, window) -> list[Quote]`
  - `engine.clv(closing_prob: float, price: int) -> float`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/unit/test_edge_engine.py
from datetime import datetime, timedelta, timezone

import pytest

from src.edge import engine
from src.edge.quotes import Quote
from src.services.mlb.clv import clv_from_closing

T0 = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)


def q(book, side, price, line=27.5, market="player_points", player="Jayson Tatum", at=T0):
    return Quote("ev1", market, player, side, line, book, price, None, at)


def market(*books):
    """books: (book, over_price, under_price)"""
    out = []
    for b, o, u in books:
        out += [q(b, "over", o), q(b, "under", u)]
    return out


class TestOddsMath:
    def test_american_to_decimal(self):
        assert engine.american_to_decimal(150) == pytest.approx(2.5)
        assert engine.american_to_decimal(-200) == pytest.approx(1.5)

    def test_prob_to_american_round_trips(self):
        assert engine.prob_to_american(0.5) == -100
        assert engine.prob_to_american(0.6) == -150
        assert engine.prob_to_american(0.4) == 150

    def test_power_devig_sums_to_one_and_respects_favourite(self):
        a, b = engine.devig_power(engine.implied_prob(-150), engine.implied_prob(130))
        assert a + b == pytest.approx(1.0, abs=1e-9)
        assert a > 0.55 and b < 0.45

    def test_power_devig_of_fair_market_is_identity(self):
        a, b = engine.devig_power(0.5, 0.5)
        assert a == pytest.approx(0.5) and b == pytest.approx(0.5)


class TestFairPrice:
    def test_leave_one_out_excludes_the_target_book(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110),
                        ("draftkings", -110, -110), ("betmgm", 300, -500))
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        with_mgm = engine.fair_prob(books, "player_points", "over", exclude_book=None)
        without_mgm = engine.fair_prob(books, "player_points", "over", exclude_book="betmgm")
        assert without_mgm.prob == pytest.approx(0.5, abs=1e-6)
        assert with_mgm.prob != pytest.approx(0.5, abs=1e-3)

    def test_thin_without_a_sharp_book(self):
        quotes = market(("fanduel", -110, -110), ("draftkings", -110, -110), ("betmgm", -110, -110))
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        f = engine.fair_prob(books, "player_points", "over", exclude_book=None)
        assert f.thin and f.n_books == 3 and not f.has_sharp

    def test_thin_with_fewer_than_three_books(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110))
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        assert engine.fair_prob(books, "player_points", "over", exclude_book=None).thin

    def test_one_sided_quotes_are_not_consensus_inputs(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110), ("draftkings", -110, -110))
        quotes.append(q("betmgm", "over", 500))  # over only
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        assert engine.fair_prob(books, "player_points", "over", exclude_book=None).n_books == 3

    def test_dfs_books_never_count(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110),
                        ("prizepicks", -137, -137), ("underdog", -137, -137))
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        assert engine.fair_prob(books, "player_points", "over", exclude_book=None).n_books == 2

    def test_different_lines_are_different_markets(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110), ("draftkings", -110, -110))
        quotes += [q("betmgm", "over", 200, line=25.5), q("betmgm", "under", -250, line=25.5)]
        groups = engine.group(quotes)
        assert ("ev1", "player_points", "Jayson Tatum", 25.5) in groups
        assert "betmgm" not in groups[("ev1", "player_points", "Jayson Tatum", 27.5)]

    def test_weights_pull_toward_pinnacle(self):
        quotes = market(("pinnacle", -150, 130), ("fanduel", -110, -110), ("draftkings", -110, -110))
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        f = engine.fair_prob(books, "player_points", "over", exclude_book=None)
        pin_over, _ = engine.devig_power(engine.implied_prob(-150), engine.implied_prob(130))
        unweighted = (pin_over + 0.5 + 0.5) / 3
        assert f.prob > unweighted


class TestPricing:
    def test_soft_book_above_consensus_is_ev(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110),
                        ("draftkings", -110, -110), ("betmgm", 120, -145))
        priced = engine.price_quotes(quotes, min_ev=0.02, verify_ev=0.15)
        hit = [p for p in priced if p.quote.book == "betmgm" and p.quote.side == "over"][0]
        assert hit.is_ev and hit.ev == pytest.approx(0.5 * 2.2 - 1, abs=1e-6)
        assert not hit.verify

    def test_huge_edges_are_flagged_verify(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110),
                        ("draftkings", -110, -110), ("betmgm", 200, -260))
        hit = [p for p in engine.price_quotes(quotes, 0.02, 0.15)
               if p.quote.book == "betmgm" and p.quote.side == "over"][0]
        assert hit.verify

    def test_thin_markets_have_no_ev(self):
        quotes = market(("fanduel", -110, -110), ("betmgm", 150, -180))
        assert all(p.ev is None and not p.is_ev for p in engine.price_quotes(quotes, 0.02, 0.15))

    def test_dfs_books_are_not_priced(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110),
                        ("draftkings", -110, -110), ("prizepicks", -137, -137))
        assert not [p for p in engine.price_quotes(quotes, 0.02, 0.15) if p.quote.book == "prizepicks"]


class TestClosing:
    def test_latest_before_takes_last_quote_per_key_inside_window(self):
        early = q("fanduel", "over", -105, at=T0 - timedelta(hours=3))
        late = q("fanduel", "over", -120, at=T0 - timedelta(minutes=5))
        after = q("fanduel", "over", -150, at=T0 + timedelta(minutes=5))
        out = engine.latest_before([early, late, after], cutoff=T0, window=timedelta(minutes=60))
        assert [x.price for x in out] == [-120]

    def test_latest_before_drops_quotes_older_than_window(self):
        stale = q("fanduel", "over", -105, at=T0 - timedelta(hours=3))
        assert engine.latest_before([stale], cutoff=T0, window=timedelta(minutes=30)) == []

    def test_clv_matches_the_mlb_definition(self):
        for price, p in ((120, 0.48), (-150, 0.62), (100, 0.5)):
            expected = clv_from_closing(engine.american_to_decimal(price), p)
            assert engine.clv(p, price) == pytest.approx(expected, abs=1e-12)
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `python3 -m pytest tests/unit/test_edge_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.edge'`

- [ ] **Step 3: Implement `config.py` and `quotes.py`**

```python
# backend/src/edge/config.py
"""TruLine Edge constants. Values are the spec's (§5, §6); do not tune here."""
SPORT_KEY = "basketball_nba"

SHARP_BOOKS = frozenset({"pinnacle", "novig", "prophetx"})
DFS_BOOKS = frozenset({"prizepicks", "underdog"})
BOOKS = (
    "pinnacle", "novig", "prophetx",
    "draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "fanatics",
    "betrivers", "hardrockbet", "betonlineag", "bovada", "lowvig",
    "prizepicks", "underdog",
)
BOOK_WEIGHTS = {"pinnacle": 3.0, "novig": 2.0, "prophetx": 2.0, "draftkings": 1.5, "fanduel": 1.5}

LINE_MARKETS = ("h2h", "spreads", "totals")
PROP_MARKETS = (
    "player_points", "player_rebounds", "player_assists", "player_threes",
    "player_points_rebounds_assists", "player_points_rebounds",
    "player_points_assists", "player_rebounds_assists",
)
MIN_BOOKS = 3
DEVIG_METHOD = "power"
# The close is read from edge_quotes, which freezes at tip because polling
# stops there. A key whose last sighting is older than this window was pulled
# before tip and is not part of the close. 30 covers the 2-minute final-hour
# cadence even at guard factor 8 (16 minutes).
CLOSING_WINDOW_MINUTES = 30
```

```python
# backend/src/edge/quotes.py
"""One book's price for one side of one market, as observed by a poll.

Line convention (spec Global Constraints): spreads carry the HOME-perspective
point so both sides of a spread share a market key; totals/props carry the
point; h2h carries 0.0.
"""
from dataclasses import dataclass
from datetime import datetime

GAME_SIDES = ("home", "away")
OU_SIDES = ("over", "under")


def sides_for(market: str) -> tuple[str, str]:
    return OU_SIDES if market == "totals" or market.startswith("player_") else GAME_SIDES


@dataclass(frozen=True)
class Quote:
    event_id: str
    market: str
    player: str        # "" for game lines
    side: str          # home|away|over|under
    line: float
    book: str
    price: int         # American
    link: str | None
    observed_at: datetime

    @property
    def market_key(self) -> tuple[str, str, str, float]:
        return (self.event_id, self.market, self.player, float(self.line))

    @property
    def key(self) -> tuple[str, str, str, float, str, str]:
        return (*self.market_key, self.side, self.book)
```

- [ ] **Step 4: Implement `engine.py`**

```python
# backend/src/edge/engine.py
"""Pure pricing functions: odds math, power devig, consensus fair price, EV, CLV.

No I/O. Spec §6.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping

from src.edge.config import BOOK_WEIGHTS, DFS_BOOKS, MIN_BOOKS, SHARP_BOOKS
from src.edge.quotes import Quote, sides_for

MarketKey = tuple[str, str, str, float]
BookSides = Mapping[str, Mapping[str, Quote]]


def american_to_decimal(price: int) -> float:
    if price == 0:
        raise ValueError("American price cannot be 0")
    return 1 + price / 100 if price > 0 else 1 + 100 / abs(price)


def implied_prob(price: int) -> float:
    return 1 / american_to_decimal(price)


def prob_to_american(p: float) -> int:
    if not 0 < p < 1:
        raise ValueError(f"probability out of range: {p}")
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def devig_power(p1: float, p2: float) -> tuple[float, float]:
    """Find k with p1**k + p2**k == 1 (bisection), return the two fair probs.

    Unlike multiplicative devig, the power method takes more margin off the
    longshot than the favourite, which is where books actually put it.
    """
    lo, hi = 0.01, 100.0
    for _ in range(200):
        k = (lo + hi) / 2
        if p1 ** k + p2 ** k > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    a, b = p1 ** k, p2 ** k
    return a / (a + b), b / (a + b)


@dataclass(frozen=True)
class Fair:
    prob: float | None   # None => thin market, no EV
    n_books: int
    has_sharp: bool

    @property
    def thin(self) -> bool:
        return self.prob is None


def group(quotes: Iterable[Quote]) -> dict[MarketKey, dict[str, dict[str, Quote]]]:
    out: dict[MarketKey, dict[str, dict[str, Quote]]] = defaultdict(lambda: defaultdict(dict))
    for q in quotes:
        out[q.market_key][q.book][q.side] = q
    return {k: {b: dict(s) for b, s in v.items()} for k, v in out.items()}


def fair_prob(book_sides: BookSides, market: str, side: str, exclude_book: str | None) -> Fair:
    a, b = sides_for(market)
    weighted, weights, n, sharp = 0.0, 0.0, 0, False
    for book, sides in book_sides.items():
        if book == exclude_book or book in DFS_BOOKS or a not in sides or b not in sides:
            continue
        pa, pb = devig_power(implied_prob(sides[a].price), implied_prob(sides[b].price))
        w = BOOK_WEIGHTS.get(book, 1.0)
        weighted += (pa if side == a else pb) * w
        weights += w
        n += 1
        sharp = sharp or book in SHARP_BOOKS
    if n < MIN_BOOKS or not sharp:
        return Fair(None, n, sharp)
    return Fair(weighted / weights, n, sharp)


@dataclass(frozen=True)
class Priced:
    quote: Quote
    fair: Fair
    ev: float | None
    is_ev: bool
    verify: bool


def price_quotes(quotes: Iterable[Quote], min_ev: float, verify_ev: float) -> list[Priced]:
    out: list[Priced] = []
    for (_, market, _, _), books in group(quotes).items():
        for book, sides in books.items():
            if book in DFS_BOOKS:
                continue
            for side, q in sides.items():
                f = fair_prob(books, market, side, exclude_book=book)
                ev = None if f.thin else f.prob * american_to_decimal(q.price) - 1
                out.append(Priced(q, f, ev,
                                  is_ev=ev is not None and ev >= min_ev,
                                  verify=ev is not None and ev > verify_ev))
    return out


def latest_before(quotes: Iterable[Quote], cutoff: datetime, window: timedelta) -> list[Quote]:
    """Last observation per price key at or before `cutoff`, no older than `window`."""
    best: dict[tuple, Quote] = {}
    for q in quotes:
        if q.observed_at > cutoff or q.observed_at < cutoff - window:
            continue
        cur = best.get(q.key)
        if cur is None or q.observed_at > cur.observed_at:
            best[q.key] = q
    return list(best.values())


def clv(closing_prob: float, price: int) -> float:
    """Same definition as src/services/mlb/clv.py: closing fair prob minus break-even prob."""
    return closing_prob - 1 / american_to_decimal(price)
```

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/unit/test_edge_engine.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/edge/__init__.py src/edge/config.py src/edge/quotes.py src/edge/engine.py tests/unit/test_edge_engine.py
git commit -m "feat(edge): power devig, leave-one-out consensus fair price, EV and CLV

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Odds API client and response parsing

**Files:**
- Create:
  - `backend/src/edge/client.py`
  - `backend/src/edge/ingest.py`
  - `backend/tests/unit/test_edge_ingest.py`
  - `backend/tests/fixtures/edge/lines.json`, `backend/tests/fixtures/edge/props.json`

**Interfaces:**
- Consumes: `Quote`, `config.BOOKS`, `LINE_MARKETS`, `PROP_MARKETS`, `SPORT_KEY`, and `redact`.
- Produces:
  - `client.Usage(remaining: int | None, used: int | None, last: int | None)`
  - `client.OddsApiError(status: int, message: str)`, where the message is already redacted.
  - `client.EdgeOddsClient(api_key: str | None = None)` with async methods:
    - `get_events() -> tuple[list[dict], Usage]`
    - `get_game_odds() -> tuple[list[dict], Usage]`
    - `get_event_odds(event_id: str) -> tuple[dict, Usage]`
    - `get_scores(days_from: int = 1) -> tuple[list[dict], Usage]`
  - `ingest.parse_event(event: dict, observed_at: datetime) -> list[Quote]`
  - `ingest.event_row(event: dict) -> dict` with keys `event_id`, `sport`, `home_team`, `away_team`, `commence_time`.

- [ ] **Step 1: Save the fixtures**

These are real responses captured on 2026-10-15: 46 events, with props for BOS @ DET. Trim the lines file to two events to keep the repo small:

```bash
mkdir -p tests/fixtures/edge
SRC=/private/tmp/claude-501/-Volumes-Samir-Ext-Sites/41d435ad-264a-43c6-8c1e-dbb6311e9c77/scratchpad/edge-fixtures
python3 -c "import json;d=json.load(open('$SRC/lines.json'));json.dump(d[:2],open('tests/fixtures/edge/lines.json','w'))"
cp $SRC/props.json tests/fixtures/edge/props.json
grep -c apiKey tests/fixtures/edge/*.json   # expect 0 for both
```

If the scratch path is gone, re-capture the fixtures. This costs 22 credits. Use `bookmakers=<the 16 books>`, `oddsFormat=american` and `includeLinks=true`, as in the spec.

- [ ] **Step 2: Write the failing tests**

```python
# backend/tests/unit/test_edge_ingest.py
import json
from datetime import datetime, timezone

import httpx
import pytest

from src.edge import client as client_mod
from src.edge.ingest import event_row, parse_event

NOW = datetime(2026, 10, 16, 2, 10, tzinfo=timezone.utc)
LINES = json.load(open("tests/fixtures/edge/lines.json"))
PROPS = json.load(open("tests/fixtures/edge/props.json"))


def test_event_row():
    row = event_row(LINES[0])
    assert row == {"event_id": "26b036ff107f3c658258eaf4a6f26228", "sport": "basketball_nba",
                   "home_team": "Detroit Pistons", "away_team": "Boston Celtics",
                   "commence_time": datetime(2026, 10, 20, 19, 0, tzinfo=timezone.utc)}


def test_h2h_maps_team_names_to_home_away():
    quotes = [x for x in parse_event(LINES[0], NOW) if x.market == "h2h" and x.book == "fanduel"]
    by_side = {x.side: x for x in quotes}
    assert by_side["away"].price == 100 and by_side["home"].price == -118
    assert by_side["home"].line == 0.0 and by_side["home"].player == ""
    assert by_side["home"].link.startswith("https://sportsbook.fanduel.com/addToBetslip")


def test_spreads_store_home_perspective_line_on_both_sides():
    quotes = [x for x in parse_event(LINES[0], NOW) if x.market == "spreads" and x.book == "fanduel"]
    assert len(quotes) == 2
    assert quotes[0].line == quotes[1].line  # same market key for both sides


def test_props_parse_player_side_and_point():
    quotes = parse_event(PROPS, NOW)
    tatum = [x for x in quotes if x.player == "Jayson Tatum" and x.market == "player_assists"
             and x.book == "fanduel"]
    assert {(x.side, x.line, x.price) for x in tatum} >= {("over", 5.5, -138), ("under", 5.5, 104)}
    assert all(x.observed_at == NOW for x in quotes)


def test_parse_skips_malformed_outcomes():
    bad = {**PROPS, "bookmakers": [{"key": "fanduel", "markets": [
        {"key": "player_points", "outcomes": [{"name": "Over", "price": -110}]}]}]}  # no description/point
    assert parse_event(bad, NOW) == []


@pytest.mark.asyncio
async def test_client_errors_are_redacted(monkeypatch):
    def handler(request):
        return httpx.Response(401, json={"message": "bad key"})

    transport = httpx.MockTransport(handler)
    c = client_mod.EdgeOddsClient(api_key="secretsecret1234", transport=transport)
    with pytest.raises(client_mod.OddsApiError) as exc:
        await c.get_events()
    assert exc.value.status == 401
    assert "secretsecret1234" not in str(exc.value)


@pytest.mark.asyncio
async def test_client_reads_usage_headers():
    def handler(request):
        assert request.url.params["bookmakers"].startswith("pinnacle,novig")
        return httpx.Response(200, json=[], headers={"x-requests-remaining": "4999000",
                                                     "x-requests-used": "1000", "x-requests-last": "6"})

    c = client_mod.EdgeOddsClient(api_key="k" * 16, transport=httpx.MockTransport(handler))
    data, usage = await c.get_game_odds()
    assert data == [] and usage == client_mod.Usage(4999000, 1000, 6)
```

- [ ] **Step 3: Run the tests to confirm they fail**

Run: `python3 -m pytest tests/unit/test_edge_ingest.py -v`
Expected: FAIL (`ModuleNotFoundError: src.edge.ingest`)

- [ ] **Step 4: Implement `client.py`**

```python
# backend/src/edge/client.py
"""The Odds API v4 client for Edge. The key travels as a query param, so every
error message is redacted before it leaves this module."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.config import settings
from src.edge.config import BOOKS, LINE_MARKETS, PROP_MARKETS, SPORT_KEY
from src.utils.redact import redact

BASE = "https://api.the-odds-api.com/v4"


@dataclass(frozen=True)
class Usage:
    remaining: int | None
    used: int | None
    last: int | None


class OddsApiError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(redact(message))


def _int(v: str | None) -> int | None:
    try:
        return int(float(v)) if v is not None else None
    except ValueError:
        return None


class EdgeOddsClient:
    def __init__(self, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.api_key = api_key if api_key is not None else settings.odds_api_key
        self.transport = transport

    async def _get(self, path: str, params: dict) -> tuple[object, Usage]:
        params = {"apiKey": self.api_key, **params}
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=30.0) as c:
                r = await c.get(f"{BASE}{path}", params=params)
        except httpx.HTTPError as e:
            raise OddsApiError(0, f"network error: {e}") from None
        if r.status_code != 200:
            raise OddsApiError(r.status_code, f"{r.status_code} on {path}: {r.text[:200]}")
        h = r.headers
        return r.json(), Usage(_int(h.get("x-requests-remaining")), _int(h.get("x-requests-used")),
                               _int(h.get("x-requests-last")))

    async def get_events(self):
        return await self._get(f"/sports/{SPORT_KEY}/events", {})

    async def get_game_odds(self):
        return await self._get(f"/sports/{SPORT_KEY}/odds", {
            "bookmakers": ",".join(BOOKS), "markets": ",".join(LINE_MARKETS),
            "oddsFormat": "american", "includeLinks": "true"})

    async def get_event_odds(self, event_id: str):
        return await self._get(f"/sports/{SPORT_KEY}/events/{event_id}/odds", {
            "bookmakers": ",".join(BOOKS), "markets": ",".join(PROP_MARKETS),
            "oddsFormat": "american", "includeLinks": "true"})

    async def get_scores(self, days_from: int = 1):
        return await self._get(f"/sports/{SPORT_KEY}/scores", {"daysFrom": days_from})
```

- [ ] **Step 5: Implement `ingest.py`**

```python
# backend/src/edge/ingest.py
"""Odds API JSON -> Quote rows. Malformed outcomes are skipped, never guessed."""
from __future__ import annotations

from datetime import datetime

from src.edge.quotes import Quote


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def event_row(event: dict) -> dict:
    return {"event_id": event["id"], "sport": event["sport_key"], "home_team": event["home_team"],
            "away_team": event["away_team"], "commence_time": _ts(event["commence_time"])}


def _outcome_quote(event: dict, book: str, market: str, o: dict, observed_at: datetime) -> Quote | None:
    price = o.get("price")
    if not isinstance(price, (int, float)) or price == 0:
        return None
    price = int(price)
    link = o.get("link")
    eid, home, away = event["id"], event["home_team"], event["away_team"]

    if market == "h2h":
        side = "home" if o.get("name") == home else "away" if o.get("name") == away else None
        return Quote(eid, market, "", side, 0.0, book, price, link, observed_at) if side else None

    point = o.get("point")
    if not isinstance(point, (int, float)):
        return None

    if market == "spreads":
        if o.get("name") == home:
            return Quote(eid, market, "", "home", float(point), book, price, link, observed_at)
        if o.get("name") == away:
            return Quote(eid, market, "", "away", -float(point), book, price, link, observed_at)
        return None

    side = {"Over": "over", "Under": "under"}.get(o.get("name"))
    if side is None:
        return None
    if market == "totals":
        return Quote(eid, market, "", side, float(point), book, price, link, observed_at)
    player = (o.get("description") or "").strip()
    if not player:
        return None
    return Quote(eid, market, player, side, float(point), book, price, link, observed_at)


def parse_event(event: dict, observed_at: datetime) -> list[Quote]:
    out: list[Quote] = []
    for bm in event.get("bookmakers") or []:
        for m in bm.get("markets") or []:
            for o in m.get("outcomes") or []:
                q = _outcome_quote(event, bm["key"], m["key"], o, observed_at)
                if q is not None:
                    out.append(q)
    return out
```

- [ ] **Step 6: Run the tests**

Run: `python3 -m pytest tests/unit/test_edge_ingest.py -v`
Expected: all PASS. If `test_spreads_...` fails because FanDuel has no spreads in the trimmed fixture, change the book in that test to one that does: `python3 -c "import json;print([b['key'] for b in json.load(open('tests/fixtures/edge/lines.json'))[0]['bookmakers'] if any(m['key']=='spreads' for m in b['markets'])])"`.

- [ ] **Step 7: Commit**

```bash
git add src/edge/client.py src/edge/ingest.py tests/unit/test_edge_ingest.py tests/fixtures/edge
git commit -m "feat(edge): Odds API client with redacted errors and quote parsing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Cadence, staleness and credit guard (pure)

**Files:**
- Create: `backend/src/edge/schedule.py`, `backend/tests/unit/test_edge_schedule.py`

**Interfaces:**
- Consumes: `Quote`, `sides_for`.
- Produces:
  - `base_intervals(minutes_to_tip: float) -> tuple[int, int] | None`, returning (lines_min, props_min).
  - `interval_for(market: str, minutes_to_tip: float, factor: int) -> timedelta | None`
  - `is_due(last_poll: datetime | None, interval: timedelta, now: datetime) -> bool`
  - `is_stale(observed_at: datetime, interval: timedelta | None, now: datetime) -> bool`
  - `fresh(quotes, commence_by_event: dict[str, datetime], now, factor) -> list[Quote]`
  - `guard_factor(cost_last_24h: int, current_factor: int, cap: int, remaining: int | None) -> int`
  - `PollPlan(lines: bool, prop_events: list[str])`
  - `plan_polls(events: list[dict], now, factor) -> PollPlan`. Each event dict has `event_id`, `commence_time`, `last_lines_poll` and `last_props_poll`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/unit/test_edge_schedule.py
from datetime import datetime, timedelta, timezone

from src.edge import schedule as s
from src.edge.quotes import Quote

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def ev(eid, hours_to_tip, last_lines=None, last_props=None):
    return {"event_id": eid, "commence_time": NOW + timedelta(hours=hours_to_tip),
            "last_lines_poll": last_lines, "last_props_poll": last_props}


def test_base_intervals_follow_the_spec_table():
    assert s.base_intervals(30 * 60) == (10, 15)
    assert s.base_intervals(6 * 60) == (3, 5)
    assert s.base_intervals(61) == (3, 5)
    assert s.base_intervals(59) == (2, 2)
    assert s.base_intervals(0) is None          # tipped
    assert s.base_intervals(37 * 60) is None    # too far out


def test_guard_factor_scales_intervals():
    assert s.interval_for("player_points", 120, 1) == timedelta(minutes=5)
    assert s.interval_for("player_points", 120, 4) == timedelta(minutes=20)
    assert s.interval_for("h2h", 120, 1) == timedelta(minutes=3)


def test_is_due():
    assert s.is_due(None, timedelta(minutes=5), NOW)
    assert not s.is_due(NOW - timedelta(minutes=4), timedelta(minutes=5), NOW)
    assert s.is_due(NOW - timedelta(minutes=5), timedelta(minutes=5), NOW)


def test_stale_after_two_intervals_and_after_tip():
    assert not s.is_stale(NOW - timedelta(minutes=9), timedelta(minutes=5), NOW)
    assert s.is_stale(NOW - timedelta(minutes=11), timedelta(minutes=5), NOW)
    assert s.is_stale(NOW, None, NOW)


def test_fresh_filters_by_each_event_and_market():
    commence = {"e1": NOW + timedelta(hours=2)}  # props 5 min, lines 3 min
    old_prop = Quote("e1", "player_points", "A", "over", 20.5, "fanduel", -110, None, NOW - timedelta(minutes=8))
    old_line = Quote("e1", "h2h", "", "home", 0.0, "fanduel", -110, None, NOW - timedelta(minutes=8))
    assert s.fresh([old_prop, old_line], commence, NOW, 1) == [old_prop]


def test_guard_factor():
    cap = 3_500_000
    assert s.guard_factor(30_000, 1, cap, 4_000_000) == 1     # 0.9M/month
    assert s.guard_factor(150_000, 1, cap, 4_000_000) == 2    # 4.5M -> 2.25M
    assert s.guard_factor(150_000, 2, cap, 4_000_000) == 4    # base 300k/day -> 9M -> /4
    assert s.guard_factor(10_000_000, 1, cap, 4_000_000) == 8  # capped at 8
    assert s.guard_factor(1_000, 1, cap, 40_000) == 8          # nearly out of credits


def test_plan_polls():
    events = [ev("near", 2, last_lines=NOW - timedelta(minutes=4), last_props=NOW - timedelta(minutes=1)),
              ev("far", 20, last_lines=NOW, last_props=None),
              ev("done", -1)]
    plan = s.plan_polls(events, NOW, 1)
    assert plan.lines is True                  # "near" lines interval (3m) elapsed
    assert plan.prop_events == ["far"]         # never polled; "near" polled 1m ago
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `python3 -m pytest tests/unit/test_edge_schedule.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement**

```python
# backend/src/edge/schedule.py
"""When to poll, when a quote is stale, and how hard to back off on credits.

Pure functions. Spec §5 (cadence table, credit guard, staleness).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from src.edge.quotes import Quote

FACTORS = (1, 2, 4, 8)
LOW_CREDITS = 50_000


def base_intervals(minutes_to_tip: float) -> tuple[int, int] | None:
    if minutes_to_tip <= 0 or minutes_to_tip > 36 * 60:
        return None
    if minutes_to_tip < 60:
        return (2, 2)
    if minutes_to_tip <= 6 * 60:
        return (3, 5)
    return (10, 15)


def interval_for(market: str, minutes_to_tip: float, factor: int) -> timedelta | None:
    base = base_intervals(minutes_to_tip)
    if base is None:
        return None
    minutes = base[1] if market.startswith("player_") else base[0]
    return timedelta(minutes=minutes * factor)


def is_due(last_poll: datetime | None, interval: timedelta, now: datetime) -> bool:
    return last_poll is None or now - last_poll >= interval


def is_stale(observed_at: datetime, interval: timedelta | None, now: datetime) -> bool:
    return interval is None or now - observed_at > 2 * interval


def _minutes_to(commence: datetime, now: datetime) -> float:
    return (commence - now).total_seconds() / 60


def fresh(quotes: Iterable[Quote], commence_by_event: dict[str, datetime], now: datetime,
          factor: int) -> list[Quote]:
    out = []
    for q in quotes:
        commence = commence_by_event.get(q.event_id)
        if commence is None:
            continue
        if not is_stale(q.observed_at, interval_for(q.market, _minutes_to(commence, now), factor), now):
            out.append(q)
    return out


def guard_factor(cost_last_24h: int, current_factor: int, cap: int, remaining: int | None) -> int:
    """Smallest factor whose projected monthly use fits the cap (max 8).

    cost_last_24h was spent at `current_factor`, so the unthrottled daily cost
    is roughly cost_last_24h * current_factor.
    """
    if remaining is not None and remaining < LOW_CREDITS:
        return FACTORS[-1]
    base_monthly = cost_last_24h * current_factor * 30
    for f in FACTORS:
        if base_monthly / f <= cap:
            return f
    return FACTORS[-1]


@dataclass(frozen=True)
class PollPlan:
    lines: bool
    prop_events: list[str]


def plan_polls(events: list[dict], now: datetime, factor: int) -> PollPlan:
    lines, props = False, []
    for e in events:
        mins = _minutes_to(e["commence_time"], now)
        li = interval_for("h2h", mins, factor)
        pi = interval_for("player_points", mins, factor)
        if li is not None and is_due(e.get("last_lines_poll"), li, now):
            lines = True
        if pi is not None and is_due(e.get("last_props_poll"), pi, now):
            props.append(e["event_id"])
    return PollPlan(lines, props)
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/unit/test_edge_schedule.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/edge/schedule.py tests/unit/test_edge_schedule.py
git commit -m "feat(edge): polling cadence, staleness and credit guard

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Storage layer and opportunity log

**Files:**
- Create:
  - `backend/src/edge/store.py`
  - `backend/src/edge/opportunities.py`
  - `backend/tests/unit/test_edge_store.py`

**Interfaces:**
- Consumes: `Quote`, `Priced`, `engine.prob_to_american`, `get_sync_database_url`, and the table columns from Task 4.
- Produces:
  - **Connection:** `store.connect() -> psycopg2 connection`.
  - **Change detection:** `store.changed_quotes(existing: dict[tuple, int], quotes: list[Quote]) -> list[Quote]` (pure).
  - **Writes:**
    - `store.upsert_events(conn, rows: list[dict]) -> int`
    - `store.write_quotes(conn, quotes: list[Quote]) -> int` (returns the number of history rows)
    - `store.mark_polled(conn, event_ids: list[str], field: str, now: datetime) -> None`, where `field` is `last_lines_poll` or `last_props_poll`.
    - `store.log_api_call(conn, endpoint: str, cost: int, remaining: int | None, now: datetime) -> None`
    - `store.set_state(conn, key: str, value: str, now: datetime) -> None`
  - **Reads:**
    - `store.upcoming_events(conn, now, hours=36) -> list[dict]`
    - `store.load_quotes(conn, event_ids: list[str], market: str | None = None) -> list[Quote]`
    - `store.cost_last_24h(conn, now) -> int`
    - `store.get_state(conn) -> dict[str, str]`
  - **Opportunity log:**
    - `opportunities.opportunity_rows(priced: list[Priced], now: datetime) -> list[tuple]` (pure)
    - `opportunities.record(conn, priced: list[Priced], now: datetime) -> int`

- [ ] **Step 1: Write the failing tests (pure parts)**

```python
# backend/tests/unit/test_edge_store.py
from datetime import datetime, timezone

from src.edge import opportunities
from src.edge.engine import Fair, Priced
from src.edge.quotes import Quote
from src.edge.store import changed_quotes

NOW = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)


def q(book="fanduel", price=-110, line=27.5, side="over"):
    return Quote("e1", "player_points", "Jayson Tatum", side, line, book, price, "https://x", NOW)


def test_new_keys_and_price_changes_are_history_but_repeats_are_not():
    existing = {q().key: -110, q(book="draftkings").key: -115}
    new = [q(), q(book="draftkings", price=-120), q(book="betmgm")]
    changed = changed_quotes(existing, new)
    assert [c.book for c in changed] == ["draftkings", "betmgm"]


def test_a_new_line_is_a_new_key():
    existing = {q().key: -110}
    assert changed_quotes(existing, [q(line=28.5)]) == [q(line=28.5)]


def test_opportunity_rows_only_for_ev():
    hit = Priced(q(book="betmgm", price=120), Fair(0.5, 4, True), 0.10, True, False)
    miss = Priced(q(), Fair(0.5, 4, True), -0.045, False, False)
    rows = opportunities.opportunity_rows([hit, miss], NOW)
    assert len(rows) == 1
    r = rows[0]
    assert r[:8] == ("e1", "player_points", "Jayson Tatum", "over", 27.5, "betmgm", 120, "https://x")
    assert r[8] == NOW and r[9] == 0.5 and r[10] == 0.10 and r[11] == 4
    assert r[12:] == (NOW, 0.10, "power", False)  # last_seen_at, max_ev, devig_method, closing_done
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `python3 -m pytest tests/unit/test_edge_store.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `store.py`**

```python
# backend/src/edge/store.py
"""psycopg2 reads/writes for Edge. Thin on purpose: logic lives in pure modules."""
from __future__ import annotations

from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from src.config import get_sync_database_url
from src.edge.quotes import Quote

_QCOLS = "event_id, market, player, side, line, book, price, link, observed_at"


def connect():
    return psycopg2.connect(get_sync_database_url())


def _quote(r) -> Quote:
    return Quote(r[0], r[1], r[2], r[3], float(r[4]), r[5], int(r[6]), r[7], r[8])


def _qtuple(q: Quote) -> tuple:
    return (q.event_id, q.market, q.player, q.side, q.line, q.book, q.price, q.link, q.observed_at)


def changed_quotes(existing: dict[tuple, int], quotes: list[Quote]) -> list[Quote]:
    return [q for q in quotes if existing.get(q.key) != q.price]


def upsert_events(conn, rows: list[dict]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO edge_events (event_id, sport, home_team, away_team, commence_time, status, completed)
            VALUES %s
            ON CONFLICT (event_id) DO UPDATE SET commence_time = EXCLUDED.commence_time,
                home_team = EXCLUDED.home_team, away_team = EXCLUDED.away_team
        """, [(r["event_id"], r["sport"], r["home_team"], r["away_team"], r["commence_time"],
               "scheduled", False) for r in rows])
    conn.commit()
    return len(rows)


def upcoming_events(conn, now: datetime, hours: int = 36) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT * FROM edge_events WHERE commence_time > %s AND commence_time <= %s
                       ORDER BY commence_time""", (now, now + timedelta(hours=hours)))
        return [dict(r) for r in cur.fetchall()]


def write_quotes(conn, quotes: list[Quote]) -> int:
    if not quotes:
        return 0
    event_ids = sorted({q.event_id for q in quotes})
    with conn.cursor() as cur:
        cur.execute("""SELECT event_id, market, player, line, side, book, price FROM edge_quotes
                       WHERE event_id = ANY(%s)""", (event_ids,))
        existing = {(r[0], r[1], r[2], float(r[3]), r[4], r[5]): r[6] for r in cur.fetchall()}
        changed = changed_quotes(existing, quotes)
        execute_values(cur, f"""
            INSERT INTO edge_quotes ({_QCOLS}) VALUES %s
            ON CONFLICT (event_id, market, player, side, line, book) DO UPDATE SET
                price = EXCLUDED.price, link = EXCLUDED.link, observed_at = EXCLUDED.observed_at
        """, [_qtuple(q) for q in quotes])
        if changed:
            execute_values(cur, f"INSERT INTO edge_quote_history ({_QCOLS}) VALUES %s",
                           [_qtuple(q) for q in changed])
    conn.commit()
    return len(changed)


def mark_polled(conn, event_ids: list[str], field: str, now: datetime) -> None:
    if field not in ("last_lines_poll", "last_props_poll") or not event_ids:
        return
    with conn.cursor() as cur:
        cur.execute(f"UPDATE edge_events SET {field} = %s WHERE event_id = ANY(%s)", (now, event_ids))
    conn.commit()


def load_quotes(conn, event_ids: list[str], market: str | None = None) -> list[Quote]:
    sql = f"SELECT {_QCOLS} FROM edge_quotes WHERE event_id = ANY(%s)"
    params: list = [event_ids]
    if market:
        sql += " AND market = %s"
        params.append(market)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [_quote(r) for r in cur.fetchall()]


def log_api_call(conn, endpoint: str, cost: int, remaining: int | None, now: datetime) -> None:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO edge_api_calls (called_at, endpoint, cost, remaining) VALUES (%s,%s,%s,%s)",
                    (now, endpoint, cost, remaining))
    conn.commit()


def cost_last_24h(conn, now: datetime) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(SUM(cost), 0) FROM edge_api_calls WHERE called_at > %s",
                    (now - timedelta(hours=24),))
        return int(cur.fetchone()[0])


def set_state(conn, key: str, value: str, now: datetime) -> None:
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_worker_state (key, value, updated_at) VALUES (%s, %s, %s)
                       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at""",
                    (key, value, now))
    conn.commit()


def get_state(conn) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT key, value FROM edge_worker_state")
        return {k: v for k, v in cur.fetchall()}
```

- [ ] **Step 4: Implement `opportunities.py`**

```python
# backend/src/edge/opportunities.py
"""Log every +EV price the first time it is seen (spec §6.4).

A new price at the same book/line is a new opportunity; a repeat sighting only
bumps last_seen_at and max_ev. Logging at first sight — not at the best moment
in hindsight — is what keeps the published record honest.
"""
from __future__ import annotations

from datetime import datetime

from psycopg2.extras import execute_values

from src.edge.config import DEVIG_METHOD
from src.edge.engine import Priced


def opportunity_rows(priced: list[Priced], now: datetime) -> list[tuple]:
    rows = []
    for p in priced:
        if not p.is_ev:
            continue
        q = p.quote
        ev = round(p.ev, 5)
        rows.append((q.event_id, q.market, q.player, q.side, q.line, q.book, q.price, q.link,
                     now, round(p.fair.prob, 5), ev, p.fair.n_books, now, ev, DEVIG_METHOD, False))
    return rows


def record(conn, priced: list[Priced], now: datetime) -> int:
    rows = opportunity_rows(priced, now)
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO edge_opportunities (event_id, market, player, side, line, book, price, link,
                first_seen_at, fair_prob, ev, n_books, last_seen_at, max_ev, devig_method, closing_done)
            VALUES %s
            ON CONFLICT (event_id, market, player, side, line, book, price) DO UPDATE SET
                last_seen_at = EXCLUDED.last_seen_at,
                max_ev = GREATEST(edge_opportunities.max_ev, EXCLUDED.ev)
        """, rows)
    conn.commit()
    return len(rows)
```

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/unit/test_edge_store.py -v`
Expected: all PASS. The SQL in `store.py` and `opportunities.py` runs for the first time against the real DB in Task 15, Step 4.

- [ ] **Step 6: Commit**

```bash
git add src/edge/store.py src/edge/opportunities.py tests/unit/test_edge_store.py
git commit -m "feat(edge): quote storage with history-on-change and the opportunity log

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Worker loop

**Files:**
- Create: `backend/src/edge/worker.py`
- Modify: `backend/tests/unit/test_edge_schedule.py` (add the backoff test)

**Interfaces:**
- Consumes:
  - `EdgeOddsClient`, `OddsApiError`, `Usage`
  - `parse_event`, `event_row`
  - `store.*`, `schedule.*`
  - `engine.price_quotes`
  - `opportunities.record`
  - `players.resolve_new_players` and `players.ingest_box_scores` (Task 10)
  - `grading.run_grading` and `grading.update_scores` (Task 11)
- Produces:
  - `Backoff` with `failure(status: int, now) -> datetime` and `success() -> None`.
  - `async tick(conn, client, bdl, now) -> dict`
  - `async ensure_backfill(conn, bdl, now) -> None`: a one-time load of the previous season's box scores, gated by `edge_worker_state['backfill_<season>']`.
  - `main()`, run as `python -m src.edge.worker`.

Worker cadence:
- Ticks every 30 s.
- Events refresh every 60 min (the call is free).
- Grading runs every 10 min.
- Scores are fetched every 30 min while a game is in progress (tipped less than 4 h ago and not completed).
- Box scores are ingested once a day at or after 09:00 UTC, for the previous ET date.
- The heartbeat is written every tick.

- [ ] **Step 1: Write the failing backoff test (append to `test_edge_schedule.py`)**

```python
from src.edge.worker import Backoff


def test_backoff_rules():
    b = Backoff()
    assert b.failure(401, NOW) == NOW + timedelta(minutes=15)
    assert b.failure(429, NOW) == NOW + timedelta(seconds=30)
    assert b.failure(503, NOW) == NOW + timedelta(seconds=60)
    for _ in range(10):
        until = b.failure(500, NOW)
    assert until == NOW + timedelta(minutes=10)
    b.success()
    assert b.failure(429, NOW) == NOW + timedelta(seconds=30)
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `python3 -m pytest tests/unit/test_edge_schedule.py::test_backoff_rules -v`
Expected: FAIL (`ModuleNotFoundError: src.edge.worker`)

- [ ] **Step 3: Implement `worker.py`**

```python
# backend/src/edge/worker.py
"""Edge ingest / pricing / grading loop. Runs as its own Railway service:

    python -m src.edge.worker

It never stops silently: every failure lands in edge_worker_state.last_error
(redacted) and the heartbeat keeps ticking so /edge/status and the frontend
banner can tell "delayed" from "dead".
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta, timezone

from src.config import settings
from src.edge import grading, opportunities, players, schedule, store
from src.edge.client import EdgeOddsClient, OddsApiError, Usage
from src.edge.engine import price_quotes
from src.edge.ingest import event_row, parse_event
from src.utils.redact import redact
from src.utils.season import current_nba_season

log = logging.getLogger("edge.worker")
TICK_SECONDS = 30


class Backoff:
    def __init__(self) -> None:
        self.seconds = 0

    def failure(self, status: int, now: datetime) -> datetime:
        if status == 401:
            return now + timedelta(minutes=15)
        self.seconds = 30 if self.seconds == 0 else min(self.seconds * 2, 600)
        return now + timedelta(seconds=self.seconds)

    def success(self) -> None:
        self.seconds = 0


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse(v: str | None) -> datetime | None:
    return datetime.fromisoformat(v) if v else None


def _record_usage(conn, endpoint: str, usage: Usage, now: datetime) -> None:
    store.log_api_call(conn, endpoint, usage.last or 0, usage.remaining, now)
    if usage.remaining is not None:
        store.set_state(conn, "credits_remaining", str(usage.remaining), now)
    if usage.used is not None:
        store.set_state(conn, "credits_used", str(usage.used), now)


def _reprice(conn, event_ids: list[str], events: list[dict], now: datetime, factor: int) -> int:
    if not event_ids:
        return 0
    commence = {e["event_id"]: e["commence_time"] for e in events}
    quotes = schedule.fresh(store.load_quotes(conn, event_ids), commence, now, factor)
    priced = price_quotes(quotes, settings.edge_min_ev, settings.edge_verify_ev)
    return opportunities.record(conn, priced, now)


async def tick(conn, client: EdgeOddsClient, bdl, now: datetime) -> dict:
    state = store.get_state(conn)
    out: dict = {}

    # Events: free call, hourly.
    last_ev = _parse(state.get("last_events_refresh"))
    if last_ev is None or now - last_ev >= timedelta(minutes=60):
        events_json, _ = await client.get_events()
        store.upsert_events(conn, [event_row(e) for e in events_json])
        store.set_state(conn, "last_events_refresh", _iso(now), now)

    factor = schedule.guard_factor(store.cost_last_24h(conn, now), int(state.get("guard_factor", "1")),
                                   settings.edge_credit_cap, int(state["credits_remaining"])
                                   if state.get("credits_remaining") else None)
    if factor != int(state.get("guard_factor", "1")):
        log.warning("edge credit guard factor -> %s", factor)
    store.set_state(conn, "guard_factor", str(factor), now)

    events = store.upcoming_events(conn, now)
    plan = schedule.plan_polls(events, now, factor)
    touched: list[str] = []

    if plan.lines:
        data, usage = await client.get_game_odds()
        _record_usage(conn, "odds", usage, now)
        upcoming = {e["event_id"] for e in events}
        quotes = [q for ev in data if ev["id"] in upcoming for q in parse_event(ev, now)]
        out["line_history"] = store.write_quotes(conn, quotes)
        store.mark_polled(conn, sorted(upcoming), "last_lines_poll", now)
        touched += sorted(upcoming)

    by_id = {e["event_id"]: e for e in events}
    for eid in plan.prop_events:
        data, usage = await client.get_event_odds(eid)
        _record_usage(conn, "event_odds", usage, now)
        quotes = parse_event(data, now)
        store.write_quotes(conn, quotes)
        store.mark_polled(conn, [eid], "last_props_poll", now)
        await players.resolve_new_players(conn, bdl, quotes, by_id[eid], now)
        touched.append(eid)

    out["opportunities"] = _reprice(conn, sorted(set(touched)), events, now, factor)

    # Scores while games are live; grading every 10 minutes.
    last_scores = _parse(state.get("last_scores_poll"))
    if grading.games_in_progress(conn, now) and (last_scores is None or now - last_scores >= timedelta(minutes=30)):
        data, usage = await client.get_scores(days_from=1)
        _record_usage(conn, "scores", usage, now)
        grading.update_scores(conn, data)
        store.set_state(conn, "last_scores_poll", _iso(now), now)

    last_grade = _parse(state.get("last_grading"))
    if last_grade is None or now - last_grade >= timedelta(minutes=10):
        out["graded"] = grading.run_grading(conn, now)
        store.set_state(conn, "last_grading", _iso(now), now)

    # Box scores once a day after 09:00 UTC, for yesterday (ET). A BallDontLie
    # failure is recorded under its own key (the loop clears last_error on every
    # good tick) and the day is marked done so we retry tomorrow, not every 30s.
    yday = players.et_date(now) - timedelta(days=1)
    if now.hour >= 9 and state.get("last_box_scores_date") != yday.isoformat():
        try:
            out["box_rows"] = await players.ingest_box_scores(conn, bdl, yday)
            store.set_state(conn, "bdl_error", "", now)
        except players.BdlError as e:
            store.set_state(conn, "bdl_error", f"balldontlie {e.status}: {e}"[:300], now)
        store.set_state(conn, "last_box_scores_date", yday.isoformat(), now)

    return out


async def ensure_backfill(conn, bdl, now: datetime) -> None:
    """One-time load of last season's box scores so hit rates work on night one.

    Runs inside the worker because only Railway can reach the private DB host.
    """
    prev = current_nba_season(now.date()) - 1
    key = f"backfill_{prev}"
    if store.get_state(conn).get(key) == "done":
        return
    try:
        total = await players.backfill(conn, bdl, date(prev, 10, 15), date(prev + 1, 6, 30))
        store.set_state(conn, key, "done", now)
        log.info("edge backfill %s: %s rows", prev, total)
    except players.BdlError as e:
        store.set_state(conn, "bdl_error", f"backfill: balldontlie {e.status}: {e}"[:300], now)


async def _run() -> None:
    client = EdgeOddsClient()
    bdl = players.BdlClient()
    backoff = Backoff()
    blocked_until: datetime | None = None
    conn = store.connect()
    await ensure_backfill(conn, bdl, datetime.now(timezone.utc))
    while True:
        now = datetime.now(timezone.utc)
        try:
            if conn.closed:
                conn = store.connect()
            store.set_state(conn, "heartbeat_at", _iso(now), now)
            if blocked_until is None or now >= blocked_until:
                result = await tick(conn, client, bdl, now)
                backoff.success()
                blocked_until = None
                store.set_state(conn, "last_error", "", now)
                log.info("edge tick %s", result)
        except OddsApiError as e:
            blocked_until = backoff.failure(e.status, now)
            log.error("edge odds api error %s: %s", e.status, e)
            store.set_state(conn, "last_error", f"odds_api {e.status}: {e}"[:500], now)
        except Exception as e:  # noqa: BLE001 - the loop must survive anything
            log.exception("edge tick failed")
            try:
                conn.rollback()
                store.set_state(conn, "last_error", redact(f"{type(e).__name__}: {e}")[:500], now)
            except Exception:  # noqa: BLE001
                conn = store.connect()
        await asyncio.sleep(TICK_SECONDS)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    while True:
        try:
            asyncio.run(_run())
        except Exception:  # noqa: BLE001
            log.exception("edge worker crashed; restarting in 30s")
            time.sleep(30)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test**

The `worker.py` imports `players` and `grading`, which don't exist until Tasks 10 and 11. Create stubs so the import works now; Tasks 10 and 11 replace them.

```bash
cat > src/edge/players.py <<'EOF'
"""Replaced in Task 10."""
class BdlError(Exception):
    status = 0
EOF
cat > src/edge/grading.py <<'EOF'
"""Replaced in Task 11."""
EOF
python3 -m pytest tests/unit/test_edge_schedule.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/edge/worker.py src/edge/players.py src/edge/grading.py tests/unit/test_edge_schedule.py
git commit -m "feat(edge): worker loop with credit guard, backoff and heartbeat

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Players: name matching, box scores, hit rates

**Files:**
- Create: `backend/src/edge/teams.py`, `backend/tests/unit/test_edge_players.py`
- Replace: `backend/src/edge/players.py`

**Interfaces:**
- Consumes: `Quote`, `store.connect`, `current_nba_season`, `settings.balldontlie_api_key`.
- Produces:
  - **Teams:** `teams.ABBR_BY_NAME: dict[str, str]` (30 teams, Odds API full name → BallDontLie abbreviation).
  - **Pure helpers:**
    - `players.normalize_name(name: str) -> str`
    - `players.Match(bdl_player_id: int, team: str, method: str, confidence: float)`
    - `players.match_player(odds_name: str, candidates: list[dict], event_teams: set[str]) -> Match | None`. Each candidate dict has `id`, `first_name`, `last_name` and `team_abbr`.
    - `players.parse_minutes(v) -> float`
    - `players.box_row(stat: dict, abbr_by_team_id: dict[int, str]) -> dict | None`
    - `players.stat_for(market: str, g: dict) -> float`
    - `players.hit_rates(games: list[dict], market: str, line: float, opponent: str | None, is_home: bool | None, season: int, min_minutes: float = 0.0) -> dict`
    - `players.et_date(dt: datetime) -> date`
  - **I/O:**
    - `players.BdlClient` with async `search(name) -> list[dict]`, `teams() -> dict[int, str]` and `stats_for_date(d: date) -> list[dict]`. It raises `BdlError(status)`.
    - `async players.resolve_new_players(conn, bdl, quotes, event: dict, now) -> int`
    - `async players.ingest_box_scores(conn, bdl, d: date) -> int`
    - `async players.backfill(conn, bdl, start: date, end: date) -> int`
    - `players.load_games(conn, bdl_player_id: int, limit: int = 82) -> list[dict]`
  - **CLI:** `python -m src.edge.players backfill 2025-10-21 2026-06-20`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/unit/test_edge_players.py
from datetime import date, datetime, timezone

import pytest

from src.edge import players as p

CANDS = [
    {"id": 246, "first_name": "Nikola", "last_name": "Jokić", "team_abbr": "DEN"},
    {"id": 3547269, "first_name": "Jaren", "last_name": "Jackson Jr.", "team_abbr": "MEM"},
    {"id": 1, "first_name": "Jalen", "last_name": "Williams", "team_abbr": "OKC"},
    {"id": 2, "first_name": "Jalen", "last_name": "Williams", "team_abbr": "DEN"},  # different player
]


def test_normalize_name_strips_accents_suffixes_and_punctuation():
    assert p.normalize_name("Nikola Jokić") == "nikola jokic"
    assert p.normalize_name("Jaren Jackson Jr.") == "jaren jackson"
    assert p.normalize_name("De'Aaron Fox") == "deaaron fox"
    assert p.normalize_name("Shai Gilgeous-Alexander") == "shai gilgeous alexander"


def test_exact_and_normalized_matches():
    m = p.match_player("Nikola Jokic", CANDS, {"DEN", "LAL"})
    assert m and m.bdl_player_id == 246 and m.method == "normalized"
    m = p.match_player("Jaren Jackson Jr", CANDS, {"MEM", "BOS"})
    assert m and m.bdl_player_id == 3547269


def test_same_name_resolved_by_event_team():
    assert p.match_player("Jalen Williams", CANDS, {"OKC", "SAS"}).bdl_player_id == 1
    assert p.match_player("Jalen Williams", CANDS, {"DEN", "UTA"}).bdl_player_id == 2


def test_no_match_when_team_disagrees_or_too_far():
    assert p.match_player("Nikola Jokic", CANDS, {"BOS", "DET"}) is None
    assert p.match_player("Nikolai Jokinen", CANDS, {"DEN", "LAL"}) is None


def test_parse_minutes():
    assert p.parse_minutes("34:30") == pytest.approx(34.5)
    assert p.parse_minutes("28") == 28.0
    assert p.parse_minutes(None) == 0.0 and p.parse_minutes("") == 0.0


def test_box_row():
    stat = {"player": {"id": 434}, "team": {"id": 2, "abbreviation": "BOS"},
            "game": {"id": 99, "date": "2026-04-10", "season": 2025, "home_team_id": 2, "visitor_team_id": 9},
            "min": "36", "pts": 31, "reb": 8, "ast": 6, "fg3m": 4, "stl": 1, "blk": 0, "turnover": 3}
    row = p.box_row(stat, {2: "BOS", 9: "DET"})
    assert row == {"bdl_player_id": 434, "bdl_game_id": 99, "season": 2025, "game_date": date(2026, 4, 10),
                   "team": "BOS", "opponent": "DET", "is_home": True, "minutes": 36.0,
                   "pts": 31, "reb": 8, "ast": 6, "fg3m": 4, "stl": 1, "blk": 0, "tov": 3}


def g(d, pts, reb=5, ast=5, mins=30.0, opp="DET", home=True, season=2026):
    return {"game_date": d, "pts": pts, "reb": reb, "ast": ast, "fg3m": 2, "minutes": mins,
            "opponent": opp, "is_home": home, "season": season}


def test_stat_for_combos():
    row = g(date(2026, 11, 1), 20, reb=7, ast=3)
    assert p.stat_for("player_points_rebounds_assists", row) == 30
    assert p.stat_for("player_rebounds_assists", row) == 10
    assert p.stat_for("player_threes", row) == 2


def test_hit_rates_windows_pushes_and_dnp():
    games = [g(date(2026, 11, 10 - i), pts) for i, pts in enumerate([30, 25, 20, 28, 10, 40])]
    games.append(g(date(2026, 11, 3), 0, mins=0.0))  # DNP: excluded
    r = p.hit_rates(games, "player_points", 25.0, opponent="DET", is_home=True, season=2026)
    assert r["l5"] == {"hits": 2, "misses": 2, "pushes": 1, "games": 5, "rate": 0.5}
    assert r["l10"]["games"] == 6
    assert r["vs_opponent"]["games"] == 6 and r["home_away"]["games"] == 6


def test_min_minutes_filter():
    games = [g(date(2026, 11, 2), 30, mins=12.0), g(date(2026, 11, 1), 30, mins=34.0)]
    assert p.hit_rates(games, "player_points", 25.0, None, None, 2026, min_minutes=20)["l5"]["games"] == 1


def test_et_date():
    assert p.et_date(datetime(2026, 10, 21, 2, 30, tzinfo=timezone.utc)) == date(2026, 10, 20)
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `python3 -m pytest tests/unit/test_edge_players.py -v`
Expected: FAIL (`AttributeError: module 'src.edge.players' has no attribute 'normalize_name'`)

- [ ] **Step 3: Implement `teams.py`**

```python
# backend/src/edge/teams.py
"""The Odds API full team names -> BallDontLie abbreviations."""
ABBR_BY_NAME = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU",
    "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}
```

- [ ] **Step 4: Implement `players.py`**

```python
# backend/src/edge/players.py
"""Player identity, BallDontLie box scores and hit rates (spec §7).

A wrong match is worse than no match — it shows another player's history next
to a price — so fuzzy matching needs >= 0.92 similarity AND a team that is
playing in this event; anything else stays `unmatched` for manual review.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
from psycopg2.extras import RealDictCursor, execute_values

from src.config import settings
from src.edge.quotes import Quote
from src.edge.teams import ABBR_BY_NAME
from src.utils.redact import redact

ET = ZoneInfo("America/New_York")
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
FUZZY_MIN = 0.92
STAT_PARTS = {
    "player_points": ("pts",), "player_rebounds": ("reb",), "player_assists": ("ast",),
    "player_threes": ("fg3m",), "player_points_rebounds_assists": ("pts", "reb", "ast"),
    "player_points_rebounds": ("pts", "reb"), "player_points_assists": ("pts", "ast"),
    "player_rebounds_assists": ("reb", "ast"),
}


def et_date(dt: datetime) -> date:
    return dt.astimezone(ET).date()


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(t for t in s.split() if t not in SUFFIXES)


@dataclass(frozen=True)
class Match:
    bdl_player_id: int
    team: str
    method: str
    confidence: float


def match_player(odds_name: str, candidates: list[dict], event_teams: set[str]) -> Match | None:
    playing = [c for c in candidates if c.get("team_abbr") in event_teams]
    for c in playing:
        if f"{c['first_name']} {c['last_name']}" == odds_name:
            return Match(c["id"], c["team_abbr"], "exact", 1.0)
    target = normalize_name(odds_name)
    for c in playing:
        if normalize_name(f"{c['first_name']} {c['last_name']}") == target:
            return Match(c["id"], c["team_abbr"], "normalized", 1.0)
    best, score = None, 0.0
    for c in playing:
        r = difflib.SequenceMatcher(None, target, normalize_name(f"{c['first_name']} {c['last_name']}")).ratio()
        if r > score:
            best, score = c, r
    if best is not None and score >= FUZZY_MIN:
        return Match(best["id"], best["team_abbr"], "fuzzy", round(score, 3))
    return None


def parse_minutes(v) -> float:
    if not v:
        return 0.0
    s = str(v)
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m or 0) + int(sec or 0) / 60
    try:
        return float(s)
    except ValueError:
        return 0.0


def box_row(stat: dict, abbr_by_team_id: dict[int, str]) -> dict | None:
    try:
        game, team_id = stat["game"], stat["team"]["id"]
        is_home = game["home_team_id"] == team_id
        opp_id = game["visitor_team_id"] if is_home else game["home_team_id"]
        return {
            "bdl_player_id": stat["player"]["id"], "bdl_game_id": game["id"], "season": game["season"],
            "game_date": date.fromisoformat(game["date"][:10]),
            "team": stat["team"].get("abbreviation") or abbr_by_team_id[team_id],
            "opponent": abbr_by_team_id[opp_id], "is_home": is_home,
            "minutes": parse_minutes(stat.get("min")),
            "pts": stat.get("pts") or 0, "reb": stat.get("reb") or 0, "ast": stat.get("ast") or 0,
            "fg3m": stat.get("fg3m") or 0, "stl": stat.get("stl") or 0, "blk": stat.get("blk") or 0,
            "tov": stat.get("turnover") or 0,
        }
    except (KeyError, TypeError, ValueError):
        return None


def stat_for(market: str, g: dict) -> float:
    return float(sum(g[k] or 0 for k in STAT_PARTS[market]))


def _rate(rows: list[dict], market: str, line: float) -> dict:
    hits = sum(1 for r in rows if stat_for(market, r) > line)
    misses = sum(1 for r in rows if stat_for(market, r) < line)
    decided = hits + misses
    return {"hits": hits, "misses": misses, "pushes": len(rows) - decided, "games": len(rows),
            "rate": round(hits / decided, 3) if decided else None}


def hit_rates(games: list[dict], market: str, line: float, opponent: str | None, is_home: bool | None,
              season: int, min_minutes: float = 0.0) -> dict:
    played = sorted((x for x in games if (x["minutes"] or 0) > 0 and (x["minutes"] or 0) >= min_minutes),
                    key=lambda x: x["game_date"], reverse=True)
    return {
        "l5": _rate(played[:5], market, line),
        "l10": _rate(played[:10], market, line),
        "l20": _rate(played[:20], market, line),
        "season": _rate([x for x in played if x["season"] == season], market, line),
        "vs_opponent": _rate([x for x in played if opponent and x["opponent"] == opponent], market, line),
        "home_away": _rate([x for x in played if is_home is not None and x["is_home"] == is_home], market, line),
    }


class BdlError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(redact(message))


class BdlClient:
    BASE = "https://api.balldontlie.io/v1"

    def __init__(self, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.api_key = api_key if api_key is not None else settings.balldontlie_api_key
        self.transport = transport
        self._teams: dict[int, str] | None = None

    async def _get(self, path: str, params: dict) -> dict:
        async with httpx.AsyncClient(transport=self.transport, timeout=30.0) as c:
            r = await c.get(f"{self.BASE}/{path}", params=params, headers={"Authorization": self.api_key})
        if r.status_code != 200:
            raise BdlError(r.status_code, f"BallDontLie {r.status_code} on {path}")
        return r.json()

    async def search(self, name: str) -> list[dict]:
        last = normalize_name(name).split()[-1]
        data = await self._get("players", {"search": last, "per_page": 100})
        return [{"id": x["id"], "first_name": x["first_name"], "last_name": x["last_name"],
                 "team_abbr": (x.get("team") or {}).get("abbreviation")} for x in data["data"]]

    async def teams(self) -> dict[int, str]:
        if self._teams is None:
            data = await self._get("teams", {"per_page": 100})
            self._teams = {t["id"]: t["abbreviation"] for t in data["data"]}
        return self._teams

    async def stats_for_date(self, d: date) -> list[dict]:
        out, cursor = [], None
        while True:
            params = {"dates[]": d.isoformat(), "per_page": 100}
            if cursor:
                params["cursor"] = cursor
            data = await self._get("stats", params)
            out += data["data"]
            cursor = (data.get("meta") or {}).get("next_cursor")
            if not cursor:
                return out


async def resolve_new_players(conn, bdl: BdlClient, quotes: list[Quote], event: dict, now: datetime) -> int:
    names = sorted({q.player for q in quotes if q.player})
    if not names:
        return 0
    with conn.cursor() as cur:
        cur.execute("SELECT odds_name FROM edge_players WHERE odds_name = ANY(%s)", (names,))
        known = {r[0] for r in cur.fetchall()}
    teams = {ABBR_BY_NAME.get(event["home_team"]), ABBR_BY_NAME.get(event["away_team"])} - {None}
    rows = []
    for name in (n for n in names if n not in known):
        m = match_player(name, await bdl.search(name), teams)
        rows.append((name, m.bdl_player_id if m else None, m.team if m else None,
                     m.method if m else "unmatched", m.confidence if m else 0.0, now))
    if rows:
        with conn.cursor() as cur:
            execute_values(cur, """INSERT INTO edge_players (odds_name, bdl_player_id, team, match_method,
                                   confidence, updated_at) VALUES %s ON CONFLICT (odds_name) DO NOTHING""", rows)
        conn.commit()
    return len(rows)


async def ingest_box_scores(conn, bdl: BdlClient, d: date) -> int:
    abbr = await bdl.teams()
    rows = [r for r in (box_row(s, abbr) for s in await bdl.stats_for_date(d)) if r]
    if not rows:
        return 0
    cols = list(rows[0].keys())
    with conn.cursor() as cur:
        execute_values(cur, f"""INSERT INTO edge_player_games ({", ".join(cols)}) VALUES %s
            ON CONFLICT (bdl_player_id, game_date) DO UPDATE SET
            {", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ("bdl_player_id", "game_date"))}""",
                       [tuple(r[c] for c in cols) for r in rows])
    conn.commit()
    return len(rows)


def load_games(conn, bdl_player_id: int, limit: int = 82) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT * FROM edge_player_games WHERE bdl_player_id = %s
                       ORDER BY game_date DESC LIMIT %s""", (bdl_player_id, limit))
        return [dict(r) for r in cur.fetchall()]


async def backfill(conn, bdl: BdlClient, start: date, end: date) -> int:
    d, total = start, 0
    while d <= end:
        n = await ingest_box_scores(conn, bdl, d)
        total += n
        print(f"{d} {n} rows (total {total})", flush=True)
        d += timedelta(days=1)
        await asyncio.sleep(1.1)  # stays under 60 req/min
    return total


if __name__ == "__main__":
    from src.edge import store
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backfill"])
    ap.add_argument("start", type=date.fromisoformat)
    ap.add_argument("end", type=date.fromisoformat)
    a = ap.parse_args()
    asyncio.run(backfill(store.connect(), BdlClient(), a.start, a.end))
```

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/unit/test_edge_players.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/edge/teams.py src/edge/players.py tests/unit/test_edge_players.py
git commit -m "feat(edge): player matching, BallDontLie box scores and hit rates

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Closing line, CLV and results

**Files:**
- Replace: `backend/src/edge/grading.py`
- Create: `backend/tests/unit/test_edge_grading.py`

**Interfaces:**
- Consumes:
  - `engine.group`, `fair_prob`, `latest_before`, `clv`, `american_to_decimal`
  - `store.load_quotes`
  - `players.stat_for`, `et_date`
  - `teams.ABBR_BY_NAME`
  - `config.CLOSING_WINDOW_MINUTES`
- Produces:
  - **Pure:**
    - `grade_over_under(actual: float, line: float, side: str) -> str`
    - `grade_game(market: str, side: str, line: float, home: int, away: int) -> str`
    - `profit_units(result: str, price: int) -> float`
    - `closing_for(quotes: list[Quote], opp: dict, tip: datetime) -> tuple[float | None, float | None, str | None]`, returning (closing_prob, clv, note).

**Why the close comes from `edge_quotes`, not `edge_quote_history`:**
- History only stores a row when a price *changes*. A book that held one price through the final hour would have no row inside the closing window, and would wrongly drop out of the close.
- `edge_quotes` keeps every key's last observation. Polling stops at tip (`schedule.base_intervals` returns None), so after tip the table is exactly the final pre-tip snapshot.
- A pulled line keeps an old `observed_at`, so `latest_before` with the 30-minute window excludes it.
    - `prop_result(opp: dict, player_game: dict | None, team_has_box: bool) -> str | None`
  - **I/O:**
    - `games_in_progress(conn, now) -> bool`
    - `update_scores(conn, scores_json: list[dict]) -> int`
    - `run_grading(conn, now) -> dict`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/unit/test_edge_grading.py
from datetime import datetime, timedelta, timezone

import pytest

from src.edge import grading
from src.edge.quotes import Quote

TIP = datetime(2026, 10, 20, 23, 0, tzinfo=timezone.utc)


def test_over_under():
    assert grading.grade_over_under(26, 25.5, "over") == "win"
    assert grading.grade_over_under(26, 25.5, "under") == "loss"
    assert grading.grade_over_under(25, 25.0, "over") == "push"


def test_spread_home_perspective_line():
    # away +5.5 is stored as line -5.5 (home -5.5). Home wins by 3 -> away covers.
    assert grading.grade_game("spreads", "away", -5.5, 103, 100) == "win"
    assert grading.grade_game("spreads", "home", -5.5, 103, 100) == "loss"
    assert grading.grade_game("spreads", "home", -3.0, 103, 100) == "push"


def test_h2h_and_totals():
    assert grading.grade_game("h2h", "home", 0.0, 110, 100) == "win"
    assert grading.grade_game("h2h", "away", 0.0, 110, 100) == "loss"
    assert grading.grade_game("totals", "over", 215.5, 110, 106) == "win"


def test_profit_units():
    assert grading.profit_units("win", 150) == pytest.approx(1.5)
    assert grading.profit_units("win", -200) == pytest.approx(0.5)
    assert grading.profit_units("loss", 150) == -1.0
    assert grading.profit_units("void", 150) == 0.0


def _hq(book, side, price, at):
    return Quote("e1", "player_points", "A", side, 25.5, book, price, None, at)


def test_a_line_pulled_hours_before_tip_is_not_in_the_close():
    at = TIP - timedelta(minutes=5)
    quotes = []
    for b in ("pinnacle", "fanduel"):
        quotes += [_hq(b, "over", -120, at), _hq(b, "under", 100, at)]
    pulled = TIP - timedelta(hours=2)
    quotes += [_hq("draftkings", "over", -120, pulled), _hq("draftkings", "under", 100, pulled)]
    opp = {"event_id": "e1", "market": "player_points", "player": "A", "side": "over",
           "line": 25.5, "book": "fanduel", "price": -110}
    prob, clv, note = grading.closing_for(quotes, opp, TIP)
    assert prob is None and note == "thin close (2 books, sharp=True)"


def test_closing_includes_own_book_and_uses_last_pre_tip_quote():
    at = TIP - timedelta(minutes=5)
    hist = []
    for b in ("pinnacle", "fanduel", "betmgm"):
        hist += [_hq(b, "over", -120, at), _hq(b, "under", 100, at)]
    hist.append(_hq("betmgm", "over", 150, TIP - timedelta(hours=5)))  # older, superseded
    opp = {"event_id": "e1", "market": "player_points", "player": "A", "side": "over",
           "line": 25.5, "book": "betmgm", "price": 120}
    prob, clv, note = grading.closing_for(hist, opp, TIP)
    assert note is None and prob > 0.5
    assert clv == pytest.approx(prob - 1 / 2.2)


def test_thin_close_has_no_clv_and_says_why():
    at = TIP - timedelta(minutes=5)
    hist = [_hq("fanduel", "over", -110, at), _hq("fanduel", "under", -110, at)]
    opp = {"event_id": "e1", "market": "player_points", "player": "A", "side": "over",
           "line": 25.5, "book": "fanduel", "price": -110}
    prob, clv, note = grading.closing_for(hist, opp, TIP)
    assert prob is None and clv is None and note.startswith("thin close")


def test_prop_result_void_only_once_team_box_scores_exist():
    opp = {"market": "player_points", "side": "over", "line": 25.5}
    assert grading.prop_result(opp, None, team_has_box=False) is None     # not ingested yet
    assert grading.prop_result(opp, None, team_has_box=True) == "void"    # DNP
    assert grading.prop_result(opp, {"pts": 30, "reb": 0, "ast": 0, "fg3m": 0, "minutes": 33.0}, True) == "win"
    assert grading.prop_result(opp, {"pts": 0, "reb": 0, "ast": 0, "fg3m": 0, "minutes": 0.0}, True) == "void"
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `python3 -m pytest tests/unit/test_edge_grading.py -v`
Expected: FAIL (`AttributeError: module 'src.edge.grading' has no attribute 'grade_over_under'`)

- [ ] **Step 3: Implement**

```python
# backend/src/edge/grading.py
"""Closing-line value and results for logged opportunities (spec §6.4)."""
from __future__ import annotations

from datetime import datetime, timedelta

from psycopg2.extras import RealDictCursor

from src.edge import store
from src.edge.config import CLOSING_WINDOW_MINUTES
from src.edge.engine import american_to_decimal, clv, fair_prob, group, latest_before
from src.edge.players import et_date, stat_for
from src.edge.quotes import Quote
from src.edge.teams import ABBR_BY_NAME


def grade_over_under(actual: float, line: float, side: str) -> str:
    if actual == line:
        return "push"
    return "win" if (actual > line) == (side == "over") else "loss"


def grade_game(market: str, side: str, line: float, home: int, away: int) -> str:
    if market == "totals":
        return grade_over_under(home + away, line, side)
    margin = home - away + (line if market == "spreads" else 0.0)
    if margin == 0:
        return "push"
    return "win" if (margin > 0) == (side == "home") else "loss"


def profit_units(result: str, price: int) -> float:
    if result == "win":
        return american_to_decimal(price) - 1
    return -1.0 if result == "loss" else 0.0


def closing_for(quotes: list[Quote], opp: dict, tip: datetime) -> tuple[float | None, float | None, str | None]:
    """quotes = the event's edge_quotes rows (frozen at tip). Own book included: only our price is on trial."""
    closing = latest_before(quotes, tip, timedelta(minutes=CLOSING_WINDOW_MINUTES))
    key = (opp["event_id"], opp["market"], opp["player"], float(opp["line"]))
    books = group(closing).get(key, {})
    f = fair_prob(books, opp["market"], opp["side"], exclude_book=None)
    if f.thin:
        return None, None, f"thin close ({f.n_books} books, sharp={f.has_sharp})"
    return f.prob, clv(f.prob, int(opp["price"])), None


def prop_result(opp: dict, player_game: dict | None, team_has_box: bool) -> str | None:
    if player_game is None:
        return "void" if team_has_box else None
    if (player_game.get("minutes") or 0) <= 0:
        return "void"
    return grade_over_under(stat_for(opp["market"], player_game), float(opp["line"]), opp["side"])


def games_in_progress(conn, now: datetime) -> bool:
    with conn.cursor() as cur:
        cur.execute("""SELECT 1 FROM edge_events WHERE NOT completed
                       AND commence_time <= %s AND commence_time > %s LIMIT 1""",
                    (now, now - timedelta(hours=4)))
        return cur.fetchone() is not None


def update_scores(conn, scores_json: list[dict]) -> int:
    n = 0
    with conn.cursor() as cur:
        for ev in scores_json:
            if not ev.get("completed") or not ev.get("scores"):
                continue
            by_name = {s["name"]: int(s["score"]) for s in ev["scores"]}
            home, away = by_name.get(ev["home_team"]), by_name.get(ev["away_team"])
            if home is None or away is None:
                continue
            cur.execute("""UPDATE edge_events SET home_score = %s, away_score = %s, completed = TRUE,
                           status = 'final' WHERE event_id = %s""", (home, away, ev["id"]))
            n += cur.rowcount
    conn.commit()
    return n


def _set(cur, opp_id: int, **cols) -> None:
    sets = ", ".join(f"{k} = %s" for k in cols)
    cur.execute(f"UPDATE edge_opportunities SET {sets} WHERE id = %s", (*cols.values(), opp_id))


def run_grading(conn, now: datetime) -> dict:
    closed = graded = 0
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # 1. Closing line for every opportunity whose game has tipped.
        cur.execute("""SELECT o.*, e.commence_time FROM edge_opportunities o
                       JOIN edge_events e USING (event_id)
                       WHERE NOT o.closing_done AND e.commence_time <= %s""", (now,))
        rows = cur.fetchall()
        quote_cache: dict[str, list[Quote]] = {}
        for o in rows:
            tip = o["commence_time"]
            if o["event_id"] not in quote_cache:
                quote_cache[o["event_id"]] = store.load_quotes(conn, [o["event_id"]])
            prob, value, note = closing_for(quote_cache[o["event_id"]], o, tip)
            _set(cur, o["id"], closing_done=True, closing_fair_prob=prob, clv=value, clv_note=note)
            closed += 1

        # 2. Game-line results from final scores.
        cur.execute("""SELECT o.id, o.market, o.side, o.line, o.price, e.home_score, e.away_score
                       FROM edge_opportunities o JOIN edge_events e USING (event_id)
                       WHERE o.result IS NULL AND o.player = '' AND e.completed""")
        for o in cur.fetchall():
            r = grade_game(o["market"], o["side"], float(o["line"]), o["home_score"], o["away_score"])
            _set(cur, o["id"], result=r, profit_units=profit_units(r, o["price"]))
            graded += 1

        # 3. Prop results from box scores.
        cur.execute("""SELECT o.id, o.market, o.side, o.line, o.price, e.commence_time, e.home_team,
                              e.away_team, p.bdl_player_id
                       FROM edge_opportunities o JOIN edge_events e USING (event_id)
                       LEFT JOIN edge_players p ON p.odds_name = o.player
                       WHERE o.result IS NULL AND o.player <> '' AND e.commence_time <= %s""",
                    (now - timedelta(hours=3),))
        for o in cur.fetchall():
            if o["bdl_player_id"] is None:
                continue  # unmatched name: stays ungraded, listed on /edge/status
            d = et_date(o["commence_time"])
            teams = [ABBR_BY_NAME.get(o["home_team"]), ABBR_BY_NAME.get(o["away_team"])]
            cur.execute("SELECT * FROM edge_player_games WHERE bdl_player_id = %s AND game_date = %s",
                        (o["bdl_player_id"], d))
            pg = cur.fetchone()
            cur.execute("SELECT 1 FROM edge_player_games WHERE game_date = %s AND team = ANY(%s) LIMIT 1",
                        (d, teams))
            r = prop_result(o, dict(pg) if pg else None, cur.fetchone() is not None)
            if r is not None:
                _set(cur, o["id"], result=r, profit_units=profit_units(r, o["price"]))
                graded += 1
    conn.commit()
    return {"closed": closed, "graded": graded}
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/unit/test_edge_grading.py tests/unit/test_edge_schedule.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/edge/grading.py tests/unit/test_edge_grading.py
git commit -m "feat(edge): closing-line value and win/loss/push/void grading

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Edge API

**Files:**
- Create:
  - `backend/src/edge/views.py`
  - `backend/src/edge/api.py`
  - `backend/tests/unit/test_edge_api.py`
- Modify: `backend/src/main.py`: `from src.edge import api as edge_api` and `app.include_router(edge_api.router, prefix=settings.api_v1_prefix, tags=["Edge"])`.

**Interfaces:**
- Consumes: everything above, plus `summarize_clv` from `src.services.mlb.clv` and `require_admin`.
- Produces:
  - `views.describe(market, player, side, line, home_team, away_team) -> str` (pure)
  - `views.odds_grid(quotes, priced_fair: dict[tuple, float]) -> list[dict]` (pure)
  - `views.results_summary(rows: list[dict]) -> dict` (pure)
  - `views.games`, `views.ev_rows`, `views.odds`, `views.player`, `views.results`, `views.status` (DB)
  - Routes `GET /api/v1/edge/{games,ev,odds/{event_id},players/{bdl_id},results,status}`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/unit/test_edge_api.py
from contextlib import contextmanager

from fastapi.testclient import TestClient

from src.api import admin as admin_api
from src.config import settings
from src.edge import api as edge_api
from src.edge import views
from src.main import app

P = f"{settings.api_v1_prefix}/edge"


def test_describe():
    assert views.describe("player_points", "Jayson Tatum", "over", 27.5, "DET", "BOS") == "Jayson Tatum Over 27.5 Points"
    assert views.describe("spreads", "", "away", -5.5, "Detroit Pistons", "Boston Celtics") == "Boston Celtics +5.5"
    assert views.describe("spreads", "", "home", -5.5, "Detroit Pistons", "Boston Celtics") == "Detroit Pistons -5.5"
    assert views.describe("h2h", "", "home", 0.0, "Detroit Pistons", "Boston Celtics") == "Detroit Pistons ML"
    assert views.describe("totals", "", "under", 221.5, "Detroit Pistons", "Boston Celtics") == "Under 221.5"


def test_results_summary_ci_and_roi():
    rows = [{"clv": 0.02, "result": "win", "profit_units": 1.2, "market": "player_points", "book": "fanduel"},
            {"clv": 0.00, "result": "loss", "profit_units": -1.0, "market": "player_points", "book": "fanduel"},
            {"clv": None, "result": None, "profit_units": None, "market": "h2h", "book": "betmgm"}]
    s = views.results_summary(rows)
    assert s["n"] == 3 and s["clv"]["measured"] == 2
    assert s["clv"]["ci95"][0] < s["clv"]["mean_clv"] < s["clv"]["ci95"][1]
    assert s["record"] == {"wins": 1, "losses": 1, "pushes": 0, "voids": 0, "win_rate": 0.5,
                           "units": 0.2, "roi": 0.1}
    assert set(s["by_market"]) == {"player_points", "h2h"}


@contextmanager
def _fake_conn():
    yield object()


def test_ev_route_uses_views_and_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)
    monkeypatch.setattr(views, "ev_rows", lambda conn, now, **kw: calls.append(kw) or [{"ev": 0.05}])
    edge_api._cache.clear()
    c = TestClient(app)
    assert c.get(f"{P}/ev", params={"min_ev": 0.03}).json() == {"rows": [{"ev": 0.05}], "access_tier": "full"}
    c.get(f"{P}/ev", params={"min_ev": 0.03})
    assert len(calls) == 1  # second call served from cache


def test_status_is_admin_only(monkeypatch):
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)
    monkeypatch.setattr(views, "status", lambda conn, now: {"heartbeat_age_s": 5})
    c = TestClient(app)
    assert c.get(f"{P}/status").status_code == 404
    r = c.get(f"{P}/status", headers={"X-Cron-Key": admin_api.CRON_API_KEY})
    assert r.json() == {"heartbeat_age_s": 5}


def test_errors_never_leak_the_key(monkeypatch):
    monkeypatch.setattr(settings, "odds_api_key", "leakyleakyleaky1", raising=False)
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)

    def boom(conn, now, **kw):
        raise RuntimeError("upstream https://x?apiKey=leakyleakyleaky1")

    monkeypatch.setattr(views, "games", boom)
    edge_api._cache.clear()
    r = TestClient(app, raise_server_exceptions=False).get(f"{P}/games")
    assert r.status_code == 503 and "leakyleakyleaky1" not in r.text
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `python3 -m pytest tests/unit/test_edge_api.py -v`
Expected: FAIL (`ModuleNotFoundError: src.edge.api`)

- [ ] **Step 3: Implement `views.py`**

```python
# backend/src/edge/views.py
"""Response builders for /api/v1/edge. Pure helpers first, DB readers after."""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

from psycopg2.extras import RealDictCursor

from src.config import settings
from src.edge import players, schedule, store
from src.edge.engine import fair_prob, group, prob_to_american, price_quotes
from src.edge.quotes import Quote, sides_for
from src.edge.teams import ABBR_BY_NAME
from src.services.mlb.clv import summarize_clv
from src.utils.season import current_nba_season

MARKET_LABEL = {
    "player_points": "Points", "player_rebounds": "Rebounds", "player_assists": "Assists",
    "player_threes": "Threes", "player_points_rebounds_assists": "Pts+Reb+Ast",
    "player_points_rebounds": "Pts+Reb", "player_points_assists": "Pts+Ast",
    "player_rebounds_assists": "Reb+Ast", "h2h": "Moneyline", "spreads": "Spread", "totals": "Total",
}


def _signed(x: float) -> str:
    return f"+{x:g}" if x > 0 else f"{x:g}"


def describe(market: str, player: str, side: str, line: float, home_team: str, away_team: str) -> str:
    if market.startswith("player_"):
        return f"{player} {side.title()} {line:g} {MARKET_LABEL[market]}"
    if market == "totals":
        return f"{side.title()} {line:g}"
    team = home_team if side == "home" else away_team
    if market == "h2h":
        return f"{team} ML"
    return f"{team} {_signed(line if side == 'home' else -line)}"


def odds_grid(quotes: list[Quote], fair_by_outcome: dict[tuple, float]) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for q in quotes:
        k = (q.player, q.line, q.side)
        r = rows.setdefault(k, {"player": q.player, "line": q.line, "side": q.side, "prices": {},
                                "fair_prob": fair_by_outcome.get(k)})
        r["prices"][q.book] = {"price": q.price, "link": q.link, "observed_at": q.observed_at.isoformat()}
    for r in rows.values():
        if r["prices"]:
            r["best_book"] = max(r["prices"], key=lambda b: r["prices"][b]["price"])
        r["fair_price"] = prob_to_american(r["fair_prob"]) if r["fair_prob"] else None
    return sorted(rows.values(), key=lambda r: (r["player"], r["line"], r["side"]))


def results_summary(rows: list[dict]) -> dict:
    def block(rs: list[dict]) -> dict:
        clv = summarize_clv([float(r["clv"]) if r["clv"] is not None else None for r in rs])
        if clv["mean_clv"] is not None and clv["std_error"] is not None:
            clv["ci95"] = [round(clv["mean_clv"] - 1.96 * clv["std_error"], 5),
                           round(clv["mean_clv"] + 1.96 * clv["std_error"], 5)]
        else:
            clv["ci95"] = None
        graded = [r for r in rs if r["result"] in ("win", "loss", "push")]
        wins = sum(r["result"] == "win" for r in graded)
        losses = sum(r["result"] == "loss" for r in graded)
        units = round(sum(float(r["profit_units"] or 0) for r in graded), 4)
        return {"n": len(rs), "clv": clv, "record": {
            "wins": wins, "losses": losses, "pushes": len(graded) - wins - losses,
            "voids": sum(r["result"] == "void" for r in rs),
            "win_rate": round(wins / (wins + losses), 3) if wins + losses else None,
            "units": units, "roi": round(units / len(graded), 4) if graded else None}}

    out = block(rows)
    for dim in ("market", "book"):
        keys = sorted({r[dim] for r in rows})
        out[f"by_{dim}"] = {k: block([r for r in rows if r[dim] == k]) for k in keys}
    return out


def _factor(conn) -> int:
    return int(store.get_state(conn).get("guard_factor", "1"))


def games(conn, now: datetime) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT e.event_id, e.home_team, e.away_team, e.commence_time, e.last_lines_poll,
                              e.last_props_poll, COUNT(DISTINCT (q.player, q.market)) FILTER
                              (WHERE q.player <> '') AS prop_markets
                       FROM edge_events e LEFT JOIN edge_quotes q USING (event_id)
                       WHERE e.commence_time > %s AND e.commence_time <= %s
                       GROUP BY e.event_id ORDER BY e.commence_time""", (now, now + timedelta(hours=36)))
        return [dict(r) for r in cur.fetchall()]


def ev_rows(conn, now: datetime, market: str | None = None, book: str | None = None,
            min_ev: float | None = None, event_id: str | None = None) -> list[dict]:
    events = {e["event_id"]: e for e in store.upcoming_events(conn, now)}
    ids = [event_id] if event_id else list(events)
    if not ids:
        return []
    commence = {k: v["commence_time"] for k, v in events.items()}
    quotes = schedule.fresh(store.load_quotes(conn, ids, market), commence, now, _factor(conn))
    threshold = settings.edge_min_ev if min_ev is None else min_ev
    player_ids = _player_ids(conn)
    out = []
    for p in price_quotes(quotes, threshold, settings.edge_verify_ev):
        q = p.quote
        if not p.is_ev or (book and q.book != book):
            continue
        e = events[q.event_id]
        out.append({
            "event_id": q.event_id, "matchup": f"{e['away_team']} @ {e['home_team']}",
            "commence_time": e["commence_time"].isoformat(), "market": q.market,
            "market_label": MARKET_LABEL[q.market], "player": q.player, "side": q.side, "line": q.line,
            "bet": describe(q.market, q.player, q.side, q.line, e["home_team"], e["away_team"]),
            "book": q.book, "price": q.price, "link": q.link, "fair_prob": round(p.fair.prob, 4),
            "fair_price": prob_to_american(p.fair.prob), "ev": round(p.ev, 4), "n_books": p.fair.n_books,
            "observed_at": q.observed_at.isoformat(), "verify": p.verify,
            "bdl_player_id": player_ids.get(q.player),
        })
    return sorted(out, key=lambda r: r["ev"], reverse=True)


def _player_ids(conn) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT odds_name, bdl_player_id FROM edge_players WHERE bdl_player_id IS NOT NULL")
        return {name: pid for name, pid in cur.fetchall()}


def odds(conn, now: datetime, event_id: str, market: str) -> dict:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM edge_events WHERE event_id = %s", (event_id,))
        e = cur.fetchone()
    if not e:
        return {"event": None, "rows": []}
    quotes = store.load_quotes(conn, [event_id], market)
    fresh_q = schedule.fresh(quotes, {event_id: e["commence_time"]}, now, _factor(conn))
    fair = {}
    for (_, mkt, player, line), books in group(fresh_q).items():
        for side in sides_for(mkt):
            f = fair_prob(books, mkt, side, exclude_book=None)
            if not f.thin:
                fair[(player, line, side)] = round(f.prob, 4)
    fresh_keys = {q.key for q in fresh_q}
    rows = odds_grid(quotes, fair)
    for r in rows:
        for b, cell in r["prices"].items():
            cell["stale"] = (e["event_id"], market, r["player"], r["line"], r["side"], b) not in fresh_keys
    return {"event": {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in dict(e).items()},
            "market": market, "rows": rows}


def player(conn, now: datetime, bdl_id: int, market: str, line: float | None, min_minutes: float) -> dict:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT odds_name, team FROM edge_players WHERE bdl_player_id = %s", (bdl_id,))
        names = [dict(r) for r in cur.fetchall()]
    games_ = players.load_games(conn, bdl_id)
    events = store.upcoming_events(conn, now)
    odds_names = {n["odds_name"] for n in names}
    team = names[0]["team"] if names else None
    event = next((e for e in events if team in (ABBR_BY_NAME.get(e["home_team"]), ABBR_BY_NAME.get(e["away_team"]))), None)
    prices = []
    if event:
        prices = [q for q in store.load_quotes(conn, [event["event_id"]], market) if q.player in odds_names]
        if line is None and prices:
            line = statistics.median(q.line for q in prices)
    opponent = is_home = None
    if event and team:
        is_home = ABBR_BY_NAME.get(event["home_team"]) == team
        opponent = ABBR_BY_NAME.get(event["away_team"] if is_home else event["home_team"])
    return {
        "bdl_player_id": bdl_id, "name": names[0]["odds_name"] if names else None, "team": team,
        "market": market, "line": line, "opponent": opponent, "is_home": is_home,
        "hit_rates": players.hit_rates(games_, market, line, opponent, is_home, current_nba_season(),
                                       min_minutes) if line is not None else None,
        "games": [{**g, "game_date": g["game_date"].isoformat(), "value": players.stat_for(market, g)}
                  for g in games_[:20]],
        "prices": [{"book": q.book, "side": q.side, "line": q.line, "price": q.price, "link": q.link}
                   for q in prices],
    }


def results(conn, now: datetime, days: int, market: str | None, book: str | None) -> dict:
    sql = """SELECT market, book, clv, result, profit_units, first_seen_at FROM edge_opportunities
             WHERE closing_done AND first_seen_at >= %s"""
    params: list = [now - timedelta(days=days)]
    if market:
        sql += " AND market = %s"
        params.append(market)
    if book:
        sql += " AND book = %s"
        params.append(book)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
    out = results_summary(rows)
    daily: dict[str, list[float]] = {}
    for r in rows:
        if r["clv"] is not None:
            daily.setdefault(r["first_seen_at"].date().isoformat(), []).append(float(r["clv"]))
    out["daily_clv"] = [{"date": d, "mean_clv": round(statistics.fmean(v), 5), "n": len(v)}
                        for d, v in sorted(daily.items())]
    return out


def status(conn, now: datetime) -> dict:
    state = store.get_state(conn)
    hb = state.get("heartbeat_at")
    with conn.cursor() as cur:
        cur.execute("SELECT odds_name FROM edge_players WHERE match_method = 'unmatched' ORDER BY odds_name")
        unmatched = [r[0] for r in cur.fetchall()]
    cost = store.cost_last_24h(conn, now)
    return {
        "heartbeat_age_s": (now - datetime.fromisoformat(hb)).total_seconds() if hb else None,
        "credits_remaining": state.get("credits_remaining"), "credits_last_24h": cost,
        "projected_monthly": cost * 30, "guard_factor": int(state.get("guard_factor", "1")),
        "last_error": state.get("last_error") or None, "bdl_error": state.get("bdl_error") or None,
        "unmatched_players": unmatched, "last_box_scores_date": state.get("last_box_scores_date"),
        "backfills": {k: v for k, v in state.items() if k.startswith("backfill_")},
    }


def public_status(conn, now: datetime) -> dict:
    """The banner's view: no credits, no errors, nothing an outsider can use."""
    s = status(conn, now)
    age = s["heartbeat_age_s"]
    return {"delayed": age is None or age > 600 or bool(s["last_error"])}
```

- [ ] **Step 4: Implement `api.py`**

```python
# backend/src/edge/api.py
"""Read-only /api/v1/edge routes (spec §8). 20-30s in-process cache per query."""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import require_admin
from src.edge import store, views
from src.utils.redact import redact

router = APIRouter(prefix="/edge", tags=["Edge"])
_cache: dict[tuple, tuple[float, object]] = {}
TTL = 25.0


@contextmanager
def _conn():
    conn = store.connect()
    try:
        yield conn
    finally:
        conn.close()


def access_tier() -> str:
    """Paywall seam: Stripe will decide this. v1 is a free beta."""
    return "full"


def _cached(key: tuple, build):
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < TTL:
        return hit[1]
    try:
        with _conn() as conn:
            value = build(conn, datetime.now(timezone.utc))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"edge data unavailable: {redact(e)[:200]}") from None
    _cache[key] = (time.monotonic(), value)
    return value


@router.get("/games")
def get_games(tier: str = Depends(access_tier)) -> dict:
    return {"games": _cached(("games",), lambda c, now: views.games(c, now)), "access_tier": tier}


@router.get("/ev")
def get_ev(market: str | None = None, book: str | None = None,
           min_ev: float | None = Query(None, ge=0, le=1), event_id: str | None = None,
           tier: str = Depends(access_tier)) -> dict:
    rows = _cached(("ev", market, book, min_ev, event_id), lambda c, now: views.ev_rows(
        c, now, market=market, book=book, min_ev=min_ev, event_id=event_id))
    return {"rows": rows, "access_tier": tier}


@router.get("/odds/{event_id}")
def get_odds(event_id: str, market: str = "h2h") -> dict:
    return _cached(("odds", event_id, market), lambda c, now: views.odds(c, now, event_id, market))


@router.get("/players/{bdl_id}")
def get_player(bdl_id: int, market: str = "player_points", line: float | None = None,
               min_minutes: float = Query(0, ge=0, le=48)) -> dict:
    return _cached(("player", bdl_id, market, line, min_minutes),
                   lambda c, now: views.player(c, now, bdl_id, market, line, min_minutes))


@router.get("/results")
def get_results(days: int = Query(365, ge=1, le=730), market: str | None = None,
                book: str | None = None) -> dict:
    return _cached(("results", days, market, book), lambda c, now: views.results(c, now, days, market, book))


@router.get("/health")
def get_public_health() -> dict:
    return _cached(("health",), lambda c, now: views.public_status(c, now))


@router.get("/status", dependencies=[Depends(require_admin)])
def get_status() -> dict:
    with _conn() as conn:
        return views.status(conn, datetime.now(timezone.utc))
```

Mount the router in `src/main.py` after the NFL router:

```python
from src.edge import api as edge_api
...
app.include_router(edge_api.router, prefix=settings.api_v1_prefix, tags=["Edge"])
```

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/unit/test_edge_api.py -v && python3 -m pytest -q tests/unit`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/edge/views.py src/edge/api.py src/main.py tests/unit/test_edge_api.py
git commit -m "feat(edge): /api/v1/edge games, ev, odds, players, results and status

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Frontend data layer

**Files:**
- Create:
  - `frontend/src/types/edge.ts`
  - `frontend/src/lib/edgeApi.ts`
  - `frontend/src/lib/edgeFormat.ts`, `frontend/src/lib/edgeFormat.test.ts`
  - `frontend/src/hooks/useEdge.ts`

**Interfaces:**
- Produces:
  - Types: `EvRow`, `EdgeGame`, `OddsGridRow`, `OddsResponse`, `PlayerResponse`, `HitRate`, `ResultsResponse`.
  - `edgeApi` with `games()`, `ev(params)`, `odds(eventId, market)`, `player(id, market, line?, minMinutes?)`, `results(days)` and `health()`.
  - Formatters: `fmtAmerican(n)`, `fmtEv(x)`, `fmtPct(x)`, `fmtAge(iso, now?)`, `bookLabel(key)`, `marketLabel(key)`.
  - Hooks: `useEdgeGames`, `useEdgeEv`, `useEdgeOdds`, `useEdgePlayer`, `useEdgeResults`, `useEdgeHealth`.

- [ ] **Step 1: Write the failing test**

```ts
// frontend/src/lib/edgeFormat.test.ts
import { describe, expect, it } from 'vitest';
import { bookLabel, fmtAge, fmtAmerican, fmtEv, fmtPct, marketLabel } from './edgeFormat';

describe('edgeFormat', () => {
  it('formats American odds with a sign', () => {
    expect(fmtAmerican(150)).toBe('+150');
    expect(fmtAmerican(-110)).toBe('-110');
    expect(fmtAmerican(null)).toBe('—');
  });
  it('formats EV and percentages', () => {
    expect(fmtEv(0.0432)).toBe('+4.3%');
    expect(fmtPct(0.5)).toBe('50%');
    expect(fmtPct(null)).toBe('—');
  });
  it('formats ages', () => {
    const now = new Date('2026-10-20T18:00:00Z');
    expect(fmtAge('2026-10-20T17:59:30Z', now)).toBe('30s');
    expect(fmtAge('2026-10-20T17:55:00Z', now)).toBe('5m');
  });
  it('labels books and markets', () => {
    expect(bookLabel('williamhill_us')).toBe('Caesars');
    expect(bookLabel('unknown_book')).toBe('unknown_book');
    expect(marketLabel('player_points_rebounds_assists')).toBe('Pts+Reb+Ast');
  });
});
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `npx vitest run src/lib/edgeFormat.test.ts`
Expected: FAIL (cannot resolve `./edgeFormat`)

- [ ] **Step 3: Implement**

```ts
// frontend/src/types/edge.ts
export interface EvRow {
  event_id: string; matchup: string; commence_time: string; market: string; market_label: string;
  player: string; side: string; line: number; bet: string; book: string; price: number;
  link: string | null; fair_prob: number; fair_price: number; ev: number; n_books: number;
  observed_at: string; verify: boolean; bdl_player_id: number | null;
}
export interface EdgeGame {
  event_id: string; home_team: string; away_team: string; commence_time: string;
  last_lines_poll: string | null; last_props_poll: string | null; prop_markets: number;
}
export interface PriceCell { price: number; link: string | null; observed_at: string; stale: boolean; }
export interface OddsGridRow {
  player: string; line: number; side: string; prices: Record<string, PriceCell>;
  best_book?: string; fair_prob: number | null; fair_price: number | null;
}
export interface OddsResponse {
  event: { event_id: string; home_team: string; away_team: string; commence_time: string } | null;
  market: string; rows: OddsGridRow[];
}
export interface HitRate { hits: number; misses: number; pushes: number; games: number; rate: number | null; }
export interface PlayerGame {
  game_date: string; opponent: string; is_home: boolean; minutes: number; value: number;
}
export interface PlayerResponse {
  bdl_player_id: number; name: string | null; team: string | null; market: string; line: number | null;
  opponent: string | null; is_home: boolean | null;
  hit_rates: Record<'l5' | 'l10' | 'l20' | 'season' | 'vs_opponent' | 'home_away', HitRate> | null;
  games: PlayerGame[]; prices: { book: string; side: string; line: number; price: number; link: string | null }[];
}
export interface ClvBlock {
  measured: number; mean_clv: number | null; median_clv: number | null; beat_close_rate: number | null;
  std_error: number | null; verdict: string; ci95: [number, number] | null;
}
export interface RecordBlock {
  wins: number; losses: number; pushes: number; voids: number; win_rate: number | null;
  units: number; roi: number | null;
}
export interface ResultsBlock { n: number; clv: ClvBlock; record: RecordBlock; }
export interface ResultsResponse extends ResultsBlock {
  by_market: Record<string, ResultsBlock>; by_book: Record<string, ResultsBlock>;
  daily_clv: { date: string; mean_clv: number; n: number }[];
}
```

```ts
// frontend/src/lib/edgeApi.ts
import axios from 'axios';
import type { EdgeGame, EvRow, OddsResponse, PlayerResponse, ResultsResponse } from '@/types/edge';

const client = axios.create({ baseURL: import.meta.env.VITE_API_URL || '/api/v1' });

export interface EvParams { market?: string; book?: string; min_ev?: number; event_id?: string; }

export const edgeApi = {
  async games(): Promise<EdgeGame[]> {
    return (await client.get<{ games: EdgeGame[] }>('/edge/games')).data.games;
  },
  async ev(params: EvParams = {}): Promise<EvRow[]> {
    return (await client.get<{ rows: EvRow[] }>('/edge/ev', { params })).data.rows;
  },
  async odds(eventId: string, market: string): Promise<OddsResponse> {
    return (await client.get<OddsResponse>(`/edge/odds/${eventId}`, { params: { market } })).data;
  },
  async player(id: number, market: string, line?: number, minMinutes?: number): Promise<PlayerResponse> {
    return (await client.get<PlayerResponse>(`/edge/players/${id}`,
      { params: { market, line, min_minutes: minMinutes } })).data;
  },
  async results(days = 365): Promise<ResultsResponse> {
    return (await client.get<ResultsResponse>('/edge/results', { params: { days } })).data;
  },
  async health(): Promise<{ delayed: boolean }> {
    return (await client.get<{ delayed: boolean }>('/edge/health')).data;
  },
};
```

```ts
// frontend/src/lib/edgeFormat.ts
const BOOKS: Record<string, string> = {
  pinnacle: 'Pinnacle', novig: 'Novig', prophetx: 'ProphetX', draftkings: 'DraftKings',
  fanduel: 'FanDuel', betmgm: 'BetMGM', williamhill_us: 'Caesars', espnbet: 'ESPN BET',
  fanatics: 'Fanatics', betrivers: 'BetRivers', hardrockbet: 'Hard Rock', betonlineag: 'BetOnline',
  bovada: 'Bovada', lowvig: 'LowVig', prizepicks: 'PrizePicks', underdog: 'Underdog',
};
const MARKETS: Record<string, string> = {
  player_points: 'Points', player_rebounds: 'Rebounds', player_assists: 'Assists',
  player_threes: 'Threes', player_points_rebounds_assists: 'Pts+Reb+Ast',
  player_points_rebounds: 'Pts+Reb', player_points_assists: 'Pts+Ast',
  player_rebounds_assists: 'Reb+Ast', h2h: 'Moneyline', spreads: 'Spread', totals: 'Total',
};
export const BOOK_KEYS = Object.keys(BOOKS);
export const MARKET_KEYS = Object.keys(MARKETS);
export const DFS_BOOKS = new Set(['prizepicks', 'underdog']);

export const fmtAmerican = (n: number | null | undefined) =>
  n == null ? '—' : n > 0 ? `+${n}` : `${n}`;
export const fmtEv = (x: number) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(1)}%`;
export const fmtPct = (x: number | null | undefined) => (x == null ? '—' : `${Math.round(x * 100)}%`);
export function fmtAge(iso: string, now: Date = new Date()): string {
  const s = Math.max(0, Math.round((now.getTime() - new Date(iso).getTime()) / 1000));
  return s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m` : `${Math.floor(s / 3600)}h`;
}
export const bookLabel = (k: string) => BOOKS[k] ?? k;
export const marketLabel = (k: string) => MARKETS[k] ?? k;
```

```ts
// frontend/src/hooks/useEdge.ts
import { useQuery } from '@tanstack/react-query';
import { edgeApi, type EvParams } from '@/lib/edgeApi';

export const useEdgeGames = () =>
  useQuery({ queryKey: ['edge', 'games'], queryFn: edgeApi.games, refetchInterval: 60000 });
export const useEdgeEv = (params: EvParams) =>
  useQuery({ queryKey: ['edge', 'ev', params], queryFn: () => edgeApi.ev(params), refetchInterval: 30000 });
export const useEdgeOdds = (eventId: string, market: string) =>
  useQuery({ queryKey: ['edge', 'odds', eventId, market], queryFn: () => edgeApi.odds(eventId, market),
    refetchInterval: 30000, enabled: !!eventId });
export const useEdgePlayer = (id: number, market: string, line?: number, minMinutes?: number) =>
  useQuery({ queryKey: ['edge', 'player', id, market, line, minMinutes],
    queryFn: () => edgeApi.player(id, market, line, minMinutes), enabled: id > 0 });
export const useEdgeResults = (days = 365) =>
  useQuery({ queryKey: ['edge', 'results', days], queryFn: () => edgeApi.results(days), staleTime: 300000 });
export const useEdgeHealth = () =>
  useQuery({ queryKey: ['edge', 'health'], queryFn: edgeApi.health, refetchInterval: 60000 });
```

- [ ] **Step 4: Run the tests and type-check**

Run: `npx vitest run src/lib/edgeFormat.test.ts && npx tsc --noEmit`
Expected: tests PASS; tsc reports no errors

- [ ] **Step 5: Commit**

```bash
git add src/types/edge.ts src/lib/edgeApi.ts src/lib/edgeFormat.ts src/lib/edgeFormat.test.ts src/hooks/useEdge.ts
git commit -m "feat(edge-ui): types, API client, formatters and hooks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Frontend pages, nav and routes

**Files:**
- Create:
  - `frontend/src/components/edge/EdgeStatusBanner.tsx`
  - `frontend/src/components/edge/EdgeFooter.tsx`
  - `frontend/src/components/edge/EvTable.tsx`
  - `frontend/src/pages/EdgeHome.tsx`, `EdgeFinder.tsx`, `EdgeOdds.tsx`, `EdgePlayer.tsx`, `EdgeResults.tsx`
- Modify:
  - `frontend/src/App.tsx`
  - `frontend/src/components/Layout.tsx:12-23` (NBA pages) and `activeSport`

**Interfaces:**
- Consumes: the hooks, formatters and types from Task 13.
- Produces these routes:
  - `/` → `EdgeHome`
  - `/edge` → `EdgeFinder`
  - `/edge/odds/:eventId` → `EdgeOdds`
  - `/edge/player/:id` → `EdgePlayer`
  - `/edge/results` → `EdgeResults`
  - `/props` → redirect to `/edge`
- Removed routes: `/evaluation` and `/bet/:marketId`. Their APIs now 404 publicly.

- [ ] **Step 1: Shared components**

```tsx
// frontend/src/components/edge/EdgeStatusBanner.tsx
import { useEdgeHealth } from '@/hooks/useEdge';

export function EdgeStatusBanner() {
  const { data, isError } = useEdgeHealth();
  if (!isError && !data?.delayed) return null;
  return (
    <div className="mb-4 rounded-lg border border-[#78350f] bg-[#1c1206] px-4 py-2 text-xs font-mono text-[#fbbf24]">
      Odds delayed — prices may be out of date. +EV is hidden for anything stale.
    </div>
  );
}
```

```tsx
// frontend/src/components/edge/EdgeFooter.tsx
export function EdgeFooter() {
  return (
    <p className="mt-10 text-[10px] text-[#64748b] font-mono leading-relaxed">
      21+ only. TruLine Edge is for research, not a guarantee of any result. Prices change fast; confirm at
      the book before betting. Gambling problem? Call 1-800-GAMBLER.
    </p>
  );
}
```

```tsx
// frontend/src/components/edge/EvTable.tsx
import { Link } from 'react-router-dom';
import type { EvRow } from '@/types/edge';
import { bookLabel, fmtAge, fmtAmerican, fmtEv } from '@/lib/edgeFormat';

export function EvTable({ rows, playerIds = {} }: { rows: EvRow[]; playerIds?: Record<string, number> }) {
  if (!rows.length) {
    return <div className="text-center py-8 text-[#64748b] font-mono text-sm">No +EV prices right now.</div>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-[10px] text-[#64748b] uppercase font-bold tracking-widest border-b border-[#1e293b]">
            <th className="pb-2">EV</th><th className="pb-2">Bet</th><th className="pb-2">Book</th>
            <th className="pb-2 text-center">Price</th><th className="pb-2 text-center">Fair</th>
            <th className="pb-2 text-center">Books</th><th className="pb-2 text-center">Age</th><th className="pb-2" />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.event_id}-${r.market}-${r.player}-${r.side}-${r.line}-${r.book}`}
                className="border-b border-[#1e293b]/50">
              <td className={`py-2 font-mono font-bold ${r.verify ? 'text-[#fbbf24]' : 'text-[#a4e6ff]'}`}>
                {fmtEv(r.ev)}{r.verify && <span className="ml-1 text-[9px] uppercase">verify</span>}
              </td>
              <td className="py-2">
                <div className="text-[#f1f5f9]">
                  {r.player && playerIds[r.player]
                    ? <Link to={`/edge/player/${playerIds[r.player]}?market=${r.market}`} className="hover:underline">{r.bet}</Link>
                    : r.bet}
                </div>
                <Link to={`/edge/odds/${r.event_id}?market=${r.market}`} className="text-[10px] text-[#64748b] font-mono hover:underline">
                  {r.matchup}
                </Link>
              </td>
              <td className="py-2 text-[#94a3b8] font-mono text-xs">{bookLabel(r.book)}</td>
              <td className="py-2 text-center font-mono text-[#f1f5f9]">{fmtAmerican(r.price)}</td>
              <td className="py-2 text-center font-mono text-[#94a3b8] text-xs">{fmtAmerican(r.fair_price)}</td>
              <td className="py-2 text-center font-mono text-[#94a3b8] text-xs">{r.n_books}</td>
              <td className="py-2 text-center font-mono text-[#64748b] text-xs">{fmtAge(r.observed_at)}</td>
              <td className="py-2 text-right">
                {r.link && (
                  <a href={r.link} target="_blank" rel="noopener noreferrer sponsored"
                     className="rounded bg-[#a4e6ff] px-2 py-1 text-[10px] font-bold uppercase text-[#0b0e14]">Bet</a>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

The "Bet" button opens the book's own bet slip. Player links use `EvRow.bdl_player_id`, which `views.ev_rows` fills from `edge_players` (Task 12). Unmatched players have `null` and render as plain text.

- [ ] **Step 2: Pages**

```tsx
// frontend/src/pages/EdgeFinder.tsx
import { useMemo, useState } from 'react';
import { useEdgeEv, useEdgeGames } from '@/hooks/useEdge';
import { EvTable } from '@/components/edge/EvTable';
import { EdgeStatusBanner } from '@/components/edge/EdgeStatusBanner';
import { EdgeFooter } from '@/components/edge/EdgeFooter';
import { BOOK_KEYS, DFS_BOOKS, MARKET_KEYS, bookLabel, marketLabel } from '@/lib/edgeFormat';

const MY_BOOKS_KEY = 'edge.myBooks';
function loadMyBooks(): string[] {
  try { return JSON.parse(localStorage.getItem(MY_BOOKS_KEY) || '[]'); } catch { return []; }
}
function saveMyBooks(v: string[]) {
  try { localStorage.setItem(MY_BOOKS_KEY, JSON.stringify(v)); } catch { /* private mode */ }
}

export function EdgeFinder({ limit }: { limit?: number }) {
  const [market, setMarket] = useState('');
  const [minEv, setMinEv] = useState(0.02);
  const [eventId, setEventId] = useState('');
  const [myBooks, setMyBooks] = useState<string[]>(loadMyBooks);
  const { data: games = [] } = useEdgeGames();
  const { data: rows = [], isLoading } = useEdgeEv({
    market: market || undefined, min_ev: minEv, event_id: eventId || undefined });

  const shown = useMemo(() => {
    const r = myBooks.length ? rows.filter((x) => myBooks.includes(x.book)) : rows;
    return limit ? r.slice(0, limit) : r;
  }, [rows, myBooks, limit]);
  const playerIds = useMemo(() => Object.fromEntries(
    rows.filter((r) => r.bdl_player_id).map((r) => [r.player, r.bdl_player_id as number])), [rows]);

  const toggleBook = (b: string) => {
    const next = myBooks.includes(b) ? myBooks.filter((x) => x !== b) : [...myBooks, b];
    setMyBooks(next); saveMyBooks(next);
  };

  return (
    <div>
      <EdgeStatusBanner />
      {!limit && (
        <>
          <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">+EV Finder</h1>
          <p className="text-sm text-[#64748b] mt-1 font-mono">
            Prices that beat the devigged sharp consensus. Updated every 30 seconds.
          </p>
          <div className="mt-4 flex flex-wrap gap-2 text-xs font-mono">
            <select value={market} onChange={(e) => setMarket(e.target.value)}
                    className="bg-[#191c22] border border-[#1e293b] rounded px-2 py-1 text-[#f1f5f9]">
              <option value="">All markets</option>
              {MARKET_KEYS.map((m) => <option key={m} value={m}>{marketLabel(m)}</option>)}
            </select>
            <select value={eventId} onChange={(e) => setEventId(e.target.value)}
                    className="bg-[#191c22] border border-[#1e293b] rounded px-2 py-1 text-[#f1f5f9]">
              <option value="">All games</option>
              {games.map((g) => <option key={g.event_id} value={g.event_id}>{g.away_team} @ {g.home_team}</option>)}
            </select>
            <select value={minEv} onChange={(e) => setMinEv(Number(e.target.value))}
                    className="bg-[#191c22] border border-[#1e293b] rounded px-2 py-1 text-[#f1f5f9]">
              {[0.02, 0.03, 0.05, 0.08].map((v) => <option key={v} value={v}>EV ≥ {v * 100}%</option>)}
            </select>
          </div>
          <div className="mt-2 flex flex-wrap gap-1">
            {BOOK_KEYS.filter((b) => !DFS_BOOKS.has(b)).map((b) => (
              <button key={b} onClick={() => toggleBook(b)}
                      className={`text-[10px] font-mono rounded px-1.5 py-0.5 border ${myBooks.includes(b)
                        ? 'border-[#a4e6ff] text-[#a4e6ff]' : 'border-[#1e293b] text-[#64748b]'}`}>
                {bookLabel(b)}
              </button>
            ))}
          </div>
        </>
      )}
      <div className="mt-4 rounded-xl bg-[#191c22] border border-[#1e293b] p-4">
        {isLoading ? <div className="text-center py-8 text-[#64748b] font-mono text-sm">Loading…</div>
          : <EvTable rows={shown} playerIds={playerIds} />}
      </div>
      {!limit && <EdgeFooter />}
    </div>
  );
}
```

```tsx
// frontend/src/pages/EdgeHome.tsx
import { Link } from 'react-router-dom';
import { useEdgeGames } from '@/hooks/useEdge';
import { EdgeFinder } from '@/pages/EdgeFinder';
import { EdgeFooter } from '@/components/edge/EdgeFooter';

export function EdgeHome() {
  const { data: games = [] } = useEdgeGames();
  return (
    <div>
      <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">Tonight's NBA slate</h1>
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {games.map((g) => (
          <Link key={g.event_id} to={`/edge/odds/${g.event_id}`}
                className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b] hover:border-[#a4e6ff]">
            <div className="text-[#f1f5f9] font-mono text-sm">{g.away_team} @ {g.home_team}</div>
            <div className="text-xs font-mono text-[#64748b] mt-1">
              {new Date(g.commence_time).toLocaleString([], { weekday: 'short', hour: 'numeric', minute: '2-digit' })}
              {' · '}{g.prop_markets} props
            </div>
          </Link>
        ))}
      </div>
      <div className="mt-8 flex items-baseline justify-between">
        <h2 className="text-base font-black font-mono text-[#94a3b8]">Top +EV right now</h2>
        <Link to="/edge" className="text-xs font-mono text-[#a4e6ff]">See all →</Link>
      </div>
      <EdgeFinder limit={10} />
      <EdgeFooter />
    </div>
  );
}
```

```tsx
// frontend/src/pages/EdgeOdds.tsx
import { useParams, useSearchParams } from 'react-router-dom';
import { useEdgeOdds } from '@/hooks/useEdge';
import { EdgeStatusBanner } from '@/components/edge/EdgeStatusBanner';
import { EdgeFooter } from '@/components/edge/EdgeFooter';
import { BOOK_KEYS, bookLabel, fmtAmerican, marketLabel, MARKET_KEYS } from '@/lib/edgeFormat';

export function EdgeOdds() {
  const { eventId = '' } = useParams();
  const [params, setParams] = useSearchParams();
  const market = params.get('market') || 'h2h';
  const { data, isLoading } = useEdgeOdds(eventId, market);
  const books = BOOK_KEYS.filter((b) => data?.rows.some((r) => r.prices[b]));

  return (
    <div>
      <EdgeStatusBanner />
      <h1 className="text-2xl font-black font-mono text-[#f1f5f9]">
        {data?.event ? `${data.event.away_team} @ ${data.event.home_team}` : 'Odds'}
      </h1>
      <div className="mt-3 flex flex-wrap gap-1">
        {MARKET_KEYS.map((m) => (
          <button key={m} onClick={() => setParams({ market: m })}
                  className={`text-[10px] font-mono rounded px-2 py-1 border ${m === market
                    ? 'border-[#a4e6ff] text-[#a4e6ff]' : 'border-[#1e293b] text-[#64748b]'}`}>
            {marketLabel(m)}
          </button>
        ))}
      </div>
      <div className="mt-4 rounded-xl bg-[#191c22] border border-[#1e293b] p-4 overflow-x-auto">
        {isLoading ? <div className="text-center py-8 text-[#64748b] font-mono text-sm">Loading…</div> : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[10px] text-[#64748b] uppercase font-bold tracking-widest border-b border-[#1e293b]">
                <th className="pb-2">Outcome</th><th className="pb-2 text-center">Fair</th>
                {books.map((b) => <th key={b} className="pb-2 text-center whitespace-nowrap">{bookLabel(b)}</th>)}
              </tr>
            </thead>
            <tbody>
              {data?.rows.map((r) => (
                <tr key={`${r.player}-${r.line}-${r.side}`} className="border-b border-[#1e293b]/50">
                  <td className="py-2 text-[#f1f5f9] whitespace-nowrap">
                    {r.player && <span className="mr-1">{r.player}</span>}
                    <span className="text-[#94a3b8] font-mono text-xs">{r.side} {market === 'h2h' ? '' : r.line}</span>
                  </td>
                  <td className="py-2 text-center font-mono text-xs text-[#94a3b8]">{fmtAmerican(r.fair_price)}</td>
                  {books.map((b) => {
                    const c = r.prices[b];
                    return (
                      <td key={b} className={`py-2 text-center font-mono text-xs whitespace-nowrap ${
                        !c ? 'text-[#475569]' : c.stale ? 'text-[#475569] line-through'
                          : b === r.best_book ? 'text-[#a4e6ff] font-bold' : 'text-[#f1f5f9]'}`}>
                        {c ? (c.link && !c.stale
                          ? <a href={c.link} target="_blank" rel="noopener noreferrer sponsored">{fmtAmerican(c.price)}</a>
                          : fmtAmerican(c.price)) : '—'}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <p className="text-[10px] text-[#64748b] font-mono mt-1.5 italic">
        PrizePicks and Underdog show the line only; their payouts are fixed, so they are never priced for EV.
      </p>
      <EdgeFooter />
    </div>
  );
}
```

```tsx
// frontend/src/pages/EdgePlayer.tsx
import { useState } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useEdgePlayer } from '@/hooks/useEdge';
import { EdgeFooter } from '@/components/edge/EdgeFooter';
import { bookLabel, fmtAmerican, fmtPct, marketLabel } from '@/lib/edgeFormat';

const WINDOWS = [['l5', 'Last 5'], ['l10', 'Last 10'], ['l20', 'Last 20'], ['season', 'Season'],
  ['vs_opponent', 'vs Opp'], ['home_away', 'Home/Away']] as const;
const PROP_MARKETS = ['player_points', 'player_rebounds', 'player_assists', 'player_threes',
  'player_points_rebounds_assists', 'player_points_rebounds', 'player_points_assists', 'player_rebounds_assists'];

export function EdgePlayer() {
  const { id = '0' } = useParams();
  const [params, setParams] = useSearchParams();
  const market = params.get('market') || 'player_points';
  const [minMinutes, setMinMinutes] = useState(0);
  const { data, isLoading } = useEdgePlayer(Number(id), market, undefined, minMinutes);
  if (isLoading || !data) return <div className="text-center py-8 text-[#64748b] font-mono text-sm">Loading…</div>;
  const chart = [...data.games].reverse().map((g) => ({ ...g, label: g.game_date.slice(5) }));

  return (
    <div>
      <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">{data.name ?? 'Player'}</h1>
      <p className="text-sm text-[#64748b] mt-1 font-mono">
        {data.team}{data.opponent ? ` · next: ${data.is_home ? 'vs' : '@'} ${data.opponent}` : ''}
        {data.line != null ? ` · line ${data.line}` : ' · no line posted'}
      </p>
      <div className="mt-3 flex flex-wrap gap-1">
        {PROP_MARKETS.map((m) => (
          <button key={m} onClick={() => setParams({ market: m })}
                  className={`text-[10px] font-mono rounded px-2 py-1 border ${m === market
                    ? 'border-[#a4e6ff] text-[#a4e6ff]' : 'border-[#1e293b] text-[#64748b]'}`}>{marketLabel(m)}</button>
        ))}
        <select value={minMinutes} onChange={(e) => setMinMinutes(Number(e.target.value))}
                className="ml-2 bg-[#191c22] border border-[#1e293b] rounded px-2 py-1 text-[10px] font-mono text-[#f1f5f9]">
          {[0, 20, 25, 30].map((v) => <option key={v} value={v}>{v ? `≥ ${v} min` : 'All minutes'}</option>)}
        </select>
      </div>
      {data.hit_rates && (
        <div className="mt-4 grid grid-cols-3 sm:grid-cols-6 gap-2">
          {WINDOWS.map(([k, label]) => (
            <div key={k} className="bg-[#0b0e14] rounded-lg p-3 border border-[#1e293b]">
              <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">{label}</div>
              <div className="text-2xl font-black font-mono text-[#f1f5f9]">{fmtPct(data.hit_rates![k].rate)}</div>
              <div className="text-xs font-mono text-[#64748b] mt-1">
                {data.hit_rates![k].hits}-{data.hit_rates![k].misses}{data.hit_rates![k].pushes ? `-${data.hit_rates![k].pushes}` : ''}
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="mt-4 h-56 rounded-xl bg-[#191c22] border border-[#1e293b] p-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chart}>
            <XAxis dataKey="label" tick={{ fill: '#64748b', fontSize: 10 }} />
            <YAxis tick={{ fill: '#64748b', fontSize: 10 }} width={28} />
            <Tooltip contentStyle={{ background: '#0b0e14', border: '1px solid #1e293b' }} />
            {data.line != null && <ReferenceLine y={data.line} stroke="#fbbf24" strokeDasharray="4 4" />}
            <Bar dataKey="value">
              {chart.map((g) => (
                <Cell key={g.game_date} fill={data.line == null ? '#475569'
                  : g.value > data.line ? '#22c55e' : g.value < data.line ? '#ef4444' : '#94a3b8'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      {data.prices.length > 0 && (
        <table className="mt-4 w-full text-sm">
          <tbody>
            {data.prices.map((p) => (
              <tr key={`${p.book}-${p.side}-${p.line}`} className="border-b border-[#1e293b]/50">
                <td className="py-2 text-[#94a3b8] font-mono text-xs">{bookLabel(p.book)}</td>
                <td className="py-2 text-[#f1f5f9] font-mono text-xs">{p.side} {p.line}</td>
                <td className="py-2 text-right font-mono text-xs">
                  {p.link ? <a href={p.link} target="_blank" rel="noopener noreferrer sponsored" className="text-[#a4e6ff]">
                    {fmtAmerican(p.price)}</a> : fmtAmerican(p.price)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <EdgeFooter />
    </div>
  );
}
```

```tsx
// frontend/src/pages/EdgeResults.tsx
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useEdgeResults } from '@/hooks/useEdge';
import { EdgeFooter } from '@/components/edge/EdgeFooter';
import { bookLabel, fmtPct, marketLabel } from '@/lib/edgeFormat';
import type { ResultsBlock } from '@/types/edge';

const pts = (x: number | null) => (x == null ? '—' : `${x >= 0 ? '+' : ''}${(x * 100).toFixed(2)} pts`);

function Breakdown({ title, data, label }: { title: string; data: Record<string, ResultsBlock>;
  label: (k: string) => string }) {
  return (
    <div className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
      <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2">{title}</div>
      <table className="w-full text-sm">
        <tbody>
          {Object.entries(data).map(([k, b]) => (
            <tr key={k} className="border-b border-[#1e293b]/50">
              <td className="py-2 text-[#94a3b8] font-mono text-xs">{label(k)}</td>
              <td className="py-2 text-center font-mono text-xs text-[#f1f5f9]">{b.n}</td>
              <td className="py-2 text-center font-mono text-xs text-[#f1f5f9]">{pts(b.clv.mean_clv)}</td>
              <td className="py-2 text-center font-mono text-xs text-[#94a3b8]">{fmtPct(b.record.win_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function EdgeResults() {
  const { data, isLoading } = useEdgeResults();
  if (isLoading || !data) return <div className="text-center py-8 text-[#64748b] font-mono text-sm">Loading…</div>;
  const c = data.clv;
  return (
    <div>
      <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">Track record</h1>
      <p className="text-sm text-[#64748b] font-mono max-w-2xl mt-2 leading-relaxed">
        Every +EV price is logged the moment we first see it and graded against the closing line. Closing-line
        value (CLV) is the gap between the market's final fair price and the price you got. It settles a betting
        edge in hundreds of bets, where win rate needs thousands. We publish it whether it is good or bad.
      </p>
      <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2">
        {[['Logged', String(data.n)], ['Mean CLV', pts(c.mean_clv)],
          ['95% CI', c.ci95 ? `${pts(c.ci95[0])} … ${pts(c.ci95[1])}` : '—'],
          ['Beat the close', fmtPct(c.beat_close_rate)],
          ['Record', `${data.record.wins}-${data.record.losses}-${data.record.pushes}`],
          ['Win rate', fmtPct(data.record.win_rate)], ['Units', data.record.units.toFixed(2)],
          ['ROI', data.record.roi == null ? '—' : `${(data.record.roi * 100).toFixed(1)}%`]].map(([k, v]) => (
          <div key={k} className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
            <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">{k}</div>
            <div className="text-lg font-black font-mono text-[#f1f5f9]">{v}</div>
          </div>
        ))}
      </div>
      <p className="text-[10px] text-[#64748b] font-mono mt-1.5 italic">
        Sample: {c.measured} measured ({c.verdict}).
      </p>
      <div className="mt-4 h-56 rounded-xl bg-[#191c22] border border-[#1e293b] p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data.daily_clv}>
            <CartesianGrid stroke="#1e293b" />
            <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 10 }} />
            <YAxis tick={{ fill: '#64748b', fontSize: 10 }} width={44} tickFormatter={(v) => `${(v * 100).toFixed(1)}`} />
            <Tooltip contentStyle={{ background: '#0b0e14', border: '1px solid #1e293b' }} />
            <ReferenceLine y={0} stroke="#475569" />
            <Line type="monotone" dataKey="mean_clv" stroke="#a4e6ff" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
        <Breakdown title="By market" data={data.by_market} label={marketLabel} />
        <Breakdown title="By book" data={data.by_book} label={bookLabel} />
      </div>
      <EdgeFooter />
    </div>
  );
}
```

- [ ] **Step 3: Routes and nav**

In `App.tsx`:
- Remove the imports and routes for `MarketBoard`, `BetDetail`, `Evaluation` and `PlayerPropsPage`.
- Add `import { Navigate } from 'react-router-dom'` (merge it into the existing import) and the five page imports.
- The routes become:

```tsx
        <Route path="/" element={<EdgeHome />} />
        <Route path="/edge" element={<EdgeFinder />} />
        <Route path="/edge/odds/:eventId" element={<EdgeOdds />} />
        <Route path="/edge/player/:id" element={<EdgePlayer />} />
        <Route path="/edge/results" element={<EdgeResults />} />
        <Route path="/props" element={<Navigate to="/edge" replace />} />
        <Route path="/evaluation" element={<Navigate to="/edge/results" replace />} />
        <Route path="/trends" element={<Trends />} />
        {/* MLB and NFL routes unchanged */}
```

In `Layout.tsx`, replace the NBA `pages` array with:

```tsx
    pages: [
      { path: '/', label: 'Slate' },
      { path: '/edge', label: '+EV' },
      { path: '/edge/results', label: 'Track record' },
      { path: '/trends', label: 'Trends' },
    ],
```

Leave `activeSport` as it is: `/edge/*` already falls through to NBA. Update its doc comment so the route list reads `'/', '/edge…', '/trends'`.

- [ ] **Step 4: Verify**

Run: `npx tsc --noEmit && npx vitest run && npm run build`
Expected: no type errors, tests pass, build succeeds.

Then run `npm run dev` with `VITE_API_URL=https://nba-value-production.up.railway.app/api/v1` (after Task 15 deploys the backend). Open `/`, `/edge`, one `/edge/odds/:id`, one `/edge/player/:id` and `/edge/results` at 390px and 1280px widths. Check that:
- nothing scrolls horizontally except the odds table;
- the banner shows if the worker is stopped.

- [ ] **Step 5: Commit**

```bash
git add src/components/edge src/pages/Edge*.tsx src/App.tsx src/components/Layout.tsx
git commit -m "feat(edge-ui): +EV finder, odds screen, player pages and track record

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Deploy, backfill and opening-day dry run

**Files:**
- Create: `backend/railway.edge.toml`. It holds the worker's own config-as-code. Without it, the worker service would inherit `backend/railway.toml`, with its uvicorn start command and `/health` healthcheck, and never run the loop.
- Modify: `backend/docs/engineering/06-roadmap-2026H2.md`: one line under NBA, pointing to the spec.

```toml
# backend/railway.edge.toml — config for the `edge-worker` service ONLY.
# Set the service's "Config file path" to /backend/railway.edge.toml in Railway.
[build]
builder = "NIXPACKS"
watchPatterns = ["src/**", "requirements.txt", "railway.edge.toml",
                 "backend/src/**", "backend/requirements.txt", "backend/railway.edge.toml"]

[deploy]
startCommand = "python -m src.edge.worker"
restartPolicyType = "ALWAYS"
```

**Prerequisites (owner actions; stop and ask if any is missing):**
1. `! railway login` has been run in this session.
2. The Odds API key `1f4b…` shows ≥ 4,000,000 credits remaining. Check with the free `/v4/sports` call:
   `curl -s -D - -o /dev/null "https://api.the-odds-api.com/v4/sports/?apiKey=$KEY" | grep -i x-requests-remaining`
3. The BallDontLie account is on the ALL-STAR tier. On 2026-10-15 the local key got 401 on `/v1/stats`, while `/v1/players` worked. If the upgrade issues a new key, set it on both services. Step 5 verifies it, through `bdl_error` on `/edge/status`.
4. `CRON_API_KEY` is set on the `nba-value` service. If it isn't, the admin key is the public default in `admin.py:22`.

- [ ] **Step 1: Push the code**

Run the full suite first:
- `python3 -m pytest -q tests/unit`
- `cd ../frontend && npx vitest run && npm run build`

Then push: `git push origin main`. Railway auto-deploys `nba-value`; check it with `railway logs --service nba-value`.

After the deploy, run: `curl -s https://nba-value-production.up.railway.app/api/v1/nfl/debug/odds`
Expected: `{"detail":"Not Found"}`. The leak is closed.

- [ ] **Step 2: Set the key and flags on `nba-value`**

Run `railway variables --service nba-value --set ODDS_API_KEY=<key> --set NBA_PICKS_PUBLIC=false`. Then confirm the old public NBA routes 404: `curl -s -o /dev/null -w '%{http_code}' https://nba-value-production.up.railway.app/api/v1/markets` should print `404`.

- [ ] **Step 3: Create the `edge-worker` service**

Use the Railway MCP `create-service` tool (or the dashboard) in project `nba` (`ba72d794-a84c-4b39-b498-1f11a83f0ef4`):
- Same GitHub repo.
- Root directory `backend`.
- **Config file path `/backend/railway.edge.toml`**. This supplies the start command, and keeps the web healthcheck off the worker.

Set its variables:
- `DATABASE_URL=${{Postgres.DATABASE_URL}}`
- `ODDS_API_KEY`, `BALLDONTLIE_API_KEY`
- `EDGE_CREDIT_CAP=3500000`

Deploy, then check that `railway logs --service edge-worker` shows `edge tick {...}` lines with no errors.

- [ ] **Step 4: Verify storage on real data (first run of the SQL)**

```bash
curl -s -H "X-Cron-Key: $CRON_API_KEY" https://nba-value-production.up.railway.app/api/v1/edge/status
curl -s https://nba-value-production.up.railway.app/api/v1/edge/games | python3 -m json.tool | head -40
curl -s "https://nba-value-production.up.railway.app/api/v1/edge/ev?min_ev=0" | python3 -c "import json,sys;r=json.load(sys.stdin)['rows'];print(len(r));print(r[:3])"
```

Expected:
- The heartbeat age is under 60 s.
- `credits_last_24h` grows by about 16 per prop poll and 6 per lines poll.
- The games list shows the opener.
- The EV rows have `n_books ≥ 3`.

- [ ] **Step 5: Verify the 2025-26 box-score backfill**

The worker runs `ensure_backfill` on its first start. It loads about 260 dates at about 1.1 s each, plus extra pages, so it takes roughly 5–10 minutes. Polling starts after it finishes, so deploy the worker no later than Sunday.

When it's done, `/edge/status` should show `"backfills": {"backfill_2025": "done"}` and `bdl_error: null`. A `bdl_error` containing `401` means the BallDontLie tier is still free. Fix the key, then delete the gate so the backfill reruns: `DELETE FROM edge_worker_state WHERE key = 'backfill_2025'`.

Then spot-check one player: `curl -s "https://nba-value-production.up.railway.app/api/v1/edge/players/434?market=player_points"` (Jayson Tatum). The `hit_rates` field should be filled in and `games` should have 20 rows.

- [ ] **Step 6: Dry run on opening-day lines (Mon 2026-10-19)**

1. Ten rows from `/edge/ev`: open the book's site for each and confirm the price. Any mismatch is a parsing or staleness bug, and must be fixed before Tuesday.
2. `/edge/status`:
   - `projected_monthly` should be below 3,500,000 and `guard_factor` should be 1.
   - Any `unmatched_players`: add a manual row, e.g. `INSERT INTO edge_players (odds_name, bdl_player_id, team, match_method, confidence, updated_at) VALUES (...,'manual',1.0,now())`.
3. Stop the worker for 11 minutes, confirm the frontend banner appears, then restart the worker.

- [ ] **Step 7: Deploy the frontend**

From the repo root (not `frontend/`), run `vercel --prod`. Then check `https://www.truline.app/`, `/edge` and `/edge/results` on a phone-width viewport.

- [ ] **Step 8: Commit and push**

```bash
git add backend/railway.edge.toml backend/docs/engineering/06-roadmap-2026H2.md
git commit -m "chore(edge): edge-worker Railway config and roadmap pointer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

- [ ] **Step 9: After the first night (Wed 2026-10-21)**

Run `/edge/results` and confirm the opportunities from Oct 20 have `closing_done = true` with CLV values. Props should be graded after the 09:00 UTC box-score run. Report the count, mean CLV, and any thin-close notes to the owner.
