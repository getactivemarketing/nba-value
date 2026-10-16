# TruLine Edge NFL Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Edge worker polls NFL game lines and 10 NFL prop markets alongside NBA; `/nfl/edge` shows NFL +EV prices and odds screens; CLV is tracked for every NFL opportunity. NBA behaviour does not change.

**Architecture:**
- **Config:** a per-sport `SportConfig` table in `src/edge/config.py`.
- **Client:** `EdgeOddsClient` methods take the sport.
- **Worker:** `tick` loops over the enabled sports with per-sport poll state. The credit guard, grading, pruning and BallDontLie stay global.
- **Views and API:** every view filters on `edge_events.sport`, and the API takes `?sport=nba|nfl` (default `nba`).
- **Frontend:** reuses `EdgeFinder`/`EdgeOdds` with a `sport` prop on new `/nfl/edge` routes.

**Tech Stack:** Python 3.13, FastAPI, psycopg2, pytest (+ throwaway Postgres 16 for integration tests), React 18 + TypeScript + Vite, vitest.

**Spec:** `backend/docs/superpowers/specs/2026-10-16-edge-nfl-phase-a-design.md`

## Global Constraints

- **Sport keys:** `basketball_nba` (slug `nba`) and `americanfootball_nfl` (slug `nfl`). API `sport` values are the slugs; default `nba`.
- **NFL prop markets (exactly these 10):** `player_pass_yds`, `player_pass_tds`, `player_pass_completions`, `player_pass_attempts`, `player_pass_interceptions`, `player_rush_yds`, `player_rush_attempts`, `player_receptions`, `player_reception_yds`, `player_rush_reception_yds`. No anytime TD.
- **NFL labels:** Pass Yds, Pass TDs, Completions, Pass Att, INTs, Rush Yds, Rush Att, Receptions, Rec Yds, Rush+Rec Yds.
- **NFL game lines** poll from 7 days (168 h) before kickoff, every 60 min until 36 h out; inside 36 h the existing cadence table applies. Props: the existing within-36 h cadence for both sports.
- **NBA state keys keep their current unprefixed names** (`last_events_refresh`, `last_scores_poll`). NFL keys are prefixed `nfl:`. Per-sport error key: `sport_error:<slug>` for every sport.
- **`EDGE_SPORTS`** (settings `edge_sports`, default `"nba,nfl"`) selects which sports the worker polls. Unknown slugs are ignored with a warning.
- **NFL props are never graded** (no void, no loss) in Phase A: `result` stays NULL. CLV applies to all NFL opportunities.
- **NBA views never contain NFL rows, and vice versa.** An unknown `sport` returns 422.
- **Commit trailer:** `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Tests on this Mac:** unit tests must never touch `localhost:5432` (it hangs). DB tests use `EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest`. Start it with `/opt/homebrew/opt/postgresql@16/bin/pg_ctl -D <scratchpad>/pg/data -o "-p 55432 -k ''" start`.

## Review Focus

1. **NBA's hourly events refresh marking every NFL game unavailable.** `drop_missing_events` updates every future event absent from the feed it was given; unscoped, NBA's feed would drop all NFL games. Expected: each sport's refresh only touches its own sport. (Task 4 test.)
2. **An NFL prop for a player who shares a name with a matched NBA player.** Prop grading joins `edge_players` by name, so it would be graded against NBA box scores. Expected: non-NBA props are skipped entirely. (Task 4 test.)
3. **An NFL `event_id` requested under `sport=nba`** (`/edge/ev?event_id=…`, `/edge/odds/{id}`). Expected: empty result, never cross-sport rows. (Task 6 test.)
4. **NFL lines on Friday for Sunday games (36–168 h out)** are dropped by `schedule.fresh` as stale because `interval_for` returns None beyond 36 h. Expected: priced and shown. (Task 2 and Task 6 tests.)
5. **One sport's Odds API failure (e.g. 500 on NFL odds) stopping the other sport's polls.** Expected: recorded under `sport_error:nfl`; NBA polls in the same tick; 401/429 still trigger the global backoff. (Task 5 test.)

---

### Task 1: Per-sport config, settings and NFL labels

**Files:**
- Modify: `backend/src/edge/config.py`
- Modify: `backend/src/config.py` (add `edge_sports`)
- Modify: `backend/src/edge/views.py:20-25` (`MARKET_LABEL`)
- Test: `backend/tests/unit/test_edge_sports.py` (create)

**Interfaces:**
- Produces:
  - `SportConfig(key: str, slug: str, prop_markets: tuple[str, ...], lines_window_hours: int, has_players: bool)`;
  - `NBA`, `NFL`, `SPORTS: dict[str, SportConfig]`, `NBA_PROP_MARKETS`, `NFL_PROP_MARKETS`;
  - `enabled_sports(csv: str) -> list[SportConfig]`;
  - `state_key(sport: SportConfig, name: str) -> str`;
  - `sport_by_key(key: str) -> SportConfig | None`;
  - `settings.edge_sports: str`;
  - `MARKET_LABEL` entries for all 10 NFL markets.
  - Legacy aliases `SPORT_KEY` and `PROP_MARKETS` stay (NBA values).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/test_edge_sports.py
import json

from src.edge import views
from src.edge.config import (NBA, NFL, NFL_PROP_MARKETS, PROP_MARKETS, SPORTS, enabled_sports,
                             sport_by_key, state_key)
from src.edge.ingest import parse_event

FIX = "tests/fixtures/edge"


def test_sports_table():
    assert SPORTS == {"nba": NBA, "nfl": NFL}
    assert NBA.key == "basketball_nba" and NBA.lines_window_hours == 36 and NBA.has_players
    assert NFL.key == "americanfootball_nfl" and NFL.lines_window_hours == 168 and not NFL.has_players
    assert PROP_MARKETS == NBA.prop_markets  # legacy alias unchanged
    assert len(NFL_PROP_MARKETS) == 10 and "player_anytime_td" not in NFL_PROP_MARKETS
    assert sport_by_key("americanfootball_nfl") is NFL and sport_by_key("icehockey_nhl") is None


def test_enabled_sports_ignores_unknown_and_blank():
    assert enabled_sports("nba,nfl") == [NBA, NFL]
    assert enabled_sports(" nfl , bogus,,") == [NFL]
    assert enabled_sports("") == []


def test_state_keys_keep_nba_names():
    assert state_key(NBA, "last_events_refresh") == "last_events_refresh"
    assert state_key(NFL, "last_events_refresh") == "nfl:last_events_refresh"


def test_every_nfl_market_has_a_label_and_parses_from_the_real_fixture():
    ev = json.load(open(f"{FIX}/nfl_props.json"))
    quotes = parse_event(ev, observed_at=None)
    markets = {q.market for q in quotes}
    assert set(NFL_PROP_MARKETS) <= markets  # all 10 present in the CHI @ ATL sample
    for m in NFL_PROP_MARKETS:
        assert m in views.MARKET_LABEL
    q = next(q for q in quotes if q.market == "player_rush_yds" and q.side == "over")
    assert views.describe(q.market, q.player, q.side, q.line, "Atlanta Falcons", "Chicago Bears") == \
        f"{q.player} Over {q.line:g} Rush Yds"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest -q tests/unit/test_edge_sports.py`
Expected: ERROR at collection: `ImportError: cannot import name 'NBA' from 'src.edge.config'`.

- [ ] **Step 3: Implement**

In `backend/src/edge/config.py`, replace the `SPORT_KEY = ...` line and the `PROP_MARKETS = (...)` block with:

```python
from dataclasses import dataclass

NBA_PROP_MARKETS = (
    "player_points", "player_rebounds", "player_assists", "player_threes",
    "player_points_rebounds_assists", "player_points_rebounds",
    "player_points_assists", "player_rebounds_assists",
)
# Anytime TD is excluded: most books quote only "Yes", so there is no
# two-sided market to devig (2026-10-16 sample: 319 Yes vs 55 No).
NFL_PROP_MARKETS = (
    "player_pass_yds", "player_pass_tds", "player_pass_completions", "player_pass_attempts",
    "player_pass_interceptions", "player_rush_yds", "player_rush_attempts",
    "player_receptions", "player_reception_yds", "player_rush_reception_yds",
)


@dataclass(frozen=True)
class SportConfig:
    key: str                         # Odds API sport key
    slug: str                        # API value and state-key prefix
    prop_markets: tuple[str, ...]
    lines_window_hours: int          # poll game lines from this far before kickoff
    has_players: bool                # player matching, pages and prop grading


NBA = SportConfig("basketball_nba", "nba", NBA_PROP_MARKETS, 36, True)
NFL = SportConfig("americanfootball_nfl", "nfl", NFL_PROP_MARKETS, 7 * 24, False)
SPORTS = {"nba": NBA, "nfl": NFL}

# Legacy aliases (NBA), kept so existing imports keep working.
SPORT_KEY = NBA.key
PROP_MARKETS = NBA_PROP_MARKETS


def enabled_sports(csv: str) -> list[SportConfig]:
    import logging
    out = []
    for slug in (s.strip() for s in csv.split(",")):
        if not slug:
            continue
        if slug in SPORTS:
            out.append(SPORTS[slug])
        else:
            logging.getLogger("edge.config").warning("EDGE_SPORTS: unknown sport %r ignored", slug)
    return out


def sport_by_key(key: str) -> SportConfig | None:
    return next((s for s in SPORTS.values() if s.key == key), None)


def state_key(sport: SportConfig, name: str) -> str:
    """NBA keeps its pre-multi-sport key names so the live worker's state carries over."""
    return name if sport.slug == "nba" else f"{sport.slug}:{name}"
```

In `backend/src/config.py`, after `edge_credit_cap: int = 3_500_000` add:

```python
    edge_sports: str = "nba,nfl"
```

In `backend/src/edge/views.py`, replace `MARKET_LABEL = {...}` with:

```python
MARKET_LABEL = {
    "player_points": "Points", "player_rebounds": "Rebounds", "player_assists": "Assists",
    "player_threes": "Threes", "player_points_rebounds_assists": "Pts+Reb+Ast",
    "player_points_rebounds": "Pts+Reb", "player_points_assists": "Pts+Ast",
    "player_rebounds_assists": "Reb+Ast", "h2h": "Moneyline", "spreads": "Spread", "totals": "Total",
    "player_pass_yds": "Pass Yds", "player_pass_tds": "Pass TDs", "player_pass_completions": "Completions",
    "player_pass_attempts": "Pass Att", "player_pass_interceptions": "INTs", "player_rush_yds": "Rush Yds",
    "player_rush_attempts": "Rush Att", "player_receptions": "Receptions", "player_reception_yds": "Rec Yds",
    "player_rush_reception_yds": "Rush+Rec Yds",
}
```

If `parse_event(ev, observed_at=None)` fails in the test because `observed_at` is required to be a datetime, pass `datetime(2026, 10, 16, tzinfo=timezone.utc)` in the test instead. That is a test-only adjustment.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python3 -m pytest -q tests/unit/test_edge_sports.py tests/unit`
Expected: all pass (4 new tests; full unit suite green).

- [ ] **Step 5: Commit**

```bash
git add backend/src/edge/config.py backend/src/config.py backend/src/edge/views.py backend/tests/unit/test_edge_sports.py
git commit -m "feat(edge): per-sport config with NFL markets and labels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Window-aware schedule (NFL lines 36–168 h out)

**Files:**
- Modify: `backend/src/edge/schedule.py` (`interval_for`, `fresh`, `plan_polls`)
- Test: `backend/tests/unit/test_edge_schedule.py` (append)

**Interfaces:**
- Consumes: none.
- Produces:
  - `interval_for(market, minutes_to_tip, factor, lines_window_hours: int = 36) -> timedelta | None`;
  - `fresh(quotes, commence_by_event, now, factor, lines_window_hours: int = 36)`;
  - `plan_polls(events, now, factor, lines_window_hours: int = 36) -> PollPlan`;
  - constant `FAR_LINES_MINUTES = 60`.
  - The defaults keep NBA behaviour identical.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/unit/test_edge_schedule.py`:

```python
def test_far_lines_poll_hourly_inside_the_sport_window():
    from src.edge.schedule import FAR_LINES_MINUTES, interval_for
    mins = 3 * 24 * 60  # Thursday for a Sunday game
    assert interval_for("h2h", mins, 1) is None                          # NBA default window: no polling
    assert interval_for("h2h", mins, 1, lines_window_hours=168) == timedelta(minutes=FAR_LINES_MINUTES)
    assert interval_for("h2h", mins, 2, lines_window_hours=168) == timedelta(minutes=2 * FAR_LINES_MINUTES)
    assert interval_for("player_pass_yds", mins, 1, lines_window_hours=168) is None  # props stay 36 h
    assert interval_for("h2h", 8 * 24 * 60, 1, lines_window_hours=168) is None      # beyond the window
    assert interval_for("h2h", 120, 1, lines_window_hours=168) == interval_for("h2h", 120, 1)  # inside 36 h unchanged


def test_fresh_and_plan_polls_use_the_window():
    from src.edge.quotes import Quote
    from src.edge.schedule import fresh, plan_polls
    now = datetime(2026, 10, 16, 18, tzinfo=timezone.utc)
    tip = now + timedelta(days=2)
    q = Quote("e1", "spreads", "", "home", -3.5, "fanduel", -110, None, now - timedelta(minutes=30))
    assert fresh([q], {"e1": tip}, now, 1) == []                           # NBA window: stale
    assert fresh([q], {"e1": tip}, now, 1, lines_window_hours=168) == [q]  # NFL window: fresh
    events = [{"event_id": "e1", "commence_time": tip, "last_lines_poll": None, "last_props_poll": None}]
    assert plan_polls(events, now, 1).lines is False
    p = plan_polls(events, now, 1, lines_window_hours=168)
    assert p.lines is True and p.prop_events == []
```

(Use the file's existing imports. If `datetime`, `timedelta` or `timezone` are not already imported at the top of the test file, add `from datetime import datetime, timedelta, timezone`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest -q tests/unit/test_edge_schedule.py`
Expected: 2 failed (`ImportError: cannot import name 'FAR_LINES_MINUTES'`, and the `fresh` call fails with an unexpected keyword argument).

- [ ] **Step 3: Implement**

In `backend/src/edge/schedule.py`:

```python
FAR_LINES_MINUTES = 60  # game lines between 36 h and the sport's window edge


def interval_for(market: str, minutes_to_tip: float, factor: int,
                 lines_window_hours: int = 36) -> timedelta | None:
    base = base_intervals(minutes_to_tip)
    if base is None:
        if (not market.startswith("player_") and 0 < minutes_to_tip <= lines_window_hours * 60):
            return timedelta(minutes=FAR_LINES_MINUTES * factor)
        return None
    minutes = base[1] if market.startswith("player_") else base[0]
    return timedelta(minutes=minutes * factor)
```

`fresh` gains `lines_window_hours: int = 36` and passes it on: `interval_for(q.market, _minutes_to(commence, now), factor, lines_window_hours)`.

`plan_polls` gains `lines_window_hours: int = 36`. Its line interval becomes `interval_for("h2h", mins, factor, lines_window_hours)`; the props interval is unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python3 -m pytest -q tests/unit/test_edge_schedule.py tests/unit`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/src/edge/schedule.py backend/tests/unit/test_edge_schedule.py
git commit -m "feat(edge): poll and price game lines across a per-sport window

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Sport-aware Odds API client

**Files:**
- Modify: `backend/src/edge/client.py`
- Test: `backend/tests/unit/test_edge_ingest.py` (append)

**Interfaces:**
- Consumes: `SportConfig`, `NBA` (Task 1).
- Produces:
  - `get_events(sport=NBA)`;
  - `get_game_odds(sport=NBA)`;
  - `get_event_odds(event_id, sport=NBA)`;
  - `get_scores(days_from=1, sport=NBA)`.
  - The defaults keep existing callers working.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/unit/test_edge_ingest.py`:

```python
@pytest.mark.asyncio
async def test_client_uses_the_sports_key_and_prop_markets():
    import httpx
    from src.edge.client import EdgeOddsClient
    from src.edge.config import NFL, NFL_PROP_MARKETS

    seen = []

    def handler(request):
        seen.append(request.url)
        return httpx.Response(200, json=[], headers={"x-requests-remaining": "5"})

    c = EdgeOddsClient(api_key="k", transport=httpx.MockTransport(handler))
    await c.get_events(NFL)
    await c.get_game_odds(NFL)
    await c.get_event_odds("e9", NFL)
    await c.get_scores(days_from=2, sport=NFL)
    paths = [u.path for u in seen]
    assert paths == ["/v4/sports/americanfootball_nfl/events", "/v4/sports/americanfootball_nfl/odds",
                     "/v4/sports/americanfootball_nfl/events/e9/odds", "/v4/sports/americanfootball_nfl/scores"]
    assert seen[2].params["markets"] == ",".join(NFL_PROP_MARKETS)
    assert seen[3].params["daysFrom"] == "2"
```

(If `pytest` is not imported at the top of that file, it already is, since the file has async tests marked `@pytest.mark.asyncio`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest -q tests/unit/test_edge_ingest.py -k sports_key`
Expected: FAIL with `TypeError: get_events() takes 1 positional argument but 2 were given`.

- [ ] **Step 3: Implement**

In `backend/src/edge/client.py`, change the import to `from src.edge.config import BOOKS, LINE_MARKETS, NBA, SportConfig` and replace the four methods:

```python
    async def get_events(self, sport: SportConfig = NBA):
        return await self._get(f"/sports/{sport.key}/events", {})

    async def get_game_odds(self, sport: SportConfig = NBA):
        return await self._get(f"/sports/{sport.key}/odds", {
            "bookmakers": ",".join(BOOKS), "markets": ",".join(LINE_MARKETS),
            "oddsFormat": "american", "includeLinks": "true"})

    async def get_event_odds(self, event_id: str, sport: SportConfig = NBA):
        return await self._get(f"/sports/{sport.key}/events/{event_id}/odds", {
            "bookmakers": ",".join(BOOKS), "markets": ",".join(sport.prop_markets),
            "oddsFormat": "american", "includeLinks": "true"})

    async def get_scores(self, days_from: int = 1, sport: SportConfig = NBA):
        return await self._get(f"/sports/{sport.key}/scores", {"daysFrom": days_from})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python3 -m pytest -q tests/unit`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/src/edge/client.py backend/tests/unit/test_edge_ingest.py
git commit -m "feat(edge): Odds API client takes the sport

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Sport-scoped store and grading

**Files:**
- Modify: `backend/src/edge/store.py` (`upcoming_events`, `drop_missing_events`)
- Modify: `backend/src/edge/grading.py` (`games_in_progress`, `run_grading` step 3, `box_scores_missing`)
- Test: `backend/tests/integration/test_edge_store_db.py` (append)

**Interfaces:**
- Consumes: `NBA`, `NFL` (Task 1).
- Produces:
  - `upcoming_events(conn, now, hours=36, sport: str | None = None)`;
  - `drop_missing_events(conn, live_ids, now, sport: str | None = None)`;
  - `games_in_progress(conn, now, sport: str | None = None)`.
  - The `sport` arguments are Odds API keys (`"americanfootball_nfl"`). `None` keeps today's unscoped behaviour.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/integration/test_edge_store_db.py`:

```python
def _nfl_event(event_id="n1", hours=2):
    return {"event_id": event_id, "sport": "americanfootball_nfl", "home_team": "Atlanta Falcons",
            "away_team": "Chicago Bears", "commence_time": NOW + timedelta(hours=hours)}


def test_one_sports_refresh_never_drops_the_others_events(conn):
    store.upsert_events(conn, [_event(), _nfl_event()])
    # NBA's feed lists only e1: NFL's n1 must survive.
    store.drop_missing_events(conn, ["e1"], NOW, sport="basketball_nba")
    nfl = store.upcoming_events(conn, NOW, hours=168, sport="americanfootball_nfl")
    nba = store.upcoming_events(conn, NOW, sport="basketball_nba")
    assert [e["event_id"] for e in nfl] == ["n1"] and [e["event_id"] for e in nba] == ["e1"]


def test_games_in_progress_is_per_sport(conn):
    store.upsert_events(conn, [_nfl_event(hours=-1)])
    from src.edge import grading
    assert grading.games_in_progress(conn, NOW, "americanfootball_nfl") is True
    assert grading.games_in_progress(conn, NOW, "basketball_nba") is False


def test_nfl_props_are_never_graded_even_when_the_name_matches_an_nba_player(conn):
    from src.edge import grading
    store.upsert_events(conn, [_nfl_event(hours=-5)])
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_players (odds_name, bdl_player_id, team, match_method, confidence, updated_at)
                       VALUES ('Josh Allen', 999, 'ORL', 'exact', 1.0, %s)""", (NOW,))
        cur.execute("""INSERT INTO edge_opportunities (event_id, market, player, side, line, book, price,
                       first_seen_at, last_seen_at, fair_prob, ev, max_ev, n_books, devig_method, closing_done)
                       VALUES ('n1', 'player_pass_yds', 'Josh Allen', 'over', 250.5, 'fanduel', 110,
                               %s, %s, 0.5, 0.05, 0.05, 5, 'power', TRUE)""", (NOW, NOW))
    conn.commit()
    grading.run_grading(conn, NOW + timedelta(hours=1))
    with conn.cursor() as cur:
        cur.execute("SELECT result FROM edge_opportunities WHERE event_id = 'n1'")
        assert cur.fetchone()[0] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py -k "never_drops or in_progress_is_per or never_graded"`
Expected:
- `never_drops` fails with `TypeError: drop_missing_events() got an unexpected keyword argument 'sport'`;
- `in_progress` fails with a TypeError;
- `never_graded` either fails because the result is not None (the NBA player-games lookup voids it) or passes. If it passes, it still pins the behaviour: keep it.

- [ ] **Step 3: Implement**

`backend/src/edge/store.py`:

```python
def upcoming_events(conn, now: datetime, hours: int = 36, sport: str | None = None) -> list[dict]:
    sql = """SELECT * FROM edge_events WHERE commence_time > %s AND commence_time <= %s
             AND status <> 'unavailable'"""
    params: list = [now, now + timedelta(hours=hours)]
    if sport:
        sql += " AND sport = %s"
        params.append(sport)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql + " ORDER BY commence_time", params)
        return [dict(r) for r in cur.fetchall()]


def drop_missing_events(conn, live_ids: list[str], now: datetime, sport: str | None = None) -> int:
    """Future events of `sport` no longer in its feed are marked unavailable.

    Scoped per sport: each sport's feed only lists its own events, so an
    unscoped update would drop every other sport's games.
    """
    sql = """UPDATE edge_events SET status = 'unavailable'
             WHERE commence_time > %s AND status = 'scheduled' AND NOT (event_id = ANY(%s))"""
    params: list = [now, live_ids]
    if sport:
        sql += " AND sport = %s"
        params.append(sport)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        n = cur.rowcount
    conn.commit()
    return n
```

`backend/src/edge/grading.py`:
- `games_in_progress(conn, now, sport: str | None = None)` appends `AND sport = %s` when `sport` is given.
- In `run_grading` step 3, change the WHERE clause to `WHERE o.result IS NULL AND o.player <> '' AND e.commence_time <= %s AND e.sport = 'basketball_nba'`, with the comment `-- prop grading needs NBA box scores; NFL props get CLV only until Phase B`.
- In `box_scores_missing`, add `AND sport = 'basketball_nba'` to the events query.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py && python3 -m pytest -q tests/unit`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/src/edge/store.py backend/src/edge/grading.py backend/tests/integration/test_edge_store_db.py
git commit -m "fix(edge): scope event refresh, live-game check and prop grading by sport

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Multi-sport worker tick

**Files:**
- Modify: `backend/src/edge/worker.py` (`tick`, `_reprice`)
- Modify: `backend/tests/integration/test_edge_store_db.py` (`_FakeOdds` signatures, new tests)

**Interfaces:**
- Consumes:
  - Task 1: `enabled_sports`, `state_key`, `SportConfig`, `NBA`, `NFL`;
  - Task 2: window params;
  - Task 3: client sport params;
  - Task 4: store/grading sport params.
- Produces:
  - `tick(conn, client, bdl, now, sports: list[SportConfig] | None = None) -> dict`, which returns `{"nba": {...}, "nfl": {...}, "graded": ..., "pruned": ...}`;
  - per-sport error state `sport_error:<slug>`.

- [ ] **Step 1: Update the fake client and write the failing tests**

In `backend/tests/integration/test_edge_store_db.py`, replace `class _FakeOdds` with a sport-aware fake (existing tests keep passing because NBA is the default):

```python
class _FakeOdds:
    def __init__(self, events, nfl_events=(), fail_nfl=None):
        self.events, self.nfl_events, self.fail_nfl = events, list(nfl_events), fail_nfl
        self.calls = []

    async def get_events(self, sport=None):
        from src.edge.client import OddsApiError, Usage
        if sport is not None and sport.slug == "nfl":
            if self.fail_nfl:
                raise OddsApiError(self.fail_nfl, "nfl upstream error")
            return self.nfl_events, Usage(None, None, 0)
        return self.events, Usage(None, None, 0)

    async def get_game_odds(self, sport=None):
        from src.edge.client import Usage
        return [], Usage(4_000_000, 1, 6)

    async def get_event_odds(self, eid, sport=None):
        from src.edge.client import OddsApiError, Usage
        self.calls.append(eid)
        if eid == "gone":
            raise OddsApiError(404, "EVENT_NOT_FOUND")
        key = sport.key if sport is not None else "basketball_nba"
        ev = {"id": eid, "sport_key": key, "home_team": "Detroit Pistons",
              "away_team": "Boston Celtics", "commence_time": "x", "bookmakers": []}
        return ev, Usage(4_000_000, 1, 16)

    async def get_scores(self, days_from=1, sport=None):
        from src.edge.client import Usage
        return [], Usage(4_000_000, 1, 2)
```

Append the tests:

```python
def _nfl_feed_event(i, tip):
    return {"id": i, "sport_key": "americanfootball_nfl", "home_team": "Atlanta Falcons",
            "away_team": "Chicago Bears", "commence_time": tip}


@pytest.mark.asyncio
async def test_tick_polls_both_sports_and_keeps_nba_state_names(conn):
    from src.edge import worker
    from src.edge.config import NBA, NFL
    now = NOW.replace(hour=5)
    tip = (now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    nba = [{"id": "b1", "sport_key": "basketball_nba", "home_team": "Detroit Pistons",
            "away_team": "Boston Celtics", "commence_time": tip}]
    client = _FakeOdds(nba, nfl_events=[_nfl_feed_event("f1", tip)])
    out = await worker.tick(conn, client, _FakeBdl(), now, sports=[NBA, NFL])
    assert sorted(client.calls) == ["b1", "f1"] and "nba" in out and "nfl" in out
    state = store.get_state(conn)
    assert "last_events_refresh" in state and "nfl:last_events_refresh" in state
    # NBA's refresh did not drop the NFL game, and vice versa
    assert {e["event_id"] for e in store.upcoming_events(conn, now, hours=168)} == {"b1", "f1"}


@pytest.mark.asyncio
async def test_an_nfl_failure_does_not_stop_nba(conn):
    from src.edge import worker
    from src.edge.config import NBA, NFL
    now = NOW.replace(hour=5)
    tip = (now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    nba = [{"id": "b1", "sport_key": "basketball_nba", "home_team": "Detroit Pistons",
            "away_team": "Boston Celtics", "commence_time": tip}]
    client = _FakeOdds(nba, fail_nfl=500)
    await worker.tick(conn, client, _FakeBdl(), now, sports=[NBA, NFL])
    assert client.calls == ["b1"]
    assert "500" in store.get_state(conn)["sport_error:nfl"]


@pytest.mark.asyncio
async def test_account_wide_errors_still_raise_for_global_backoff(conn):
    from src.edge import worker
    from src.edge.client import OddsApiError
    from src.edge.config import NBA, NFL
    with pytest.raises(OddsApiError):
        await worker.tick(conn, _FakeOdds([], fail_nfl=401), _FakeBdl(), NOW, sports=[NBA, NFL])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py -k "both_sports or nfl_failure or account_wide"`
Expected: 3 failed (`tick() got an unexpected keyword argument 'sports'`).

- [ ] **Step 3: Implement**

In `backend/src/edge/worker.py`:
- Change the import to `from src.edge.config import LINE_MARKETS, SportConfig, enabled_sports, state_key`. `PROP_MARKETS` is no longer used here.
- Change `_reprice` to take `lines_window_hours: int` and pass it to `schedule.fresh(...)`.
- Restructure `tick` as follows. The bodies are the existing code, moved into `_tick_sport`, with the sport threaded through.

```python
async def tick(conn, client: EdgeOddsClient, bdl, now: datetime,
               sports: list[SportConfig] | None = None) -> dict:
    sports = enabled_sports(settings.edge_sports) if sports is None else sports
    state = store.get_state(conn)
    out: dict = {}

    current = int(state.get("guard_factor", "1"))
    factor = schedule.guard_factor(
        store.cost_last_24h(conn, now), current, settings.edge_credit_cap,
        int(state["credits_remaining"]) if state.get("credits_remaining") else None,
        _parse(state.get("guard_changed_at")), now)
    if factor != current:
        log.warning("edge credit guard factor %s -> %s", current, factor)
        store.set_state(conn, "guard_factor", str(factor), now)
        store.set_state(conn, "guard_changed_at", _iso(now), now)

    for sport in sports:
        try:
            out[sport.slug] = await _tick_sport(conn, client, bdl, sport, state, factor, now)
            store.set_state(conn, f"sport_error:{sport.slug}", "", now)
        except OddsApiError as e:
            if e.status in (401, 429):
                raise  # account-wide: the key is shared, back off globally
            conn.rollback()
            store.set_state(conn, f"sport_error:{sport.slug}", f"odds_api {e.status}: {e}"[:300], now)
        except Exception as e:  # noqa: BLE001 - one sport never blocks the other
            log.exception("edge %s tick failed", sport.slug)
            conn.rollback()
            store.set_state(conn, f"sport_error:{sport.slug}", redact(f"{type(e).__name__}: {e}")[:300], now)

    last_grade = _parse(state.get("last_grading"))
    if last_grade is None or now - last_grade >= timedelta(minutes=10):
        out["graded"] = grading.run_grading(conn, now)
        store.set_state(conn, "last_grading", _iso(now), now)

    out["pruned"] = prune_step(conn, now)

    # BallDontLie work never blocks odds: failures land in bdl_error.
    for step in (box_scores_step, backfill_step):
        try:
            await step(conn, bdl, now)
        except Exception as e:  # noqa: BLE001
            store.set_state(conn, "bdl_error", redact(f"{step.__name__}: {type(e).__name__}: {e}")[:300], now)

    return out


async def _tick_sport(conn, client, bdl, sport: SportConfig, state: dict, factor: int, now: datetime) -> dict:
    out: dict = {}
    window = sport.lines_window_hours

    # Events: free call, hourly.
    last_ev = _parse(state.get(state_key(sport, "last_events_refresh")))
    if last_ev is None or now - last_ev >= timedelta(minutes=60):
        events_json, _ = await client.get_events(sport)
        store.upsert_events(conn, [event_row(e) for e in events_json])
        store.drop_missing_events(conn, [e["id"] for e in events_json], now, sport=sport.key)
        store.set_state(conn, state_key(sport, "last_events_refresh"), _iso(now), now)

    events = store.upcoming_events(conn, now, hours=window, sport=sport.key)
    plan = schedule.plan_polls(events, now, factor, window)
    touched: list[str] = []

    if plan.lines:
        data, usage = await client.get_game_odds(sport)
        _record_usage(conn, "odds", usage, now)
        upcoming = {e["event_id"] for e in events}
        quotes = [q for ev in data if ev["id"] in upcoming for q in parse_event(ev, now)]
        out["line_history"] = store.write_quotes(conn, quotes)
        store.retire_missing(conn, sorted(upcoming), list(LINE_MARKETS), now)
        store.mark_polled(conn, sorted(upcoming), "last_lines_poll", now)
        touched += sorted(upcoming)

    by_id = {e["event_id"]: e for e in events}
    for eid in plan.prop_events:
        try:
            data, usage = await client.get_event_odds(eid, sport)
        except OddsApiError as e:
            if e.status in (401, 429):
                raise  # account-wide: global backoff
            if e.status in (404, 422):
                log.warning("edge event %s gone upstream (%s); dropping", eid, e.status)
                store.mark_unavailable(conn, [eid])
            else:
                log.error("edge props poll failed for %s: %s", eid, e)
                store.set_state(conn, "last_event_error", f"{eid}: {e}"[:300], now)
            continue  # other events still poll; reprice and grading still run
        _record_usage(conn, "event_odds", usage, now)
        quotes = parse_event(data, now)
        store.write_quotes(conn, quotes)
        store.retire_missing(conn, [eid], list(sport.prop_markets), now)
        store.mark_polled(conn, [eid], "last_props_poll", now)
        if sport.has_players:
            await players.resolve_new_players(conn, bdl, quotes, by_id[eid], now)
        touched.append(eid)

    out["opportunities"] = _reprice(conn, sorted(set(touched)), events, now, factor, window)

    # Scores while this sport's games are live.
    last_scores = _parse(state.get(state_key(sport, "last_scores_poll")))
    if grading.games_in_progress(conn, now, sport.key) and (
            last_scores is None or now - last_scores >= timedelta(minutes=30)):
        data, usage = await client.get_scores(days_from=1, sport=sport)
        _record_usage(conn, "scores", usage, now)
        grading.update_scores(conn, data)
        store.set_state(conn, state_key(sport, "last_scores_poll"), _iso(now), now)

    return out
```

Copy the props-loop body from the current `tick` exactly. The block above shows it as it stands at commit `HEAD`; only `eid, sport`, `sport.prop_markets` and the `has_players` guard are new. If the current body differs (read it first), keep its lines and apply only those three changes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py tests/integration/test_startup_migrations_db.py && python3 -m pytest -q tests/unit`
Expected: all pass, including the pre-existing `test_one_bad_event_does_not_stop_the_tick` and `test_events_missing_from_the_feed_drop_out`. Those call `tick` without `sports`; with `settings.edge_sports = "nba,nfl"` they now also run NFL against the fake, which returns no NFL events, so their assertions hold.

- [ ] **Step 5: Commit**

```bash
git add backend/src/edge/worker.py backend/tests/integration/test_edge_store_db.py
git commit -m "feat(edge): worker polls every enabled sport; one sport's failure never blocks another

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Sport-filtered views and `?sport=` on the API

**Files:**
- Modify: `backend/src/edge/views.py` (`games`, `ev_rows`, `odds`, `results`, `public_status`)
- Modify: `backend/src/edge/api.py`
- Test: `backend/tests/unit/test_edge_api.py` (append), `backend/tests/integration/test_edge_store_db.py` (append)

**Interfaces:**
- Consumes: Task 1 (`SPORTS`, `NBA`, `SportConfig`), Task 2 (window), Task 4 (`upcoming_events(..., sport=)`).
- Produces:
  - `games(conn, now, sport=NBA)`;
  - `ev_rows(conn, now, market=None, book=None, min_ev=None, event_id=None, sport=NBA)`;
  - `odds(conn, now, event_id, market, sport=NBA)`;
  - `results(conn, now, days, market, book, sport=NBA)`;
  - `public_status(conn, now, sport=NBA)`.
  - API query param `sport` (`nba`|`nfl`, default `nba`; anything else returns 422) on `/games`, `/ev`, `/odds/{id}`, `/results` and `/health`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/unit/test_edge_api.py`:

```python
def test_unknown_sport_is_422_and_sport_is_part_of_the_cache_key(monkeypatch):
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)
    seen = []
    monkeypatch.setattr(views, "games", lambda conn, now, sport: seen.append(sport.slug) or [])
    edge_api._cache.clear()
    edge_api._reset_limits()
    c = TestClient(app)
    assert c.get(f"{P}/games", params={"sport": "nhl"}).status_code == 422
    c.get(f"{P}/games")
    c.get(f"{P}/games", params={"sport": "nfl"})
    assert seen == ["nba", "nfl"]  # separate cache entries per sport
```

Append to `backend/tests/integration/test_edge_store_db.py`:

```python
def _seed_two_sports(conn):
    """One NBA and one NFL game 2 h out, each with a clear +EV over at fanduel."""
    store.upsert_events(conn, [_event(), _nfl_event()])
    def qs(eid, market, player):
        return [Quote(eid, market, player, side, 10.5, book, price, "https://x", NOW - timedelta(minutes=1))
                for book, over, under in (("pinnacle", -110, -110), ("novig", -110, -110),
                                          ("draftkings", -112, -108), ("fanduel", 140, -170))
                for side, price in (("over", over), ("under", under))]
    store.write_quotes(conn, qs("e1", "player_points", "Jayson Tatum") + qs("n1", "player_rush_yds", "Bijan Robinson"))


def test_views_never_mix_sports(conn):
    from src.edge import views
    from src.edge.config import NBA, NFL
    _seed_two_sports(conn)
    nba = views.ev_rows(conn, NOW, min_ev=0.0, sport=NBA)
    nfl = views.ev_rows(conn, NOW, min_ev=0.0, sport=NFL)
    assert nba and {r["event_id"] for r in nba} == {"e1"}
    assert nfl and {r["event_id"] for r in nfl} == {"n1"}
    assert all(r["bdl_player_id"] is None for r in nfl)
    assert nfl[0]["market_label"] == "Rush Yds"
    # an NFL event id under NBA returns nothing
    assert views.ev_rows(conn, NOW, min_ev=0.0, event_id="n1", sport=NBA) == []
    assert views.odds(conn, NOW, "n1", "player_rush_yds", sport=NBA)["rows"] == []
    assert views.odds(conn, NOW, "n1", "player_rush_yds", sport=NFL)["rows"]
    assert [g["event_id"] for g in views.games(conn, NOW, sport=NFL)] == ["n1"]
    assert [g["event_id"] for g in views.games(conn, NOW, sport=NBA)] == ["e1"]


def test_nfl_lines_two_days_out_are_priced(conn):
    from src.edge import views
    from src.edge.config import NFL
    store.upsert_events(conn, [_nfl_event(hours=48)])
    qs = [Quote("n1", "spreads", "", side, -3.5, book, price, None, NOW - timedelta(minutes=20))
          for book, h, a in (("pinnacle", -110, -110), ("novig", -110, -110), ("draftkings", -110, -110),
                             ("fanduel", 130, -160))
          for side, price in (("home", h), ("away", a))]
    store.write_quotes(conn, qs)
    rows = views.ev_rows(conn, NOW, min_ev=0.0, sport=NFL)
    assert any(r["market"] == "spreads" and r["book"] == "fanduel" for r in rows)


def test_results_are_per_sport(conn):
    from src.edge import views
    from src.edge.config import NBA, NFL
    store.upsert_events(conn, [_event(), _nfl_event()])
    with conn.cursor() as cur:
        for eid in ("e1", "n1"):
            cur.execute("""INSERT INTO edge_opportunities (event_id, market, player, side, line, book, price,
                           first_seen_at, last_seen_at, fair_prob, ev, max_ev, n_books, devig_method,
                           closing_done, clv) VALUES (%s, 'h2h', '', 'home', 0, 'fanduel', 110, %s, %s,
                           0.5, 0.05, 0.05, 5, 'power', TRUE, 0.01)""", (eid, NOW, NOW))
    conn.commit()
    assert views.results(conn, NOW, 30, None, None, sport=NBA)["n"] == 1
    assert views.results(conn, NOW, 30, None, None, sport=NFL)["n"] == 1
```

`Quote` and `NOW` are already imported or defined at the top of the integration file.

- [ ] **Step 2: Run tests to verify they fail**

Run:
- `cd backend && python3 -m pytest -q tests/unit/test_edge_api.py -k unknown_sport`
- `EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py -k "never_mix or two_days_out or per_sport"`

Expected: all fail with TypeErrors (unexpected keyword `sport`).

- [ ] **Step 3: Implement views**

In `backend/src/edge/views.py`, import `from src.edge.config import NBA, SportConfig`, then:

```python
def games(conn, now: datetime, sport: SportConfig = NBA) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT e.event_id, e.home_team, e.away_team, e.commence_time, e.last_lines_poll,
                              e.last_props_poll, COUNT(DISTINCT (q.player, q.market)) FILTER
                              (WHERE q.player <> '') AS prop_markets
                       FROM edge_events e LEFT JOIN edge_quotes q USING (event_id)
                       WHERE e.commence_time > %s AND e.commence_time <= %s AND e.sport = %s
                         AND e.status <> 'unavailable'
                       GROUP BY e.event_id ORDER BY e.commence_time""",
                    (now, now + timedelta(hours=sport.lines_window_hours), sport.key))
        return [dict(r) for r in cur.fetchall()]
```

`ev_rows` gains `sport: SportConfig = NBA` as its last parameter. The changed lines:

```python
    events = {e["event_id"]: e for e in store.upcoming_events(
        conn, now, hours=sport.lines_window_hours, sport=sport.key)}
    if event_id and event_id not in events:
        return []  # another sport's (or a past) event
    ids = [event_id] if event_id else list(events)
    ...
    quotes = schedule.fresh(store.load_quotes(conn, ids, market), commence, now, _factor(conn),
                            sport.lines_window_hours)
    ...
    player_ids = _player_ids(conn) if sport.has_players else {}
```

`odds` gains `sport: SportConfig = NBA`. Select the event with `WHERE event_id = %s AND sport = %s` (params `(event_id, sport.key)`). Pass `sport.lines_window_hours` to `schedule.fresh`.

`results` gains `sport: SportConfig = NBA`. Its SQL becomes:

```python
    sql = """SELECT o.event_id, o.market, o.player, o.line, o.side, o.book, o.price, o.clv, o.result,
                    o.profit_units, o.first_seen_at
             FROM edge_opportunities o JOIN edge_events e USING (event_id)
             WHERE o.closing_done AND o.first_seen_at >= %s AND e.sport = %s"""
    params: list = [now - timedelta(days=days), sport.key]
    if market:
        sql += " AND o.market = %s"
        params.append(market)
    if book:
        sql += " AND o.book = %s"
        params.append(book)
```

`public_status` gains `sport: SportConfig = NBA`:

```python
def public_status(conn, now: datetime, sport: SportConfig = NBA) -> dict:
    """The banner's view: no credits, no errors, nothing an outsider can use."""
    s = status(conn, now)
    age = s["heartbeat_age_s"]
    sport_error = store.get_state(conn).get(f"sport_error:{sport.slug}")
    return {"delayed": age is None or age > 600 or bool(s["last_error"]) or bool(sport_error)}
```

- [ ] **Step 4: Implement the API**

In `backend/src/edge/api.py`, import `from src.edge.config import SPORTS, SportConfig` and add:

```python
def _sport(sport: str = Query("nba")) -> SportConfig:
    if sport not in SPORTS:
        raise HTTPException(status_code=422, detail=f"unknown sport {sport!r}; one of {sorted(SPORTS)}")
    return SPORTS[sport]
```

Change the routes (cache keys gain `sp.slug`):

```python
@router.get("/games")
def get_games(sp: SportConfig = Depends(_sport), tier: str = Depends(access_tier)) -> dict:
    return {"games": _cached(("games", sp.slug), lambda c, now: views.games(c, now, sp)), "access_tier": tier}


@router.get("/ev")
def get_ev(market: str | None = None, book: str | None = None,
           min_ev: float | None = Query(None, ge=0, le=1), event_id: str | None = None,
           sp: SportConfig = Depends(_sport), tier: str = Depends(access_tier)) -> dict:
    rows = _cached(("ev", sp.slug, market, book, min_ev, event_id), lambda c, now: views.ev_rows(
        c, now, market=market, book=book, min_ev=min_ev, event_id=event_id, sport=sp))
    return {"rows": rows, "access_tier": tier}


@router.get("/odds/{event_id}")
def get_odds(event_id: str, market: str = "h2h", sp: SportConfig = Depends(_sport)) -> dict:
    return _cached(("odds", sp.slug, event_id, market), lambda c, now: views.odds(c, now, event_id, market, sp))


@router.get("/results")
def get_results(days: int = Query(365, ge=1, le=730), market: str | None = None,
                book: str | None = None, sp: SportConfig = Depends(_sport)) -> dict:
    return _cached(("results", sp.slug, days, market, book),
                   lambda c, now: views.results(c, now, days, market, book, sp))


@router.get("/health")
def get_public_health(sp: SportConfig = Depends(_sport)) -> dict:
    return _cached(("health", sp.slug), lambda c, now: views.public_status(c, now, sp))
```

The existing unit test `test_ev_route_uses_views_and_cache` monkeypatches `views.ev_rows` with `lambda conn, now, **kw`, which still accepts `sport=` as a keyword. Any other existing test lambda that breaks on the new positional `sport` argument gets an extra parameter. That is a test-only change; record it in the ledger.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && python3 -m pytest -q tests/unit && EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backend/src/edge/views.py backend/src/edge/api.py backend/tests/unit/test_edge_api.py backend/tests/integration/test_edge_store_db.py
git commit -m "feat(edge): sport parameter on the Edge API; views never mix sports

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Frontend `/nfl/edge` and odds screen

**Files:**
- Modify: `frontend/src/lib/edgeFormat.ts`, `frontend/src/lib/edgeFormat.test.ts`
- Modify: `frontend/src/lib/edgeApi.ts`, `frontend/src/hooks/useEdge.ts`
- Modify: `frontend/src/pages/EdgeFinder.tsx`, `frontend/src/pages/EdgeOdds.tsx`
- Modify: `frontend/src/components/edge/EvTable.tsx`, `frontend/src/components/edge/EdgeStatusBanner.tsx`
- Modify: `frontend/src/App.tsx`, `frontend/src/components/Layout.tsx`

**Interfaces:**
- Consumes: the API `sport` param (Task 6).
- Produces:
  - `type Sport = 'nba' | 'nfl'`;
  - `marketKeysFor(sport: Sport): string[]`;
  - `edgeBase(sport: Sport): string` (`'/edge'` | `'/nfl/edge'`);
  - routes `/nfl/edge` and `/nfl/edge/odds/:eventId`.

- [ ] **Step 1: Write the failing test**

Append to `frontend/src/lib/edgeFormat.test.ts`:

```ts
import { edgeBase, marketKeysFor } from './edgeFormat';

describe('sports', () => {
  it('labels NFL markets', () => {
    expect(marketLabel('player_rush_reception_yds')).toBe('Rush+Rec Yds');
    expect(marketLabel('player_pass_interceptions')).toBe('INTs');
  });
  it('lists each sport\'s own markets, lines first', () => {
    expect(marketKeysFor('nfl')).toEqual(['h2h', 'spreads', 'totals', 'player_pass_yds', 'player_pass_tds',
      'player_pass_completions', 'player_pass_attempts', 'player_pass_interceptions', 'player_rush_yds',
      'player_rush_attempts', 'player_receptions', 'player_reception_yds', 'player_rush_reception_yds']);
    expect(marketKeysFor('nba')).toContain('player_points');
    expect(marketKeysFor('nba')).not.toContain('player_pass_yds');
  });
  it('routes each sport under its own base', () => {
    expect(edgeBase('nba')).toBe('/edge');
    expect(edgeBase('nfl')).toBe('/nfl/edge');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run`
Expected: FAIL (`edgeBase` / `marketKeysFor` are not exported).

- [ ] **Step 3: Implement format, API and hooks**

`frontend/src/lib/edgeFormat.ts`: add the NFL labels to `MARKETS`, and add the sport helpers. `MARKET_KEYS` stays exported with its current (NBA) value so nothing else breaks.

```ts
const NBA_MARKETS = ['h2h', 'spreads', 'totals', 'player_points', 'player_rebounds', 'player_assists',
  'player_threes', 'player_points_rebounds_assists', 'player_points_rebounds', 'player_points_assists',
  'player_rebounds_assists'];
const NFL_MARKETS = ['h2h', 'spreads', 'totals', 'player_pass_yds', 'player_pass_tds',
  'player_pass_completions', 'player_pass_attempts', 'player_pass_interceptions', 'player_rush_yds',
  'player_rush_attempts', 'player_receptions', 'player_reception_yds', 'player_rush_reception_yds'];
export type Sport = 'nba' | 'nfl';
export const marketKeysFor = (sport: Sport) => (sport === 'nfl' ? NFL_MARKETS : NBA_MARKETS);
export const edgeBase = (sport: Sport) => (sport === 'nfl' ? '/nfl/edge' : '/edge');
```

Add to the `MARKETS` record:

```ts
  player_pass_yds: 'Pass Yds', player_pass_tds: 'Pass TDs', player_pass_completions: 'Completions',
  player_pass_attempts: 'Pass Att', player_pass_interceptions: 'INTs', player_rush_yds: 'Rush Yds',
  player_rush_attempts: 'Rush Att', player_receptions: 'Receptions', player_reception_yds: 'Rec Yds',
  player_rush_reception_yds: 'Rush+Rec Yds',
```

`frontend/src/lib/edgeApi.ts`: import `type Sport` from `./edgeFormat`. `games`, `ev`, `odds`, `results` and `health` take a trailing `sport: Sport = 'nba'` and add `sport` to `params`:

```ts
  async games(sport: Sport = 'nba'): Promise<EdgeGame[]> {
    return (await client.get<{ games: EdgeGame[] }>('/edge/games', { params: { sport } })).data.games;
  },
  async ev(params: EvParams = {}, sport: Sport = 'nba'): Promise<EvRow[]> {
    return (await client.get<{ rows: EvRow[] }>('/edge/ev', { params: { ...params, sport } })).data.rows;
  },
  async odds(eventId: string, market: string, sport: Sport = 'nba'): Promise<OddsResponse> {
    return (await client.get<OddsResponse>(`/edge/odds/${eventId}`, { params: { market, sport } })).data;
  },
  async results(days = 365, sport: Sport = 'nba'): Promise<ResultsResponse> {
    return (await client.get<ResultsResponse>('/edge/results', { params: { days, sport } })).data;
  },
  async health(sport: Sport = 'nba'): Promise<{ delayed: boolean }> {
    return (await client.get<{ delayed: boolean }>('/edge/health', { params: { sport } })).data;
  },
```

`frontend/src/hooks/useEdge.ts`: the hooks take `sport: Sport = 'nba'`, and it is part of every query key:

```ts
export const useEdgeGames = (sport: Sport = 'nba') =>
  useQuery({ queryKey: ['edge', sport, 'games'], queryFn: () => edgeApi.games(sport), refetchInterval: 60000 });
export const useEdgeEv = (params: EvParams, sport: Sport = 'nba') =>
  useQuery({ queryKey: ['edge', sport, 'ev', params], queryFn: () => edgeApi.ev(params, sport), refetchInterval: 30000 });
export const useEdgeOdds = (eventId: string, market: string, sport: Sport = 'nba') =>
  useQuery({ queryKey: ['edge', sport, 'odds', eventId, market], queryFn: () => edgeApi.odds(eventId, market, sport),
    refetchInterval: 30000, enabled: !!eventId });
export const useEdgeResults = (days = 365, sport: Sport = 'nba') =>
  useQuery({ queryKey: ['edge', sport, 'results', days], queryFn: () => edgeApi.results(days, sport), staleTime: 300000 });
export const useEdgeHealth = (sport: Sport = 'nba') =>
  useQuery({ queryKey: ['edge', sport, 'health'], queryFn: () => edgeApi.health(sport), refetchInterval: 60000 });
```

`useEdgePlayer` is unchanged.

- [ ] **Step 4: Implement pages, routes and nav**

- **`EdgeStatusBanner`:** `export function EdgeStatusBanner({ sport = 'nba' }: { sport?: Sport })` and `useEdgeHealth(sport)`.
- **`EvTable`:**
  - Props become `{ rows, playerIds = {}, sport = 'nba' }`.
  - The odds link becomes ``to={`${edgeBase(sport)}/odds/${r.event_id}?market=${r.market}`}``.
  - Player links render only when `sport === 'nba' && playerIds[r.player]`.
- **`EdgeFinder`:**
  - Signature `export function EdgeFinder({ limit, sport = 'nba' }: { limit?: number; sport?: Sport })`.
  - Pass `sport` to `useEdgeGames(sport)`, `useEdgeEv({...}, sport)`, `<EdgeStatusBanner sport={sport} />` and `<EvTable ... sport={sport} />`.
  - The market `<select>` maps `marketKeysFor(sport)` instead of `MARKET_KEYS`.
  - The heading reads `{sport === 'nfl' ? 'NFL +EV Finder' : '+EV Finder'}`.
  - Under the description paragraph, when `sport === 'nfl'`, render `<p className="text-xs text-[#64748b] mt-1 font-mono">NFL player pages and prop results are coming next week.</p>`.
- **`EdgeOdds`:**
  - Signature `export function EdgeOdds({ sport = 'nba' }: { sport?: Sport })`.
  - `useEdgeOdds(eventId, market, sport)`, `<EdgeStatusBanner sport={sport} />`.
  - The market buttons map `marketKeysFor(sport)`.
- **`App.tsx`:** add the two routes next to the existing `/nfl` routes:
  ```tsx
  <Route path="/nfl/edge" element={<EdgeFinder sport="nfl" />} />
  <Route path="/nfl/edge/odds/:eventId" element={<EdgeOdds sport="nfl" />} />
  ```
- **`Layout.tsx`:** the NFL pages become:
  ```ts
  pages: [
    { path: '/nfl', label: 'Picks' },
    { path: '/nfl/edge', label: '+EV' },
    { path: '/nfl/performance', label: 'Results' },
  ],
  ```

- [ ] **Step 5: Run tests, type check and build**

Run: `cd frontend && npx vitest run && npx tsc --noEmit && npm run build`
Expected: vitest all pass (9 tests); tsc exits 0; build succeeds.

- [ ] **Step 6: Commit**

```bash
git add frontend/src
git commit -m "feat(frontend): NFL +EV finder and odds screen at /nfl/edge

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Ship and verify live (Sat 2026-10-18)

**Files:** none (deploy and verification). Results go in the ledger.

**Interfaces:**
- Consumes: everything above. Production `edge-worker` service `e43e039f-a2db-4149-99dd-60c812ccda67`, project `ba72d794-a84c-4b39-b498-1f11a83f0ef4`. The cron key is in `Truline/.cron_api_key`.

- [ ] **Step 1: Full verification on the branch**

Run:
- `cd backend && python3 -m pytest -q tests/unit`
- `EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py tests/integration/test_startup_migrations_db.py`
- `cd ../frontend && npx vitest run && npx tsc --noEmit && npm run build`

Expected: all green. (`tests/integration/test_nfl_*` need `localhost:5432` and the old key; they are environmental and not part of this gate.)

- [ ] **Step 2: Merge and deploy (with Samir's go-ahead)**

`git checkout main && git merge --ff-only edge-nfl-a && git push origin main`. This deploys `nba-value`, `edge-worker` and Vercel. Wait until `curl -s "https://nba-value-production.up.railway.app/api/v1/edge/games?sport=nfl"` returns 200 with a `games` list.

- [ ] **Step 3: First NFL tick**

Check the `edge-worker` logs (Railway MCP `get-logs`, service `e43e039f…`) for an `edge tick` line containing `'nfl'`. Then run `curl -s -H "X-Cron-Key: $(cat .cron_api_key)" .../api/v1/edge/status`.

Expected: no `sport_error:nfl`; `credits_last_24h` rising by about 20 per NFL props poll.

- [ ] **Step 4: Live check**

1. `curl -s ".../api/v1/edge/ev?sport=nfl&min_ev=0"` returns rows only for NFL matchups.
2. `curl -s ".../api/v1/edge/ev?min_ev=0"` (NBA) contains no NFL teams.
3. Open `https://www.truline.app/nfl/edge` and an odds screen. No console errors.
4. Spot-check 10 NFL +EV rows against the books' own sites (price and line). Record the matches in the ledger.
5. Credits: `projected_monthly` on `/edge/status` stays at or below about 1.2M.

- [ ] **Step 5: Hand-off**

Report the live-check results. Next up: Phase B (NFL player pages + prop grading) and Phase C (backtest), each starting with its own spec.
