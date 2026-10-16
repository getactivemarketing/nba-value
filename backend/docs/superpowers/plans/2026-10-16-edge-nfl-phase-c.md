# TruLine Edge NFL Phase C (Historical Backtest) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A resumable offline CLI (`python -m src.edge.backtest fetch|replay|report`) that buys 2023–2025 NFL historical odds snapshots once, replays them through the live +EV engine, grades every pick with nflverse data, and writes a tuning report.

**Architecture:** New package `src/edge/backtest/` with five focused modules:
- `nflverse` (grading data);
- `plan` (snapshot jobs and costs, pure);
- `fetch` (Odds API historical calls into `edge_bt_snapshots`, with budget stops);
- `replay` (live engine functions into `edge_bt_picks`);
- `report` (markdown).

Raw responses are stored once, so re-tuning costs 0 credits. Nothing writes to live tables or live worker state.

**Tech Stack:** Python 3.11, psycopg2, SQLAlchemy models (create_all), httpx (existing `EdgeOddsClient`), pandas, nfl_data_py, pytest.

**Spec:** `backend/docs/superpowers/specs/2026-10-16-edge-nfl-phase-c-design.md`

## Global Constraints

- **Seasons:** 2023, 2024 and 2025, regular season only (`game_type == "REG"` / `season_type == "REG"`).
- **Snapshot labels:** `t24h` = kickoff − 24 h, `t6h` = kickoff − 6 h, `t1h` = kickoff − 1 h, `close` = kickoff − 5 min. Picks are flagged only at `t24h`/`t6h`/`t1h`; `close` only supplies CLV.
- **Request contents:** props calls request all 10 `NFL_PROP_MARKETS` and lines calls request `h2h,spreads,totals`. Both use the 16 `config.BOOKS` and `oddsFormat=american`.
- **Planned costs:** events 1 credit; props 10 × 2 × 10 = **200**; lines 10 × 2 × 3 = **60**. Accounting uses the actual `x-requests-last` when the API sends it.
- **Budget stops:** `BT_CREDIT_CAP` (env, default **800000**) and `BT_MIN_REMAINING` (env, default **3600000**).
  - Before each call: stop if stored credits plus the planned cost would exceed the cap, or if the last seen `x-requests-remaining` is below the floor.
- **Error handling:** 401/429 stops the run. 404 stores an empty stub (0 credits) so the call isn't repeated. Any other error is logged and skipped, then retried by the next run.
- **Pacing:** at most 2 requests per second (`min_interval=0.5`).
- **Live isolation:** the backtest never calls `store.log_api_call` or `store.set_state` and never writes `edge_quotes`, `edge_quote_history`, `edge_opportunities`, `edge_api_calls` or `edge_worker_state`.
- **Engine reuse:** use `ingest.parse_event`, `engine.price_quotes`/`fair_prob`/`group`/`clv`/`american_to_decimal`, `grading.grade_game`/`profit_units`, `nfl_players.nfl_prop_result` and `players.normalize_name` unchanged. Do not copy or edit them.
- **EV settings:** from `settings.edge_min_ev` (0.02) and `settings.edge_verify_ev` (0.15).
- **Report:** written to `backend/docs/backtests/nfl-2023-2025.md`.
  - EV buckets: `2–4%`, `4–6%`, `6%+`.
  - "small sample" when picks < 50.
  - ROI 95% CI from the normal approximation on per-pick profit.
  - Market and book recommendations require average CLV > 0 and ≥ 200 picks.
- **The API key is never logged:** the CLI entry point (`if __name__ == "__main__"`) calls `install_redaction()` before `main()`. Not inside `main()`, because it wraps stdout process-wide and would break pytest capture.
- **Commit trailer:** `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Rulings against the spec (made while planning; carry into the ledger)

- **§8, grading data:** weekly stats come from the nflverse `stats_player_week_{season}.parquet` release for **all** seasons, not just as a 2025 fallback.
  - **Why:** `nfl_data_py.import_weekly_data([2025])` returns HTTP 404 (checked 2026-10-16). The release file loads 2023 and 2025, and its `game_id` matches schedules and snap counts exactly.
  - **Benefit:** one source keeps grading uniform across seasons.
- **§6, schema:**
  - `edge_bt_snapshots` gains `game_id` (nflverse id) so replay joins to schedules without guessing.
  - `edge_bt_picks.player` is `''` for game lines, not NULL, so the unique key works. This matches the `Quote` convention.
  - `edge_bt_picks` gains `verify` (EV > 15%, as the live badge) and `pin_ev`, which supports the §7 "Pinnacle-only consensus" re-price.
  - Events-list calls are stored as `kind='events'` rows (`event_id='day:YYYY-MM-DD'`, `label='list'`), so their credits count toward the cap and resume skips them.
- **§6, lines credits:** a lines call's credits go on the first stored row of its slate, and every other row gets 0, so `SUM(credits)` is exact. Events missing from a lines response get a stub row (`bookmakers: []`), so the job counts as done.
- **§11, de-duplication:** a pick flagged at more than one lead time counts **once** (at its earliest label) in every table except the lead-time tables, which count each label on its own. Otherwise overall ROI would count the same bet three times.

## Review Focus

1. **Odds API team names whose nflverse abbreviation differs** (Rams `LA`, Commanders `WAS`), and kickoffs that drift from the schedule time: they must still match their event. The test is in Task 2.
2. **A `close` snapshot timestamped after kickoff:** it must give no CLV, never a CLV from in-game prices. The test is in Task 4.
3. **A fetch that crashes partway and is re-run:** no snapshot is paid for twice, and a lines job with an event missing from its response still counts as done. The test is in Task 3.
4. **A pick flagged at `t24h`, `t6h` and `t1h`:** it counts once in the overall, season, market, book and EV tables. The test is in Task 5.
5. **Player names with suffixes or punctuation** ("Kenneth Walker III", "D.J. Moore", "Ja'Marr Chase"): they must match nflverse display names, never void wrongly. The test is in Task 4.

---

## File structure

| File | Responsibility |
|---|---|
| `src/edge/backtest/__init__.py` | Package marker and `SEASONS = (2023, 2024, 2025)` |
| `src/edge/backtest/nflverse.py` | `Game`; schedule, stats and snaps frames → indexes; network loaders |
| `src/edge/backtest/plan.py` | Labels, costs, `Job`, `day_jobs`, `match_events`, `snapshot_jobs`, `dry_run` |
| `src/edge/backtest/fetch.py` | `Repo` (SQL), `run_fetch`, `ensure_tables` |
| `src/edge/backtest/replay.py` | `Snap`, `grade`, `replay_game` (pure), `run_replay` (DB) |
| `src/edge/backtest/report.py` | `metrics`, `build_report` (pure), `run_report` (DB) |
| `src/edge/backtest/__main__.py` | argparse CLI |
| `src/edge/client.py` | + 3 historical methods |
| `src/models/edge.py`, `src/models/__init__.py` | + `EdgeBtSnapshot`, `EdgeBtPick` |
| `tests/unit/test_edge_backtest_{nflverse,plan,fetch,replay,report,cli}.py` | unit tests |
| `tests/integration/test_edge_backtest_db.py` | real-Postgres round trip |

Test commands, run from `backend/`:
- **Unit:** `python3 -m pytest -q tests/unit`
- **DB:** `EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py tests/integration/test_startup_migrations_db.py tests/integration/test_edge_backtest_db.py`

---

### Task 1: nflverse grading data and backtest tables

**Files:**
- Create: `src/edge/backtest/__init__.py`, `src/edge/backtest/nflverse.py`
- Modify: `src/models/edge.py` (append two models), `src/models/__init__.py` (import them)
- Test: `tests/unit/test_edge_backtest_nflverse.py`, `tests/unit/test_edge_models.py`

**Interfaces:**
- Produces:
  - `nflverse.Game(game_id: str, season: int, week: int, home: str, away: str, kickoff: datetime, home_score: int | None, away_score: int | None)`, frozen. `home`/`away` use BallDontLie abbreviations, matching `teams.NFL_ABBR_BY_NAME` values. `kickoff` is in UTC.
  - `nflverse.schedule_games(frame) -> list[Game]`.
  - `nflverse.stats_index(frame) -> dict[str, list[dict]]`: maps game_id to rows `{"name": str, "team": str, pass_att, pass_cmp, pass_yds, pass_tds, pass_int, rush_att, rush_yds, receptions, rec_yds}`.
  - `nflverse.snaps_index(frame) -> dict[str, set[str]]`: maps game_id to the normalized names with snaps > 0.
  - `nflverse.load_schedule(seasons) -> list[Game]`, `load_stats(seasons) -> dict[str, list[dict]]` and `load_snaps(seasons) -> dict[str, set[str]]`. These make network calls.
  - `EdgeBtSnapshot` (`edge_bt_snapshots`) and `EdgeBtPick` (`edge_bt_picks`) models.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_edge_backtest_nflverse.py`:

```python
from datetime import datetime, timezone

import pandas as pd

from src.edge.backtest import SEASONS, nflverse


def test_seasons():
    assert SEASONS == (2023, 2024, 2025)


def test_schedule_games_converts_et_to_utc_maps_teams_and_skips_playoffs():
    frame = pd.DataFrame([
        {"game_id": "2023_10_CLE_BAL", "season": 2023, "game_type": "REG", "week": 10, "gameday": "2023-11-12",
         "gametime": "13:00", "home_team": "BAL", "away_team": "CLE", "home_score": 31.0, "away_score": 33.0},
        {"game_id": "2023_10_WAS_LA", "season": 2023, "game_type": "REG", "week": 10, "gameday": "2023-11-13",
         "gametime": "20:15", "home_team": "LA", "away_team": "WAS", "home_score": float("nan"),
         "away_score": float("nan")},
        {"game_id": "2023_19_X_Y", "season": 2023, "game_type": "WC", "week": 19, "gameday": "2024-01-13",
         "gametime": "16:30", "home_team": "KC", "away_team": "MIA", "home_score": 26.0, "away_score": 7.0},
    ])
    games = nflverse.schedule_games(frame)
    assert [g.game_id for g in games] == ["2023_10_CLE_BAL", "2023_10_WAS_LA"]
    a, b = games
    assert a.kickoff == datetime(2023, 11, 12, 18, 0, tzinfo=timezone.utc)
    assert (a.home, a.away, a.home_score, a.away_score) == ("BAL", "CLE", 31, 33)
    assert b.kickoff == datetime(2023, 11, 14, 1, 15, tzinfo=timezone.utc)  # Monday 20:15 ET
    assert (b.home, b.away, b.home_score, b.away_score) == ("LAR", "WSH", None, None)


def test_stats_index_maps_columns_normalizes_names_and_keeps_regular_season():
    frame = pd.DataFrame([
        {"game_id": "g1", "season_type": "REG", "player_display_name": "Ja'Marr Chase", "team": "CIN",
         "attempts": 0, "completions": 0, "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
         "carries": 1, "rushing_yards": 4.0, "receptions": 7, "receiving_yards": 81.0},
        {"game_id": "g1", "season_type": "REG", "player_display_name": "Kenneth Walker III", "team": "SEA",
         "attempts": 0, "completions": 0, "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
         "carries": float("nan"), "rushing_yards": 55, "receptions": 1, "receiving_yards": 6},
        {"game_id": "g2", "season_type": "POST", "player_display_name": "Joe Burrow", "team": "CIN",
         "attempts": 30, "completions": 20, "passing_yards": 250, "passing_tds": 2, "passing_interceptions": 1,
         "carries": 2, "rushing_yards": 5, "receptions": 0, "receiving_yards": 0},
    ])
    idx = nflverse.stats_index(frame)
    assert set(idx) == {"g1"}
    chase, walker = idx["g1"]
    assert chase == {"name": "jamarr chase", "team": "CIN", "pass_att": 0, "pass_cmp": 0, "pass_yds": 0,
                     "pass_tds": 0, "pass_int": 0, "rush_att": 1, "rush_yds": 4, "receptions": 7, "rec_yds": 81}
    assert walker["name"] == "kenneth walker" and walker["rush_att"] == 0 and walker["rush_yds"] == 55


def test_snaps_index_keeps_players_with_snaps_only():
    frame = pd.DataFrame([
        {"game_id": "g1", "game_type": "REG", "player": "D.J. Moore", "offense_snaps": 60.0,
         "defense_snaps": 0.0, "st_snaps": 0.0},
        {"game_id": "g1", "game_type": "REG", "player": "Bench Guy", "offense_snaps": 0.0,
         "defense_snaps": 0.0, "st_snaps": 0.0},
        {"game_id": "g9", "game_type": "POST", "player": "X", "offense_snaps": 10.0,
         "defense_snaps": 0.0, "st_snaps": 0.0},
    ])
    assert nflverse.snaps_index(frame) == {"g1": {"dj moore"}}
```

Append to `tests/unit/test_edge_models.py`:

```python
def test_backtest_tables_registered_with_keys():
    t = Base.metadata.tables
    assert ("event_id", "kind", "label") in _uniques(t["edge_bt_snapshots"])
    assert ("run_id", "event_id", "label", "market", "player", "side", "line", "book") in _uniques(t["edge_bt_picks"])
    assert t["edge_bt_snapshots"].c.payload.type.__class__.__name__ == "JSONB"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -q tests/unit/test_edge_backtest_nflverse.py tests/unit/test_edge_models.py`
Expected: FAIL. `ModuleNotFoundError: No module named 'src.edge.backtest'`, and a `KeyError: 'edge_bt_snapshots'`.

- [ ] **Step 3: Implement**

`src/edge/backtest/__init__.py`:

```python
"""TruLine Edge NFL historical backtest (spec 2026-10-16-edge-nfl-phase-c-design.md)."""
SEASONS = (2023, 2024, 2025)
```

`src/edge/backtest/nflverse.py`:

```python
"""nflverse data for the backtest: schedules, weekly player stats and snap counts.

Weekly stats come from the `stats_player` release file for every season: the
nfl_data_py weekly loader 404s for 2025 (checked 2026-10-16), and one source
for all three seasons keeps grading uniform.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from src.edge.players import normalize_name
from src.edge.teams import NFLVERSE_TO_BDL

ET = ZoneInfo("America/New_York")
STATS_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
             "stats_player/stats_player_week_{season}.parquet")
STAT_MAP = {"attempts": "pass_att", "completions": "pass_cmp", "passing_yards": "pass_yds",
            "passing_tds": "pass_tds", "passing_interceptions": "pass_int", "carries": "rush_att",
            "rushing_yards": "rush_yds", "receptions": "receptions", "receiving_yards": "rec_yds"}


@dataclass(frozen=True)
class Game:
    game_id: str
    season: int
    week: int
    home: str              # BallDontLie abbreviation, as teams.NFL_ABBR_BY_NAME
    away: str
    kickoff: datetime      # UTC
    home_score: int | None
    away_score: int | None


def _team(abbr: str) -> str:
    return NFLVERSE_TO_BDL.get(abbr, abbr)


def _int(v) -> int:
    return 0 if pd.isna(v) else int(v)


def _score(v) -> int | None:
    return None if pd.isna(v) else int(v)


def schedule_games(frame) -> list[Game]:
    out = []
    for r in frame.itertuples(index=False):
        if r.game_type != "REG":
            continue
        kickoff = (datetime.strptime(f"{r.gameday} {r.gametime}", "%Y-%m-%d %H:%M")
                   .replace(tzinfo=ET).astimezone(timezone.utc))
        out.append(Game(r.game_id, int(r.season), int(r.week), _team(r.home_team), _team(r.away_team),
                        kickoff, _score(r.home_score), _score(r.away_score)))
    return out


def stats_index(frame) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in frame.to_dict("records"):
        if r["season_type"] != "REG":
            continue
        row = {"name": normalize_name(r["player_display_name"]), "team": _team(r["team"])}
        row.update({col: _int(r[src]) for src, col in STAT_MAP.items()})
        out.setdefault(r["game_id"], []).append(row)
    return out


def snaps_index(frame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for r in frame.itertuples(index=False):
        if r.game_type != "REG":
            continue
        if _int(r.offense_snaps) + _int(r.defense_snaps) + _int(r.st_snaps) > 0:
            out.setdefault(r.game_id, set()).add(normalize_name(r.player))
    return out


def load_schedule(seasons) -> list[Game]:
    import nfl_data_py
    return schedule_games(nfl_data_py.import_schedules(list(seasons)))


def load_stats(seasons) -> dict[str, list[dict]]:
    return stats_index(pd.concat([pd.read_parquet(STATS_URL.format(season=s)) for s in seasons]))


def load_snaps(seasons) -> dict[str, set[str]]:
    import nfl_data_py
    return snaps_index(nfl_data_py.import_snap_counts(list(seasons)))
```

Append to `src/models/edge.py`. Add `from sqlalchemy import func` and `from sqlalchemy.dialects.postgresql import JSONB` to its imports.

```python
class EdgeBtSnapshot(Base):
    """Phase C backtest: one raw Odds API historical response per (event, kind, label). Never read by live code."""
    __tablename__ = "edge_bt_snapshots"
    __table_args__ = (UniqueConstraint("event_id", "kind", "label", name="uq_edge_bt_snapshots"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    game_id: Mapped[str | None] = mapped_column(String(32), index=True)   # nflverse id; NULL for events rows
    event_id: Mapped[str] = mapped_column(String(64), nullable=False)    # Odds API id, or 'day:YYYY-MM-DD'
    kind: Mapped[str] = mapped_column(String(8), nullable=False)         # events | props | lines
    label: Mapped[str] = mapped_column(String(8), nullable=False)        # t24h | t6h | t1h | close | list
    kickoff: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    snapshot_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    home: Mapped[str] = mapped_column(String(4), nullable=False, default="")
    away: Mapped[str] = mapped_column(String(4), nullable=False, default="")
    credits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class EdgeBtPick(Base):
    """Phase C backtest: one +EV pick found by a replay run."""
    __tablename__ = "edge_bt_picks"
    __table_args__ = (UniqueConstraint("run_id", "event_id", "label", "market", "player", "side", "line", "book",
                                       name="uq_edge_bt_picks"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str] = mapped_column(String(8), nullable=False)
    market: Mapped[str] = mapped_column(String(40), nullable=False)
    player: Mapped[str] = mapped_column(String(80), nullable=False, default="")  # '' for game lines
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    line: Mapped[float] = mapped_column(Numeric(7, 2), nullable=False)
    book: Mapped[str] = mapped_column(String(24), nullable=False)
    price: Mapped[int] = mapped_column(Integer, nullable=False)
    fair_prob: Mapped[float] = mapped_column(Float, nullable=False)
    ev: Mapped[float] = mapped_column(Float, nullable=False)
    verify: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    pin_ev: Mapped[float | None] = mapped_column(Float)          # EV with Novig/ProphetX left out of the consensus
    n_books: Mapped[int] = mapped_column(Integer, nullable=False)
    sharps: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    closing_prob: Mapped[float | None] = mapped_column(Float)
    clv: Mapped[float | None] = mapped_column(Float)
    result: Mapped[str | None] = mapped_column(String(8))
    profit: Mapped[float | None] = mapped_column(Float)
```

In `src/models/__init__.py`, add `EdgeBtPick, EdgeBtSnapshot` to the `from src.models.edge import (...)` list, and to `__all__` if the edge models are listed there.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -q tests/unit/test_edge_backtest_nflverse.py tests/unit/test_edge_models.py`
Expected: PASS (all).

- [ ] **Step 5: Live data smoke check (network, no credits)**

Run:

```bash
python3 -c "
from src.edge.backtest import SEASONS, nflverse
g = nflverse.load_schedule(SEASONS); s = nflverse.load_stats(SEASONS); n = nflverse.load_snaps(SEASONS)
print(len(g), sum(1 for x in g if x.game_id in s), sum(1 for x in g if x.game_id in n))" 2>&1 | grep -v Downcast
```

Expected: `816 816 816`. The second and third numbers must equal 816: every regular-season game has stats and snaps. If not, record the shortfall as a ruling.

- [ ] **Step 6: Run the unit suite and commit**

Run: `python3 -m pytest -q tests/unit`. Expected: all pass (722 + new).

```bash
git add src/edge/backtest/__init__.py src/edge/backtest/nflverse.py src/models/edge.py src/models/__init__.py \
        tests/unit/test_edge_backtest_nflverse.py tests/unit/test_edge_models.py
git commit -m "feat(edge-bt): nflverse grading data and backtest tables

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Snapshot planning (labels, event matching, slates, cost)

**Files:**
- Create: `src/edge/backtest/plan.py`
- Test: `tests/unit/test_edge_backtest_plan.py`

**Interfaces:**
- Consumes: `nflverse.Game`.
- Produces:
  - `plan.LABELS: dict[str, timedelta]` and `plan.FLAG_LABELS = ("t24h", "t6h", "t1h")`.
  - `plan.EVENTS_COST = 1`, `plan.PROPS_COST = 200` and `plan.LINES_COST = 60`.
  - `plan.Job(kind, label, requested_at, games: tuple[Game, ...], event_ids: tuple[str, ...])` with `.cost` and `.keys -> list[tuple[str, str, str]]`, the `(event_id, kind, label)` rows it produces.
  - `plan.day_jobs(games) -> list[Job]`.
  - `plan.match_events(games, events: list[dict]) -> tuple[list[tuple[Game, str]], list[str]]`. It returns the matched (Game with kickoff = event commence_time, event_id) pairs and the unmatched game_ids.
  - `plan.snapshot_jobs(pairs) -> list[Job]`, sorted by `requested_at`.
  - `plan.dry_run(games) -> dict` with keys `games`, `events_calls`, `props_calls`, `lines_calls` and `credits`.
  - `plan.iso(dt) -> str` (`YYYY-MM-DDTHH:MM:SSZ`).

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime, timedelta, timezone

from src.edge.backtest import plan
from src.edge.backtest.nflverse import Game

UTC = timezone.utc


def g(gid="2023_10_CLE_BAL", home="BAL", away="CLE", kick=datetime(2023, 11, 12, 18, 0, tzinfo=UTC)):
    return Game(gid, 2023, 10, home, away, kick, 31, 33)


def ev(eid, home, away, commence):
    return {"id": eid, "home_team": home, "away_team": away, "commence_time": commence}


def test_costs():
    assert (plan.EVENTS_COST, plan.PROPS_COST, plan.LINES_COST) == (1, 200, 60)


def test_props_jobs_use_the_four_offsets():
    jobs = [j for j in plan.snapshot_jobs([(g(), "e1")]) if j.kind == "props"]
    got = {j.label: j.requested_at for j in jobs}
    k = datetime(2023, 11, 12, 18, 0, tzinfo=UTC)
    assert got == {"t24h": k - timedelta(hours=24), "t6h": k - timedelta(hours=6),
                   "t1h": k - timedelta(hours=1), "close": k - timedelta(minutes=5)}
    assert jobs[0].keys == [("e1", "props", jobs[0].label)]


def test_lines_jobs_group_kickoffs_within_15_minutes_into_one_slate():
    a = g("a", kick=datetime(2023, 11, 12, 18, 0, tzinfo=UTC))
    b = g("b", "NE", "IND", kick=datetime(2023, 11, 12, 18, 5, tzinfo=UTC))
    c = g("c", "SF", "JAX", kick=datetime(2023, 11, 12, 21, 25, tzinfo=UTC))
    lines = [j for j in plan.snapshot_jobs([(a, "ea"), (b, "eb"), (c, "ec")]) if j.kind == "lines"]
    assert len(lines) == 8
    close_ab = next(j for j in lines if j.label == "close" and "ea" in j.event_ids)
    assert close_ab.event_ids == ("ea", "eb")
    assert close_ab.requested_at == datetime(2023, 11, 12, 17, 55, tzinfo=UTC)
    assert close_ab.keys == [("ea", "lines", "close"), ("eb", "lines", "close")]


def test_snapshot_jobs_sorted_by_request_time():
    jobs = plan.snapshot_jobs([(g(), "e1")])
    assert [j.requested_at for j in jobs] == sorted(j.requested_at for j in jobs)


def test_day_jobs_use_the_eastern_game_day_at_10_utc():
    mnf = g("m", "LAR", "WSH", kick=datetime(2023, 11, 14, 1, 15, tzinfo=UTC))  # Monday 20:15 ET
    jobs = plan.day_jobs([g(), mnf])
    assert [(j.event_ids, j.requested_at) for j in jobs] == [
        (("day:2023-11-12",), datetime(2023, 11, 12, 10, 0, tzinfo=UTC)),
        (("day:2023-11-13",), datetime(2023, 11, 13, 10, 0, tzinfo=UTC))]
    assert jobs[0].keys == [("day:2023-11-12", "events", "list")]


def test_match_events_by_teams_and_time_including_renamed_abbreviations():
    games = [g(), g("2023_10_WAS_LA", "LAR", "WSH", datetime(2023, 11, 14, 1, 15, tzinfo=UTC)),
             g("2023_10_X", "KC", "MIA", datetime(2023, 11, 5, 14, 30, tzinfo=UTC))]
    events = [ev("e1", "Baltimore Ravens", "Cleveland Browns", "2023-11-12T18:03:00Z"),
              ev("e2", "Los Angeles Rams", "Washington Commanders", "2023-11-14T01:20:00Z"),
              ev("e3", "Miami Dolphins", "Kansas City Chiefs", "2023-11-05T14:30:00Z")]  # home/away swapped
    pairs, unmatched = plan.match_events(games, events)
    assert [(p[0].game_id, p[1]) for p in pairs] == [("2023_10_CLE_BAL", "e1"), ("2023_10_WAS_LA", "e2")]
    assert pairs[0][0].kickoff == datetime(2023, 11, 12, 18, 3, tzinfo=UTC)  # event time wins
    assert unmatched == ["2023_10_X"]


def test_match_events_rejects_kickoff_more_than_3_hours_off():
    pairs, unmatched = plan.match_events([g()], [ev("e1", "Baltimore Ravens", "Cleveland Browns",
                                                    "2023-11-12T22:30:00Z")])
    assert pairs == [] and unmatched == ["2023_10_CLE_BAL"]


def test_dry_run_counts_calls_and_credits():
    a = g("a")
    b = g("b", "NE", "IND", kick=datetime(2023, 11, 12, 18, 5, tzinfo=UTC))
    assert plan.dry_run([a, b]) == {"games": 2, "events_calls": 1, "props_calls": 8, "lines_calls": 4,
                                    "credits": 1 + 8 * 200 + 4 * 60}


def test_iso():
    assert plan.iso(datetime(2023, 11, 12, 17, 55, tzinfo=UTC)) == "2023-11-12T17:55:00Z"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -q tests/unit/test_edge_backtest_plan.py`
Expected: FAIL with `ImportError: cannot import name 'plan'`.

- [ ] **Step 3: Implement `src/edge/backtest/plan.py`**

```python
"""Which historical snapshots to buy, and what they cost (spec §5, §9). Pure."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta, timezone

from src.edge.backtest.nflverse import ET, Game
from src.edge.config import LINE_MARKETS, NFL_PROP_MARKETS
from src.edge.ingest import _ts
from src.edge.teams import NFL_ABBR_BY_NAME

LABELS = {"t24h": timedelta(hours=24), "t6h": timedelta(hours=6),
          "t1h": timedelta(hours=1), "close": timedelta(minutes=5)}
FLAG_LABELS = ("t24h", "t6h", "t1h")
REGION_EQUIVALENTS = 2          # 16 bookmakers; every 10 count as one region
EVENTS_COST = 1
PROPS_COST = 10 * REGION_EQUIVALENTS * len(NFL_PROP_MARKETS)
LINES_COST = 10 * REGION_EQUIVALENTS * len(LINE_MARKETS)
SLATE_GAP = timedelta(minutes=15)
MATCH_TOLERANCE = timedelta(hours=3)
EVENTS_AT = time(10, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Job:
    kind: str                       # events | props | lines
    label: str                      # LABELS key, or 'list' for events
    requested_at: datetime
    games: tuple[Game, ...]
    event_ids: tuple[str, ...]      # parallel to games; ('day:YYYY-MM-DD',) for events

    @property
    def cost(self) -> int:
        return {"events": EVENTS_COST, "props": PROPS_COST, "lines": LINES_COST}[self.kind]

    @property
    def keys(self) -> list[tuple[str, str, str]]:
        return [(e, self.kind, self.label) for e in self.event_ids]


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def day_jobs(games: list[Game]) -> list[Job]:
    days: dict = {}
    for gm in games:
        days.setdefault(gm.kickoff.astimezone(ET).date(), []).append(gm)
    return [Job("events", "list", datetime.combine(d, EVENTS_AT), tuple(gs), (f"day:{d.isoformat()}",))
            for d, gs in sorted(days.items())]


def match_events(games: list[Game], events: list[dict]) -> tuple[list[tuple[Game, str]], list[str]]:
    pairs, unmatched = [], []
    for gm in games:
        hit = None
        for e in events:
            if (NFL_ABBR_BY_NAME.get(e.get("home_team")) == gm.home
                    and NFL_ABBR_BY_NAME.get(e.get("away_team")) == gm.away
                    and abs(_ts(e["commence_time"]) - gm.kickoff) <= MATCH_TOLERANCE):
                hit = e
                break
        if hit is None:
            unmatched.append(gm.game_id)
        else:
            pairs.append((replace(gm, kickoff=_ts(hit["commence_time"])), hit["id"]))
    return pairs, unmatched


def _slates(pairs: list[tuple[Game, str]]) -> list[list[tuple[Game, str]]]:
    out: list[list[tuple[Game, str]]] = []
    for p in sorted(pairs, key=lambda p: p[0].kickoff):
        if out and p[0].kickoff - out[-1][0][0].kickoff <= SLATE_GAP:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def snapshot_jobs(pairs: list[tuple[Game, str]]) -> list[Job]:
    jobs = [Job("props", label, gm.kickoff - off, (gm,), (eid,))
            for gm, eid in pairs for label, off in LABELS.items()]
    for slate in _slates(pairs):
        first = slate[0][0].kickoff
        jobs += [Job("lines", label, first - off, tuple(p[0] for p in slate), tuple(p[1] for p in slate))
                 for label, off in LABELS.items()]
    return sorted(jobs, key=lambda j: (j.requested_at, j.kind, j.event_ids))


def dry_run(games: list[Game]) -> dict:
    days = day_jobs(games)
    snaps = snapshot_jobs([(gm, gm.game_id) for gm in games])
    props = sum(1 for j in snaps if j.kind == "props")
    lines = sum(1 for j in snaps if j.kind == "lines")
    return {"games": len(games), "events_calls": len(days), "props_calls": props, "lines_calls": lines,
            "credits": sum(j.cost for j in days) + sum(j.cost for j in snaps)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -q tests/unit/test_edge_backtest_plan.py`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/edge/backtest/plan.py tests/unit/test_edge_backtest_plan.py
git commit -m "feat(edge-bt): snapshot planning, event matching and credit cost

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Historical client and resumable fetch with budget stops

**Files:**
- Modify: `src/edge/client.py` (add three methods)
- Create: `src/edge/backtest/fetch.py`
- Test: `tests/unit/test_edge_backtest_fetch.py`

**Interfaces:**
- Consumes:
  - `plan.Job`, `plan.day_jobs`, `plan.match_events`, `plan.snapshot_jobs` and `plan.iso`;
  - `client.Usage(remaining, used, last)` and `client.OddsApiError(status, message)`.
- Produces:
  - `EdgeOddsClient.get_historical_events(sport, date: str)`, `.get_historical_game_odds(sport, date: str)` and `.get_historical_event_odds(event_id, sport, date: str)`. Each returns `(json, Usage)`.
  - `fetch.Repo(conn)` with `done_keys() -> set[tuple[str, str, str]]`, `spent() -> int`, `events_payload(day_key: str) -> dict | None` and `insert(rows: list[dict]) -> None`.
  - `fetch.Summary` (fields `calls`, `credits`, `errors`, `stopped`, `unmatched`, `remaining`).
  - `async fetch.run_fetch(repo, client, games, *, cap: int, min_remaining: int, sleep=asyncio.sleep, min_interval=0.5) -> Summary`.
  - `fetch.ensure_tables(url: str) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_edge_backtest_fetch.py`:

```python
import asyncio
from datetime import datetime, timezone

import pytest

from src.edge import store
from src.edge.backtest import fetch
from src.edge.backtest.nflverse import Game
from src.edge.client import OddsApiError, Usage

UTC = timezone.utc
GAME = Game("2023_10_CLE_BAL", 2023, 10, "BAL", "CLE", datetime(2023, 11, 12, 18, 0, tzinfo=UTC), 31, 33)
EVENT = {"id": "e1", "sport_key": "americanfootball_nfl", "home_team": "Baltimore Ravens",
         "away_team": "Cleveland Browns", "commence_time": "2023-11-12T18:03:00Z"}


class FakeRepo:
    def __init__(self):
        self.rows = []

    def done_keys(self):
        return {(r["event_id"], r["kind"], r["label"]) for r in self.rows}

    def spent(self):
        return sum(r["credits"] for r in self.rows)

    def events_payload(self, day_key):
        return next((r["payload"] for r in self.rows if r["event_id"] == day_key), None)

    def insert(self, rows):
        have = self.done_keys()
        self.rows += [r for r in rows if (r["event_id"], r["kind"], r["label"]) not in have]


class FakeClient:
    def __init__(self, events=(EVENT,), fail=None, remaining=4_000_000, lines_events=("e1",)):
        self.events, self.fail, self.remaining, self.lines_events = list(events), fail or {}, remaining, lines_events
        self.calls = []

    def _usage(self, cost):
        self.remaining -= cost
        return Usage(self.remaining, None, cost)

    async def get_historical_events(self, sport, date):
        self.calls.append(("events", date))
        return {"timestamp": date, "data": self.events}, self._usage(1)

    async def get_historical_game_odds(self, sport, date):
        self.calls.append(("lines", date))
        if ("lines", date) in self.fail:
            raise self.fail[("lines", date)]
        data = [{**EVENT, "id": e, "bookmakers": [{"key": "pinnacle", "markets": []}]} for e in self.lines_events]
        return {"timestamp": date, "data": data}, self._usage(60)

    async def get_historical_event_odds(self, event_id, sport, date):
        self.calls.append(("props", date))
        if ("props", date) in self.fail:
            raise self.fail[("props", date)]
        return {"timestamp": date, "data": {**EVENT, "bookmakers": [{"key": "pinnacle", "markets": []}]}}, \
            self._usage(200)


async def _nosleep(_):
    return None


def run(repo, client, cap=800_000, min_remaining=3_600_000):
    return asyncio.run(fetch.run_fetch(repo, client, [GAME], cap=cap, min_remaining=min_remaining,
                                       sleep=_nosleep))


def test_fetch_stores_every_snapshot_once_and_resume_spends_nothing():
    repo, client = FakeRepo(), FakeClient()
    s = run(repo, client)
    kinds = sorted((r["kind"], r["label"]) for r in repo.rows)
    assert kinds.count(("events", "list")) == 1
    assert sorted(l for k, l in kinds if k == "props") == ["close", "t1h", "t24h", "t6h"]
    assert sorted(l for k, l in kinds if k == "lines") == ["close", "t1h", "t24h", "t6h"]
    assert s.credits == repo.spent() == 1 + 4 * 200 + 4 * 60 and s.stopped is None
    props_close = next(r for r in repo.rows if r["kind"] == "props" and r["label"] == "close")
    assert props_close["requested_at"] == datetime(2023, 11, 12, 17, 58, tzinfo=UTC)  # event kickoff 18:03
    assert props_close["game_id"] == "2023_10_CLE_BAL" and props_close["home"] == "BAL"
    assert props_close["snapshot_at"] == datetime(2023, 11, 12, 17, 58, tzinfo=UTC)
    client2 = FakeClient()
    s2 = run(repo, client2)
    assert client2.calls == [] and s2.credits == 0


def test_cap_stops_before_the_call_that_would_exceed_it():
    repo, client = FakeRepo(), FakeClient()
    s = run(repo, client, cap=1 + 200 + 60 + 100)
    assert s.stopped.startswith("credit cap")
    assert repo.spent() <= 1 + 200 + 60 + 100


def test_min_remaining_floor_stops_the_run():
    repo, client = FakeRepo(), FakeClient(remaining=3_600_100)
    s = run(repo, client)
    assert s.stopped.startswith("credits remaining")
    assert len(client.calls) == 2  # events (remaining 3,600,099) + one 60/200 call drops below the floor


def test_429_stops_and_stores_nothing_for_that_job():
    t24 = "2023-11-11T18:03:00Z"
    repo, client = FakeRepo(), FakeClient(fail={("props", t24): OddsApiError(429, "slow down")})
    s = run(repo, client)
    assert s.stopped == "odds_api 429"
    assert ("e1", "props", "t24h") not in repo.done_keys()


def test_other_errors_skip_the_job_and_the_next_run_retries_it():
    t6 = "2023-11-12T12:03:00Z"
    repo = FakeRepo()
    s = run(repo, FakeClient(fail={("props", t6): OddsApiError(500, "boom")}))
    assert s.errors == 1 and s.stopped is None
    assert ("e1", "props", "t6h") not in repo.done_keys()
    client = FakeClient()
    run(repo, client)
    assert client.calls == [("props", t6)]


def test_404_stores_an_empty_stub_so_it_is_not_bought_again():
    t1 = "2023-11-12T17:03:00Z"
    repo = FakeRepo()
    run(repo, FakeClient(fail={("props", t1): OddsApiError(404, "not found")}))
    stub = next(r for r in repo.rows if r["kind"] == "props" and r["label"] == "t1h")
    assert stub["credits"] == 0 and stub["payload"]["bookmakers"] == [] and stub["snapshot_at"] is None


def test_event_missing_from_a_lines_response_gets_a_stub_and_credits_land_once():
    other = Game("2023_10_IND_NE", 2023, 10, "NE", "IND", datetime(2023, 11, 12, 18, 5, tzinfo=UTC), 10, 6)
    ev2 = {**EVENT, "id": "e2", "home_team": "New England Patriots", "away_team": "Indianapolis Colts",
           "commence_time": "2023-11-12T18:05:00Z"}
    repo, client = FakeRepo(), FakeClient(events=(EVENT, ev2), lines_events=("e1",))
    asyncio.run(fetch.run_fetch(repo, client, [GAME, other], cap=800_000, min_remaining=0, sleep=_nosleep))
    lines = [r for r in repo.rows if r["kind"] == "lines" and r["label"] == "close"]
    assert {r["event_id"] for r in lines} == {"e1", "e2"}
    assert sorted(r["credits"] for r in lines) == [0, 60]
    assert next(r for r in lines if r["event_id"] == "e2")["payload"]["bookmakers"] == []


def test_unmatched_games_are_reported_and_not_fetched():
    repo, client = FakeRepo(), FakeClient(events=())
    s = run(repo, client)
    assert s.unmatched == ["2023_10_CLE_BAL"]
    assert [c[0] for c in client.calls] == ["events"]


def test_fetch_never_writes_live_worker_state(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("live state touched")
    monkeypatch.setattr(store, "log_api_call", boom)
    monkeypatch.setattr(store, "set_state", boom)
    s = run(FakeRepo(), FakeClient())
    assert s.stopped is None
```

The `t24h`/`t6h`/`t1h` timestamps in these tests are computed from the event's kickoff (18:03), not the schedule's (18:00).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -q tests/unit/test_edge_backtest_fetch.py`
Expected: FAIL with `ImportError: cannot import name 'fetch'`.

- [ ] **Step 3: Add the historical methods to `EdgeOddsClient` in `src/edge/client.py`** (after `get_scores`)

```python
    async def get_historical_events(self, sport: SportConfig, date: str):
        return await self._get(f"/historical/sports/{sport.key}/events", {"date": date})

    async def get_historical_game_odds(self, sport: SportConfig, date: str):
        return await self._get(f"/historical/sports/{sport.key}/odds", {
            "date": date, "bookmakers": ",".join(BOOKS), "markets": ",".join(LINE_MARKETS),
            "oddsFormat": "american"})

    async def get_historical_event_odds(self, event_id: str, sport: SportConfig, date: str):
        return await self._get(f"/historical/sports/{sport.key}/events/{event_id}/odds", {
            "date": date, "bookmakers": ",".join(BOOKS), "markets": ",".join(sport.prop_markets),
            "oddsFormat": "american"})
```

- [ ] **Step 4: Implement `src/edge/backtest/fetch.py`**

```python
"""Buys historical snapshots once and stores them raw (spec §5, §6, §9).

Never touches live worker state: no store.log_api_call, no credits_* keys, so
the worker's credit guard projects live spend only. The budget is the sum of
edge_bt_snapshots.credits.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial

from psycopg2.extras import Json, execute_values

from src.edge.backtest import plan
from src.edge.backtest.nflverse import Game
from src.edge.client import OddsApiError
from src.edge.config import NFL
from src.edge.ingest import _ts

log = logging.getLogger("edge.backtest.fetch")
COLS = ("season", "week", "game_id", "event_id", "kind", "label", "kickoff", "requested_at", "snapshot_at",
        "home", "away", "credits", "payload")


class Stop(Exception):
    pass


@dataclass
class Summary:
    calls: int = 0
    credits: int = 0
    errors: int = 0
    stopped: str | None = None
    unmatched: list[str] = field(default_factory=list)
    remaining: int | None = None


class Repo:
    def __init__(self, conn):
        self.conn = conn

    def done_keys(self) -> set[tuple[str, str, str]]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT event_id, kind, label FROM edge_bt_snapshots")
            return set(cur.fetchall())

    def spent(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COALESCE(SUM(credits), 0) FROM edge_bt_snapshots")
            return int(cur.fetchone()[0])

    def events_payload(self, day_key: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT payload FROM edge_bt_snapshots WHERE event_id = %s AND kind = 'events'", (day_key,))
            row = cur.fetchone()
        return row[0] if row else None

    def insert(self, rows: list[dict]) -> None:
        with self.conn.cursor() as cur:
            execute_values(cur, f"INSERT INTO edge_bt_snapshots ({', '.join(COLS)}) VALUES %s "
                                "ON CONFLICT (event_id, kind, label) DO NOTHING",
                           [tuple(Json(r[c]) if c == "payload" else r[c] for c in COLS) for r in rows])
        self.conn.commit()


def ensure_tables(url: str) -> None:
    from sqlalchemy import create_engine

    from src.database import Base
    from src.models.edge import EdgeBtPick, EdgeBtSnapshot
    engine = create_engine(url)
    Base.metadata.create_all(engine, tables=[EdgeBtSnapshot.__table__, EdgeBtPick.__table__])
    engine.dispose()


def _row(job: plan.Job, game: Game | None, event_id: str, payload: dict, snapshot_at: datetime | None,
         credits: int) -> dict:
    first = job.games[0]
    return {"season": first.season, "week": first.week, "game_id": game.game_id if game else None,
            "event_id": event_id, "kind": job.kind, "label": job.label,
            "kickoff": game.kickoff if game else job.requested_at, "requested_at": job.requested_at,
            "snapshot_at": snapshot_at, "home": game.home if game else "", "away": game.away if game else "",
            "credits": credits, "payload": payload}


def _stub(event_id: str, status: int | None = None) -> dict:
    return {"id": event_id, "bookmakers": [], "missing": status or True}


async def run_fetch(repo, client, games: list[Game], *, cap: int, min_remaining: int,
                    sleep=asyncio.sleep, min_interval: float = 0.5) -> Summary:
    s = Summary()
    done = repo.done_keys()
    spent = repo.spent()

    async def call(job: plan.Job, request):
        nonlocal spent
        if spent + job.cost > cap:
            raise Stop(f"credit cap {cap:,} reached ({spent:,} spent)")
        if s.remaining is not None and s.remaining < min_remaining:
            raise Stop(f"credits remaining {s.remaining:,} below floor {min_remaining:,}")
        await sleep(min_interval)
        data, usage = await request()
        cost = usage.last if usage.last is not None else job.cost
        spent += cost
        s.calls += 1
        s.credits += cost
        if usage.remaining is not None:
            s.remaining = usage.remaining
        if s.calls % 100 == 0:
            log.info("backtest fetch: %d calls, %d credits this run, %s remaining", s.calls, s.credits, s.remaining)
        return data, cost

    try:
        events: list[dict] = []
        for job in plan.day_jobs(games):
            day_key = job.event_ids[0]
            payload = repo.events_payload(day_key) if job.keys[0] in done else None
            if payload is None:
                data, cost = await call(job, partial(client.get_historical_events, NFL, plan.iso(job.requested_at)))
                payload = data
                repo.insert([_row(job, None, day_key, data, _ts(data["timestamp"]), cost)])
            events += payload.get("data") or []

        pairs, s.unmatched = plan.match_events(games, events)
        for gid in s.unmatched:
            log.warning("backtest: no Odds API event matched %s", gid)

        for job in plan.snapshot_jobs(pairs):
            if all(k in done for k in job.keys):
                continue
            date = plan.iso(job.requested_at)
            try:
                if job.kind == "props":
                    request = partial(client.get_historical_event_odds, job.event_ids[0], NFL, date)
                else:
                    request = partial(client.get_historical_game_odds, NFL, date)
                data, cost = await call(job, request)
            except OddsApiError as e:
                if e.status in (401, 429):
                    s.stopped = f"odds_api {e.status}"
                    return s
                if e.status == 404:
                    repo.insert([_row(job, gm, eid, _stub(eid, 404), None, 0)
                                 for gm, eid in zip(job.games, job.event_ids)])
                    continue
                s.errors += 1
                log.warning("backtest %s %s %s skipped: %s", job.kind, job.label, job.event_ids, e)
                continue
            at = _ts(data["timestamp"])
            if job.kind == "props":
                rows = [_row(job, job.games[0], job.event_ids[0], data["data"], at, cost)]
            else:
                by_id = {e["id"]: e for e in data.get("data") or []}
                rows = [_row(job, gm, eid, by_id.get(eid) or _stub(eid), at, cost if i == 0 else 0)
                        for i, (gm, eid) in enumerate(zip(job.games, job.event_ids))]
            repo.insert(rows)
    except Stop as e:
        s.stopped = str(e)
    return s
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest -q tests/unit/test_edge_backtest_fetch.py`
Expected: PASS (9 tests).

On `test_min_remaining_floor_stops_the_run`: the events call leaves 3,600,099, which is at or above the 3,600,000 floor. The next call (lines or props, whichever sorts first) drops below it, and the third call is refused. So the client sees exactly 2 calls.

- [ ] **Step 6: Run the unit suite and commit**

Run: `python3 -m pytest -q tests/unit`. Expected: all pass.

```bash
git add src/edge/client.py src/edge/backtest/fetch.py tests/unit/test_edge_backtest_fetch.py
git commit -m "feat(edge-bt): resumable historical fetch with credit cap and remaining floor

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Replay through the live engine, with CLV and grading

**Files:**
- Create: `src/edge/backtest/replay.py`
- Test: `tests/unit/test_edge_backtest_replay.py`

**Interfaces:**
- Consumes: `nflverse.Game`, `plan.FLAG_LABELS`, the stats rows from `nflverse.stats_index` and the name sets from `nflverse.snaps_index`.
- Produces:
  - `replay.Snap(payload: dict, snapshot_at: datetime | None)`.
  - `replay.grade(market, player, side, line, game, stats: list[dict], snap_names: set[str] | None) -> str | None`.
  - `replay.replay_game(game, snaps: dict[str, dict[str, Snap]], stats, snap_names, min_ev, verify_ev) -> list[dict]`. `snaps` is keyed by label, then kind. Each pick dict has the `edge_bt_picks` columns except `run_id`, `season` and `week`.
  - `replay.run_replay(conn, run_id, games: dict[str, Game], stats, snaps_idx, min_ev, verify_ev) -> int`.
  - `replay.PICK_COLS`.

- [ ] **Step 1: Write the failing tests**

```python
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.edge.backtest import replay
from src.edge.backtest.nflverse import Game
from src.edge.engine import american_to_decimal, price_quotes
from src.edge.ingest import parse_event

UTC = timezone.utc
KICK = datetime(2023, 11, 12, 18, 3, tzinfo=UTC)
GAME = Game("2023_10_CLE_BAL", 2023, 10, "BAL", "CLE", KICK, 31, 33)
FIX = Path(__file__).resolve().parents[1] / "fixtures" / "edge" / "bt_props_2023.json"


def event(prices, player="Amari Cooper", line=55.5, market="player_reception_yds"):
    """prices: {book: (over, under)}"""
    return {"id": "e1", "home_team": "Baltimore Ravens", "away_team": "Cleveland Browns",
            "commence_time": "2023-11-12T18:03:00Z",
            "bookmakers": [{"key": b, "markets": [{"key": market, "outcomes": [
                {"name": "Over", "description": player, "price": o, "point": line},
                {"name": "Under", "description": player, "price": u, "point": line}]}]}
                for b, (o, u) in prices.items()]}


EVEN = {"pinnacle": (-110, -110), "draftkings": (-110, -110), "betmgm": (-110, -110)}
SOFT = {**EVEN, "fanduel": (120, -150)}
STAT = {"name": "amari cooper", "team": "CLE", "pass_att": 0, "pass_cmp": 0, "pass_yds": 0, "pass_tds": 0,
        "pass_int": 0, "rush_att": 0, "rush_yds": 0, "receptions": 4, "rec_yds": 60}


def snaps(t6h=None, close=None, close_at=KICK - timedelta(minutes=5)):
    out = {}
    if t6h is not None:
        out["t6h"] = {"props": replay.Snap(t6h, KICK - timedelta(hours=6))}
    if close is not None:
        out["close"] = {"props": replay.Snap(close, close_at)}
    return out


def fd_over(picks):
    return next(p for p in picks if p["book"] == "fanduel" and p["side"] == "over")


def test_picks_equal_live_engine_output_on_real_2023_payload():
    payload = json.loads(FIX.read_text())["data"]
    at = KICK - timedelta(hours=6)
    picks = replay.replay_game(GAME, {"t6h": {"props": replay.Snap(payload, at)}}, [], None, -0.05, 0.15)
    want = {(p.quote.book, p.quote.player, p.quote.side, p.quote.line)
            for p in price_quotes(parse_event(payload, at), -0.05, 0.15) if p.is_ev}
    assert want and {(p["book"], p["player"], p["side"], p["line"]) for p in picks} == want
    assert all(p["label"] == "t6h" and "pinnacle" in p["sharps"].split(",") for p in picks)


def test_pick_fields_and_clv_against_the_close():
    picks = replay.replay_game(GAME, snaps(event(SOFT), event(SOFT)), [STAT], {"amari cooper"}, 0.02, 0.15)
    p = fd_over(picks)
    assert (p["event_id"], p["market"], p["player"], p["line"], p["price"]) == \
        ("e1", "player_reception_yds", "Amari Cooper", 55.5, 120)
    assert p["fair_prob"] == pytest.approx(0.5) and p["ev"] == pytest.approx(0.1) and p["verify"] is False
    assert p["n_books"] == 3 and p["sharps"] == "pinnacle"
    assert p["closing_prob"] is not None
    assert p["clv"] == pytest.approx(p["closing_prob"] - 1 / american_to_decimal(120))
    assert p["result"] == "win" and p["profit"] == pytest.approx(1.2)


def test_no_clv_when_the_line_moved_before_close():
    picks = replay.replay_game(GAME, snaps(event(SOFT), event(SOFT, line=57.5)), [STAT], {"amari cooper"},
                               0.02, 0.15)
    p = fd_over(picks)
    assert p["closing_prob"] is None and p["clv"] is None


def test_close_snapshot_after_kickoff_is_ignored():
    picks = replay.replay_game(GAME, snaps(event(SOFT), event(SOFT), close_at=KICK + timedelta(minutes=1)),
                               [STAT], {"amari cooper"}, 0.02, 0.15)
    assert fd_over(picks)["clv"] is None


def test_flag_snapshot_at_or_after_kickoff_flags_nothing():
    late = {"t1h": {"props": replay.Snap(event(SOFT), KICK)}}
    assert replay.replay_game(GAME, late, [STAT], {"amari cooper"}, 0.02, 0.15) == []


def test_empty_stub_payloads_are_skipped():
    stub = {"t6h": {"props": replay.Snap({"id": "e1", "bookmakers": [], "missing": 404}, None)}}
    assert replay.replay_game(GAME, stub, [], None, 0.02, 0.15) == []


def test_pinnacle_only_ev_leaves_novig_and_prophetx_out():
    books = {**SOFT, "novig": (110, -130), "prophetx": (110, -130)}
    p = fd_over(replay.replay_game(GAME, snaps(event(books)), [STAT], {"amari cooper"}, -1.0, 0.15))
    assert p["sharps"] == "novig,pinnacle,prophetx"
    assert p["pin_ev"] == pytest.approx(0.1)
    assert p["ev"] < p["pin_ev"]


@pytest.mark.parametrize("stats,names,side,want", [
    ([STAT], {"amari cooper"}, "over", "win"),             # stat row grades
    ([], {"amari cooper"}, "over", "loss"),                # played, no row: graded as 0
    ([], {"someone else"}, "over", "void"),                # snaps published, not in them: inactive
    ([], None, "over", None),                              # no snap counts for the game: ungradable
    ([STAT, {**STAT, "team": "BAL"}], {"amari cooper"}, "over", None),  # two players share the name
])
def test_prop_grading(stats, names, side, want):
    assert replay.grade("player_reception_yds", "Amari Cooper", side, 55.5, GAME, stats, names) == want


@pytest.mark.parametrize("odds_name,display", [
    ("Kenneth Walker III", "Kenneth Walker III"), ("D.J. Moore", "DJ Moore"), ("Ja'Marr Chase", "Ja'Marr Chase")])
def test_prop_grading_matches_names_with_suffixes_and_punctuation(odds_name, display):
    from src.edge.players import normalize_name
    row = {**STAT, "name": normalize_name(display), "rec_yds": 80}
    assert replay.grade("player_reception_yds", odds_name, "over", 55.5, GAME, [row], {row["name"]}) == "win"


def test_game_line_grading_uses_final_scores():
    assert replay.grade("h2h", "", "away", 0.0, GAME, [], None) == "win"         # CLE 33, BAL 31
    assert replay.grade("spreads", "", "home", 3.5, GAME, [], None) == "win"     # 31 + 3.5 > 33
    assert replay.grade("totals", "", "over", 63.5, GAME, [], None) == "win"
    no_score = Game(**{**GAME.__dict__, "home_score": None})
    assert replay.grade("h2h", "", "home", 0.0, no_score, [], None) is None


def test_void_and_ungradable_picks_have_no_profit():
    picks = replay.replay_game(GAME, snaps(event(SOFT)), [], {"someone else"}, 0.02, 0.15)
    p = fd_over(picks)
    assert p["result"] == "void" and p["profit"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -q tests/unit/test_edge_backtest_replay.py`
Expected: FAIL with `ImportError: cannot import name 'replay'`.

- [ ] **Step 3: Implement `src/edge/backtest/replay.py`**

```python
"""Replays stored snapshots through the live engine (spec §7).

`replay_game` is pure; `run_replay` reads edge_bt_snapshots one event at a time
(the payloads are large) and writes edge_bt_picks under a fresh run_id.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime

from psycopg2.extras import execute_values

from src.edge.backtest.nflverse import Game
from src.edge.backtest.plan import FLAG_LABELS
from src.edge.config import DFS_BOOKS, SHARP_BOOKS
from src.edge.engine import american_to_decimal, fair_prob, group, price_quotes
from src.edge.engine import clv as clv_of
from src.edge.grading import grade_game, profit_units
from src.edge.ingest import parse_event
from src.edge.nfl_players import nfl_prop_result
from src.edge.players import normalize_name
from src.edge.quotes import Quote, sides_for

log = logging.getLogger("edge.backtest.replay")
PIN_EXCLUDE = frozenset({"novig", "prophetx"})
PICK_COLS = ("run_id", "season", "week", "event_id", "label", "market", "player", "side", "line", "book",
             "price", "fair_prob", "ev", "verify", "pin_ev", "n_books", "sharps", "closing_prob", "clv",
             "result", "profit")


@dataclass(frozen=True)
class Snap:
    payload: dict
    snapshot_at: datetime | None


def _quotes(by_kind: dict[str, Snap], before: datetime, inclusive: bool) -> list[Quote]:
    out: list[Quote] = []
    for s in by_kind.values():
        if s.snapshot_at is None or not s.payload.get("bookmakers"):
            continue
        if s.snapshot_at > before or (not inclusive and s.snapshot_at == before):
            continue
        out += parse_event(s.payload, s.snapshot_at)
    return out


def _sharps(books: dict, market: str, exclude_book: str) -> str:
    a, b = sides_for(market)
    return ",".join(sorted(bk for bk, sides in books.items()
                           if bk in SHARP_BOOKS and bk != exclude_book and bk not in DFS_BOOKS
                           and a in sides and b in sides))


def grade(market: str, player: str, side: str, line: float, game: Game, stats: list[dict],
          snap_names: set[str] | None) -> str | None:
    if not player:
        if game.home_score is None or game.away_score is None:
            return None
        return grade_game(market, side, line, game.home_score, game.away_score)
    key = normalize_name(player)
    rows = [r for r in stats if r["name"] == key]
    if len(rows) > 1:
        return None  # two players in this game share the name key: never guess
    return nfl_prop_result({"market": market, "line": line, "side": side}, rows[0] if rows else None,
                           snaps_loaded=snap_names is not None,
                           played=snap_names is not None and key in snap_names)


def replay_game(game: Game, snaps: dict[str, dict[str, Snap]], stats: list[dict], snap_names: set[str] | None,
                min_ev: float, verify_ev: float) -> list[dict]:
    # The close may sit at kickoff; a snapshot after kickoff holds in-game prices and is never the close.
    closing = group(_quotes(snaps.get("close", {}), game.kickoff, inclusive=True))
    picks = []
    for label in FLAG_LABELS:
        quotes = _quotes(snaps.get(label, {}), game.kickoff, inclusive=False)
        if not quotes:
            continue
        grouped = group(quotes)
        for p in price_quotes(quotes, min_ev, verify_ev):
            if not p.is_ev:
                continue
            q = p.quote
            books = grouped[q.market_key]
            pin = fair_prob({b: v for b, v in books.items() if b not in PIN_EXCLUDE}, q.market, q.side,
                            exclude_book=q.book)
            close = fair_prob(closing.get(q.market_key, {}), q.market, q.side, exclude_book=None)
            closing_prob = None if close.thin else close.prob
            result = grade(q.market, q.player, q.side, q.line, game, stats, snap_names)
            picks.append({
                "event_id": q.event_id, "label": label, "market": q.market, "player": q.player, "side": q.side,
                "line": q.line, "book": q.book, "price": q.price, "fair_prob": p.fair.prob, "ev": p.ev,
                "verify": p.verify,
                "pin_ev": None if pin.thin else pin.prob * american_to_decimal(q.price) - 1,
                "n_books": p.fair.n_books, "sharps": _sharps(books, q.market, q.book),
                "closing_prob": closing_prob,
                "clv": None if closing_prob is None else clv_of(closing_prob, q.price),
                "result": result,
                "profit": profit_units(result, q.price) if result in ("win", "loss", "push") else None,
            })
    return picks


def run_replay(conn, run_id: str, games: dict[str, Game], stats: dict[str, list[dict]],
               snaps_idx: dict[str, set[str]], min_ev: float, verify_ev: float) -> int:
    with conn.cursor() as cur:
        cur.execute("""SELECT DISTINCT event_id, game_id, kickoff FROM edge_bt_snapshots
                       WHERE kind IN ('props', 'lines') AND game_id = ANY(%s) ORDER BY game_id""",
                    (sorted(games),))
        targets = cur.fetchall()
    total = 0
    for event_id, game_id, kickoff in targets:
        game = replace(games[game_id], kickoff=kickoff)
        with conn.cursor() as cur:
            cur.execute("""SELECT kind, label, payload, snapshot_at FROM edge_bt_snapshots
                           WHERE event_id = %s AND kind IN ('props', 'lines')""", (event_id,))
            by_label: dict[str, dict[str, Snap]] = defaultdict(dict)
            for kind, label, payload, at in cur.fetchall():
                by_label[label][kind] = Snap(payload, at)
        picks = replay_game(game, by_label, stats.get(game_id, []), snaps_idx.get(game_id), min_ev, verify_ev)
        rows = [tuple({**p, "run_id": run_id, "season": game.season, "week": game.week}[c] for c in PICK_COLS)
                for p in picks]
        if rows:
            with conn.cursor() as cur:
                execute_values(cur, f"INSERT INTO edge_bt_picks ({', '.join(PICK_COLS)}) VALUES %s "
                                    "ON CONFLICT DO NOTHING", rows)
        conn.commit()
        total += len(rows)
    log.info("backtest replay %s: %d picks from %d events", run_id, total, len(targets))
    return total
```

Notes for the implementer:
- `price_quotes` already leaves the pick's own book out of its consensus (`exclude_book=book`), and `p.fair.n_books` counts the books that were used.
- In `test_pinnacle_only_ev_leaves_novig_and_prophetx_out`, Novig and ProphetX lean Under, so their presence lowers the Over consensus. Removing them restores the 0.5 fair price and a 10% EV.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -q tests/unit/test_edge_backtest_replay.py`
Expected: PASS (all).

If `test_picks_equal_live_engine_output_on_real_2023_payload` finds `want` empty at −5%, lower `min_ev` in that test to `-1.0` and record the change as a ruling. That test checks equality with the engine; it doesn't check whether this payload has edges.

- [ ] **Step 5: Run the unit suite and commit**

Run: `python3 -m pytest -q tests/unit`. Expected: all pass.

```bash
git add src/edge/backtest/replay.py tests/unit/test_edge_backtest_replay.py
git commit -m "feat(edge-bt): replay snapshots through the live engine with CLV and grading

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Report

**Files:**
- Create: `src/edge/backtest/report.py`
- Test: `tests/unit/test_edge_backtest_report.py`

**Interfaces:**
- Consumes: pick dicts with the `edge_bt_picks` columns.
- Produces:
  - `report.bucket(ev) -> str`.
  - `report.first_flag(picks) -> list[dict]`.
  - `report.metrics(rows) -> dict` with keys `picks`, `graded`, `win_pct`, `roi`, `roi_ci`, `avg_ev`, `avg_clv`, `beat_close` and `small`.
  - `report.recommend(picks) -> list[str]`.
  - `report.build_report(picks, meta: dict) -> str`.
  - `report.run_report(conn, run_id: str | None, out_path) -> str`, which returns the run_id it used.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from src.edge.backtest import report


def pick(label="t6h", ev=0.03, clv=0.01, result="win", profit=0.9, market="player_receptions",
         book="fanduel", season=2023, player="A", pin_ev=0.03, verify=False, event_id="e1", side="over", line=4.5):
    return {"season": season, "week": 1, "event_id": event_id, "label": label, "market": market,
            "player": player, "side": side, "line": line, "book": book, "price": -110, "ev": ev, "clv": clv,
            "result": result, "profit": profit, "pin_ev": pin_ev, "verify": verify}


def test_bucket_edges():
    assert [report.bucket(x) for x in (0.02, 0.0399, 0.04, 0.0599, 0.06, 0.3)] == \
        ["2–4%", "2–4%", "4–6%", "4–6%", "6%+", "6%+"]


def test_first_flag_counts_a_pick_once_at_its_earliest_label():
    rows = [pick("t1h"), pick("t24h"), pick("t6h"), pick("t6h", player="B")]
    got = report.first_flag(rows)
    assert sorted((r["player"], r["label"]) for r in got) == [("A", "t24h"), ("B", "t6h")]


def test_metrics_roi_excludes_voids_and_counts_pushes_as_staked():
    rows = [pick(result="win", profit=1.0), pick(result="loss", profit=-1.0), pick(result="push", profit=0.0),
            pick(result="void", profit=None, clv=None), pick(result=None, profit=None, clv=-0.02)]
    m = report.metrics(rows)
    assert m["picks"] == 5 and m["graded"] == 3
    assert m["win_pct"] == pytest.approx(0.5)
    assert m["roi"] == pytest.approx(0.0)
    assert m["avg_clv"] == pytest.approx((0.01 * 3 - 0.02) / 4)
    assert m["beat_close"] == pytest.approx(3 / 4)
    assert m["small"] is True and m["roi_ci"] is not None


def test_metrics_on_empty_rows():
    m = report.metrics([])
    assert m["picks"] == 0 and m["roi"] is None and m["avg_clv"] is None and m["win_pct"] is None


def test_recommendations_follow_the_fixed_rules():
    good = [pick(market="player_receptions", book="fanduel", ev=0.03, clv=0.02, player=f"p{i}")
            for i in range(200)]
    bad = [pick(market="player_pass_tds", book="bovada", ev=0.07, clv=-0.01, player=f"q{i}")
           for i in range(250)]
    early = [pick(label="t24h", market="player_rush_yds", book="betmgm", ev=0.05, clv=0.05, player=f"r{i}")
             for i in range(10)]
    text = "\n".join(report.recommend(good + bad + early))
    assert "player_receptions" in text and "player_pass_tds" not in text.split("Markets to show")[1].split("\n")[0]
    assert "fanduel" in text
    assert "EV ≥ 2%" in text
    assert "t24h" in text


def test_build_report_has_every_section_and_flags_small_samples():
    rows = [pick(), pick(label="t24h", season=2025, book="pinnacle", pin_ev=None)]
    md = report.build_report(rows, {"run_id": "r1", "snapshots": 10, "credits": 2000})
    for heading in ("## Overall", "## By season", "## By market", "## By book", "## By EV bucket",
                    "## By lead time", "## Season × lead time", "## Pinnacle-only consensus",
                    "## Recommendations"):
        assert heading in md, heading
    assert "small sample" in md and "r1" in md and "2,000" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -q tests/unit/test_edge_backtest_report.py`
Expected: FAIL with `ImportError: cannot import name 'report'`.

- [ ] **Step 3: Implement `src/edge/backtest/report.py`**

```python
"""Backtest report (spec §11). Pure aggregation plus one DB entry point.

A pick flagged at several lead times counts once (its earliest label) in every
table except the lead-time ones, which treat each label on its own.
"""
from __future__ import annotations

from math import sqrt
from pathlib import Path
from statistics import mean, pstdev

from psycopg2.extras import RealDictCursor

from src.config import settings
from src.edge.backtest.plan import FLAG_LABELS

BUCKETS = ("2–4%", "4–6%", "6%+")
BUCKET_FLOOR = {"2–4%": 2, "4–6%": 4, "6%+": 6}
SMALL = 50
MIN_RECOMMEND = 200
_ORDER = {label: i for i, label in enumerate(FLAG_LABELS)}


def bucket(ev: float) -> str:
    return "2–4%" if ev < 0.04 else "4–6%" if ev < 0.06 else "6%+"


def first_flag(picks: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for p in picks:
        k = (p["event_id"], p["market"], p["player"], p["side"], float(p["line"]), p["book"])
        if k not in best or _ORDER[p["label"]] < _ORDER[best[k]["label"]]:
            best[k] = p
    return list(best.values())


def metrics(rows: list[dict]) -> dict:
    graded = [r for r in rows if r["result"] in ("win", "loss", "push")]
    wl = [r for r in graded if r["result"] in ("win", "loss")]
    profits = [float(r["profit"]) for r in graded]
    clvs = [float(r["clv"]) for r in rows if r["clv"] is not None]
    return {
        "picks": len(rows), "graded": len(graded),
        "win_pct": sum(r["result"] == "win" for r in wl) / len(wl) if wl else None,
        "roi": mean(profits) if profits else None,
        "roi_ci": 1.96 * pstdev(profits) / sqrt(len(profits)) if len(profits) > 1 else None,
        "avg_ev": mean(float(r["ev"]) for r in rows) if rows else None,
        "avg_clv": mean(clvs) if clvs else None,
        "beat_close": sum(c > 0 for c in clvs) / len(clvs) if clvs else None,
        "small": len(rows) < SMALL,
    }


def _pct(v, signed=False) -> str:
    if v is None:
        return "–"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def _table(groups: list[tuple[str, list[dict]]], first_col: str) -> str:
    out = [f"| {first_col} | Picks | Win% | ROI (95% CI) | Avg EV | Avg CLV | Beat close | |",
           "|---|---|---|---|---|---|---|---|"]
    for name, rows in groups:
        m = metrics(rows)
        roi = "–" if m["roi"] is None else f"{_pct(m['roi'], True)} ± {_pct(m['roi_ci'])}"
        out.append(f"| {name} | {m['picks']:,} | {_pct(m['win_pct'])} | {roi} | {_pct(m['avg_ev'])} | "
                   f"{_pct(m['avg_clv'], True)} | {_pct(m['beat_close'])} | "
                   f"{'small sample' if m['small'] else ''} |")
    return "\n".join(out)


def _by(rows: list[dict], key) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(key(r)), []).append(r)
    return sorted(groups.items())


def recommend(picks: list[dict]) -> list[str]:
    once = first_flag(picks)

    def winners(key) -> list[str]:
        return [name for name, rows in _by(once, key)
                if len(rows) >= MIN_RECOMMEND and (metrics(rows)["avg_clv"] or 0) > 0]

    out = [f"- **Markets to show:** {', '.join(winners(lambda r: r['market'])) or 'none qualify'} "
           f"(average CLV > 0, at least {MIN_RECOMMEND} picks)",
           f"- **Books to show:** {', '.join(winners(lambda r: r['book'])) or 'none qualify'}"]
    by_bucket = dict(_by(once, lambda r: bucket(float(r["ev"]))))
    floor = next((b for b in BUCKETS if b in by_bucket and (metrics(by_bucket[b])["avg_clv"] or 0) > 0), None)
    out.append(f"- **EV threshold:** EV ≥ {BUCKET_FLOOR[floor]}% (lowest bucket with average CLV > 0)" if floor
               else "- **EV threshold:** no EV bucket beats the close on average")
    leads = [(metrics(rows)["avg_clv"], name) for name, rows in _by(picks, lambda r: r["label"])
             if metrics(rows)["avg_clv"] is not None]
    out.append(f"- **Lead time:** {max(leads)[1]} has the best average CLV" if leads
               else "- **Lead time:** no lead time has CLV data")
    return out


def build_report(picks: list[dict], meta: dict) -> str:
    once = first_flag(picks)
    pin = [r for r in once if r["pin_ev"] is not None and float(r["pin_ev"]) >= settings.edge_min_ev]
    sections = [
        "# NFL +EV backtest, 2023–2025 regular seasons",
        f"- Run `{meta['run_id']}` · {meta['snapshots']:,} snapshots · {meta['credits']:,} credits spent",
        f"- {len(picks):,} flagged rows; {len(once):,} distinct picks (counted once, at the earliest lead time)",
        f"- Ungradable: {sum(r['result'] is None for r in once):,} · Void: "
        f"{sum(r['result'] == 'void' for r in once):,} · No close (line moved or thin): "
        f"{sum(r['clv'] is None for r in once):,} · Verify-flagged (EV > 15%): "
        f"{sum(bool(r['verify']) for r in once):,}",
        "", "## Overall", _table([("All", once)], "Scope"),
        "", "## By season", _table(_by(once, lambda r: r["season"]), "Season"),
        "", "## By market", _table(_by(once, lambda r: r["market"]), "Market"),
        "", "## By book", _table(_by(once, lambda r: r["book"]), "Book"),
        "", "## By EV bucket", _table(_by(once, lambda r: bucket(float(r["ev"]))), "EV"),
        "", "## By lead time", _table(_by(picks, lambda r: r["label"]), "Lead time"),
        "", "## Season × lead time", _table(_by(picks, lambda r: f"{r['season']} {r['label']}"), "Season, lead"),
        "", "## Pinnacle-only consensus",
        "Picks that stay +EV with Novig and ProphetX left out of the consensus, for a like-for-like 2023 vs 2025 view.",
        _table(_by(pin, lambda r: r["season"]), "Season"),
        "", "## Recommendations", *recommend(picks), "",
    ]
    return "\n".join(sections)


def run_report(conn, run_id: str | None, out_path) -> str:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if run_id is None:
            cur.execute("SELECT run_id FROM edge_bt_picks ORDER BY run_id DESC LIMIT 1")
            row = cur.fetchone()
            if row is None:
                raise SystemExit("no replay run found; run `replay` first")
            run_id = row["run_id"]
        cur.execute("SELECT * FROM edge_bt_picks WHERE run_id = %s", (run_id,))
        picks = cur.fetchall()
        cur.execute("SELECT COUNT(*) AS n, COALESCE(SUM(credits), 0) AS c FROM edge_bt_snapshots")
        snap = cur.fetchone()
    md = build_report(picks, {"run_id": run_id, "snapshots": int(snap["n"]), "credits": int(snap["c"])})
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    return run_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -q tests/unit/test_edge_backtest_report.py`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the unit suite and commit**

Run: `python3 -m pytest -q tests/unit`. Expected: all pass.

```bash
git add src/edge/backtest/report.py tests/unit/test_edge_backtest_report.py
git commit -m "feat(edge-bt): backtest report with breakdowns and rule-based recommendations

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: CLI and the real-Postgres round trip

**Files:**
- Create: `src/edge/backtest/__main__.py`, `tests/unit/test_edge_backtest_cli.py`, `tests/integration/test_edge_backtest_db.py`

**Interfaces:**
- Consumes: everything above; `store.connect()` and `config.get_sync_database_url()`; `utils.redact.install_redaction()`.
- Produces: `__main__.main(argv=None) -> int`, `__main__.DEFAULT_OUT`.
  - `fetch [--season S ...] [--week W] [--dry-run]`
  - `replay [--season S ...]`
  - `report [--run ID] [--out PATH]`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_edge_backtest_cli.py`:

```python
from datetime import datetime, timezone

from src.edge.backtest import __main__ as cli
from src.edge.backtest import nflverse
from src.edge.backtest.nflverse import Game

UTC = timezone.utc
GAMES = [Game("a", 2023, 10, "BAL", "CLE", datetime(2023, 11, 12, 18, 0, tzinfo=UTC), 31, 33),
         Game("b", 2023, 11, "NE", "IND", datetime(2023, 11, 19, 18, 0, tzinfo=UTC), 10, 6)]


def test_default_report_path_is_backend_docs():
    assert cli.DEFAULT_OUT.as_posix().endswith("backend/docs/backtests/nfl-2023-2025.md")


def test_dry_run_prints_cost_and_never_connects(monkeypatch, capsys):
    monkeypatch.setattr(nflverse, "load_schedule", lambda seasons: GAMES)
    monkeypatch.setattr(cli.store, "connect", lambda: (_ for _ in ()).throw(AssertionError("connected")))
    assert cli.main(["fetch", "--season", "2023", "--week", "10", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "games: 1" in out and "credits: 1,041" in out
```

The arithmetic for a single week-10 game: 1 events call + 4 props × 200 + 4 lines × 60 = 1,041.

`tests/integration/test_edge_backtest_db.py`:

```python
"""Backtest SQL against a real Postgres (EDGE_TEST_DB_URL, disposable)."""
import asyncio
import os
from datetime import datetime, timezone

import psycopg2
import pytest
from sqlalchemy import create_engine

from src.database import Base
import src.models  # noqa: F401
from src.edge.backtest import fetch, replay, report
from src.edge.backtest.nflverse import Game
from src.edge.client import Usage

URL = os.environ.get("EDGE_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not URL, reason="EDGE_TEST_DB_URL not set")
UTC = timezone.utc
GAME = Game("2023_10_CLE_BAL", 2023, 10, "BAL", "CLE", datetime(2023, 11, 12, 18, 0, tzinfo=UTC), 31, 33)
EDGE_TABLES = [t for name, t in Base.metadata.tables.items() if name.startswith("edge_")]
LIVE = ("edge_quotes", "edge_quote_history", "edge_opportunities", "edge_api_calls", "edge_worker_state")


@pytest.fixture
def conn():
    engine = create_engine(URL)
    Base.metadata.drop_all(engine, tables=EDGE_TABLES)
    Base.metadata.create_all(engine, tables=EDGE_TABLES)
    engine.dispose()
    c = psycopg2.connect(URL)
    yield c
    c.close()


EVENT = {"id": "e1", "home_team": "Baltimore Ravens", "away_team": "Cleveland Browns",
         "commence_time": "2023-11-12T18:03:00Z"}
SOFT = {"pinnacle": (-110, -110), "draftkings": (-110, -110), "betmgm": (-110, -110), "fanduel": (120, -150)}
STAT = {"name": "amari cooper", "team": "CLE", "pass_att": 0, "pass_cmp": 0, "pass_yds": 0, "pass_tds": 0,
        "pass_int": 0, "rush_att": 0, "rush_yds": 0, "receptions": 4, "rec_yds": 60}


def props_event():
    return {**EVENT, "bookmakers": [{"key": b, "markets": [{"key": "player_reception_yds", "outcomes": [
        {"name": "Over", "description": "Amari Cooper", "price": o, "point": 55.5},
        {"name": "Under", "description": "Amari Cooper", "price": u, "point": 55.5}]}]}
        for b, (o, u) in SOFT.items()]}


class PropsClient:
    """Self-contained: tests/ is not a package, so unit-test helpers can't be imported here."""
    async def get_historical_events(self, sport, date):
        return {"timestamp": date, "data": [EVENT]}, Usage(4_000_000, None, 1)

    async def get_historical_game_odds(self, sport, date):
        return {"timestamp": date, "data": [{**EVENT, "bookmakers": []}]}, Usage(4_000_000, None, 60)

    async def get_historical_event_odds(self, event_id, sport, date):
        return {"timestamp": date, "data": props_event()}, Usage(4_000_000, None, 200)


async def _nosleep(_):
    return None


def test_ensure_tables_is_idempotent(conn):
    fetch.ensure_tables(URL)
    fetch.ensure_tables(URL)


def test_repo_round_trip_and_conflict(conn):
    repo = fetch.Repo(conn)
    row = {"season": 2023, "week": 10, "game_id": "g", "event_id": "e1", "kind": "props", "label": "t6h",
           "kickoff": GAME.kickoff, "requested_at": GAME.kickoff, "snapshot_at": GAME.kickoff, "home": "BAL",
           "away": "CLE", "credits": 200, "payload": {"id": "e1", "bookmakers": []}}
    repo.insert([row])
    repo.insert([{**row, "credits": 999}])
    assert repo.done_keys() == {("e1", "props", "t6h")} and repo.spent() == 200


def test_fetch_replay_report_round_trip_leaves_live_tables_empty(conn, tmp_path):
    s = asyncio.run(fetch.run_fetch(fetch.Repo(conn), PropsClient(), [GAME], cap=800_000,
                                    min_remaining=0, sleep=_nosleep))
    assert s.stopped is None and s.unmatched == []
    n = replay.run_replay(conn, "r1", {GAME.game_id: GAME}, {GAME.game_id: [STAT]},
                          {GAME.game_id: {"amari cooper"}}, 0.02, 0.15)
    assert n > 0
    with conn.cursor() as cur:
        cur.execute("SELECT label, book, side, result, clv FROM edge_bt_picks WHERE book = 'fanduel' "
                    "AND side = 'over' ORDER BY label")
        rows = cur.fetchall()
        assert [r[0] for r in rows] == ["t1h", "t24h", "t6h"]
        assert all(r[3] == "win" and r[4] is not None for r in rows)
        for t in LIVE:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            assert cur.fetchone()[0] == 0, t
    assert report.run_report(conn, None, tmp_path / "r.md") == "r1"
    assert "## Recommendations" in (tmp_path / "r.md").read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -q tests/unit/test_edge_backtest_cli.py` and the integration file with `EDGE_TEST_DB_URL` set.
Expected: the CLI test FAILS (`ImportError` on `__main__`). The DB round trip should already pass, since its modules exist; if it fails, fix the cause before continuing.

- [ ] **Step 3: Implement `src/edge/backtest/__main__.py`**

```python
"""python -m src.edge.backtest fetch|replay|report (spec §10).

Run from backend/ with DATABASE_URL pointing at the target database and
ODDS_API_KEY set (fetch only). BT_CREDIT_CAP / BT_MIN_REMAINING bound spend.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import get_sync_database_url, settings
from src.edge import store
from src.edge.backtest import SEASONS, fetch, nflverse, plan, replay, report
from src.edge.client import EdgeOddsClient
from src.utils.redact import install_redaction

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "docs" / "backtests" / "nfl-2023-2025.md"


def _games(seasons, week):
    games = nflverse.load_schedule(seasons)
    return [g for g in games if week is None or g.week == week]


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m src.edge.backtest")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--season", type=int, action="append")
    f.add_argument("--week", type=int)
    f.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("replay")
    r.add_argument("--season", type=int, action="append")
    rp = sub.add_parser("report")
    rp.add_argument("--run")
    rp.add_argument("--out", default=str(DEFAULT_OUT))
    a = p.parse_args(argv)
    seasons = a.season if getattr(a, "season", None) else list(SEASONS)

    if a.cmd == "fetch":
        games = _games(seasons, a.week)
        if a.dry_run:
            for k, v in plan.dry_run(games).items():
                print(f"{k}: {v:,}")
            return 0
        fetch.ensure_tables(get_sync_database_url())
        conn = store.connect()
        try:
            s = asyncio.run(fetch.run_fetch(
                fetch.Repo(conn), EdgeOddsClient(), games,
                cap=int(os.environ.get("BT_CREDIT_CAP", "800000")),
                min_remaining=int(os.environ.get("BT_MIN_REMAINING", "3600000"))))
        finally:
            conn.close()
        print(f"calls: {s.calls:,}  credits: {s.credits:,}  errors: {s.errors}  remaining: {s.remaining}")
        print(f"unmatched games: {len(s.unmatched)} {s.unmatched[:10]}")
        if s.stopped:
            print(f"stopped: {s.stopped}")
            return 1
        return 0

    conn = store.connect()
    try:
        if a.cmd == "replay":
            games = {g.game_id: g for g in nflverse.load_schedule(seasons)}
            run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            n = replay.run_replay(conn, run_id, games, nflverse.load_stats(seasons), nflverse.load_snaps(seasons),
                                  settings.edge_min_ev, settings.edge_verify_ev)
            print(f"run {run_id}: {n:,} picks")
        else:
            run_id = report.run_report(conn, a.run, a.out)
            print(f"report for run {run_id} written to {a.out}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    install_redaction()  # process-wide; only for real runs, never inside pytest
    sys.exit(main())
```

- [ ] **Step 4: Run all backtest tests**

Run: `python3 -m pytest -q tests/unit/test_edge_backtest_cli.py`
Expected: PASS (2).

Run: `EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py tests/integration/test_startup_migrations_db.py tests/integration/test_edge_backtest_db.py`
Expected: PASS (45 existing + 3 new).

- [ ] **Step 5: Real dry run (network only, 0 credits)**

Run: `python3 -m src.edge.backtest fetch --dry-run 2>&1 | grep -v Downcast`
Expected: `games: 816`, `props_calls: 3,264`, and `credits` between 650,000 and 720,000.

Record the exact figures in the ledger.

- [ ] **Step 6: Run the unit suite and commit**

Run: `python3 -m pytest -q tests/unit`. Expected: all pass.

```bash
git add src/edge/backtest/__main__.py tests/unit/test_edge_backtest_cli.py tests/integration/test_edge_backtest_db.py
git commit -m "feat(edge-bt): fetch/replay/report CLI and Postgres round-trip tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## After the tasks (outside this plan; each needs Samir's go-ahead)

1. **Merge** to `main`. Startup `create_all` creates the two tables in prod; live behaviour doesn't change.
2. **Pilot** (about 13K credits; writes to the prod DB): `fetch --season 2023 --week 10`, then `replay --season 2023` and `report`.
   - Spot-check 10 picks against the raw payloads.
   - Spot-check 5 grades against nflverse.
3. **Full run:** `fetch`, then `replay` and `report`. Commit the report and present the findings.
