# TruLine Edge NFL Phase B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** NFL props are graded under sportsbook rules (BallDontLie box scores + nflverse snap counts); NFL +EV rows link to player pages with hit rates; `/nfl/edge/results` shows the NFL track record. No NBA table or behaviour changes.

**Architecture:**
- **Storage:** four new NFL tables, created by `create_all`.
- **Module:** a new `src/edge/nfl_players.py` holds the NFL BallDontLie client, row parsing, name resolution, schedule/stats/snaps ingest, hit rates and prop grading. It reuses `normalize_name`, `match_player` and `_rate`-style maths from `players.py` without changing them.
- **Worker:** three NFL steps, gated by `settings.edge_nfl_players`.
- **Grading:** `run_grading` calls `nfl_players.grade_props`.
- **Views/API:** dispatch on `sport` for player pages and player links.
- **Frontend:** reuses `EdgePlayer`/`EdgeResults` with a `sport` prop.

**Tech Stack:** Python 3.13, FastAPI, psycopg2, SQLAlchemy models (create_all), `nfl_data_py` (already a dependency), pytest + throwaway Postgres 16, React 18 + TypeScript + vitest.

**Spec:** `backend/docs/superpowers/specs/2026-10-16-edge-nfl-phase-b-design.md`

## Global Constraints

- **No NBA table is altered.** NBA grading, player pages and tests stay unchanged; the full NBA suite must pass untouched.
- **New tables:** `edge_nfl_players` (PK `odds_name`+`team`), `edge_nfl_player_games` (unique `bdl_player_id`+`bdl_game_id`), `edge_nfl_snaps` (unique `season`+`week`+`team`+`player_norm`), `edge_nfl_games` (PK `bdl_game_id`).
- **NFL stat columns** (integers, NULL from BallDontLie stored as 0): `pass_att`, `pass_cmp`, `pass_yds`, `pass_tds`, `pass_int`, `rush_att`, `rush_yds`, `receptions`, `rec_yds`.
- **Market → columns:**
  - `player_pass_yds`→pass_yds; `player_pass_tds`→pass_tds; `player_pass_completions`→pass_cmp; `player_pass_attempts`→pass_att; `player_pass_interceptions`→pass_int;
  - `player_rush_yds`→rush_yds; `player_rush_attempts`→rush_att;
  - `player_receptions`→receptions; `player_reception_yds`→rec_yds;
  - `player_rush_reception_yds`→rush_yds+rec_yds.
- **Stats are fetched only by `game_ids` from BallDontLie `/games`.** Preseason never enters.
- **Team codes:** BallDontLie abbreviations everywhere. nflverse → BallDontLie: `{"LA": "LAR", "WAS": "WSH"}`.
- **Grading rules (spec §5), in order:**
  1. Unmatched player: wait.
  2. Game stats not complete: wait.
  3. Stat row: graded.
  4. No row and the player is in the game's snap counts (snaps > 0): graded with 0.
  5. No row, snaps loaded, player absent: void.
  6. No row, snaps not loaded: wait (listed as `stale_ungraded` after 7 days).
- **Stats complete** = BallDontLie `status_state == 'final'`, first seen final ≥ 2 h before the stats were loaded.
- **Kill switch:** `EDGE_NFL_PLAYERS` (settings `edge_nfl_players: bool = True`).
- **NFL season label** = the calendar year the season starts. Rollover Mar 1: Jan–Feb belong to the previous season.
- **Commit trailer:** `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Tests on this Mac:** never touch `localhost:5432`. DB tests use `EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest`.

## Review Focus

1. **Zero-stat players in hit rates.** A receiver who played but caught nothing has no box-score row. Expected: he still counts as a played game with 0, using the snap counts; otherwise Over hit rates are inflated. (Task 6 test.)
2. **Name spelling differs between nflverse and the sportsbook** ("D.J. Moore" vs "DJ Moore"). Expected: matching on the normalized name finds him. A miss must never void a player who has a stat row: rule 3 runs before rules 4–5. (Task 5 test.)
3. **A traded player.** Expected: the sportsbook name is matched per `(odds_name, team)`, so his new-team props resolve against the new roster and his old games keep their old team. (Task 3 test.)
4. **Odds API event ↔ BallDontLie game link** for London/Thursday/Monday games (date-only BallDontLie dates). Expected: matched by home/away abbreviation and |date difference| ≤ 1 day. (Task 5 test.)
5. **BallDontLie 429 mid-backfill.** Expected: the cursor stays put and the next tick resumes. Odds polling is unaffected. (Task 4 test.)

---

### Task 1: NFL tables, team maps, season helper, settings

**Files:**
- Modify: `backend/src/models/edge.py` (4 models), `backend/src/models/__init__.py` (import + `__all__`)
- Modify: `backend/src/edge/teams.py` (`NFL_ABBR_BY_NAME`, `NFLVERSE_TO_BDL`)
- Modify: `backend/src/utils/season.py` (`current_nfl_season`)
- Modify: `backend/src/config.py` (`edge_nfl_players: bool = True`)
- Test: `backend/tests/unit/test_edge_nfl_base.py` (create)

**Interfaces:**
- Produces:
  - models `EdgeNflPlayer`, `EdgeNflPlayerGame`, `EdgeNflSnap`, `EdgeNflGame`;
  - `NFL_ABBR_BY_NAME: dict[str, str]` (Odds API full name → BallDontLie abbreviation);
  - `NFLVERSE_TO_BDL: dict[str, str]`;
  - `current_nfl_season(today: date | None = None) -> int`;
  - `settings.edge_nfl_players`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/test_edge_nfl_base.py
import json
from datetime import date

from src.database import Base
import src.models  # noqa: F401
from src.edge.teams import NFL_ABBR_BY_NAME, NFLVERSE_TO_BDL
from src.utils.season import current_nfl_season

FIX = "tests/fixtures/edge"


def test_nfl_team_names_cover_bdl_and_the_odds_api():
    bdl = {t["full_name"]: t["abbreviation"] for t in json.load(open(f"{FIX}/bdl_nfl_teams.json"))["data"]}
    assert NFL_ABBR_BY_NAME == bdl and len(bdl) == 32
    for e in json.load(open(f"{FIX}/nfl_events.json")):
        assert e["home_team"] in NFL_ABBR_BY_NAME and e["away_team"] in NFL_ABBR_BY_NAME


def test_nflverse_codes_map_onto_bdl():
    bdl = set(NFL_ABBR_BY_NAME.values())
    nflverse = (bdl - {"LAR", "WSH"}) | {"LA", "WAS"}
    assert {NFLVERSE_TO_BDL.get(t, t) for t in nflverse} == bdl


def test_current_nfl_season():
    assert current_nfl_season(date(2026, 10, 16)) == 2026
    assert current_nfl_season(date(2027, 1, 20)) == 2026   # playoffs
    assert current_nfl_season(date(2027, 3, 1)) == 2027


def test_nfl_tables_are_registered():
    names = set(Base.metadata.tables)
    assert {"edge_nfl_players", "edge_nfl_player_games", "edge_nfl_snaps", "edge_nfl_games"} <= names
    pk = [c.name for c in Base.metadata.tables["edge_nfl_players"].primary_key.columns]
    assert pk == ["odds_name", "team"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python3 -m pytest -q tests/unit/test_edge_nfl_base.py`
Expected: collection ERROR (`cannot import name 'NFL_ABBR_BY_NAME'`).

- [ ] **Step 3: Implement**

Append to `backend/src/edge/teams.py`. Build the dict from the fixture's 32 `full_name`→`abbreviation` pairs; write it out literally, sorted by name:

```python
# NFL: The Odds API full names -> BallDontLie NFL abbreviations (bdl_nfl_teams.json, 2026-10-16).
NFL_ABBR_BY_NAME = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WSH",
}
# nflverse uses LA / WAS where BallDontLie uses LAR / WSH (checked 2026-10-16).
NFLVERSE_TO_BDL = {"LA": "LAR", "WAS": "WSH"}
```

If the fixture disagrees with any abbreviation above, the fixture wins: fix the literal, and ledger the difference.

Append to `backend/src/utils/season.py`:

```python
def current_nfl_season(today: date | None = None) -> int:
    """NFL season label = the year it kicks off; Jan-Feb playoffs belong to the previous one."""
    d = today or date.today()
    return d.year if d.month >= 3 else d.year - 1
```

Append to `backend/src/models/edge.py`:

```python
class EdgeNflPlayer(Base):
    """Odds API NFL player name -> BallDontLie NFL id. Keyed with team: NFL names repeat across teams."""
    __tablename__ = "edge_nfl_players"
    odds_name: Mapped[str] = mapped_column(String(80), primary_key=True)
    team: Mapped[str] = mapped_column(String(4), primary_key=True)
    bdl_player_id: Mapped[int | None] = mapped_column(Integer, index=True)
    position: Mapped[str | None] = mapped_column(String(8))
    match_method: Mapped[str] = mapped_column(String(12), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EdgeNflGame(Base):
    __tablename__ = "edge_nfl_games"
    bdl_game_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    home: Mapped[str] = mapped_column(String(4), nullable=False)
    away: Mapped[str] = mapped_column(String(4), nullable=False)
    postseason: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status_state: Mapped[str | None] = mapped_column(String(16))
    final_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stats_loaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    snaps_loaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EdgeNflPlayerGame(Base):
    __tablename__ = "edge_nfl_player_games"
    __table_args__ = (UniqueConstraint("bdl_player_id", "bdl_game_id", name="uq_edge_nfl_player_games"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    bdl_player_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    bdl_game_id: Mapped[int] = mapped_column(Integer, nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    team: Mapped[str] = mapped_column(String(4), nullable=False)
    opponent: Mapped[str] = mapped_column(String(4), nullable=False)
    is_home: Mapped[bool] = mapped_column(Boolean, nullable=False)
    pass_att: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pass_cmp: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pass_yds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pass_tds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pass_int: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rush_att: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rush_yds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    receptions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rec_yds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class EdgeNflSnap(Base):
    """nflverse snap counts: the participation record that tells 'played, no stats' from 'inactive'."""
    __tablename__ = "edge_nfl_snaps"
    __table_args__ = (UniqueConstraint("season", "week", "team", "player_norm", name="uq_edge_nfl_snaps"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    team: Mapped[str] = mapped_column(String(4), nullable=False)
    player_norm: Mapped[str] = mapped_column(String(80), nullable=False)
    snaps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
```

In `backend/src/models/__init__.py`, add the four models to the `from src.models.edge import (...)` line and to `__all__`.

In `backend/src/config.py`, after `edge_sports: str = "nba,nfl"`, add `    edge_nfl_players: bool = True`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && python3 -m pytest -q tests/unit/test_edge_nfl_base.py tests/unit`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/src/models backend/src/edge/teams.py backend/src/utils/season.py backend/src/config.py backend/tests/unit/test_edge_nfl_base.py
git commit -m "feat(edge): NFL player/stat/snap/game tables, team maps and season helper

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: BallDontLie NFL client and row parsing

**Files:**
- Create: `backend/src/edge/nfl_players.py`
- Test: `backend/tests/unit/test_edge_nfl_players.py` (create)

**Interfaces:**
- Consumes: Task 1 maps.
- Produces:
  - `NFL_STAT_PARTS: dict[str, tuple[str, ...]]`;
  - `stat_for(market, g: dict) -> float`;
  - `game_row(g: dict) -> dict`;
  - `stat_row(s: dict) -> dict | None`;
  - `class BdlNflClient(api_key=None, transport=None, min_interval=1.05)`, with async methods:
    - `teams() -> dict[int, str]`;
    - `roster(team_abbr) -> list[dict]` (each `{"id","first_name","last_name","team_abbr","position"}`);
    - `games(season, week=None) -> list[dict]`;
    - `stats(game_ids: list[int]) -> list[dict]`.
  - `BdlNflClient` raises `players.BdlError`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/test_edge_nfl_players.py
import json
from datetime import date

import httpx
import pytest

from src.edge import nfl_players as nfl
from src.edge.config import NFL_PROP_MARKETS

FIX = "tests/fixtures/edge"


def test_every_nfl_market_has_stat_parts():
    assert set(nfl.NFL_STAT_PARTS) == set(NFL_PROP_MARKETS)
    g = {"pass_att": 30, "pass_cmp": 20, "pass_yds": 250, "pass_tds": 2, "pass_int": 1,
         "rush_att": 5, "rush_yds": 22, "receptions": 0, "rec_yds": 0}
    assert nfl.stat_for("player_rush_reception_yds", {**g, "rec_yds": 8}) == 30
    assert nfl.stat_for("player_pass_completions", g) == 20


def test_game_and_stat_rows_from_the_real_fixtures():
    games = json.load(open(f"{FIX}/bdl_nfl_games_w5.json"))["data"]
    gr = nfl.game_row(games[0])
    assert gr["bdl_game_id"] == games[0]["id"] and gr["week"] == 5 and gr["season"] == 2026
    assert gr["home"] == games[0]["home_team"]["abbreviation"] and isinstance(gr["game_date"], date)
    stats = json.load(open(f"{FIX}/bdl_nfl_stats_game.json"))["data"]
    rows = [r for r in (nfl.stat_row(s) for s in stats) if r]
    assert len(rows) == len(stats)
    qb = max(rows, key=lambda r: r["pass_att"])
    assert qb["pass_yds"] > 0 and qb["opponent"] in {"TB", "DAL"} and qb["opponent"] != qb["team"]
    assert all(isinstance(r[c], int) for r in rows for c in nfl.STAT_COLS)  # NULL -> 0


@pytest.mark.asyncio
async def test_client_follows_cursors_and_uses_the_nfl_base():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if "players" in request.url.path and "cursor" not in request.url.params:
            return httpx.Response(200, json={"data": [{"id": 1, "first_name": "A", "last_name": "B",
                                                       "position_abbreviation": "WR", "team": {"abbreviation": "ATL"}}],
                                             "meta": {"next_cursor": 9}})
        if "players" in request.url.path:
            return httpx.Response(200, json={"data": [{"id": 2, "first_name": "C", "last_name": "D",
                                                       "position_abbreviation": "RB", "team": {"abbreviation": "ATL"}}],
                                             "meta": {}})
        if "teams" in request.url.path:
            return httpx.Response(200, json={"data": [{"id": 7, "abbreviation": "ATL"}]})
        return httpx.Response(200, json={"data": [], "meta": {}})

    c = nfl.BdlNflClient(api_key="k", transport=httpx.MockTransport(handler), min_interval=0)
    roster = await c.roster("ATL")
    assert [p["id"] for p in roster] == [1, 2] and roster[0]["position"] == "WR"
    await c.stats([11, 12])
    assert all("/nfl/v1/" in u for u in seen)
    assert "game_ids%5B%5D=11" in seen[-1] and "game_ids%5B%5D=12" in seen[-1]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python3 -m pytest -q tests/unit/test_edge_nfl_players.py`
Expected: collection ERROR (`No module named 'src.edge.nfl_players'`).

- [ ] **Step 3: Implement**

Create `backend/src/edge/nfl_players.py`:

```python
# backend/src/edge/nfl_players.py
"""NFL player identity, BallDontLie NFL box scores, nflverse snap counts,
hit rates and prop grading (Phase B spec).

Stats come only from games listed by BallDontLie /games (regular season and
playoffs): its /stats?seasons[] feed also carries preseason games.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date

import httpx

from src.config import settings
from src.edge.players import BdlError

log = logging.getLogger("edge.nfl_players")
ROSTER_TTL = 6 * 3600

STAT_COLS = ("pass_att", "pass_cmp", "pass_yds", "pass_tds", "pass_int",
             "rush_att", "rush_yds", "receptions", "rec_yds")
NFL_STAT_PARTS = {
    "player_pass_yds": ("pass_yds",), "player_pass_tds": ("pass_tds",),
    "player_pass_completions": ("pass_cmp",), "player_pass_attempts": ("pass_att",),
    "player_pass_interceptions": ("pass_int",), "player_rush_yds": ("rush_yds",),
    "player_rush_attempts": ("rush_att",), "player_receptions": ("receptions",),
    "player_reception_yds": ("rec_yds",), "player_rush_reception_yds": ("rush_yds", "rec_yds"),
}
_BDL_FIELDS = {"pass_att": "passing_attempts", "pass_cmp": "passing_completions", "pass_yds": "passing_yards",
               "pass_tds": "passing_touchdowns", "pass_int": "passing_interceptions",
               "rush_att": "rushing_attempts", "rush_yds": "rushing_yards",
               "receptions": "receptions", "rec_yds": "receiving_yards"}


def stat_for(market: str, g: dict) -> float:
    return float(sum(g[k] or 0 for k in NFL_STAT_PARTS[market]))


def game_row(g: dict) -> dict:
    return {"bdl_game_id": g["id"], "season": g["season"], "week": g["week"],
            "game_date": date.fromisoformat(g["date"][:10]), "home": g["home_team"]["abbreviation"],
            "away": g["visitor_team"]["abbreviation"], "postseason": bool(g.get("postseason")),
            "status_state": g.get("status_state")}


def stat_row(s: dict) -> dict | None:
    try:
        game, team = s["game"], s["team"]["abbreviation"]
        home, away = game["home_team"]["abbreviation"], game["visitor_team"]["abbreviation"]
        row = {"bdl_player_id": s["player"]["id"], "bdl_game_id": game["id"], "season": game["season"],
               "week": game["week"], "game_date": date.fromisoformat(game["date"][:10]), "team": team,
               "opponent": away if team == home else home, "is_home": team == home}
        row.update({c: int(s.get(f) or 0) for c, f in _BDL_FIELDS.items()})
        return row
    except (KeyError, TypeError, ValueError):
        return None


class BdlNflClient:
    BASE = "https://api.balldontlie.io/nfl/v1"

    def __init__(self, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None,
                 min_interval: float = 1.05):
        self.api_key = api_key if api_key is not None else settings.balldontlie_api_key
        self.transport = transport
        self.min_interval = min_interval
        self._last_call = 0.0
        self._teams: dict[int, str] | None = None
        self._rosters: dict[str, tuple[float, list[dict]]] = {}

    async def _get(self, path: str, params) -> dict:
        wait = self._last_call + self.min_interval - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = time.monotonic()
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=30.0) as c:
                r = await c.get(f"{self.BASE}/{path}", params=params, headers={"Authorization": self.api_key})
        except httpx.HTTPError as e:
            raise BdlError(0, f"BallDontLie NFL network error on {path}: {type(e).__name__}") from None
        if r.status_code != 200:
            raise BdlError(r.status_code, f"BallDontLie NFL {r.status_code} on {path}")
        return r.json()

    async def _paged(self, path: str, params: list[tuple[str, object]]) -> list[dict]:
        out, cursor = [], None
        while True:
            p = params + [("per_page", 100)] + ([("cursor", cursor)] if cursor else [])
            data = await self._get(path, p)
            out += data["data"]
            cursor = (data.get("meta") or {}).get("next_cursor")
            if not cursor:
                return out

    async def teams(self) -> dict[int, str]:
        if self._teams is None:
            data = await self._get("teams", [])
            self._teams = {t["id"]: t["abbreviation"] for t in data["data"]}
        return self._teams

    async def roster(self, team_abbr: str) -> list[dict]:
        hit = self._rosters.get(team_abbr)
        if hit and time.monotonic() - hit[0] < ROSTER_TTL:
            return hit[1]
        team_id = next((i for i, a in (await self.teams()).items() if a == team_abbr), None)
        if team_id is None:
            return []
        out = [{"id": x["id"], "first_name": x["first_name"], "last_name": x["last_name"],
                "team_abbr": (x.get("team") or {}).get("abbreviation"), "position": x.get("position_abbreviation")}
               for x in await self._paged("players", [("team_ids[]", team_id)])]
        self._rosters[team_abbr] = (time.monotonic(), out)
        return out

    async def games(self, season: int, week: int | None = None) -> list[dict]:
        params: list[tuple[str, object]] = [("seasons[]", season)]
        if week is not None:
            params.append(("weeks[]", week))
        return await self._paged("games", params)

    async def stats(self, game_ids: list[int]) -> list[dict]:
        return await self._paged("stats", [("game_ids[]", g) for g in game_ids])
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && python3 -m pytest -q tests/unit/test_edge_nfl_players.py tests/unit`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/src/edge/nfl_players.py backend/tests/unit/test_edge_nfl_players.py
git commit -m "feat(edge): BallDontLie NFL client and box-score parsing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: NFL name resolution and worker dispatch

**Files:**
- Modify: `backend/src/edge/nfl_players.py` (`resolve_new_players`)
- Modify: `backend/src/edge/config.py` (`NFL` → `has_players=True`)
- Modify: `backend/src/edge/worker.py` (dispatch resolution by sport; NFL client)
- Modify: `backend/src/edge/views.py` (`ev_rows` player ids for NFL)
- Test: `backend/tests/integration/test_edge_store_db.py` (append)

**Interfaces:**
- Consumes: Task 1 (`NFL_ABBR_BY_NAME`, tables), Task 2 (`BdlNflClient.roster`).
- Produces:
  - `nfl_players.resolve_new_players(conn, bdl: BdlNflClient, quotes, event, now) -> int`;
  - `nfl_players.player_ids(conn, teams: set[str]) -> dict[str, int]` (name → id for players on `teams`);
  - worker `tick(..., nfl_bdl=None)`.
  - The `tick` signature gains `nfl_bdl` as a keyword with default `None`. The worker builds a `BdlNflClient` in `_run`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/integration/test_edge_store_db.py`:

```python
class _FakeNflBdl:
    def __init__(self, rosters, fail=None):
        self.rosters, self.fail, self.calls = rosters, fail, []

    async def roster(self, team):
        from src.edge.players import BdlError
        self.calls.append(team)
        if self.fail:
            raise BdlError(self.fail, f"BallDontLie NFL {self.fail} rate limited")
        return self.rosters.get(team, [])


def _q(player, eid="n1", market="player_rush_yds"):
    return Quote(eid, market, player, "over", 60.5, "fanduel", -110, None, NOW)


@pytest.mark.asyncio
async def test_nfl_names_resolve_per_team_and_never_touch_nba_players(conn):
    from src.edge import nfl_players
    ev = {"home_team": "Atlanta Falcons", "away_team": "Chicago Bears"}
    rosters = {"ATL": [{"id": 10, "first_name": "Bijan", "last_name": "Robinson", "team_abbr": "ATL", "position": "RB"}],
               "CHI": [{"id": 20, "first_name": "D.J.", "last_name": "Moore", "team_abbr": "CHI", "position": "WR"}]}
    n = await nfl_players.resolve_new_players(conn, _FakeNflBdl(rosters), [_q("Bijan Robinson"), _q("DJ Moore"),
                                                                          _q("Nobody Here")], ev, NOW)
    assert n == 3
    with conn.cursor() as cur:
        cur.execute("SELECT odds_name, team, bdl_player_id, match_method FROM edge_nfl_players ORDER BY odds_name")
        rows = cur.fetchall()
        cur.execute("SELECT COUNT(*) FROM edge_players")
        assert cur.fetchone()[0] == 0  # NBA table untouched
    assert ("Bijan Robinson", "ATL", 10, "exact") in rows
    assert ("DJ Moore", "CHI", 20, "normalized") in rows
    assert any(r[0] == "Nobody Here" and r[3] == "unmatched" for r in rows)
    assert nfl_players.player_ids(conn, {"ATL", "CHI"}) == {"Bijan Robinson": 10, "DJ Moore": 20}


@pytest.mark.asyncio
async def test_a_traded_player_resolves_again_on_his_new_team(conn):
    from src.edge import nfl_players
    old = {"KC": [{"id": 30, "first_name": "Amari", "last_name": "Cooper", "team_abbr": "KC", "position": "WR"}]}
    await nfl_players.resolve_new_players(conn, _FakeNflBdl(old), [_q("Amari Cooper")],
                                          {"home_team": "Kansas City Chiefs", "away_team": "Denver Broncos"}, NOW)
    new = {"BUF": [{"id": 30, "first_name": "Amari", "last_name": "Cooper", "team_abbr": "BUF", "position": "WR"}]}
    n = await nfl_players.resolve_new_players(conn, _FakeNflBdl(new), [_q("Amari Cooper")],
                                              {"home_team": "Buffalo Bills", "away_team": "Miami Dolphins"}, NOW)
    assert n == 1 and nfl_players.player_ids(conn, {"BUF"}) == {"Amari Cooper": 30}


@pytest.mark.asyncio
async def test_a_roster_error_leaves_names_unseen_not_unmatched(conn):
    from src.edge import nfl_players
    n = await nfl_players.resolve_new_players(conn, _FakeNflBdl({}, fail=429), [_q("Bijan Robinson")],
                                              {"home_team": "Atlanta Falcons", "away_team": "Chicago Bears"}, NOW)
    assert n == 0
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM edge_nfl_players")
        assert cur.fetchone()[0] == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd backend && EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py -k "nfl_names or traded or roster_error"`
Expected: 3 failed (`module 'src.edge.nfl_players' has no attribute 'resolve_new_players'`).

- [ ] **Step 3: Implement resolution**

Append to `backend/src/edge/nfl_players.py` (add imports `from datetime import datetime`, `from psycopg2.extras import execute_values`, `from src.edge.players import match_player`, `from src.edge.quotes import Quote`, `from src.edge.teams import NFL_ABBR_BY_NAME`):

```python
async def resolve_new_players(conn, bdl: BdlNflClient, quotes: list[Quote], event: dict, now: datetime) -> int:
    """Match unseen (name, team) pairs from one props poll against both teams' rosters.

    A name is 'seen' per team: a traded player gets a fresh match on his new team.
    Roster errors leave names unseen (retried next poll), never 'unmatched'.
    """
    names = sorted({q.player for q in quotes if q.player})
    teams = {NFL_ABBR_BY_NAME.get(event["home_team"]), NFL_ABBR_BY_NAME.get(event["away_team"])} - {None}
    if not names or not teams:
        return 0
    with conn.cursor() as cur:
        cur.execute("SELECT odds_name FROM edge_nfl_players WHERE odds_name = ANY(%s) AND team = ANY(%s)",
                    (names, sorted(teams)))
        known = {r[0] for r in cur.fetchall()}
    new = [n for n in names if n not in known]
    if not new:
        return 0
    try:
        candidates = [c for t in sorted(teams) for c in await bdl.roster(t)]
    except BdlError as e:
        log.warning("edge NFL player resolve skipped: %s", e)
        return 0
    by_id = {c["id"]: c for c in candidates}
    rows = []
    for name in new:
        m = match_player(name, candidates, teams)
        if m:
            rows.append((name, m.team, m.bdl_player_id, by_id[m.bdl_player_id].get("position"),
                         m.method, m.confidence, now))
        else:
            # No roster match: stored once under the first of the two teams (alphabetical), so the
            # `known` check above stops it being retried every poll; listed on /edge/status.
            rows.append((name, sorted(teams)[0], None, None, "unmatched", 0.0, now))
    with conn.cursor() as cur:
        execute_values(cur, """INSERT INTO edge_nfl_players (odds_name, team, bdl_player_id, position,
                               match_method, confidence, updated_at) VALUES %s
                               ON CONFLICT (odds_name, team) DO NOTHING""", rows)
    conn.commit()
    return len(rows)


def player_ids(conn, teams: set[str]) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("""SELECT odds_name, bdl_player_id FROM edge_nfl_players
                       WHERE bdl_player_id IS NOT NULL AND team = ANY(%s)""", (sorted(teams),))
        return {n: pid for n, pid in cur.fetchall()}
```

- [ ] **Step 4: Wire the worker, config and views**

- **`backend/src/edge/config.py`:** `NFL = SportConfig("americanfootball_nfl", "nfl", NFL_PROP_MARKETS, 7 * 24, True)`. Update `test_sports_table` in `tests/unit/test_edge_sports.py` to `assert ... NFL.has_players` (ledger as a spec-mandated test change).
- **`backend/src/edge/worker.py`:**
  - `tick` and `_tick_sport` gain `nfl_bdl=None` (keyword, passed through).
  - In `_tick_sport`, replace the resolution line with:

```python
        if sport.has_players:
            if sport.slug == "nba":
                await players.resolve_new_players(conn, bdl, quotes, by_id[eid], now)
            elif nfl_bdl is not None and settings.edge_nfl_players:
                await nfl_players.resolve_new_players(conn, nfl_bdl, quotes, by_id[eid], now)
```

  - `_run` creates `nfl_bdl = nfl_players.BdlNflClient()` and passes `nfl_bdl=nfl_bdl` to `tick`. Import `nfl_players`.
- **`backend/src/edge/views.py` `ev_rows`:** replace the `player_ids = …` line with:

```python
    if not sport.has_players:
        player_ids = {}
    elif sport.slug == "nfl":
        teams = {NFL_ABBR_BY_NAME.get(e[k]) for e in events.values() for k in ("home_team", "away_team")} - {None}
        player_ids = nfl_players.player_ids(conn, teams)
    else:
        player_ids = _player_ids(conn)
```

  Import `nfl_players` and `NFL_ABBR_BY_NAME`. In Phase A's `test_views_never_mix_sports`, the assertion `all(r["bdl_player_id"] is None for r in nfl)` still holds, because no NFL players are seeded.

- [ ] **Step 5: Run to verify they pass**

Run: `cd backend && EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py && python3 -m pytest -q tests/unit`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backend/src/edge backend/tests
git commit -m "feat(edge): resolve NFL player names per team; NFL rows link to players

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: NFL schedule, stats and snaps ingest + worker steps + backfill

**Files:**
- Modify: `backend/src/edge/nfl_players.py` (`refresh_schedule`, `load_due_stats`, `load_snaps`)
- Modify: `backend/src/edge/worker.py` (`nfl_stats_step`, `nfl_snaps_step`, `nfl_backfill_step`)
- Test: `backend/tests/integration/test_edge_store_db.py` (append)

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `refresh_schedule(conn, bdl, season, now) -> int`: upserts `edge_nfl_games`; sets `final_seen_at` the first time a game is `final`.
  - `load_due_stats(conn, bdl, now, limit=6) -> int`: loads stats for final games whose `final_seen_at <= now - 2h` and `stats_loaded_at IS NULL`, oldest first, at most `limit` games; returns player rows written.
  - `load_snaps(conn, season, now, frame=None) -> int`: `frame` is an injectable DataFrame for tests; when None it calls `nfl_data_py.import_snap_counts([season])`. Sets `snaps_loaded_at` on games whose two teams have snaps for that week.
  - worker `nfl_stats_step(conn, nfl_bdl, now)`, `nfl_snaps_step(conn, now, frame=None)`, `nfl_backfill_step(conn, nfl_bdl, now)`.

- [ ] **Step 1: Write the failing tests**

```python
def _nfl_games_fixture():
    return json.load(open("tests/fixtures/edge/bdl_nfl_games_w5.json"))["data"]


class _FakeNflSchedule(_FakeNflBdl):
    def __init__(self, games, stats_by_game, fail_stats=None):
        super().__init__({})
        self._games, self._stats, self.fail_stats = games, stats_by_game, fail_stats

    async def games(self, season, week=None):
        return [g for g in self._games if g["season"] == season and (week is None or g["week"] == week)]

    async def stats(self, game_ids):
        from src.edge.players import BdlError
        if self.fail_stats:
            raise BdlError(self.fail_stats, f"BallDontLie NFL {self.fail_stats} on stats")
        return [s for g in game_ids for s in self._stats.get(g, [])]


def test_stats_load_only_two_hours_after_final_and_only_for_listed_games(conn):
    import asyncio
    from src.edge import nfl_players
    games = _nfl_games_fixture()
    stats = json.load(open("tests/fixtures/edge/bdl_nfl_stats_game.json"))["data"]
    gid = stats[0]["game"]["id"]
    preseason = dict(stats[0], game=dict(stats[0]["game"], id=999999))  # not in /games
    bdl = _FakeNflSchedule(games, {gid: stats, 999999: [preseason]})
    asyncio.run(nfl_players.refresh_schedule(conn, bdl, 2026, NOW))
    assert asyncio.run(nfl_players.load_due_stats(conn, bdl, NOW)) == 0          # final just seen: wait 2 h
    n = asyncio.run(nfl_players.load_due_stats(conn, bdl, NOW + timedelta(hours=3), limit=100))
    assert n >= len(stats)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM edge_nfl_player_games WHERE bdl_game_id = 999999")
        assert cur.fetchone()[0] == 0                                             # preseason never enters
        cur.execute("SELECT COUNT(*) FROM edge_nfl_games WHERE stats_loaded_at IS NOT NULL")
        assert cur.fetchone()[0] >= 1
    assert asyncio.run(nfl_players.load_due_stats(conn, bdl, NOW + timedelta(hours=4))) == 0  # idempotent


def test_snaps_map_team_codes_and_mark_games(conn):
    import asyncio
    import pandas as pd
    from src.edge import nfl_players
    g = {"id": 5001, "season": 2026, "week": 6, "date": "2026-10-18T00:00:00.000Z", "postseason": False,
         "status_state": "final", "home_team": {"abbreviation": "LAR"}, "visitor_team": {"abbreviation": "WSH"}}
    asyncio.run(nfl_players.refresh_schedule(conn, _FakeNflSchedule([g], {}), 2026, NOW))
    frame = pd.DataFrame([
        {"season": 2026, "week": 6, "game_type": "REG", "team": "LA", "player": "Puka Nacua",
         "offense_snaps": 60, "defense_snaps": 0, "st_snaps": 0},
        {"season": 2026, "week": 6, "game_type": "REG", "team": "WAS", "player": "Terry McLaurin",
         "offense_snaps": 55, "defense_snaps": 0, "st_snaps": 2}])
    assert nfl_players.load_snaps(conn, 2026, NOW, frame=frame) == 2
    with conn.cursor() as cur:
        cur.execute("SELECT team, player_norm, snaps FROM edge_nfl_snaps ORDER BY team")
        assert cur.fetchall() == [("LAR", "puka nacua", 60), ("WSH", "terry mclaurin", 57)]
        cur.execute("SELECT snaps_loaded_at IS NOT NULL FROM edge_nfl_games WHERE bdl_game_id = 5001")
        assert cur.fetchone()[0] is True


@pytest.mark.asyncio
async def test_backfill_resumes_after_a_rate_limit(conn, monkeypatch):
    from src.edge import nfl_players, worker
    monkeypatch.setattr(nfl_players, "load_snaps", lambda *a, **k: 0)  # never call nflverse in tests
    bdl = _FakeNflSchedule(_nfl_games_fixture(), {}, fail_stats=429)
    await nfl_players.refresh_schedule(conn, bdl, 2026, NOW - timedelta(days=3))  # finals seen long ago
    later = NOW + timedelta(days=30)

    def loaded():
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM edge_nfl_games WHERE stats_loaded_at IS NOT NULL")
            return cur.fetchone()[0]

    await worker.nfl_backfill_step(conn, bdl, later)
    assert "429" in (store.get_state(conn).get("bdl_error") or "") and loaded() == 0  # nothing marked
    bdl.fail_stats = None
    await worker.nfl_backfill_step(conn, bdl, later)
    assert loaded() == 6  # resumes: 6 games per tick
```

(Add `import json` at the top of the integration file if it is not already imported.)

- [ ] **Step 2: Run to verify they fail**

Run: `cd backend && EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py -k "two_hours_after or snaps_map or backfill_resumes_after_a_rate"`
Expected: 3 failed (missing attributes).

- [ ] **Step 3: Implement ingest**

Append to `backend/src/edge/nfl_players.py` (imports: `from datetime import timedelta`, `from psycopg2.extras import RealDictCursor`, `from src.edge.players import normalize_name`, `from src.edge.teams import NFLVERSE_TO_BDL`):

```python
STATS_SETTLE = timedelta(hours=2)  # stat corrections land after the whistle


async def refresh_schedule(conn, bdl: BdlNflClient, season: int, now: datetime) -> int:
    rows = [game_row(g) for g in await bdl.games(season)]
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO edge_nfl_games (bdl_game_id, season, week, game_date, home, away, postseason,
                                        status_state, final_seen_at)
            VALUES %s
            ON CONFLICT (bdl_game_id) DO UPDATE SET
                week = EXCLUDED.week, game_date = EXCLUDED.game_date, status_state = EXCLUDED.status_state,
                final_seen_at = COALESCE(edge_nfl_games.final_seen_at, EXCLUDED.final_seen_at)
        """, [(r["bdl_game_id"], r["season"], r["week"], r["game_date"], r["home"], r["away"], r["postseason"],
               r["status_state"], now if r["status_state"] == "final" else None) for r in rows])
    conn.commit()
    return len(rows)


async def load_due_stats(conn, bdl: BdlNflClient, now: datetime, limit: int = 6) -> int:
    with conn.cursor() as cur:
        cur.execute("""SELECT bdl_game_id FROM edge_nfl_games
                       WHERE status_state = 'final' AND stats_loaded_at IS NULL AND final_seen_at <= %s
                       ORDER BY game_date LIMIT %s""", (now - STATS_SETTLE, limit))
        ids = [r[0] for r in cur.fetchall()]
    if not ids:
        return 0
    rows = [r for r in (stat_row(s) for s in await bdl.stats(ids)) if r and r["bdl_game_id"] in ids]
    cols = ["bdl_player_id", "bdl_game_id", "season", "week", "game_date", "team", "opponent", "is_home", *STAT_COLS]
    with conn.cursor() as cur:
        if rows:
            execute_values(cur, f"""INSERT INTO edge_nfl_player_games ({", ".join(cols)}) VALUES %s
                ON CONFLICT (bdl_player_id, bdl_game_id) DO UPDATE SET
                {", ".join(f"{c} = EXCLUDED.{c}" for c in cols[2:])}""",
                           [tuple(r[c] for c in cols) for r in rows])
        cur.execute("UPDATE edge_nfl_games SET stats_loaded_at = %s WHERE bdl_game_id = ANY(%s)", (now, ids))
    conn.commit()
    return len(rows)


def load_snaps(conn, season: int, now: datetime, frame=None) -> int:
    if frame is None:
        import nfl_data_py as nfl_data  # heavy import, only in the worker
        frame = nfl_data.import_snap_counts([season])
    rows: dict[tuple, int] = {}
    for r in frame.itertuples(index=False):
        team = NFLVERSE_TO_BDL.get(r.team, r.team)
        snaps = int((r.offense_snaps or 0) + (r.defense_snaps or 0) + (r.st_snaps or 0))
        if snaps > 0:
            key = (int(r.season), int(r.week), team, normalize_name(r.player))
            rows[key] = rows.get(key, 0) + snaps
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, """INSERT INTO edge_nfl_snaps (season, week, team, player_norm, snaps, loaded_at)
                               VALUES %s ON CONFLICT (season, week, team, player_norm)
                               DO UPDATE SET snaps = EXCLUDED.snaps, loaded_at = EXCLUDED.loaded_at""",
                       [(*k, v, now) for k, v in rows.items()])
        cur.execute("""UPDATE edge_nfl_games g SET snaps_loaded_at = %s
                       WHERE g.snaps_loaded_at IS NULL AND g.season = %s
                         AND EXISTS (SELECT 1 FROM edge_nfl_snaps s WHERE s.season = g.season AND s.week = g.week AND s.team = g.home)
                         AND EXISTS (SELECT 1 FROM edge_nfl_snaps s WHERE s.season = g.season AND s.week = g.week AND s.team = g.away)""",
                    (now, season))
    conn.commit()
    return len(rows)
```

- [ ] **Step 4: Implement the worker steps**

In `backend/src/edge/worker.py` (import `from src.utils.season import current_nfl_season`):

```python
async def nfl_stats_step(conn, nfl_bdl, now: datetime) -> int:
    """Hourly: refresh this season's NFL schedule, then load settled box scores."""
    last = _parse(store.get_state(conn).get("nfl:last_stats_step"))
    if last is not None and now - last < timedelta(minutes=60):
        return 0
    store.set_state(conn, "nfl:last_stats_step", _iso(now), now)
    await nfl_players.refresh_schedule(conn, nfl_bdl, current_nfl_season(now.date()), now)
    return await nfl_players.load_due_stats(conn, nfl_bdl, now)


def nfl_snaps_step(conn, now: datetime, frame=None) -> int:
    """Every 6 h: nflverse snap counts. Errors land in snaps_error, never block."""
    last = _parse(store.get_state(conn).get("nfl:last_snaps_step"))
    if last is not None and now - last < timedelta(hours=6):
        return 0
    store.set_state(conn, "nfl:last_snaps_step", _iso(now), now)
    try:
        n = nfl_players.load_snaps(conn, current_nfl_season(now.date()), now, frame)
        store.set_state(conn, "snaps_error", "", now)
        return n
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        store.set_state(conn, "snaps_error", redact(f"{type(e).__name__}: {e}")[:300], now)
        return 0


async def nfl_backfill_step(conn, nfl_bdl, now: datetime) -> int:
    """Until done: last season's schedule + snaps once, then settled box scores a few games per tick."""
    season = current_nfl_season(now.date())
    state = store.get_state(conn)
    try:
        if state.get("backfill_nfl_schedule") != str(season - 1):
            await nfl_players.refresh_schedule(conn, nfl_bdl, season - 1, now)
            nfl_players.load_snaps(conn, season - 1, now)
            store.set_state(conn, "backfill_nfl_schedule", str(season - 1), now)
        n = await nfl_players.load_due_stats(conn, nfl_bdl, now, limit=6)
        store.set_state(conn, "backfill_nfl_next", "done" if n == 0 else _iso(now), now)
        return n
    except Exception as e:  # noqa: BLE001 - cursor stays put; next tick retries
        conn.rollback()
        store.set_state(conn, "bdl_error", redact(f"nfl backfill: {type(e).__name__}: {e}")[:300], now)
        return 0
```

`load_due_stats` loads every final, settled game without stats in any season, oldest first. That makes the hourly step and the backfill step the same mechanism, and the backfill finishes when nothing is due.

In `tick`, after the NBA BallDontLie steps loop, add:

```python
    if settings.edge_nfl_players and nfl_bdl is not None and any(s.slug == "nfl" for s in sports):
        for name, coro in (("nfl_stats_step", nfl_stats_step(conn, nfl_bdl, now)),
                           ("nfl_backfill_step", nfl_backfill_step(conn, nfl_bdl, now))):
            try:
                await coro
            except Exception as e:  # noqa: BLE001 - BallDontLie never blocks odds
                conn.rollback()
                store.set_state(conn, "bdl_error", redact(f"{name}: {type(e).__name__}: {e}")[:300], now)
        nfl_snaps_step(conn, now)
```

- [ ] **Step 5: Run to verify they pass**

Run: `cd backend && EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py && python3 -m pytest -q tests/unit`
Expected: all pass. The existing tick tests pass `nfl_bdl=None` by default, so the NFL steps are skipped.

- [ ] **Step 6: Commit**

```bash
git add backend/src/edge backend/tests
git commit -m "feat(edge): NFL schedule, settled box scores, snap counts and backfill

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Grade NFL props

**Files:**
- Modify: `backend/src/edge/nfl_players.py` (`nfl_prop_result`, `grade_props`)
- Modify: `backend/src/edge/grading.py` (`run_grading` calls `nfl_players.grade_props` when `settings.edge_nfl_players`)
- Test: `backend/tests/unit/test_edge_nfl_players.py` (append), `backend/tests/integration/test_edge_store_db.py` (append)

**Interfaces:**
- Consumes: Tasks 1–4 tables.
- Produces:
  - `nfl_prop_result(opp: dict, stat: dict | None, snaps_loaded: bool, played: bool) -> str | None`;
  - `grade_props(conn, now) -> int` (number graded).

- [ ] **Step 1: Write the failing tests**

Unit (append to `test_edge_nfl_players.py`):

```python
def test_nfl_prop_result_rules():
    opp = {"market": "player_reception_yds", "line": 40.5, "side": "over"}
    row = {c: 0 for c in nfl.STAT_COLS} | {"rec_yds": 55}
    assert nfl.nfl_prop_result(opp, row, snaps_loaded=False, played=False) == "win"     # rule 3
    assert nfl.nfl_prop_result(opp, None, snaps_loaded=True, played=True) == "loss"     # rule 4: played, 0 yards
    assert nfl.nfl_prop_result({**opp, "side": "under"}, None, True, True) == "win"
    assert nfl.nfl_prop_result(opp, None, snaps_loaded=True, played=False) == "void"    # rule 5
    assert nfl.nfl_prop_result(opp, None, snaps_loaded=False, played=False) is None     # rule 6: wait
```

Integration (append):

```python
def _seed_graded_nfl_game(conn, status="final", stats_loaded=True, snaps_loaded=True):
    """BDL game 7001: CHI @ ATL on 2026-10-18; opportunity n1 on the matching Odds API event."""
    kickoff = datetime(2026, 10, 18, 17, tzinfo=timezone.utc)
    store.upsert_events(conn, [dict(_nfl_event(), commence_time=kickoff)])
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_nfl_games (bdl_game_id, season, week, game_date, home, away, postseason,
                       status_state, final_seen_at, stats_loaded_at, snaps_loaded_at)
                       VALUES (7001, 2026, 6, '2026-10-18', 'ATL', 'CHI', FALSE, %s, %s, %s, %s)""",
                    (status, kickoff, kickoff + timedelta(hours=6) if stats_loaded else None,
                     kickoff + timedelta(hours=20) if snaps_loaded else None))
        for name, pid, team in (("Bijan Robinson", 10, "ATL"), ("DJ Moore", 20, "CHI"), ("Benched Guy", 30, "ATL"),
                                ("Late Scratch", 40, "CHI")):
            cur.execute("""INSERT INTO edge_nfl_players (odds_name, team, bdl_player_id, match_method, confidence,
                           updated_at) VALUES (%s, %s, %s, 'exact', 1.0, %s)""", (name, team, pid, NOW))
        cur.execute("""INSERT INTO edge_nfl_player_games (bdl_player_id, bdl_game_id, season, week, game_date, team,
                       opponent, is_home, rush_yds, rec_yds) VALUES (10, 7001, 2026, 6, '2026-10-18', 'ATL', 'CHI',
                       TRUE, 88, 12)""")
        cur.execute("""INSERT INTO edge_nfl_snaps (season, week, team, player_norm, snaps, loaded_at) VALUES
                       (2026, 6, 'CHI', 'dj moore', 51, %s), (2026, 6, 'ATL', 'benched guy', 3, %s)""", (NOW, NOW))
        for player, market in (("Bijan Robinson", "player_rush_yds"), ("DJ Moore", "player_receptions"),
                                ("Benched Guy", "player_reception_yds"), ("Late Scratch", "player_rush_yds")):
            cur.execute("""INSERT INTO edge_opportunities (event_id, market, player, side, line, book, price,
                           first_seen_at, last_seen_at, fair_prob, ev, max_ev, n_books, devig_method, closing_done)
                           VALUES ('n1', %s, %s, 'over', 0.5, 'fanduel', 100, %s, %s, 0.5, 0.05, 0.05, 5,
                           'power', TRUE)""", (market, player, NOW, NOW))
    conn.commit()
    return kickoff


def _results(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT player, result FROM edge_opportunities WHERE event_id = 'n1' ORDER BY player")
        return dict(cur.fetchall())


def test_nfl_props_grade_under_book_rules(conn):
    from src.edge import grading
    kickoff = _seed_graded_nfl_game(conn)
    grading.run_grading(conn, kickoff + timedelta(days=1))
    assert _results(conn) == {"Benched Guy": "loss",     # played (3 snaps), no stat row -> 0 < 0.5
                              "Bijan Robinson": "win",   # 88 rush yards
                              "DJ Moore": "loss",        # played 51 snaps, 0 catches
                              "Late Scratch": "void"}    # not in snaps once loaded


def test_nfl_props_wait_for_stats_and_snaps(conn):
    from src.edge import grading
    kickoff = _seed_graded_nfl_game(conn, stats_loaded=False)
    grading.run_grading(conn, kickoff + timedelta(days=1))
    assert set(_results(conn).values()) == {None}
    with conn.cursor() as cur:
        cur.execute("UPDATE edge_nfl_games SET stats_loaded_at = %s, snaps_loaded_at = NULL", (kickoff + timedelta(hours=6),))
        cur.execute("DELETE FROM edge_nfl_snaps")  # snaps not published yet
    conn.commit()
    grading.run_grading(conn, kickoff + timedelta(days=1))
    r = _results(conn)
    assert r["Bijan Robinson"] == "win" and r["Late Scratch"] is None and r["DJ Moore"] is None  # rule 6
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd backend && python3 -m pytest -q tests/unit/test_edge_nfl_players.py -k rules && EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py -k "book_rules or wait_for_stats"`
Expected: failures (missing `nfl_prop_result`; results stay None).

- [ ] **Step 3: Implement**

Append to `backend/src/edge/nfl_players.py`. `grade_over_under` and `profit_units` live in `src/edge/grading.py`, which imports this module, so import them lazily inside the functions to avoid a circular import. `et_date` comes from `src.edge.players`:

```python
def nfl_prop_result(opp: dict, stat: dict | None, snaps_loaded: bool, played: bool) -> str | None:
    from src.edge.grading import grade_over_under
    if stat is not None:
        return grade_over_under(stat_for(opp["market"], stat), float(opp["line"]), opp["side"])
    if played:
        return grade_over_under(0.0, float(opp["line"]), opp["side"])
    if snaps_loaded:
        return "void"
    return None


def grade_props(conn, now: datetime) -> int:
    from src.edge.grading import profit_units
    graded = 0
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT o.id, o.market, o.side, o.line, o.price, o.player, e.home_team, e.away_team,
                              e.commence_time
                       FROM edge_opportunities o JOIN edge_events e USING (event_id)
                       WHERE o.result IS NULL AND o.player <> '' AND e.sport = 'americanfootball_nfl'
                         AND e.commence_time <= %s""", (now - timedelta(hours=3),))
        for o in cur.fetchall():
            home, away = NFL_ABBR_BY_NAME.get(o["home_team"]), NFL_ABBR_BY_NAME.get(o["away_team"])
            cur.execute("""SELECT bdl_player_id, team FROM edge_nfl_players
                           WHERE odds_name = %s AND team = ANY(%s) AND bdl_player_id IS NOT NULL""",
                        (o["player"], [home, away]))
            p = cur.fetchone()
            if p is None:
                continue  # rule 1: unmatched
            kickoff_et = et_date(o["commence_time"])
            cur.execute("""SELECT * FROM edge_nfl_games WHERE home = %s AND away = %s
                           AND game_date BETWEEN %s AND %s""",
                        (home, away, kickoff_et - timedelta(days=1), kickoff_et + timedelta(days=1)))
            g = cur.fetchone()
            if g is None or g["status_state"] != "final" or g["stats_loaded_at"] is None:
                continue  # rule 2: stats not complete
            cur.execute("SELECT * FROM edge_nfl_player_games WHERE bdl_player_id = %s AND bdl_game_id = %s",
                        (p["bdl_player_id"], g["bdl_game_id"]))
            stat = cur.fetchone()
            cur.execute("""SELECT 1 FROM edge_nfl_snaps WHERE season = %s AND week = %s AND team = %s
                           AND player_norm = %s AND snaps > 0""",
                        (g["season"], g["week"], p["team"], normalize_name(o["player"])))
            played = cur.fetchone() is not None
            r = nfl_prop_result(o, dict(stat) if stat else None, g["snaps_loaded_at"] is not None, played)
            if r is not None:
                cur.execute("UPDATE edge_opportunities SET result = %s, profit_units = %s WHERE id = %s",
                            (r, profit_units(r, o["price"]), o["id"]))
                graded += 1
    conn.commit()
    return graded
```

In `backend/src/edge/grading.py` `run_grading`, before `return`:

```python
    if settings.edge_nfl_players:
        from src.edge import nfl_players
        graded += nfl_players.grade_props(conn, now)
```

Import `settings` if `grading.py` does not already.

- [ ] **Step 4: Run to verify they pass**

Run: `cd backend && python3 -m pytest -q tests/unit && EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py`
Expected: all pass. Phase A's `test_nfl_props_are_never_graded_even_when_the_name_matches_an_nba_player` still passes: that NFL "Josh Allen" has no `edge_nfl_players` row, so rule 1 applies.

- [ ] **Step 5: Commit**

```bash
git add backend/src/edge backend/tests
git commit -m "feat(edge): grade NFL props by box score and snap counts (book rules)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: NFL hit rates, player page view/API, status

**Files:**
- Modify: `backend/src/edge/nfl_players.py` (`load_games`, `hit_rates`)
- Modify: `backend/src/edge/views.py` (`nfl_player`, `status` NFL additions)
- Modify: `backend/src/edge/api.py` (`/players/{id}` takes `sport`)
- Test: `backend/tests/unit/test_edge_nfl_players.py`, `backend/tests/integration/test_edge_store_db.py`, `backend/tests/unit/test_edge_api.py` (append)

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  - `nfl_players.load_games(conn, bdl_player_id, team, odds_name) -> list[dict]`: stat rows, plus zero rows for snap-only games.
  - `nfl_players.hit_rates(games, market, line, opponent, is_home, season) -> dict` (keys `l5, l10, l20, season, vs_opponent, home_away`, same shape as NBA).
  - `views.nfl_player(conn, now, bdl_id, market, line) -> dict` (same response shape as `views.player`, plus `position` and per-game `week`).
  - `GET /edge/players/{id}?sport=nfl`.
  - `status()` gains `nfl_unmatched_players`, `nfl_stale_ungraded`, `snaps_error`.

- [ ] **Step 1: Write the failing tests**

Unit:

```python
def test_nfl_hit_rates_count_zero_games():
    rows = [{"season": 2026, "week": w, "game_date": date(2026, 9, 7 * w), "opponent": "TB" if w == 3 else "NO",
             "is_home": w % 2 == 0, **{c: 0 for c in nfl.STAT_COLS}, "rec_yds": y}
            for w, y in ((1, 80), (2, 0), (3, 45), (4, 60))]
    hr = nfl.hit_rates(rows, "player_reception_yds", 50.5, "TB", True, 2026)
    assert hr["l5"] == {"hits": 2, "misses": 2, "pushes": 0, "games": 4, "rate": 0.5}  # the 0 counts
    assert hr["vs_opponent"]["games"] == 1 and hr["home_away"]["games"] == 2
```

Integration:

```python
def test_nfl_player_page_includes_snap_only_games_as_zero(conn):
    from src.edge import views
    _seed_graded_nfl_game(conn)
    page = views.nfl_player(conn, NOW, 20, "player_receptions", 3.5)   # DJ Moore: snaps, no stat row
    assert page["name"] == "DJ Moore" and page["team"] == "CHI"
    assert [g["value"] for g in page["games"]] == [0.0] and page["hit_rates"]["l5"]["games"] == 1
```

API (unit):

```python
def test_player_route_dispatches_on_sport(monkeypatch):
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)
    monkeypatch.setattr(views, "nfl_player", lambda conn, now, pid, market, line: {"sport": "nfl", "id": pid})
    edge_api._cache.clear()
    edge_api._reset_limits()
    r = TestClient(app).get(f"{P}/players/20", params={"sport": "nfl", "market": "player_receptions"})
    assert r.json() == {"sport": "nfl", "id": 20}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd backend && python3 -m pytest -q tests/unit -k "zero_games or dispatches_on_sport" && EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py -k snap_only`
Expected: failures (missing functions).

- [ ] **Step 3: Implement**

`nfl_players.py`:

```python
def _rate(rows: list[dict], market: str, line: float) -> dict:
    hits = sum(1 for r in rows if stat_for(market, r) > line)
    misses = sum(1 for r in rows if stat_for(market, r) < line)
    decided = hits + misses
    return {"hits": hits, "misses": misses, "pushes": len(rows) - decided, "games": len(rows),
            "rate": round(hits / decided, 3) if decided else None}


def hit_rates(games: list[dict], market: str, line: float, opponent: str | None, is_home: bool | None,
              season: int) -> dict:
    played = sorted(games, key=lambda x: x["game_date"], reverse=True)
    return {
        "l5": _rate(played[:5], market, line), "l10": _rate(played[:10], market, line),
        "l20": _rate(played[:20], market, line),
        "season": _rate([x for x in played if x["season"] == season], market, line),
        "vs_opponent": _rate([x for x in played if opponent and x["opponent"] == opponent], market, line),
        "home_away": _rate([x for x in played if is_home is not None and x["is_home"] == is_home], market, line),
    }


def load_games(conn, bdl_player_id: int, team: str, odds_name: str, limit: int = 40) -> list[dict]:
    """Box-score rows plus zero rows for games he played (snap counts) without recording a stat."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM edge_nfl_player_games WHERE bdl_player_id = %s", (bdl_player_id,))
        rows = [dict(r) for r in cur.fetchall()]
        have = {r["bdl_game_id"] for r in rows}
        cur.execute("""SELECT g.* FROM edge_nfl_snaps s JOIN edge_nfl_games g
                         ON g.season = s.season AND g.week = s.week AND s.team IN (g.home, g.away)
                       WHERE s.team = %s AND s.player_norm = %s AND s.snaps > 0""",
                    (team, normalize_name(odds_name)))
        for g in cur.fetchall():
            if g["bdl_game_id"] not in have:
                rows.append({"bdl_game_id": g["bdl_game_id"], "season": g["season"], "week": g["week"],
                             "game_date": g["game_date"], "team": team, "is_home": g["home"] == team,
                             "opponent": g["away"] if g["home"] == team else g["home"],
                             **{c: 0 for c in STAT_COLS}})
    return sorted(rows, key=lambda r: r["game_date"], reverse=True)[:limit]
```

`views.py`, add:

```python
def nfl_player(conn, now: datetime, bdl_id: int, market: str, line: float | None) -> dict:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT odds_name, team, position FROM edge_nfl_players WHERE bdl_player_id = %s
                       ORDER BY updated_at DESC""", (bdl_id,))
        names = [dict(r) for r in cur.fetchall()]
    if not names:
        return {"bdl_player_id": bdl_id, "name": None, "team": None, "position": None, "market": market,
                "line": line, "opponent": None, "is_home": None, "hit_rates": None, "games": [], "prices": []}
    team, odds_names = names[0]["team"], {n["odds_name"] for n in names}
    games_ = nfl_players.load_games(conn, bdl_id, team, names[0]["odds_name"])
    events = store.upcoming_events(conn, now, hours=NFL.lines_window_hours, sport=NFL.key)
    event = next((e for e in events if team in (NFL_ABBR_BY_NAME.get(e["home_team"]),
                                                NFL_ABBR_BY_NAME.get(e["away_team"]))), None)
    prices = []
    if event:
        prices = [q for q in store.load_quotes(conn, [event["event_id"]], market) if q.player in odds_names]
        if line is None and prices:
            line = statistics.median(q.line for q in prices)
    opponent = is_home = None
    if event:
        is_home = NFL_ABBR_BY_NAME.get(event["home_team"]) == team
        opponent = NFL_ABBR_BY_NAME.get(event["away_team"] if is_home else event["home_team"])
    return {
        "bdl_player_id": bdl_id, "name": names[0]["odds_name"], "team": team, "position": names[0]["position"],
        "market": market, "line": line, "opponent": opponent, "is_home": is_home,
        "hit_rates": nfl_players.hit_rates(games_, market, line, opponent, is_home,
                                           current_nfl_season(now.date())) if line is not None else None,
        "games": [{"game_date": g["game_date"].isoformat(), "week": g["week"], "opponent": g["opponent"],
                   "is_home": g["is_home"], "minutes": 0, "value": nfl_players.stat_for(market, g)}
                  for g in games_[:20]],
        "prices": [{"book": q.book, "side": q.side, "line": q.line, "price": q.price, "link": q.link}
                   for q in prices],
    }
```

Import `NFL` from config, and `current_nfl_season`. In `status()`, add three keys:

```python
    with conn.cursor() as cur:
        cur.execute("SELECT odds_name || ' (' || team || ')' FROM edge_nfl_players WHERE match_method = 'unmatched' ORDER BY 1")
        nfl_unmatched = [r[0] for r in cur.fetchall()]
        cur.execute("""SELECT COUNT(*) FROM edge_opportunities o JOIN edge_events e USING (event_id)
                       WHERE o.result IS NULL AND o.player <> '' AND e.sport = 'americanfootball_nfl'
                         AND e.commence_time < %s""", (now - timedelta(days=7),))
        nfl_stale = cur.fetchone()[0]
```

and return `"nfl_unmatched_players": nfl_unmatched, "nfl_stale_ungraded": nfl_stale, "snaps_error": state.get("snaps_error") or None`.

`api.py`: `get_player` gains `sp: SportConfig = Depends(_sport)`. When `sp.slug == "nfl"`, it returns `_cached(("player", "nfl", bdl_id, market, line), lambda c, now: views.nfl_player(c, now, bdl_id, market, line))`. Otherwise it runs the existing NBA path with its existing cache key prefixed by `"nba"`.

- [ ] **Step 4: Run to verify they pass**

Run: `cd backend && python3 -m pytest -q tests/unit && EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/src/edge backend/tests
git commit -m "feat(edge): NFL player pages with hit rates; status lists NFL unmatched/stale

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Frontend: NFL player page, track record, links, nav

**Files:**
- Modify: `frontend/src/lib/edgeApi.ts` (`player` takes `sport`), `frontend/src/hooks/useEdge.ts` (`useEdgePlayer` takes `sport`)
- Modify: `frontend/src/types/edge.ts` (`PlayerGame.week?`, `PlayerResponse.position?`)
- Modify: `frontend/src/pages/EdgePlayer.tsx`, `frontend/src/pages/EdgeResults.tsx`, `frontend/src/pages/EdgeFinder.tsx` (remove the "coming next week" note)
- Modify: `frontend/src/components/edge/EvTable.tsx` (player links for any sport, under `edgeBase`)
- Modify: `frontend/src/App.tsx`, `frontend/src/components/Layout.tsx`
- Modify: `frontend/src/lib/edgeFormat.ts` (`propMarketsFor(sport)`)
- Test: `frontend/src/lib/edgeFormat.test.ts`, `frontend/src/App.sportRoutes.test.tsx` (append)

**Interfaces:**
- Consumes: the API from Task 6.
- Produces:
  - routes `/nfl/edge/player/:id` and `/nfl/edge/results`;
  - `propMarketsFor(sport: Sport): string[]` (the prop part of `marketKeysFor`).

- [ ] **Step 1: Write the failing tests**

Append to `edgeFormat.test.ts`:

```ts
import { propMarketsFor } from './edgeFormat';
describe('prop markets', () => {
  it('lists only player markets per sport', () => {
    expect(propMarketsFor('nfl')).toHaveLength(10);
    expect(propMarketsFor('nfl').every((m) => m.startsWith('player_'))).toBe(true);
    expect(propMarketsFor('nba')).toContain('player_points');
  });
});
```

Append to `App.sportRoutes.test.tsx` (inside the existing `describe`):

```tsx
  it('serves the NFL player page and track record from NFL data', async () => {
    (edgeApi.player as any).mockResolvedValue({ bdl_player_id: 20, name: 'DJ Moore', team: 'CHI', position: 'WR',
      market: 'player_receptions', line: 4.5, opponent: 'ATL', is_home: false, hit_rates: null, games: [], prices: [] });
    (edgeApi.results as any).mockResolvedValue({ n: 0, clv: { measured: 0, bets: 0, mean_clv: null, median_clv: null,
      beat_close_rate: null, std_error: null, verdict: '', ci95: null }, record: { wins: 0, losses: 0, pushes: 0,
      voids: 0, win_rate: null, units: 0, roi: null }, by_market: {}, by_book: {}, daily_clv: [] });
    render(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter initialEntries={['/nfl/edge/player/20?market=player_receptions']}><Nav /><App /></MemoryRouter>
      </QueryClientProvider>,
    );
    await screen.findByText('DJ Moore');
    expect(edgeApi.player).toHaveBeenLastCalledWith(20, 'player_receptions', undefined, undefined, 'nfl');
    expect(screen.queryByText('All minutes')).toBeNull();          // no minutes filter on NFL
    act(() => nav('/nfl/edge/results'));
    await waitFor(() => expect(edgeApi.results).toHaveBeenLastCalledWith(365, 'nfl'));
  });
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd frontend && npx vitest run`
Expected: failures (`propMarketsFor` missing; route not found).

- [ ] **Step 3: Implement**

- **`edgeFormat.ts`:** `export const propMarketsFor = (sport: Sport) => marketKeysFor(sport).filter((m) => m.startsWith('player_'));`
- **`edgeApi.player`:** `async player(id, market, line?, minMinutes?, sport: Sport = 'nba')`; params add `sport`.
- **`useEdgePlayer`:**

```ts
export const useEdgePlayer = (id: number, market: string, line?: number, minMinutes?: number, sport: Sport = 'nba') =>
  useQuery({ queryKey: ['edge', sport, 'player', id, market, line, minMinutes],
    queryFn: () => edgeApi.player(id, market, line, minMinutes, sport), enabled: id > 0 });
```

- **`types/edge.ts`:** `PlayerGame` gains `week?: number`; `PlayerResponse` gains `position?: string | null`.
- **`EdgePlayer`:**
  - Signature `export function EdgePlayer({ sport = 'nba' }: { sport?: Sport })`.
  - The default market is `sport === 'nfl' ? 'player_reception_yds' : 'player_points'`.
  - Its buttons map `propMarketsFor(sport)` (delete the local `PROP_MARKETS`).
  - `useEdgePlayer(Number(id), market, undefined, sport === 'nba' ? minMinutes : undefined, sport)`.
  - The minutes `<select>` renders only when `sport === 'nba'`.
  - The chart label is `sport === 'nfl' && g.week ? \`W${g.week}\` : g.game_date.slice(5)`.
  - The header shows `{data.position ? \`${data.position} · \` : ''}{data.team}`.
  - On NFL, a footer note under the chart: `Stats: BallDontLie. Participation: nflverse snap counts.`
- **`EdgeResults`:** `export function EdgeResults({ sport = 'nba' }: { sport?: Sport })`; `useEdgeResults(365, sport)`; heading `{sport === 'nfl' ? 'NFL track record' : 'Track record'}`.
- **`EdgeFinder`:** delete the `{sport === 'nfl' && (<p …>coming next week</p>)}` block.
- **`EvTable`:** the player link condition becomes `r.player && playerIds[r.player]` (any sport), and the URL is ``to={`${edgeBase(sport)}/player/${playerIds[r.player]}?market=${r.market}`}``.
- **`App.tsx`:** add `<Route path="/nfl/edge/player/:id" element={<EdgePlayer key="nfl" sport="nfl" />} />` and `<Route path="/nfl/edge/results" element={<EdgeResults key="nfl" sport="nfl" />} />`, and give the NBA `EdgePlayer`/`EdgeResults` routes `key="nba"`.
- **`Layout.tsx`** NFL pages: `[{ path: '/nfl', label: 'Picks' }, { path: '/nfl/edge', label: '+EV' }, { path: '/nfl/edge/results', label: 'Track record' }, { path: '/nfl/performance', label: 'Results' }]`.

- [ ] **Step 4: Run tests, type check, build**

Run: `cd frontend && npx vitest run && npx tsc --noEmit && npm run build`
Expected: all pass; tsc clean; build ok.

- [ ] **Step 5: Commit**

```bash
git add frontend/src
git commit -m "feat(frontend): NFL player pages and track record

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Ship and verify live (by Sat 2026-10-24)

**Files:** none. Results go in the ledger.

- [ ] **Step 1: Full verification on the branch**

Run:
- `cd backend && python3 -m pytest -q tests/unit`
- `EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest python3 -m pytest -q tests/integration/test_edge_store_db.py tests/integration/test_startup_migrations_db.py`
- `cd ../frontend && npx vitest run && npx tsc --noEmit && npm run build`

Expected: all green.

- [ ] **Step 2: Merge and deploy (with Samir's go-ahead)**

`git checkout main && git merge --ff-only edge-nfl-b && git push origin main`. This deploys `nba-value`, `edge-worker` and Vercel, and the new tables are created on startup.

- [ ] **Step 3: Watch ingest**

Check `/edge/status` (X-Cron-Key):
- `backfill_nfl_schedule` = `2025`;
- `backfill_nfl_next` reaches `done`;
- no lasting `bdl_error`/`snaps_error`;
- `nfl_unmatched_players` reviewed.

- [ ] **Step 4: Live check**

1. Pick 10 graded NFL props (`edge_opportunities` joined to `edge_events` with `sport = 'americanfootball_nfl'` and `result` set; read through `/edge/results?sport=nfl` and the DB tools available). Compare each against the official box score (and snap counts for any void).
2. Open one NFL player page. Compare the L5 hit rate with a hand count from its game log.
3. Confirm `/edge/results` (NBA) is unchanged and has no NFL rows.

Record all three in the ledger.
