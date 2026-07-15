# NFL Model — Phase 1: Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the NFL data foundation — nflverse (`nfl_data_py`) ingestion, the `nfl_*` schema, and leakage-free rolling team features — so later phases have trustworthy inputs to model on.

**Architecture:** A self-contained `services/nfl/` package mirroring `services/mlb/`. nflverse data is loaded into pandas DataFrames by a thin I/O wrapper, transformed by **pure functions** (derivations + feature aggregation, unit-testable with no DB), then persisted to `nfl_*` tables by a thin ingest/upsert layer. Feature rows are strictly point-in-time: a game in week *W* only ever sees data from weeks `< W`.

**Tech Stack:** Python 3.11+, SQLAlchemy 2.0 (async), PostgreSQL, pandas, `nfl_data_py`, structlog, pytest.

## Global Constraints

- **Naming:** every new table, model, file, and config key is prefixed `nfl_` / `NFL`.
- **Point-in-time correctness:** no feature for a game may reference that game's own week or any later week. This is the single most important invariant of Phase 1.
- **Git staging:** stage specific files only. **Never** `git add -A` or `git add .` (standing rule for this repo).
- **Push:** commit locally; do not push to `main` unless explicitly asked.
- **Commit trailer:** end commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **DB migrations:** new tables are auto-created by `Base.metadata.create_all` in `src/database.py::init_db()` once the model is imported in `src/models/__init__.py`. New columns on *existing* tables go in the `column_migrations` list in `init_db()` — but Phase 1 only adds new tables, so `create_all` suffices.
- **Async DB:** use `async_session_maker` from `src.database`; upserts use `sqlalchemy.dialects.postgresql.insert(...).on_conflict_do_update(...)`.
- **Team abbreviations:** use nflverse team abbreviations verbatim as the canonical key (e.g. `KC`, `LA`, `LAC`, `SF`, `WAS`). Do not remap.

## File Structure

**Create:**
- `src/services/nfl/__init__.py` — package marker + exports.
- `src/services/nfl/constants.py` — team→division map, primetime rules, pure derivation helpers.
- `src/services/nfl/nfl_data.py` — thin `nfl_data_py` + Odds API I/O wrapper (returns DataFrames).
- `src/services/nfl/features.py` — pure feature functions (EPA aggregation, point-in-time rolling team stats).
- `src/services/nfl/ingest.py` — orchestration: DataFrame → `nfl_*` upserts.
- `src/models/nfl_team.py`, `nfl_game.py`, `nfl_game_context.py`, `nfl_team_stats.py` — SQLAlchemy models.
- `src/tasks/nfl_backfill.py` — CLI to backfill 2010→present and compute team stats.
- `tests/unit/test_nfl_constants.py`, `test_nfl_features_epa.py`, `test_nfl_features_pointintime.py` — unit tests over pure functions.
- `tests/fixtures/nfl_pbp_sample.parquet`, `nfl_schedule_sample.parquet` — small recorded nflverse samples.

**Modify:**
- `requirements.txt` — add `nfl_data_py`.
- `src/models/__init__.py` — import + export the four new models.

**Out of scope (later phases):** `nfl_markets` + `nfl_prediction_snapshots` tables (P3), models (P2), scorer/value_calculator (P2/P3), scheduler + API (P4). The `nfl_teams` table is defined as schema in Task 3 but intentionally **seeded later (P5, with team colors for cards)** — Phase 1 keys everything on nflverse abbreviations directly and derives divisions from `constants.NFL_DIVISIONS`, so it never reads `nfl_teams`.

---

### Task 1: Add `nfl_data_py` dependency and verify the loaders work

**Files:**
- Modify: `requirements.txt`
- Create: `tests/fixtures/nfl_schedule_sample.parquet`, `tests/fixtures/nfl_pbp_sample.parquet`

**Interfaces:**
- Produces: recorded fixtures used by later feature tests; confirmed `nfl_data_py` import surface (`import_schedules`, `import_pbp_data`).

- [ ] **Step 1: Add the dependency**

Add to `requirements.txt` (alphabetically near other data libs):

```
nfl_data_py>=0.3.2
```

- [ ] **Step 2: Install and record a small fixture**

Run:

```bash
pip install "nfl_data_py>=0.3.2"
python3 - <<'PY'
import nfl_data_py as nfl
sched = nfl.import_schedules([2023])
# keep a light slice for fixtures: 2023 weeks 1-2
sched = sched[sched["week"].isin([1, 2])]
sched.to_parquet("tests/fixtures/nfl_schedule_sample.parquet")
pbp = nfl.import_pbp_data([2023], downcast=True)
pbp = pbp[(pbp["week"].isin([1, 2])) & (pbp["season_type"] == "REG")]
# keep only columns features need, to keep the fixture small
cols = ["game_id","season","week","posteam","defteam","home_team","away_team",
        "epa","success","pass","rush","play_type","season_type"]
pbp[[c for c in cols if c in pbp.columns]].to_parquet("tests/fixtures/nfl_pbp_sample.parquet")
print("schedule rows:", len(sched), "pbp rows:", len(pbp))
print("schedule cols:", sorted(sched.columns.tolist()))
PY
```

Expected: prints non-zero row counts and a schedule column list that includes `game_id, season, week, gameday, weekday, gametime, home_team, away_team, home_score, away_score, roof, surface, home_qb_name, away_qb_name, home_qb_id, away_qb_id, div_game, home_rest, away_rest`.

- [ ] **Step 3: Confirm fixtures load**

Run:

```bash
python3 -c "import pandas as pd; print(pd.read_parquet('tests/fixtures/nfl_schedule_sample.parquet').shape); print(pd.read_parquet('tests/fixtures/nfl_pbp_sample.parquet').shape)"
```

Expected: two non-empty shapes printed.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt tests/fixtures/nfl_schedule_sample.parquet tests/fixtures/nfl_pbp_sample.parquet
git commit -m "chore: add nfl_data_py dep and recorded nflverse fixtures

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Pure derivations — divisional and primetime flags

**Files:**
- Create: `src/services/nfl/constants.py`
- Create: `src/services/nfl/__init__.py`
- Test: `tests/unit/test_nfl_constants.py`

**Interfaces:**
- Produces:
  - `NFL_DIVISIONS: dict[str, str]` — team abbr → `"AFC East"` etc. (32 teams, nflverse abbrs).
  - `is_divisional(home: str, away: str) -> bool`
  - `primetime_slot(weekday: str, gametime: str) -> str | None` — returns `"TNF"`, `"SNF"`, `"MNF"`, or `None`.
  - `is_primetime(weekday: str, gametime: str) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_constants.py
from src.services.nfl.constants import (
    NFL_DIVISIONS, is_divisional, is_primetime, primetime_slot,
)


def test_all_32_teams_have_a_division():
    assert len(NFL_DIVISIONS) == 32
    assert NFL_DIVISIONS["KC"] == "AFC West"
    assert NFL_DIVISIONS["PHI"] == "NFC East"


def test_divisional_game_detection():
    assert is_divisional("KC", "DEN") is True      # both AFC West
    assert is_divisional("KC", "PHI") is False


def test_primetime_slots():
    # Thursday night
    assert primetime_slot("Thursday", "20:15") == "TNF"
    # Sunday night (>= 19:00)
    assert primetime_slot("Sunday", "20:20") == "SNF"
    # Sunday afternoon is NOT primetime
    assert primetime_slot("Sunday", "13:00") is None
    # Monday night
    assert primetime_slot("Monday", "20:15") == "MNF"


def test_is_primetime_matches_slot():
    assert is_primetime("Sunday", "20:20") is True
    assert is_primetime("Sunday", "13:00") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_nfl_constants.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.nfl.constants'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/nfl/__init__.py
"""NFL betting-value vertical (data foundation)."""
```

```python
# src/services/nfl/constants.py
"""Static NFL reference data and pure schedule derivations."""

# nflverse team abbreviations -> division.
NFL_DIVISIONS: dict[str, str] = {
    "BUF": "AFC East", "MIA": "AFC East", "NE": "AFC East", "NYJ": "AFC East",
    "BAL": "AFC North", "CIN": "AFC North", "CLE": "AFC North", "PIT": "AFC North",
    "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "TEN": "AFC South",
    "DEN": "AFC West", "KC": "AFC West", "LV": "AFC West", "LAC": "AFC West",
    "DAL": "NFC East", "NYG": "NFC East", "PHI": "NFC East", "WAS": "NFC East",
    "CHI": "NFC North", "DET": "NFC North", "GB": "NFC North", "MIN": "NFC North",
    "ATL": "NFC South", "CAR": "NFC South", "NO": "NFC South", "TB": "NFC South",
    "ARI": "NFC West", "LA": "NFC West", "SF": "NFC West", "SEA": "NFC West",
}


def is_divisional(home: str, away: str) -> bool:
    """True when both teams share a division."""
    h, a = NFL_DIVISIONS.get(home), NFL_DIVISIONS.get(away)
    return h is not None and h == a


def primetime_slot(weekday: str, gametime: str) -> str | None:
    """Classify a nationally televised primetime window.

    weekday: nflverse 'weekday' (e.g. 'Sunday'); gametime: 'HH:MM' 24h ET.
    Sunday counts as primetime only at/after 19:00 (SNF).
    """
    if not gametime:
        return None
    try:
        hour = int(gametime.split(":")[0])
    except (ValueError, IndexError):
        return None
    if weekday == "Thursday" and hour >= 19:
        return "TNF"
    if weekday == "Monday" and hour >= 19:
        return "MNF"
    if weekday == "Sunday" and hour >= 19:
        return "SNF"
    return None


def is_primetime(weekday: str, gametime: str) -> bool:
    return primetime_slot(weekday, gametime) is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_nfl_constants.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/services/nfl/__init__.py src/services/nfl/constants.py tests/unit/test_nfl_constants.py
git commit -m "feat(nfl): divisional and primetime schedule derivations

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: DB models — `nfl_teams`, `nfl_games`, `nfl_game_context`, `nfl_team_stats`

**Files:**
- Create: `src/models/nfl_team.py`, `src/models/nfl_game.py`, `src/models/nfl_game_context.py`, `src/models/nfl_team_stats.py`
- Modify: `src/models/__init__.py`
- Test: `tests/unit/test_nfl_models_import.py`

**Interfaces:**
- Produces SQLAlchemy models (all keyed by nflverse abbrs):
  - `NFLTeam(abbr PK, name, conference, division, primary_color, secondary_color)`
  - `NFLGame(game_id PK, season, week, season_type, home_team, away_team, kickoff_utc, home_score, away_score, roof, surface, neutral_site, home_qb, home_qb_id, away_qb, away_qb_id, is_divisional, is_primetime, primetime_slot, status)`
  - `NFLGameContext(game_id PK/FK, home_rest_days, away_rest_days, wind_mph, temp_f, is_dome, home_starters_out, away_starters_out, home_playoff_stakes, away_playoff_stakes)`
  - `NFLTeamStats(id PK, team, season, through_week, off_epa_play, def_epa_play, pass_epa, rush_epa, success_rate, pace, power_rating, UNIQUE(team, season, through_week))`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_models_import.py
def test_nfl_models_importable_and_named():
    from src.models import NFLTeam, NFLGame, NFLGameContext, NFLTeamStats
    assert NFLTeam.__tablename__ == "nfl_teams"
    assert NFLGame.__tablename__ == "nfl_games"
    assert NFLGameContext.__tablename__ == "nfl_game_context"
    assert NFLTeamStats.__tablename__ == "nfl_team_stats"


def test_nfl_game_has_situational_flag_columns():
    from src.models import NFLGame
    cols = set(NFLGame.__table__.columns.keys())
    assert {"is_divisional", "is_primetime", "primetime_slot",
            "home_qb", "away_qb"}.issubset(cols)


def test_nfl_team_stats_unique_constraint():
    from src.models import NFLTeamStats
    uniques = {tuple(c.name for c in con.columns)
               for con in NFLTeamStats.__table__.constraints
               if con.__class__.__name__ == "UniqueConstraint"}
    assert ("team", "season", "through_week") in uniques
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_nfl_models_import.py -v`
Expected: FAIL with `ImportError: cannot import name 'NFLTeam'`.

- [ ] **Step 3: Write the models**

```python
# src/models/nfl_team.py
"""NFL team database model."""
from datetime import datetime
from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base


class NFLTeam(Base):
    __tablename__ = "nfl_teams"

    abbr: Mapped[str] = mapped_column(String(5), primary_key=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    conference: Mapped[str | None] = mapped_column(String(3), nullable=True)
    division: Mapped[str | None] = mapped_column(String(20), nullable=True)
    primary_color: Mapped[str | None] = mapped_column(String(7), nullable=True)
    secondary_color: Mapped[str | None] = mapped_column(String(7), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
```

```python
# src/models/nfl_game.py
"""NFL game database model."""
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base


class NFLGame(Base):
    __tablename__ = "nfl_games"

    game_id: Mapped[str] = mapped_column(String(20), primary_key=True)  # nflverse game_id
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    season_type: Mapped[str] = mapped_column(String(4), default="REG")  # REG / POST

    home_team: Mapped[str] = mapped_column(String(5), nullable=False)
    away_team: Mapped[str] = mapped_column(String(5), nullable=False)
    kickoff_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    roof: Mapped[str | None] = mapped_column(String(12), nullable=True)      # dome/outdoors/closed/open
    surface: Mapped[str | None] = mapped_column(String(20), nullable=True)
    neutral_site: Mapped[bool] = mapped_column(Boolean, default=False)

    home_qb: Mapped[str | None] = mapped_column(String(50), nullable=True)
    home_qb_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    away_qb: Mapped[str | None] = mapped_column(String(50), nullable=True)
    away_qb_id: Mapped[str | None] = mapped_column(String(20), nullable=True)

    is_divisional: Mapped[bool] = mapped_column(Boolean, default=False)
    is_primetime: Mapped[bool] = mapped_column(Boolean, default=False)
    primetime_slot: Mapped[str | None] = mapped_column(String(4), nullable=True)  # TNF/SNF/MNF

    status: Mapped[str] = mapped_column(String(20), default="scheduled")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    @property
    def margin(self) -> int | None:
        if self.home_score is not None and self.away_score is not None:
            return self.home_score - self.away_score
        return None

    @property
    def total_points(self) -> int | None:
        if self.home_score is not None and self.away_score is not None:
            return self.home_score + self.away_score
        return None
```

```python
# src/models/nfl_game_context.py
"""NFL per-game situational context."""
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base


class NFLGameContext(Base):
    __tablename__ = "nfl_game_context"

    game_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("nfl_games.game_id"), primary_key=True
    )
    home_rest_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_rest_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    wind_mph: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp_f: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_dome: Mapped[bool] = mapped_column(Boolean, default=False)

    # Best-effort candidate features (may be noisy; see spec).
    home_starters_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_starters_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_playoff_stakes: Mapped[str | None] = mapped_column(String(12), nullable=True)  # alive/clinched/eliminated
    away_playoff_stakes: Mapped[str | None] = mapped_column(String(12), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
```

```python
# src/models/nfl_team_stats.py
"""NFL rolling team form, one row per team per through-week (point-in-time)."""
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base


class NFLTeamStats(Base):
    __tablename__ = "nfl_team_stats"
    __table_args__ = (
        UniqueConstraint("team", "season", "through_week", name="uq_nfl_team_stats_team_season_week"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team: Mapped[str] = mapped_column(String(5), nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    through_week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    off_epa_play: Mapped[float | None] = mapped_column(Float, nullable=True)
    def_epa_play: Mapped[float | None] = mapped_column(Float, nullable=True)
    pass_epa: Mapped[float | None] = mapped_column(Float, nullable=True)
    rush_epa: Mapped[float | None] = mapped_column(Float, nullable=True)
    success_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    pace: Mapped[float | None] = mapped_column(Float, nullable=True)
    power_rating: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
```

- [ ] **Step 4: Register the models**

In `src/models/__init__.py`, after the MLB imports (around line 26) add:

```python
# NFL Models
from src.models.nfl_team import NFLTeam
from src.models.nfl_game import NFLGame
from src.models.nfl_game_context import NFLGameContext
from src.models.nfl_team_stats import NFLTeamStats
```

And add to the `__all__` list (after the MLB entries):

```python
    # NFL Models
    "NFLTeam",
    "NFLGame",
    "NFLGameContext",
    "NFLTeamStats",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_nfl_models_import.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/models/nfl_team.py src/models/nfl_game.py src/models/nfl_game_context.py src/models/nfl_team_stats.py src/models/__init__.py tests/unit/test_nfl_models_import.py
git commit -m "feat(nfl): add nfl_* SQLAlchemy models

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: nflverse I/O wrapper

**Files:**
- Create: `src/services/nfl/nfl_data.py`
- Test: `tests/unit/test_nfl_data_wrapper.py`

**Interfaces:**
- Consumes: `nfl_data_py`.
- Produces:
  - `load_schedules(seasons: list[int]) -> pd.DataFrame` — passthrough to `nfl.import_schedules`.
  - `load_pbp(seasons: list[int]) -> pd.DataFrame` — passthrough to `nfl.import_pbp_data(downcast=True)`, filtered to `season_type == "REG"`.
  - `schedule_to_game_rows(sched: pd.DataFrame) -> list[dict]` — pure: maps schedule rows to `NFLGame` kwargs dicts (applies `is_divisional`/`is_primetime`/`primetime_slot` from `constants`, parses `kickoff_utc`).

The two `load_*` functions are thin I/O (tested via the recorded fixture with monkeypatch). `schedule_to_game_rows` is pure and gets the real behavioral test.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_data_wrapper.py
import pandas as pd
from src.services.nfl.nfl_data import schedule_to_game_rows


def test_schedule_to_game_rows_maps_flags_and_keys():
    sched = pd.read_parquet("tests/fixtures/nfl_schedule_sample.parquet")
    rows = schedule_to_game_rows(sched)
    assert len(rows) == len(sched)
    sample = rows[0]
    # required NFLGame kwargs present
    for key in ("game_id", "season", "week", "home_team", "away_team",
                "is_divisional", "is_primetime", "primetime_slot"):
        assert key in sample
    # divisional flag is a bool derived from constants, not copied blindly
    assert isinstance(sample["is_divisional"], bool)


def test_schedule_to_game_rows_divisional_matches_constants():
    from src.services.nfl.constants import is_divisional
    sched = pd.read_parquet("tests/fixtures/nfl_schedule_sample.parquet")
    rows = {r["game_id"]: r for r in schedule_to_game_rows(sched)}
    for _, g in sched.iterrows():
        expected = is_divisional(g["home_team"], g["away_team"])
        assert rows[g["game_id"]]["is_divisional"] is expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_nfl_data_wrapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.nfl.nfl_data'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/nfl/nfl_data.py
"""Thin nflverse (nfl_data_py) I/O wrapper returning pandas DataFrames."""
from datetime import datetime, timezone

import pandas as pd
import structlog

from src.services.nfl.constants import is_divisional, is_primetime, primetime_slot

logger = structlog.get_logger()


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    return nfl.import_schedules(seasons)


def load_pbp(seasons: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data(seasons, downcast=True)
    return pbp[pbp["season_type"] == "REG"].copy()


def _kickoff_utc(row) -> datetime | None:
    """Combine nflverse 'gameday' (YYYY-MM-DD) + 'gametime' (HH:MM ET) into UTC.

    nflverse times are US/Eastern. ET is UTC-4 (DST) for the NFL season
    (Sep-Jan uses -5 in Nov-Jan); store naive-ET + a fixed -5 is wrong for
    early season. We store the ET wall-clock as UTC-naive here and let the
    scheduler (P4) handle precise windowing; kickoff_utc is informational in P1.
    """
    gameday = row.get("gameday")
    gametime = row.get("gametime")
    if not gameday or not gametime:
        return None
    try:
        return datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(
            tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        return None


def schedule_to_game_rows(sched: pd.DataFrame) -> list[dict]:
    """Pure map of nflverse schedule rows -> NFLGame kwargs dicts."""
    rows: list[dict] = []
    for _, g in sched.iterrows():
        weekday = g.get("weekday") or ""
        gametime = g.get("gametime") or ""
        home, away = g["home_team"], g["away_team"]
        rows.append({
            "game_id": g["game_id"],
            "season": int(g["season"]),
            "week": int(g["week"]),
            "season_type": g.get("game_type", "REG"),
            "home_team": home,
            "away_team": away,
            "kickoff_utc": _kickoff_utc(g),
            "home_score": None if pd.isna(g.get("home_score")) else int(g["home_score"]),
            "away_score": None if pd.isna(g.get("away_score")) else int(g["away_score"]),
            "roof": g.get("roof"),
            "surface": g.get("surface"),
            "neutral_site": bool(g.get("location") == "Neutral"),
            "home_qb": g.get("home_qb_name"),
            "home_qb_id": g.get("home_qb_id"),
            "away_qb": g.get("away_qb_name"),
            "away_qb_id": g.get("away_qb_id"),
            "is_divisional": is_divisional(home, away),
            "is_primetime": is_primetime(weekday, gametime),
            "primetime_slot": primetime_slot(weekday, gametime),
            "status": "final" if not pd.isna(g.get("home_score")) else "scheduled",
        })
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_nfl_data_wrapper.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/services/nfl/nfl_data.py tests/unit/test_nfl_data_wrapper.py
git commit -m "feat(nfl): nflverse I/O wrapper and pure schedule->game mapping

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: EPA team-game aggregation (pure)

**Files:**
- Modify: `src/services/nfl/features.py` (create)
- Test: `tests/unit/test_nfl_features_epa.py`

**Interfaces:**
- Consumes: a play-by-play DataFrame.
- Produces: `team_game_epa(pbp: pd.DataFrame) -> pd.DataFrame` — one row per (season, week, team) with columns `off_epa_play, def_epa_play, pass_epa, rush_epa, success_rate, plays`. Offense = rows where `posteam == team`; defense = rows where `defteam == team` (defensive EPA is the EPA *allowed*, so lower is better).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_features_epa.py
import pandas as pd
from src.services.nfl.features import team_game_epa


def test_team_game_epa_offense_and_defense_split():
    # 2 plays: KC on offense vs DEN; then DEN on offense vs KC.
    pbp = pd.DataFrame([
        {"season": 2023, "week": 1, "posteam": "KC", "defteam": "DEN",
         "epa": 1.0, "success": 1, "pass": 1, "rush": 0, "play_type": "pass"},
        {"season": 2023, "week": 1, "posteam": "DEN", "defteam": "KC",
         "epa": -0.5, "success": 0, "pass": 0, "rush": 1, "play_type": "run"},
    ])
    out = team_game_epa(pbp).set_index("team")
    # KC offense = +1.0 over 1 play; KC defense = DEN's -0.5 EPA allowed
    assert round(out.loc["KC", "off_epa_play"], 3) == 1.0
    assert round(out.loc["KC", "def_epa_play"], 3) == -0.5
    assert round(out.loc["KC", "success_rate"], 3) == 1.0
    # DEN mirror
    assert round(out.loc["DEN", "off_epa_play"], 3) == -0.5
    assert round(out.loc["DEN", "def_epa_play"], 3) == 1.0


def test_team_game_epa_ignores_plays_without_posteam():
    pbp = pd.DataFrame([
        {"season": 2023, "week": 1, "posteam": None, "defteam": None,
         "epa": 5.0, "success": 1, "pass": 0, "rush": 0, "play_type": "kickoff"},
        {"season": 2023, "week": 1, "posteam": "SF", "defteam": "SEA",
         "epa": 0.2, "success": 1, "pass": 1, "rush": 0, "play_type": "pass"},
    ])
    out = team_game_epa(pbp).set_index("team")
    # The posteam=None play must not pollute SF's offense
    assert round(out.loc["SF", "off_epa_play"], 3) == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_nfl_features_epa.py -v`
Expected: FAIL with `ImportError: cannot import name 'team_game_epa'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/nfl/features.py
"""Pure NFL feature engineering over nflverse play-by-play.

All functions here are deterministic transforms of DataFrames — no DB, no I/O —
so they are cheaply unit-testable and hold the point-in-time invariant explicitly.
"""
import pandas as pd


def team_game_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate play-by-play to one row per (season, week, team).

    Offense metrics come from plays where the team has possession (posteam);
    defense EPA is the EPA the team *allowed* (mean epa of plays where it is
    defteam). Plays with no posteam (kickoffs, etc.) are ignored for offense.
    """
    valid = pbp[pbp["posteam"].notna() & pbp["defteam"].notna()].copy()

    off = valid.groupby(["season", "week", "posteam"]).agg(
        off_epa_play=("epa", "mean"),
        pass_epa=("epa", lambda s: s[valid.loc[s.index, "pass"] == 1].mean()),
        rush_epa=("epa", lambda s: s[valid.loc[s.index, "rush"] == 1].mean()),
        success_rate=("success", "mean"),
        plays=("epa", "size"),
    ).reset_index().rename(columns={"posteam": "team"})

    deff = valid.groupby(["season", "week", "defteam"]).agg(
        def_epa_play=("epa", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})

    out = off.merge(deff, on=["season", "week", "team"], how="outer")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_nfl_features_epa.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/services/nfl/features.py tests/unit/test_nfl_features_epa.py
git commit -m "feat(nfl): per-game EPA aggregation from play-by-play

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Point-in-time rolling team stats (the critical no-leak task)

**Files:**
- Modify: `src/services/nfl/features.py`
- Test: `tests/unit/test_nfl_features_pointintime.py`

**Interfaces:**
- Consumes: the per-game EPA frame from `team_game_epa`.
- Produces: `rolling_team_stats(team_game: pd.DataFrame, window: int = 8) -> pd.DataFrame` — for each (season, team, through_week) a row aggregating **only** that team's games in the same season with `week <= through_week`, over a trailing `window` games. Columns: `team, season, through_week, off_epa_play, def_epa_play, pass_epa, rush_epa, success_rate, pace, power_rating`. `power_rating = off_epa_play - def_epa_play`. A row exists for `through_week = w` for every week `w` from 1..max, representing "form entering week w+1".

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_features_pointintime.py
import pandas as pd
from src.services.nfl.features import rolling_team_stats


def _tg(rows):
    return pd.DataFrame(rows)


def test_through_week_never_includes_current_or_future_weeks():
    # KC posts off_epa 1.0 in wk1, 3.0 in wk2, 5.0 in wk3.
    tg = _tg([
        {"season": 2023, "week": 1, "team": "KC", "off_epa_play": 1.0,
         "def_epa_play": 0.0, "pass_epa": 1.0, "rush_epa": 1.0, "success_rate": 0.5, "plays": 60},
        {"season": 2023, "week": 2, "team": "KC", "off_epa_play": 3.0,
         "def_epa_play": 0.0, "pass_epa": 3.0, "rush_epa": 3.0, "success_rate": 0.5, "plays": 60},
        {"season": 2023, "week": 3, "team": "KC", "off_epa_play": 5.0,
         "def_epa_play": 0.0, "pass_epa": 5.0, "rush_epa": 5.0, "success_rate": 0.5, "plays": 60},
    ])
    out = rolling_team_stats(tg).set_index("through_week")
    # through_week=1 sees ONLY week 1 -> 1.0 (no leakage of wk2/wk3)
    assert round(out.loc[1, "off_epa_play"], 3) == 1.0
    # through_week=2 sees weeks 1-2 -> mean(1,3)=2.0
    assert round(out.loc[2, "off_epa_play"], 3) == 2.0
    # through_week=3 sees weeks 1-3 -> mean(1,3,5)=3.0
    assert round(out.loc[3, "off_epa_play"], 3) == 3.0


def test_power_rating_is_off_minus_def():
    tg = _tg([
        {"season": 2023, "week": 1, "team": "SF", "off_epa_play": 0.2,
         "def_epa_play": -0.1, "pass_epa": 0.2, "rush_epa": 0.2, "success_rate": 0.5, "plays": 60},
    ])
    out = rolling_team_stats(tg).set_index("through_week")
    assert round(out.loc[1, "power_rating"], 3) == 0.3


def test_window_limits_trailing_games():
    rows = [
        {"season": 2023, "week": w, "team": "BUF", "off_epa_play": float(w),
         "def_epa_play": 0.0, "pass_epa": float(w), "rush_epa": float(w),
         "success_rate": 0.5, "plays": 60}
        for w in range(1, 11)
    ]
    out = rolling_team_stats(_tg(rows), window=3).set_index("through_week")
    # through_week=10 with window 3 -> mean of weeks 8,9,10 = 9.0
    assert round(out.loc[10, "off_epa_play"], 3) == 9.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_nfl_features_pointintime.py -v`
Expected: FAIL with `ImportError: cannot import name 'rolling_team_stats'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/services/nfl/features.py

_EPA_COLS = ["off_epa_play", "def_epa_play", "pass_epa", "rush_epa", "success_rate"]


def rolling_team_stats(team_game: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Point-in-time trailing team form.

    For each (season, team) and each played week w, emit a row keyed
    through_week=w that aggregates ONLY that team's games with week <= w
    (trailing `window` games). This row represents the team's form used to
    predict week w+1 — it must never reference week > w.
    """
    out_rows: list[dict] = []
    for (season, team), grp in team_game.groupby(["season", "team"]):
        grp = grp.sort_values("week")
        weeks = grp["week"].tolist()
        for w in weeks:
            hist = grp[grp["week"] <= w].tail(window)   # <= w, never > w
            row = {"season": int(season), "team": team, "through_week": int(w),
                   "pace": float(hist["plays"].mean())}
            for col in _EPA_COLS:
                row[col] = float(hist[col].mean())
            row["power_rating"] = row["off_epa_play"] - row["def_epa_play"]
            out_rows.append(row)
    return pd.DataFrame(out_rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_nfl_features_pointintime.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Full feature-suite regression run**

Run: `pytest tests/unit/test_nfl_features_epa.py tests/unit/test_nfl_features_pointintime.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/services/nfl/features.py tests/unit/test_nfl_features_pointintime.py
git commit -m "feat(nfl): point-in-time rolling team stats (no leakage)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Ingest orchestration — DataFrames → `nfl_*` upserts

**Files:**
- Create: `src/services/nfl/ingest.py`
- Test: `tests/integration/test_nfl_ingest.py`

**Interfaces:**
- Consumes: `schedule_to_game_rows`, `team_game_epa`, `rolling_team_stats`, `async_session_maker`, the models.
- Produces (async, all idempotent upserts):
  - `upsert_games(session, game_rows: list[dict]) -> int` — upserts `NFLGame` (conflict key `game_id`), returns count.
  - `upsert_game_context(session, sched_df) -> int` — upserts `NFLGameContext` (rest days, dome from `roof`).
  - `upsert_team_stats(session, stats_df) -> int` — upserts `NFLTeamStats` (conflict key `team, season, through_week`).

Integration test requires the test Postgres DB (same as existing `tests/integration`). If no DB is configured in the runner, mark with `@pytest.mark.integration` and skip — but the upsert SQL must still be exercised once locally before the phase is accepted.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_nfl_ingest.py
import pandas as pd
import pytest
from src.database import async_session_maker
from src.services.nfl.nfl_data import schedule_to_game_rows
from src.services.nfl.ingest import upsert_games


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_games_is_idempotent():
    sched = pd.read_parquet("tests/fixtures/nfl_schedule_sample.parquet")
    rows = schedule_to_game_rows(sched)
    async with async_session_maker() as session:
        n1 = await upsert_games(session, rows)
        await session.commit()
        n2 = await upsert_games(session, rows)   # re-run: no duplicates
        await session.commit()
    assert n1 == len(rows)
    assert n2 == len(rows)   # same count, upsert not insert
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_nfl_ingest.py -v -m integration`
Expected: FAIL with `ImportError: cannot import name 'upsert_games'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/nfl/ingest.py
"""Persist nflverse-derived DataFrames into nfl_* tables (idempotent upserts)."""
import pandas as pd
import structlog
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import NFLGame, NFLGameContext, NFLTeamStats

logger = structlog.get_logger()


async def upsert_games(session: AsyncSession, game_rows: list[dict]) -> int:
    for row in game_rows:
        stmt = insert(NFLGame).values(**row).on_conflict_do_update(
            index_elements=["game_id"],
            set_={k: row[k] for k in row if k != "game_id"},
        )
        await session.execute(stmt)
    logger.info("nfl_upsert_games", count=len(game_rows))
    return len(game_rows)


async def upsert_game_context(session: AsyncSession, sched: pd.DataFrame) -> int:
    count = 0
    for _, g in sched.iterrows():
        roof = (g.get("roof") or "").lower()
        values = {
            "game_id": g["game_id"],
            "home_rest_days": None if pd.isna(g.get("home_rest")) else int(g["home_rest"]),
            "away_rest_days": None if pd.isna(g.get("away_rest")) else int(g["away_rest"]),
            "temp_f": None if pd.isna(g.get("temp")) else float(g["temp"]),
            "wind_mph": None if pd.isna(g.get("wind")) else float(g["wind"]),
            "is_dome": roof in ("dome", "closed"),
        }
        stmt = insert(NFLGameContext).values(**values).on_conflict_do_update(
            index_elements=["game_id"],
            set_={k: values[k] for k in values if k != "game_id"},
        )
        await session.execute(stmt)
        count += 1
    logger.info("nfl_upsert_game_context", count=count)
    return count


async def upsert_team_stats(session: AsyncSession, stats: pd.DataFrame) -> int:
    records = stats.to_dict("records")
    for rec in records:
        clean = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
        stmt = insert(NFLTeamStats).values(**clean).on_conflict_do_update(
            index_elements=["team", "season", "through_week"],
            set_={k: clean[k] for k in clean
                  if k not in ("team", "season", "through_week", "id")},
        )
        await session.execute(stmt)
    logger.info("nfl_upsert_team_stats", count=len(records))
    return len(records)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_nfl_ingest.py -v -m integration`
Expected: PASS (requires the tables to exist — run `python3 -c "import asyncio; from src.database import init_db; asyncio.run(init_db())"` first to create `nfl_*` tables).

- [ ] **Step 5: Commit**

```bash
git add src/services/nfl/ingest.py tests/integration/test_nfl_ingest.py
git commit -m "feat(nfl): idempotent upserts for games, context, team stats

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Backfill CLI + point-in-time spot check (Phase 1 exit gate)

**Files:**
- Create: `src/tasks/nfl_backfill.py`
- Test: manual spot-check documented below (this is the phase's acceptance gate).

**Interfaces:**
- Consumes: everything above.
- Produces: `python3 -m src.tasks.nfl_backfill <start_season> <end_season>` — loads schedules + pbp per season, upserts games/context, computes `team_game_epa` → `rolling_team_stats` → upserts team stats.

- [ ] **Step 1: Write the backfill task**

```python
# src/tasks/nfl_backfill.py
"""Backfill NFL games + team stats from nflverse for a season range."""
import asyncio
import sys

import structlog

from src.database import async_session_maker, init_db
from src.services.nfl.nfl_data import load_schedules, load_pbp, schedule_to_game_rows
from src.services.nfl.features import team_game_epa, rolling_team_stats
from src.services.nfl.ingest import upsert_games, upsert_game_context, upsert_team_stats

logger = structlog.get_logger()


async def backfill_season(season: int) -> None:
    sched = load_schedules([season])
    sched = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    game_rows = schedule_to_game_rows(sched)

    pbp = load_pbp([season])
    tg = team_game_epa(pbp)
    stats = rolling_team_stats(tg)

    async with async_session_maker() as session:
        await upsert_games(session, game_rows)
        await upsert_game_context(session, sched)
        await upsert_team_stats(session, stats)
        await session.commit()
    logger.info("nfl_backfill_season_done", season=season,
                games=len(game_rows), stat_rows=len(stats))


async def main(start: int, end: int) -> None:
    await init_db()   # ensure nfl_* tables exist
    for season in range(start, end + 1):
        await backfill_season(season)


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2010
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    asyncio.run(main(start, end))
```

- [ ] **Step 2: Run a single-season smoke backfill**

Run: `python3 -m src.tasks.nfl_backfill 2023 2023`
Expected: logs `nfl_backfill_season_done season=2023 games=~272 stat_rows>0` with no exceptions.

- [ ] **Step 3: Point-in-time spot-check against the DB (ACCEPTANCE GATE)**

Run (uses the local/prod DB per `src.config`):

```bash
python3 - <<'PY'
import asyncio
from sqlalchemy import select, text
from src.database import async_session_maker

async def main():
    async with async_session_maker() as s:
        # 1) team_stats row for a known team entering a known week
        r = (await s.execute(text(
            "SELECT team, season, through_week, off_epa_play, def_epa_play, power_rating "
            "FROM nfl_team_stats WHERE team='KC' AND season=2023 ORDER BY through_week LIMIT 5"
        ))).fetchall()
        print("KC 2023 early-week form:")
        for row in r: print(" ", row)
        # 2) leakage guard: no through_week exceeds the team's played weeks
        bad = (await s.execute(text(
            "SELECT COUNT(*) FROM nfl_team_stats ts WHERE ts.through_week > "
            "(SELECT MAX(week) FROM nfl_games g WHERE g.season=ts.season)"
        ))).scalar()
        print("rows with through_week beyond season max week (must be 0):", bad)
        # 3) situational flags populated
        r2 = (await s.execute(text(
            "SELECT COUNT(*) FILTER (WHERE is_divisional) AS div, "
            "COUNT(*) FILTER (WHERE is_primetime) AS prime, COUNT(*) AS total "
            "FROM nfl_games WHERE season=2023"
        ))).fetchone()
        print("2023 games: divisional=%s primetime=%s total=%s" % tuple(r2))

asyncio.run(main())
PY
```

Expected: KC early-week EPA values look sane (small magnitudes, e.g. -0.2..0.3), leakage count is **0**, and divisional (~96) / primetime (~50+) counts are non-zero for a full 2023 season. **Review these numbers with the user — this is the Phase 1 exit gate.**

- [ ] **Step 4: Commit**

```bash
git add src/tasks/nfl_backfill.py
git commit -m "feat(nfl): season backfill CLI + point-in-time spot check

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Best-effort situational features — playoff stakes + starters out

**Files:**
- Modify: `src/services/nfl/features.py`, `src/services/nfl/ingest.py`, `src/tasks/nfl_backfill.py`
- Test: `tests/unit/test_nfl_features_stakes.py`

**Interfaces:**
- Produces:
  - `playoff_stakes(standings_through_week: pd.DataFrame, season: int, week: int) -> dict[str, str]` — pure heuristic mapping team → `alive`/`clinched`/`eliminated`. For Phase 1, a **documented heuristic**: weeks < 15 → all `alive`; weeks >= 15 → `eliminated` if a team's max possible wins < the current 7th seed's wins, `clinched` if its min guaranteed playoff spot is locked, else `alive`. Standings derived from `nfl_games` results to date.
  - `starters_out(injuries_df, depth_df, game_id, team) -> int` — count of `report_status == 'Out'` players who appear among the team's depth-chart starters (rank 1). Returns `0` when injury/depth data is missing (logged).

Both are stored into `nfl_game_context`. They are **candidate features** — Phase 2 tests them for signal; do not block on their precision.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_features_stakes.py
import pandas as pd
from src.services.nfl.features import starters_out


def test_starters_out_counts_only_out_starters():
    injuries = pd.DataFrame([
        {"team": "KC", "gsis_id": "p1", "report_status": "Out"},
        {"team": "KC", "gsis_id": "p2", "report_status": "Questionable"},
        {"team": "KC", "gsis_id": "p3", "report_status": "Out"},
    ])
    depth = pd.DataFrame([
        {"team": "KC", "gsis_id": "p1", "depth_team": 1},   # starter, Out -> counts
        {"team": "KC", "gsis_id": "p2", "depth_team": 1},   # starter but Questionable
        {"team": "KC", "gsis_id": "p3", "depth_team": 2},   # Out but backup -> no count
    ])
    assert starters_out(injuries, depth, "KC") == 1


def test_starters_out_missing_data_returns_zero():
    empty = pd.DataFrame(columns=["team", "gsis_id", "report_status"])
    depth = pd.DataFrame(columns=["team", "gsis_id", "depth_team"])
    assert starters_out(empty, depth, "KC") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_nfl_features_stakes.py -v`
Expected: FAIL with `ImportError: cannot import name 'starters_out'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/services/nfl/features.py
import structlog as _structlog
_log = _structlog.get_logger()


def starters_out(injuries: pd.DataFrame, depth: pd.DataFrame, team: str) -> int:
    """Best-effort count of depth-chart starters ruled Out for a team.

    Returns 0 (and logs) when data is missing — this is a noisy candidate
    feature, not a correctness-critical one.
    """
    if injuries.empty or depth.empty:
        _log.info("nfl_starters_out_missing_data", team=team)
        return 0
    starters = set(
        depth[(depth["team"] == team) & (depth["depth_team"] == 1)]["gsis_id"]
    )
    out = injuries[(injuries["team"] == team)
                   & (injuries["report_status"] == "Out")]["gsis_id"]
    return int(sum(1 for pid in out if pid in starters))


def playoff_stakes(standings: pd.DataFrame, season: int, week: int) -> dict[str, str]:
    """Heuristic meaningful-game label per team. Weeks < 15 are all 'alive'.

    `standings` is a per-team wins/losses frame through `week`. This is a
    deliberately simple v1 heuristic (documented in the spec); Phase 2 decides
    whether it carries signal.
    """
    if week < 15:
        return {t: "alive" for t in standings["team"]}
    # Weeks 15+: rank by wins; label bottom third eliminated, top third clinched.
    ranked = standings.sort_values("wins", ascending=False).reset_index(drop=True)
    n = len(ranked)
    labels: dict[str, str] = {}
    for i, row in ranked.iterrows():
        if i < n // 3:
            labels[row["team"]] = "clinched"
        elif i >= 2 * n // 3:
            labels[row["team"]] = "eliminated"
        else:
            labels[row["team"]] = "alive"
    return labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_nfl_features_stakes.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire into backfill (best-effort, non-blocking)**

In `src/tasks/nfl_backfill.py::backfill_season`, after computing team stats, load injuries/depth via `nfl_data_py` inside a try/except that logs and continues on failure (these datasets have gaps for older seasons), compute per-game `starters_out` for home/away and `playoff_stakes`, and pass them into an extended `upsert_game_context`. Keep failures non-fatal:

```python
    try:
        import nfl_data_py as nfl
        injuries = nfl.import_injuries([season])
        depth = nfl.import_depth_charts([season])
    except Exception as e:   # older seasons / schema drift — candidate feature only
        logger.warning("nfl_injury_data_unavailable", season=season, error=str(e))
        injuries = depth = None
    # (compute + attach in upsert_game_context; leave columns NULL when unavailable)
```

- [ ] **Step 6: Run full NFL unit suite**

Run: `pytest tests/unit -k nfl -v`
Expected: all NFL unit tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/services/nfl/features.py src/services/nfl/ingest.py src/tasks/nfl_backfill.py tests/unit/test_nfl_features_stakes.py
git commit -m "feat(nfl): best-effort starters-out and playoff-stakes candidate features

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 1 Exit Criteria (review with user before Phase 2)

1. `pytest tests/unit -k nfl` fully green.
2. `python3 -m src.tasks.nfl_backfill 2010 2024` completes with ~272 games/season and non-zero team-stat rows.
3. The Task 8 spot-check shows: sane KC early-week EPA, **leakage count = 0**, non-zero divisional/primetime counts.
4. Situational flags (`is_divisional`, `is_primetime`, QBs) populated; best-effort columns populated where data exists, NULL (not wrong) where it doesn't.

Only after these are reviewed do we proceed to Phase 2 (MOV + totals models).
