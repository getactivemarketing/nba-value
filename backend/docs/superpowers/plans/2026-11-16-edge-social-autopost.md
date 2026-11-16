# Edge Social Autopost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** @trulineapp posts NBA and NFL cheat sheets, value plays, blind plays, final recaps and receipts to X automatically, driving people to truline.app.

**Architecture:** A new `src/edge/social/` package runs as one step of the `edge-worker` tick (the worker holds an advisory lock, so posting runs exactly once). Pure modules pick content, compute matchup panels, check safety and render PIL cards; a runner records every post in `edge_social_posts` (unique per kind+key) before calling Blotato, and in dry-run mode stores the PNG instead of posting. The legacy MLB/NBA social thread in the API process is switched off. The frontend gets public blind-play reveal pages.

**Tech Stack:** Python 3.11, psycopg2, SQLAlchemy models + `create_all`, Pillow, httpx, FastAPI; React 18 + Vite + React Query + Vitest.

**Spec:** `backend/docs/superpowers/specs/2026-11-16-edge-social-autopost-design.md` (approved 2026-11-16). Read it before any task.

## Global Constraints

- Cards are 1080×1350 PNG, TruLine dark style (bg `#05070a`/`#0b0e14`, accent `#a4e6ff`, win `#4ade80`, loss `#f87171`), footer `truline.app` + `@trulineapp`; Final Recap uses the light style of the MLB final card.
- Never post a wrong score, a pick after its tip, or a play that already lost. Any failed check → skip + log, never post.
- Blind cards never carry name, team, opponent, logo, headshot, team colors, matchup string or venue. Enforced by type: `BlindCard` has no such fields.
- Blind plays: favorable matchup required; exclude Top 5 players; one per game and player; up to 3 per sport per slate; posted ≥ 60 min before tip with EV > 0 at current best price; zero is fine.
- Value Plays post at nightly lock at locked prices; every row's tip must still be in the future, else skip the whole post.
- Cheat Sheet rows: +EV on the board now, sorted by L10 hit rate, 6-10 rows, else skip. Not "posted plays".
- Receipts: wins only. Final Recaps: only games with a posted blind or value play; show W, L, P and void honestly.
- Runaway brakes: max 25 posts per sport per ET day, max 6 per rolling hour. Tripping a brake stops posting.
- All times scheduled in `America/New_York` (DST-safe); never hard-code UTC offsets.
- Links carry `utm_source=x&utm_medium=social&utm_campaign=<kind>`.
- No LLM copy; text comes from templates over DB values only.
- Blind plays may not post until worker state `social_blind_gate:<sport>` = `pass` (set by Task 5's gate script).
- Legacy MLB / legacy NBA model posts and founder SMS alerts must not fire.
- Follow `CLAUDE.md`: tests first, throwaway Postgres on 55432 for SQL tests, unit tests never touch DB routes, commit per task, push only when Samir asks.

## Review Focus

1. **A game that crosses midnight ET** (late West Coast tip): its Final Recap and receipts still post, attributed to the slate it tipped on. → Task 9 test `test_final_recap_for_game_finishing_after_midnight`.
2. **Postponed or cancelled game / voided prop:** no Final Recap, no receipt; the blind reveal page shows "void". → Task 9 test `test_postponed_game_never_recapped_or_receipted`.
3. **Blotato call fails or the worker restarts mid-post:** the ledger row exists as `pending`/`failed`; the same key is never posted twice; a failed key retries at most 3 times. → Task 2 test `test_claim_is_once_per_key_and_failed_retries_capped`.
4. **Two players share a name (e.g. "Jalen Williams") or ESPN has no match:** silhouette, never a wrong face. → Task 7 test `test_ambiguous_name_gets_no_headshot`.
5. **Score sources disagree** (Odds API score vs box-score sum / BDL NFL score): no Final Recap or receipt until they agree. → Task 9 test `test_score_mismatch_blocks_recap`.

---

### Task 1: Switch off legacy social posting

**Files:**
- Modify: `src/config.py` (settings block near `twitter_posting_enabled`, line ~54)
- Modify: `src/main.py:195-198` (`_start_social_scheduler_thread()` call)
- Test: `tests/unit/test_legacy_social_off.py`

**Interfaces:**
- Produces settings: `legacy_social_enabled: bool = False`, `social_dry_run: bool = True`, `social_sports: str = "nba"` (comma list), `social_kinds: str = "cheatsheet,value,blind,final,receipt"`, `social_max_per_day: int = 25`, `social_max_per_hour: int = 6`, `site_url: str = "https://www.truline.app"`.

- [ ] **Step 1: Write the failing test**

```python
def test_legacy_social_thread_not_started_by_default(monkeypatch):
    started = []
    monkeypatch.setattr(main, "_start_social_scheduler_thread", lambda: started.append(1))
    main._maybe_start_legacy_social()
    assert started == []

def test_legacy_social_thread_starts_only_when_flag_on(monkeypatch):
    monkeypatch.setattr(settings, "legacy_social_enabled", True)
    started = []
    monkeypatch.setattr(main, "_start_social_scheduler_thread", lambda: started.append(1))
    main._maybe_start_legacy_social()
    assert started == [1]

def test_social_defaults():
    s = Settings()
    assert s.social_dry_run is True and s.social_max_per_day == 25 and s.social_max_per_hour == 6
```

- [ ] **Step 2: Run** `python3 -m pytest tests/unit/test_legacy_social_off.py -q -p no:cacheprovider` → FAIL (no `_maybe_start_legacy_social`).
- [ ] **Step 3: Add the settings; add `_maybe_start_legacy_social() -> None` in `src/main.py` and call it where `_start_social_scheduler_thread()` is called today.**
- [ ] **Step 4: Run the test** → PASS.
- [ ] **Step 5: Commit** `feat(social): legacy MLB/NBA social thread off unless LEGACY_SOCIAL_ENABLED`

---

### Task 2: Post ledger, switches and brakes

**Files:**
- Modify: `src/models/edge.py` (new `EdgeSocialPost`), `src/models/__init__.py`
- Create: `src/edge/social/__init__.py`, `src/edge/social/ledger.py`
- Test: `tests/unit/test_social_ledger.py`, `tests/integration/test_edge_store_db.py` (append)

**Interfaces:**
- Table `edge_social_posts`: `id bigserial`, `sport varchar(8)`, `kind varchar(16)`, `key varchar(80)`, `status varchar(10)` (`pending|posted|dry_run|failed|skipped`), `attempts int default 0`, `text text`, `image bytea null`, `media_url text null`, `submission_id varchar(80) null`, `post_url text null`, `detail jsonb null`, `created_at`, `updated_at`; unique `(kind, key)`.
- `ledger.claim(conn, sport: str, kind: str, key: str, now: datetime) -> int | None` — inserts `pending` or re-claims a `failed` row with `attempts < 3`; returns row id, else None.
- `ledger.finish(conn, row_id: int, status: str, now: datetime, *, text: str, image: bytes | None = None, media_url: str | None = None, submission_id: str | None = None, post_url: str | None = None, detail: dict | None = None) -> None`
- `ledger.brake_tripped(conn, sport: str, now: datetime) -> bool` — counts `posted` + `pending` rows: ≥ `social_max_per_day` in the ET day or ≥ `social_max_per_hour` in the last hour.
- `ledger.enabled(kind: str, sport: str) -> bool` — `kind in settings.social_kinds` and `sport in settings.social_sports`.
- `ledger.find(conn, kind: str, key: str) -> dict | None`
- Key formats (used by later tasks): `cheatsheet:{sport}:{slate}`, `value:{sport}:{slate}`, `blind:{sport}:{n}`, `final:{event_id}`, `receipt:blind:{sport}:{n}`, `receipt:value:{sport}:{slate}:{rank}`.

- [ ] **Step 1: Write failing tests**

```python
# unit
def test_enabled_respects_kind_and_sport_lists(monkeypatch):
    monkeypatch.setattr(settings, "social_kinds", "blind,final"); monkeypatch.setattr(settings, "social_sports", "nba")
    assert ledger.enabled("blind", "nba") and not ledger.enabled("value", "nba") and not ledger.enabled("blind", "nfl")

# DB (append to test_edge_store_db.py)
def test_claim_is_once_per_key_and_failed_retries_capped(conn):
    a = ledger.claim(conn, "nba", "blind", "blind:nba:1", NOW)
    assert a and ledger.claim(conn, "nba", "blind", "blind:nba:1", NOW) is None
    for _ in range(3):
        ledger.finish(conn, a, "failed", NOW, text="x")
        a = ledger.claim(conn, "nba", "blind", "blind:nba:1", NOW) or a
    assert ledger.claim(conn, "nba", "blind", "blind:nba:1", NOW) is None  # 3 attempts used

def test_brake_trips_at_six_an_hour(conn):
    for i in range(6):
        ledger.finish(conn, ledger.claim(conn, "nba", "final", f"final:e{i}", NOW), "posted", NOW, text="x")
    assert ledger.brake_tripped(conn, "nba", NOW) and not ledger.brake_tripped(conn, "nfl", NOW)
```

- [ ] **Step 2: Run** unit test + DB tests (`EDGE_TEST_DB_URL=... -k "claim or brake"`) → FAIL.
- [ ] **Step 3: Implement the model and `ledger.py`.** `dry_run` rows count toward neither brake. `pending` rows older than 15 min count as `failed` for re-claim (worker died mid-post).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(social): edge_social_posts ledger with once-per-key claims and runaway brakes`

---

### Task 3: NBA box-score fields and positions

**Files:**
- Modify: `src/models/edge.py` (`EdgePlayerGame` + `fgm`, `fga`, `fta`, `oreb`; `EdgePlayer` + `position varchar(4)`), `src/database.py` (`EDGE_COLUMN_MIGRATIONS`), `src/edge/players.py` (`box_row`, `ingest_box_scores` insert, `resolve_new_players` stores position)
- Create: `src/edge/scripts/backfill_box_fields.py` (one-off: reload box scores since 2025-10-01 and set positions from rosters)
- Test: `tests/unit/test_edge_players.py` (extend), DB test in `tests/integration/test_edge_store_db.py`

**Interfaces:**
- `players.box_row(stat, abbr_by_team_id)` dict gains `fgm`, `fga`, `fta`, `oreb` (BDL keys `fgm`, `fga`, `fta`, `oreb`; missing → 0).
- `players.position_group(bdl_position: str | None) -> str | None` — `"G"|"F"|"C"` from the first letter of BDL position (`"G-F"` → `"G"`), else None.
- Migrations: `("edge_player_games","fgm","INTEGER NOT NULL DEFAULT 0")` and same for `fga`, `fta`, `oreb`; `("edge_players","position","VARCHAR(4)")`.

- [ ] **Step 1: Failing tests**

```python
def test_box_row_keeps_shooting_fields():
    r = players.box_row(_stat(fgm=9, fga=18, fta=6, oreb=2), TEAMS)
    assert (r["fgm"], r["fga"], r["fta"], r["oreb"]) == (9, 18, 6, 2)

@pytest.mark.parametrize("raw,want", [("G", "G"), ("F-C", "F"), ("C", "C"), ("", None), (None, None)])
def test_position_group(raw, want):
    assert players.position_group(raw) == want
```
(`_stat` / `TEAMS`: reuse the fixtures already in `test_edge_players.py`.)

- [ ] **Step 2: Run** `python3 -m pytest tests/unit/test_edge_players.py -q -p no:cacheprovider` → FAIL.
- [ ] **Step 3: Implement**; the backfill script takes `--since YYYY-MM-DD` and `--dry-run`, logs counts, and uses `players.backfill` for box scores plus one roster call per team for positions.
- [ ] **Step 4: Run** unit + DB tests (`-k "box or position"`) → PASS.
- [ ] **Step 5: Commit** `feat(edge-nba): store FGM/FGA/FTA/OREB and player position group`

---

### Task 4: NBA matchup context and favorable rule

**Files:**
- Create: `src/edge/social/nba_matchup.py`
- Test: `tests/unit/test_social_nba_matchup.py`

**Interfaces:**
- `TeamRanks = dataclass(team: str, vs_pos: dict[tuple[str, str], int], drtg_rank: int, pace_rank: int)` — `vs_pos[(position, market)]` rank 1-30, **30 = allows the most**; `drtg_rank` 1 = best defense; `pace_rank` 1 = fastest.
- `PlayerUsage = dataclass(usage_l10: float, usage_season: float, minutes_l5: float, minutes_season: float, games: int)`
- `build_ranks(games: list[dict]) -> dict[str, TeamRanks]` — `games` = `edge_player_games` rows (with `position` joined) for the current season. Opponent-vs-position = per-game average of `players.stat_for(market, row)` allowed to that position. Possessions = FGA + 0.44·FTA − OREB + TOV of team totals; DRtg = 100·opp pts / opp possessions; pace = possessions per game.
- `player_usage(rows: list[dict], team_totals: dict[tuple[int, str], dict]) -> PlayerUsage` — standard usage: 100·((FGA + 0.44·FTA + TOV)·(team_min/5)) / (min·(team FGA + 0.44·team FTA + team TOV)).
- `load_context(conn, season: int) -> tuple[dict[str, TeamRanks], Callable[[int], PlayerUsage | None]]` — the only DB function here.
- `Panel = dataclass(favorable: bool, opp_vs_pos_rank: int, drtg_rank: int, pace_rank: int, usage_l10: float, usage_change: float, minutes_l5: float)`
- `panel(row: dict, position: str, opp: TeamRanks, usage: PlayerUsage) -> Panel` — favorable for **over**: `opp_vs_pos_rank >= 21` and `drtg_rank > 10`; for **under**: `opp_vs_pos_rank <= 10` and `drtg_rank <= 20`; both sides also need `minutes_l5 >= 0.85·minutes_season` and `usage_l10 >= usage_season − 3`.

- [ ] **Step 1: Failing tests** (synthetic box rows, no DB)

```python
def test_team_allowing_most_forward_pra_ranks_30(): ...   # build 30 teams; team "WAS" allows most F PRA → vs_pos[("F","player_points_rebounds_assists")] == 30
def test_drtg_rank_1_is_fewest_points_per_possession(): ...
def test_usage_formula_matches_hand_calc():
    assert round(nba_matchup.player_usage(ROWS, TOTALS).usage_l10, 1) == 26.8   # hand-computed fixture
def test_over_favorable_needs_bottom_third_and_not_top10_defense():
    assert nba_matchup.panel(OVER, "F", ranks(vs=27, drtg=22), USAGE_OK).favorable
    assert not nba_matchup.panel(OVER, "F", ranks(vs=27, drtg=5), USAGE_OK).favorable
    assert not nba_matchup.panel(OVER, "F", ranks(vs=15, drtg=22), USAGE_OK).favorable
def test_minutes_drop_kills_favorable():
    assert not nba_matchup.panel(OVER, "F", ranks(vs=27, drtg=22), usage(min_l5=28, min_season=35)).favorable
def test_under_is_mirrored(): ...
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** Rank windows: whole current season (document in module docstring; spec leaves it to the builder). **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(social): NBA matchup ranks, usage and the favorable rule`

---

### Task 5: Matchup filter validation gate

**Files:**
- Create: `src/edge/backtest/matchup_gate.py` (CLI: `python -m src.edge.backtest.matchup_gate --sport nba|nfl [--write-state]`)
- Create (output): `docs/backtests/nba-matchup-filter-2026-11.md`
- Test: `tests/unit/test_matchup_gate.py`

**Interfaces:**
- Consumes: `nba_matchup.load_context`, `nba_matchup.panel` (Task 4). The `--sport nfl` branch is added in Task 6 (it needs `nfl_matchup.favorable`); until then `--sport nfl` exits with "not implemented".
- `dedupe(rows: list[dict]) -> list[dict]` — one row per (event_id, market, player, side, line), earliest `first_seen` sighting.
- `gate(fav_clv: list[float], rest_clv: list[float]) -> dict` — returns `{"pass": bool, "n_fav": int, "fav_mean": float, "rest_mean": float, "fav_ci95": [lo, hi], ...}`. **Pre-registered rule, copy exactly:** pass iff `fav_mean >= rest_mean - 0.0025` **and** `fav_ci95[0] > 0` **and** `n_fav >= 200`. For NFL: pass iff `fav_mean >= 0` on whatever n exists (spec §7 "lighter").
- Ranks must be computed **as of each game's date** (only games before it), not end-of-window, or the backtest peeks ahead.
- `--write-state` sets worker state `social_blind_gate:<sport>` to `pass` or `fail` via `store.set_state`.

- [ ] **Step 1: Failing tests**

```python
def test_gate_rule_exact():
    assert matchup_gate.gate([0.02]*250, [0.015]*250)["pass"]
    assert not matchup_gate.gate([0.02]*150, [0.0]*250)["pass"]          # n < 200
    assert not matchup_gate.gate([0.010]*250, [0.0130]*250)["pass"]      # trails by 0.3pp
def test_dedupe_keeps_earliest_sighting(): ...
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** Report includes CLV, win rate and units per group with CIs (win rate / units reported, not gated). **Step 4: Run** → PASS.
- [ ] **Step 5: Run it against prod data** (read-only; `DATABASE_URL` = Railway public proxy) and write the report. **If NBA fails, stop and tell Samir before Task 8+ blind posting is enabled.**
- [ ] **Step 6: Commit** `feat(social): pre-registered matchup filter gate + NBA report`

---

### Task 6: NFL favorable rule and panel

**Files:**
- Create: `src/edge/social/nfl_matchup.py`
- Modify: `docs/engineering/07-nfl-promotion-gates.md` (Operational log entry, **not** §7)
- Test: `tests/unit/test_social_nfl_matchup.py`

**Interfaces:**
- `favorable(row: dict) -> bool` — `row["tags"]` (list of `{"key","direction","detail"}` from `tagging.row_tags`) contains `def_soft` with direction `tailwind` and no tag with direction `headwind`.
- `NflPanel = dataclass(favorable: bool, opp_rank: int | None, share_l4: float | None, snap_share: float | None, tags: list[str])` and `panel(row: dict, ctx: matchups.TagContext) -> NflPanel` — `opp_rank` and shares read from `ctx` the same way `matchups.tags_for` does.

- [ ] **Step 1: Failing tests:** `test_def_soft_tailwind_is_favorable`, `test_any_headwind_blocks`, `test_no_tags_not_favorable`.
- [ ] **Step 2-4:** run → FAIL, implement, run → PASS.
- [ ] **Step 4b: Add the `--sport nfl` branch to `matchup_gate.py`** using `favorable(row)` on 2026 graded NFL props; test `test_nfl_gate_passes_on_nonnegative_mean`.
- [ ] **Step 5: Gates doc** — append to the Operational log: "2026-11-16 — Samir approved posting NFL Edge blind plays on X (tracking-only product; requires `def_soft` tailwind and no headwind). Not an amendment; no gate threshold changed."
- [ ] **Step 6: Commit** `feat(social): NFL favorable rule from def_soft tags; log Samir's decision`

---

### Task 7: ESPN headshots

**Files:**
- Modify: `src/models/edge.py` (`EdgeHeadshot`: `sport`, `odds_name`, `team`, `espn_id int null`, `checked_at`; PK `(sport, odds_name)`)
- Create: `src/edge/social/headshots.py`
- Test: `tests/unit/test_social_headshots.py`

**Interfaces:**
- `ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/{path}/teams/{abbr}/roster"`, `path` = `basketball/nba` | `football/nfl`.
- `HEADSHOT_URL = "https://a.espncdn.com/i/headshots/{league}/players/full/{espn_id}.png"`
- `match(odds_name: str, team: str, roster: list[dict]) -> int | None` — uses `players.normalize_name` and `NAME_ALIASES`; exactly one roster entry must match, else None.
- `async refresh(conn, sport: str, teams: set[str], now: datetime, http: httpx.AsyncClient) -> int` — refreshes rows older than 7 days.
- `async fetch_png(conn, sport: str, odds_name: str, http) -> bytes | None` — None when no id or HTTP ≠ 200.
- `LOGO_URL = "https://a.espncdn.com/i/teamlogos/{league}/500/{abbr}.png"`; `async fetch_logo(sport, abbr, http) -> bytes | None` (in-process cache).
- ESPN abbreviations differ for a few teams (NBA: `GSW→gs`, `NOP→no`, `NYK→ny`, `SAS→sa`, `UTA→utah`, `WAS→wsh`; NFL: `WAS→wsh`, `LAR→lar`, `JAX→jax`). Reuse the mapping in `image_generator.py` if present; else a dict here.

- [ ] **Step 1: Failing tests**

```python
def test_exact_name_and_team_match(): assert headshots.match("Pascal Siakam", "IND", ROSTER) == 3149673
def test_ambiguous_name_gets_no_headshot():
    assert headshots.match("Jalen Williams", "OKC", ROSTER_WITH_TWO_JALEN_WILLIAMS) is None
def test_no_match_returns_none(): assert headshots.match("Nobody Here", "IND", ROSTER) is None
```

- [ ] **Step 2-4:** FAIL → implement (httpx with `transport` injectable for tests, 10 s timeout) → PASS.
- [ ] **Step 5: Commit** `feat(social): ESPN headshot ids matched by name+team, logo fetch`

---

### Task 8: Blind plays: selection, storage, API

**Files:**
- Modify: `src/models/edge.py` (`EdgeBlindPlay`: `sport`, `n` (PK with sport), `slate_date`, `event_id`, `market`, `market_label`, `player`, `side`, `line`, `book`, `price`, `fair_prob`, `ev`, `n_books`, `form jsonb`, `position varchar(4)`, `panel jsonb`, `commence_time`, `created_at`)
- Create: `src/edge/social/blind.py`
- Modify: `src/edge/api.py` (routes), `src/edge/signups.py` (`SOURCES` += `"blind_reveal"`)
- Test: `tests/unit/test_social_blind.py`, `tests/unit/test_edge_api.py` (extend), DB test

**Interfaces:**
- Consumes: `nightly.eligible`, `views.ev_rows`, Task 4 `panel`, Task 6 `favorable`/`panel`, `ledger`.
- `select(rows: list[dict], top5_players: set[str], panels: dict[tuple, object], now: datetime, n: int = 3) -> list[dict]` — pure: `nightly.eligible` → props only → not in `top5_players` → `panels[key].favorable` → `commence_time - now >= 60 min` → `nightly._one_per_game` (rename to public `one_per_game`; keep `top_picks` calling it).
- `record(conn, sport: str, picks: list[dict], panels, now) -> list[int]` — assigns the next season `n` per sport; skips a bet already recorded.
- `BlindCard = dataclass(sport, n, position, market_label, side, line, price, ev, n_books, l10_hits, l10_games, l10_avg, tip_label, panel)` — **no name/team fields**; `tip_label` is the ET tip time unless only one game tips at that time, then `"Tonight"`.
- `to_card(play: dict, slate_tips: list[datetime]) -> BlindCard`
- `result(conn, sport, n) -> dict` — joins `edge_opportunities` grading (same bet, any book) like `nightly`.
- API (public, rate-limited, cached like other routes): `GET /edge/blind?sport=` → `{"plays": [...], "record": {...}}`; `GET /edge/blind/{n}?sport=` → full play incl. name, teams, panel, explanation, result. **Before tip the name is still returned** (the reveal is the point); the X card is what's blind.
- `explain(play: dict) -> str` — "9 books price this at 49%; {book} pays +116, which needs only 46%." Template only.

- [ ] **Step 1: Failing tests**

```python
def test_select_excludes_top5_players_and_unfavorable(): ...
def test_select_requires_60_min_before_tip(): ...
def test_select_one_per_game_and_max_three(): ...
def test_zero_qualifying_returns_empty(): assert blind.select([], set(), {}, NOW) == []
def test_blind_card_has_no_identity_fields():
    names = {f.name for f in dataclasses.fields(blind.BlindCard)}
    assert not names & {"player", "team", "opponent", "matchup", "event_id", "home_team", "away_team"}
def test_tip_label_hides_a_lone_tip_time(): ...
def test_reveal_route_404_for_unknown_n(monkeypatch): ...
```

- [ ] **Step 2-4:** FAIL → implement → PASS (DB test: `record` numbering and no duplicate bet).
- [ ] **Step 5: Commit** `feat(social): blind play selection, numbering, reveal API`

---

### Task 9: Content builders and safety checks

**Files:**
- Create: `src/edge/social/content.py`, `src/edge/social/safety.py`
- Test: `tests/unit/test_social_content.py`, `tests/unit/test_social_safety.py`

**Interfaces:**
- `safety.pick_ok(row: dict, now: datetime, current_ev: float | None) -> bool` — blind: `commence - now >= 60 min` and `current_ev > 0`.
- `safety.value_ok(legs: list[dict], now) -> bool` — every leg's `commence_time > now`.
- `safety.final_ok(event: dict, second_source: tuple[int, int] | None) -> bool` — `event.completed` and `status` not postponed/cancelled and `(home_score, away_score) == second_source`.
- `content.second_source_nba(conn, event: dict) -> tuple[int, int] | None` — sums `pts` in `edge_player_games` per team for that date + teams (all players stored); None if any team has no rows.
- `content.second_source_nfl(conn, event) -> tuple[int, int] | None` — from `edge_nfl_games` (add `home_score`, `away_score` columns via migration if absent; BDL games return them).
- `content.cheatsheet_rows(rows: list[dict], games_by_player: dict[int, list[dict]]) -> list[dict]` — `ev > 0`, player props, sort by L10 hit rate desc then ev; 6-10 rows else `[]`.
- `content.slate_for(commence: datetime) -> date` — `nightly.slate_date` semantics (games crossing midnight stay on the tip's slate).
- `content.posted_plays(conn, event_id: str) -> list[dict]` — blind + value plays recorded for that game with results.
- Text templates (exact copy, `{}` filled from data, URLs with UTM):
  - Blind post: `"🙈 Blind Play #{n} · {SPORT}\n\nNo names. No teams. No bias. Who is it? 👇"`; reply: `"Reveal + full breakdown → {site}/blind/{sport}/{n}?utm_source=x&utm_medium=social&utm_campaign=blind"`
  - Cheat Sheet: `"📋 {SPORT} Cheat Sheet · {Day} {m/d}\n\nTonight's hottest props by last-10 hit rate, all priced as value on the board right now."`
  - Value Plays: `"💰 Tonight's Top 5 value plays ({SPORT})\n\nBooks paying more than the market says these are worth. Locked {time} ET.\n\nAll plays + odds → {site}/?utm_...=value"`
  - Final: `"FINAL\n\n{Away} {a}, {Home} {h}\n\nTruLine plays in this game: {w}-{l}{p} {✅ if units>0 else ''} {units:+.2f}u\n\n#{SPORT} #{AwayNick} #{HomeNick}"`
  - Receipt (blind): `"✅ CASHED. Blind Play #{n} was {Name}.\n\nCalled it before tip at {price}. Did you guess right?"` (+ original post URL if known, else reveal link).
  - Receipt (value): `"✅ CASHED. {bet} at {price} from tonight's Top 5."`

- [ ] **Step 1: Failing tests** (pure, synthetic data):
  `test_blind_pick_rejected_59_minutes_before_tip`, `test_blind_pick_rejected_when_ev_gone`, `test_value_post_skipped_if_any_leg_started`, `test_score_mismatch_blocks_recap`, `test_postponed_game_never_recapped_or_receipted`, `test_final_recap_for_game_finishing_after_midnight` (slate = tip date), `test_cheatsheet_skips_under_six_rows`, `test_cheatsheet_excludes_negative_ev`, `test_final_text_shows_push_and_units`, `test_links_carry_utm`.
- [ ] **Step 2-4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(social): post content templates and pre-post safety checks`

---

### Task 10: Card rendering

**Files:**
- Create: `src/edge/social/cards.py`, `src/edge/social/fonts/` (bundle the TTFs `image_generator.py` already uses; reuse its font loader if one exists)
- Test: `tests/unit/test_social_cards.py`

**Interfaces:**
- `render_blind(card: blind.BlindCard) -> bytes`
- `render_cheatsheet(sport: str, slate: date, rows: list[dict], faces: dict[str, bytes | None]) -> bytes`
- `render_value(sport: str, slate: date, legs: list[dict], faces, locked_at: datetime) -> bytes`
- `render_final(event: dict, records: dict[str, str], logos: dict[str, bytes | None], plays: list[dict], faces) -> bytes`
- `render_receipt(play: dict, face: bytes | None, logo: bytes | None, original: bytes | None = None) -> bytes`
- All return PNG bytes, 1080×1350, < 1.5 MB, layout per the approved mockups (spec §5, `post-lineup-v2.html`). `face=None` → neutral silhouette.

- [ ] **Step 1: Failing tests:** each renderer returns a decodable PNG of size (1080, 1350) under 1.5 MB; `render_*` with every face/logo `None` still renders; renders in < 5 s.
- [ ] **Step 2-4:** FAIL → implement → PASS.
- [ ] **Step 5: Visual check:** `python -m src.edge.social.cards --sample out/` writes one of each card from fixture data; open them and compare with the mockups. Fix layout until close.
- [ ] **Step 6: Commit** `feat(social): PIL card renderers for blind, cheat sheet, value, final, receipt`

---

### Task 11: Poster and runner in the worker

**Files:**
- Create: `src/edge/social/poster.py`, `src/edge/social/runner.py`, `src/edge/social/__main__.py` (CLI `preview --out DIR --since YYYY-MM-DD`: export dry-run rows as PNG + TXT)
- Modify: `src/edge/worker.py` (call `runner.social_step` once per tick after sports, own try/except so a social error never breaks odds polling)
- Modify: `src/services/social/blotato.py` only if needed for a published-URL lookup
- Test: `tests/unit/test_social_runner.py`, `tests/unit/test_edge_worker.py` (extend)

**Interfaces:**
- `class Poster(Protocol): def post(self, texts: list[str], images: list[list[bytes]]) -> PostResult` with `PostResult = dataclass(ok: bool, submission_id: str | None, media_urls: list[str], post_url: str | None)`.
- `BlotatoPoster` — `upload_media` per image, `post_thread(texts, schedule_at=None, media_urls_per_post=...)`; then tries Blotato's post-status lookup for the public URL (verify in Blotato docs; if none exists, `post_url=None`).
- `DryRunPoster` — returns `ok=True` without network; runner stores image bytes in the ledger with status `dry_run`.
- `runner.social_step(conn, now: datetime, poster: Poster | None = None) -> dict` — `poster` defaults to Dry-run when `settings.social_dry_run` or not `settings.twitter_posting_enabled`. For each sport in `social_sports`, in order: cheat sheet (11:00 ET NBA; NFL Thu/Sun 10:00 ET and Mon 10:00 ET), value (when `edge_nightly_legs` for today exists and not yet posted; NBA only), blind (gate state must be `pass`; select/record/post each new play), finals (completed events in last 36 h with posted plays), receipts (graded wins of posted blind/value plays). Checks `ledger.enabled`, `ledger.brake_tripped` before every post; claim → render → post → finish.
- Receipt with known `post_url`: text includes it. Unknown: two images (receipt + original blind card from the ledger's stored image or a re-render).
- Worker needs env on `edge-worker`: `BLOTATO_API_KEY`, `BLOTATO_TWITTER_ACCOUNT_ID` (reference `${{nba-value.…}}`), `TWITTER_POSTING_ENABLED`, `SOCIAL_DRY_RUN=true`, `SOCIAL_SPORTS=nba`.

- [ ] **Step 1: Failing tests** (fake conn functions via monkeypatch, `FakePoster` recording calls):
  `test_dry_run_never_calls_blotato`, `test_brake_stops_all_posting`, `test_blind_waits_for_gate_pass`, `test_same_key_not_posted_twice_across_ticks`, `test_cheatsheet_fires_at_11_et_across_dst` (one date in EDT, one in EST), `test_social_error_does_not_break_tick`, `test_receipt_uses_two_images_without_post_url`.
- [ ] **Step 2-4:** FAIL → implement → PASS. Then full suite: `python3 -m pytest tests/unit -q -p no:cacheprovider` → all pass; DB suite with `EDGE_TEST_DB_URL` → all pass.
- [ ] **Step 5: Commit** `feat(social): runner in edge-worker with dry-run poster, brakes and ledger`

---

### Task 12: Reveal pages on truline.app

**Files:**
- Create: `frontend/src/pages/BlindPlay.tsx`, `frontend/src/pages/BlindList.tsx`, `frontend/src/pages/BlindPlay.test.tsx`
- Modify: `frontend/src/App.tsx` (routes `/blind`, `/blind/:sport/:n`), `frontend/src/lib/edgeApi.ts` (`blind(sport)`, `blindPlay(sport, n)`; `SignupSource` += `'blind_reveal'`), `frontend/src/hooks/useEdge.ts`, `frontend/src/types/edge.ts`

**Interfaces:**
- Consumes Task 8 API shapes.
- `BlindPlay` page: headshot (ESPN URL from API field `headshot_url`, silhouette fallback), name, team + opponent logos, card numbers, matchup panel, `explain` text, links to odds screen and player page, result once graded, `<EmailSignup source="blind_reveal" />`, `<EdgeFooter />`.
- `BlindList`: all blind plays newest first, running record (W-L-P, units) shown always (wins and losses).

- [ ] **Step 1: Failing tests:** renders name + explanation for a revealed play; shows "void" for a void result; list shows losses as well as wins; signup uses source `blind_reveal`.
- [ ] **Step 2-4:** `npx vitest run src/pages/BlindPlay.test.tsx` FAIL → implement → PASS; `npm run build` passes; `npx eslint` on new files clean.
- [ ] **Step 5: Commit** `feat(social): public blind play reveal and list pages`

---

### Task 13: Ship in dry-run, then go live

- [ ] **Step 1:** Push `main` (ask Samir first). Check both Railway services deploy (`nba-value`, `edge-worker`) and Vercel.
- [ ] **Step 2:** On `edge-worker` set `SOCIAL_DRY_RUN=true`, `SOCIAL_SPORTS=nba`, Blotato vars, `TWITTER_POSTING_ENABLED=true`. Run Task 3's backfill script once.
- [ ] **Step 3:** Run Task 5 gate for NBA with `--write-state`; commit the report. If `fail`, stop and report to Samir.
- [ ] **Step 4:** After 2-3 nights, `python -m src.edge.social preview --out <scratch> --since <date>` and show Samir the cards and texts.
- [ ] **Step 5:** On Samir's OK: `SOCIAL_DRY_RUN=false`. Watch the first night's posts live on @trulineapp. Update memory.
- [ ] **Step 6:** After one clean NBA week: run the NFL gate, then `SOCIAL_SPORTS=nba,nfl`.
