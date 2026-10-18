# NFL Matchup Tags and Anytime TD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the anytime TD market to NFL Edge (live and backtest), and attach seven matchup, usage and injury tags to +EV rows. Each tag is graded on the 2023–25 backtest, and only proven tags affect ranking.

**Architecture:**
- `src/edge/matchups.py` is pure. It turns nflverse play-by-play into per-game team and player aggregates, and computes tags point-in-time from a `TagContext`.
- The backtest builds a `TagContext` per (season, week) from nflverse files. The live worker stores aggregates and BallDontLie injuries in Postgres, and `src/edge/tagging.py` builds the live context from them.
- Anytime TD reuses the over/under pipeline as Yes = over 0.5 and No = under 0.5, with a per-market consensus rule.

**Tech Stack:** Python 3.11, psycopg2, SQLAlchemy models (`src/models/edge.py`), pandas/pyarrow (nflverse parquet), httpx (BallDontLie), pytest; React 18 + TypeScript + Vitest.

**Spec:** `backend/docs/superpowers/specs/2026-10-18-edge-nfl-matchup-tags-design.md`

## Global Constraints

Run all commands from `backend/` unless a step says otherwise.

**Tags (never change the price):**
- Tags never change `fair_prob`, `ev` or `is_ev`.
- Tag cutoffs are fixed. Do not tune them:
  - rank top 8 / bottom 8 of 32;
  - defense window: last 8 games;
  - minimum 4 games;
  - `usage_up`: last 3 games, ≥ 5 percentage points above the earlier same-season share, with ≥ 2 earlier games;
  - `rz_role`: ≥ 25% over the last 4 games;
  - `teammate_out`: ≥ 30% over the last 4 games, status Out or Doubtful.
- Point-in-time: a tag for a game in (season, week) uses only rows with `(season, week) < (that season, that week)`.
- Red zone = `yardline_100 <= 20`. FB counts as RB. Positions other than QB/RB/WR/TE get no matchup or usage tags.
- Teams use BallDontLie abbreviations everywhere (map nflverse codes with `teams.NFLVERSE_TO_BDL`). Players use `players.normalize_name` keys.

**Anytime TD:**
- Market key `player_anytime_td`. "Yes" maps to side `over`, line 0.5. "No" maps to `under`, line 0.5. The display label is "Anytime TD".
- Fair rule: consensus inputs are `{"pinnacle", "novig", "prophetx"}` only, quoting both sides, at least 2 books, `MAX_HOLD` applies. Every other market keeps `MIN_BOOKS = 3` with a sharp book required.
- Grading: rush TDs + rec TDs + return TDs ≥ 1 is a win. Passing TDs never count.

**Backtest TD fetch:**
- `kind = 'td'`, credit cap `BT_CREDIT_CAP=80000`.
- Pilot week first. The full run happens **only after Samir confirms**; that is outside this plan's tasks.

**Tag verdicts:**
- **Proven:** n ≥ 200 tagged, and the CLV lift lower bound > 0 at z = `NormalDist().inv_cdf(1 - 0.05 / (2 * m))`, where m = the number of (tag, direction) pairs with n ≥ 200. Also, the lift is > 0 in ≥ 2 of 3 seasons.
- **Negative:** the mirror image.
- **Neutral:** everything else.

**Live serving:**
- Hide tags when the last successful matchups refresh is more than 8 days old.
- If `edge_tag_status` is empty, every verdict is neutral.

**Workflow:**
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never push or merge without Samir's go-ahead. Never touch localhost:5432.
- DB tests: `EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest`.
- Unit suite: `python3 -m pytest tests/unit -q`.

**Spec clarifications (decided here):**
- **Signed lift in the sort key.** The sort key is `ev + Σ clv_lift` over the row's tags whose verdict is proven or negative. `clv_lift` is signed and measured per (tag, direction), so a proven headwind lowers a row and a proven tailwind raises it. This is what spec §7's tailwind-minus-headwind wording means once lift is measured per direction.
- **`questionable` uses the standard directions.** It is a headwind for over/Yes and a tailwind for under/No. The chip's colour comes from its verdict, as for every tag.
- **Live injuries are a current snapshot.** `edge_nfl_injuries` stores BallDontLie's current status per player, not per week. A status other than IR older than 14 days is ignored. IR, IR-R, PUP-R, NFI-R and Reserve-Sus count as Out.
- **TD tags apply to RB/WR/TE.** `def_rz`, `def_td`, `rz_role` and the anytime TD part of `usage_up`/`teammate_out` cover only these positions.

## Review Focus

1. **A player traded mid-season.** His latest team in the aggregates is not one of the event's two teams. He gets no player tags, and never his old team's teammates or opponent. Pinned in Task 5.
2. **Duplicate name keys.** Two players with the same `normalize_name` key on the event's teams get no player tags, and are never guessed. Pinned in Task 5.
3. **Week 1 and byes.** Windows count games, not weeks. A bye never counts as a zero game. Week 1 fills from last season's final games. Pinned in Task 4 (`window_games`) and Task 5.
4. **Stale live injury rows.** A BallDontLie "Questionable" from three weeks ago must not fire `questionable` today, while IR stays Out indefinitely. Pinned in Task 6.
5. **Exchange ladders inside CLV.** `group()` keeps the best price per book and side for every market, so duplicate ProphetX quotes can't flip the close, and the odds grid shows the price a bettor can actually take. Pinned in Task 1.

---

### Task 1: Anytime TD — parsing, best-price grouping, per-market fair rule

**Files:**
- Modify: `src/edge/config.py` (add the market, `TD_MARKET`, `MarketRule`, `MARKET_RULES`; delete the "Anytime TD is excluded" comment)
- Modify: `src/edge/ingest.py:_outcome_quote`, `src/edge/engine.py:group`, `src/edge/engine.py:fair_prob`, `src/edge/views.py:MARKET_LABEL`, `src/edge/views.py:describe`, `src/edge/backtest/plan.py` (`BT_PROP_MARKETS`; `PROPS_COST` uses it so it stays 200)
- Test: `tests/unit/test_edge_engine.py`, `tests/unit/test_edge_ingest.py` (create it if absent, matching the engine test style)

**Interfaces:**
- Produces:
  - `config.TD_MARKET = "player_anytime_td"` (appended last in `NFL_PROP_MARKETS`);
  - `@dataclass(frozen=True) class MarketRule: min_books: int; inputs: frozenset[str] | None` (None = any non-DFS book) and `require_sharp: bool`;
  - `config.MARKET_RULES: dict[str, MarketRule] = {TD_MARKET: MarketRule(2, SHARP_BOOKS, False)}`;
  - `config.DEFAULT_RULE = MarketRule(MIN_BOOKS, None, True)`;
  - `MARKET_LABEL[TD_MARKET] = "Anytime TD"`.
  - `plan.BT_PROP_MARKETS = tuple(m for m in NFL_PROP_MARKETS if m != TD_MARKET)`, with `PROPS_COST = 10 * REGION_EQUIVALENTS * len(BT_PROP_MARKETS)` (unchanged at 200).

- [ ] **Step 1: Write the failing tests**

```python
# test_edge_ingest.py
def test_anytime_td_yes_no_map_to_over_under_half():
    ev = {"id": "e1", "home_team": "H", "away_team": "A", "bookmakers": [{"key": "novig", "markets": [
        {"key": "player_anytime_td", "outcomes": [
            {"name": "Yes", "description": "Woody Marks", "price": 141},
            {"name": "No", "description": "Woody Marks", "price": -144}]}]}]}
    qs = parse_event(ev, T0)
    assert {(q.side, q.line, q.price) for q in qs} == {("over", 0.5, 141), ("under", 0.5, -144)}

def test_anytime_td_yes_without_point_is_kept_but_other_props_still_need_a_point():
    ...  # player_rush_yds outcome without "point" still returns no quote

# test_edge_engine.py
def test_group_keeps_the_best_price_per_book_side():
    qs = [q("prophetx", "over", 130, line=0.5, market="player_anytime_td"),
          q("prophetx", "over", 132, line=0.5, market="player_anytime_td"),
          q("prophetx", "over", 129, line=0.5, market="player_anytime_td")]
    assert engine.group(qs)[("ev1", "player_anytime_td", "Jayson Tatum", 0.5)]["prophetx"]["over"].price == 132

def test_td_consensus_needs_two_two_sided_sharps_and_ignores_retail():
    td = lambda b, o, u: [q(b, "over", o, 0.5, "player_anytime_td"), q(b, "under", u, 0.5, "player_anytime_td")]
    books = engine.group(td("pinnacle", 125, -152) + td("novig", 141, -144)
                         + td("draftkings", 125, -160) + [q("fanduel", "over", 130, 0.5, "player_anytime_td")])
    f = engine.fair_prob(books[("ev1", "player_anytime_td", "Jayson Tatum", 0.5)], "player_anytime_td", "over", None)
    assert f.n_books == 2 and not f.thin   # draftkings two-sided but not an input; fanduel yes-only

def test_td_with_one_sharp_is_thin():
    ...  # pinnacle only (two-sided) + 3 retail yes -> thin

def test_other_markets_keep_three_books_and_a_sharp():
    ...  # existing behaviour: player_rush_yds with pinnacle+novig only -> thin

def test_backtest_props_cost_stays_200_without_td():
    assert plan.PROPS_COST == 200 and TD_MARKET not in plan.BT_PROP_MARKETS

def test_describe_anytime_td_reads_yes_no():
    assert describe("player_anytime_td", "Woody Marks", "over", 0.5, "H", "A") == "Woody Marks Anytime TD — Yes"
    assert describe("player_anytime_td", "Woody Marks", "under", 0.5, "H", "A") == "Woody Marks Anytime TD — No"
```

- [ ] **Step 2: Run** `python3 -m pytest tests/unit/test_edge_engine.py tests/unit/test_edge_ingest.py -q`. Expected: the new tests FAIL. Yes/No outcomes are dropped, the last ladder price wins, and the TD rule is not applied.

- [ ] **Step 3: Implement**
  - **`ingest._outcome_quote`:** when `market == TD_MARKET`, map `{"Yes": "over", "No": "under"}`, use line `0.5`, and skip the point check.
  - **`engine.group`:** on a duplicate (book, side), keep the higher `american_to_decimal(price)`.
  - **`engine.fair_prob`:** look up `rule = MARKET_RULES.get(market, DEFAULT_RULE)`. Skip books outside `rule.inputs` when it is set. Thin when `n < rule.min_books`, or when `rule.require_sharp` is set and there is no sharp.
  - **`views.describe`:** for `TD_MARKET`, return `f"{player} Anytime TD — {'Yes' if side == 'over' else 'No'}"`.

- [ ] **Step 4: Run** `python3 -m pytest tests/unit -q`. Expected: all pass. The existing tests stay green, since `DEFAULT_RULE` equals today's behaviour.

- [ ] **Step 5: Commit** `feat(edge): anytime TD market — Yes/No parsing, best-price grouping, 2-of-3 sharp consensus`

### Task 2: Anytime TD grading — TD stat columns live and in the backtest

**Files:**
- Modify:
  - `src/models/edge.py:EdgeNflPlayerGame` (add `rush_tds`, `rec_tds`, `ret_tds`, each Integer, default 0, server_default "0");
  - `src/database.py:init_db` (`column_migrations`: the three columns on `edge_nfl_player_games`, each `"INTEGER NOT NULL DEFAULT 0"`);
  - `src/edge/nfl_players.py` (`STAT_COLS`, `_BDL_FIELDS`, `NFL_STAT_PARTS`, `stat_row`);
  - `src/edge/backtest/nflverse.py:STAT_MAP`;
  - `src/edge/worker.py` (one-time reload).
- Test: `tests/unit/test_edge_nfl_players.py`, `tests/unit/test_edge_backtest_nflverse.py`, `tests/unit/test_edge_worker.py` (the existing files for these modules; create the worker test file if absent)

**Interfaces:**
- Produces:
  - `NFL_STAT_PARTS["player_anytime_td"] = ("rush_tds", "rec_tds", "ret_tds")`;
  - stat dicts carry `rush_tds`, `rec_tds` and `ret_tds`, in both live (`stat_row`) and backtest (`stats_index`).
- BallDontLie fields (verified 2026-10-18):
  - `rush_tds ← rushing_touchdowns`;
  - `rec_tds ← receiving_touchdowns`;
  - `ret_tds ← kick_return_touchdowns + punt_return_touchdowns`.
- nflverse:
  - `rushing_tds → rush_tds`;
  - `receiving_tds → rec_tds`;
  - `special_teams_tds → ret_tds`.

- [ ] **Step 1: Write the failing tests**
  - `test_stat_row_reads_td_fields_and_sums_return_tds`: a BallDontLie stat with `kick_return_touchdowns=1, punt_return_touchdowns=1` gives `ret_tds == 2`.
  - `test_anytime_td_grades_rush_rec_return_and_ignores_passing`:
    - `nfl_prop_result({"market": TD_MARKET, "line": 0.5, "side": "over"}, {..., "pass_tds": 3, "rush_tds": 0, "rec_tds": 0, "ret_tds": 0}, True, True) == "loss"`;
    - `ret_tds: 1` gives `"win"`;
    - under/No with `rec_tds: 1` gives `"loss"`.
  - `test_anytime_td_inactive_is_void`: no stat row, snaps loaded, not played, gives `"void"`.
  - `test_stats_index_maps_nflverse_td_columns`.
  - `test_td_reload_clears_stats_loaded_once`: `nfl_td_reload_step(conn, now)` runs `UPDATE edge_nfl_games SET stats_loaded_at = NULL` exactly once, gated by the state key `nfl:td_reload` = `"1"`. This needs the DB, so put it in `tests/integration/test_edge_store_db.py`.

- [ ] **Step 2: Run** `python3 -m pytest tests/unit -q -k "td"`. Expected: FAIL. The columns and mapping are missing.

- [ ] **Step 3: Implement**
  - Add the columns and the BallDontLie/nflverse mappings above. `stat_row` sums the two return fields into `ret_tds`.
  - Add `def nfl_td_reload_step(conn, now: datetime) -> bool` to `worker.py`, and call it from the NFL tick next to `nfl_backfill_step`. It reloads box scores so past games gain TD counts; `load_due_stats` re-pulls 6 games per tick.

- [ ] **Step 4: Run** the unit suite and `EDGE_TEST_DB_URL=… python3 -m pytest tests/integration/test_edge_store_db.py -q`. Expected: all pass.

- [ ] **Step 5: Commit** `feat(edge): grade anytime TD (rush+rec+return TDs) live and in the backtest`

### Task 3: Backtest TD fetch kind

**Files:**
- Modify:
  - `src/edge/client.py:get_historical_event_odds` (add `markets: tuple[str, ...] | None = None`, defaulting to `sport.prop_markets`);
  - `src/edge/backtest/plan.py`;
  - `src/edge/backtest/fetch.py`;
  - `src/edge/backtest/__main__.py`;
  - `src/edge/backtest/replay.py`.
- Test: `tests/unit/test_edge_backtest_plan.py`, `tests/unit/test_edge_backtest_fetch.py`, `tests/unit/test_edge_backtest_cli.py`, `tests/unit/test_edge_backtest_replay.py`

**Interfaces:**
- Produces:
  - Consumes `plan.BT_PROP_MARKETS` from Task 1.
  - `plan.TD_COST = 10 * REGION_EQUIVALENTS * 1` (= 20). `Job.cost` covers `"td"`.
  - `plan.snapshot_jobs(pairs, kinds: tuple[str, ...] = ("props", "lines"))`. A `td` job is one event per job, like `props`.
  - `fetch.run_fetch(..., kinds: tuple[str, ...] = ("props", "lines"))`. `td` jobs call `client.get_historical_event_odds(eid, NFL, date, markets=(TD_MARKET,))`; `props` jobs pass `markets=plan.BT_PROP_MARKETS`.
  - CLI: `fetch --kind {all,td}` (default `all` = props+lines). `--kind td` passes `kinds=("td",)`.
  - `replay.run_replay` and `replay_game` read `kind IN ('props','lines','td')`.

- [ ] **Step 1: Write the failing tests**
  - `test_td_jobs_cost_20_and_one_per_event_label`: 2 games × 4 labels gives 8 `td` jobs, each `cost == 20`.
  - `test_fetch_td_kind_requests_only_the_td_market_and_never_rebuys_props`: the fake client records `markets`; stored `props` keys are untouched, and `td` rows are inserted with `kind == "td"`.
  - `test_cli_kind_td_passes_td_kinds`.
  - `test_replay_merges_td_snapshot_quotes`: a `td` Snap with a pinnacle and novig Yes/No and a retail Yes at a better price gives an `is_ev` pick with `market == TD_MARKET`.

- [ ] **Step 2: Run** `python3 -m pytest tests/unit -q -k "backtest"`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement** as the interfaces above say. `dry_run` reports `td` counts and credits when `kinds` includes `td`.

- [ ] **Step 4: Run** the unit suite and `EDGE_TEST_DB_URL=… python3 -m pytest tests/integration/test_edge_backtest_db.py -q`. Expected: all pass.

- [ ] **Step 5: Commit** `feat(edge-bt): td snapshot kind — fetch --kind td, 20 credits/call, replayed with props`

### Task 4: `matchups.py` — per-game aggregates from play-by-play

**Files:**
- Create: `src/edge/matchups.py`
- Test: `tests/unit/test_edge_matchups.py`

**Interfaces:**
- Produces:
  - `POSITIONS = ("QB", "RB", "WR", "TE")`;
  - `PBP_COLS = ("season", "week", "season_type", "game_id", "posteam", "defteam", "play_type", "two_point_attempt", "yardline_100", "rusher_player_id", "receiver_player_id", "passer_player_id", "rushing_yards", "receiving_yards", "passing_yards", "rush_touchdown", "pass_touchdown", "td_player_id")`.
  - `@dataclass(frozen=True) class TeamGame: season: int; week: int; team: str; pos: str; yds: float; rz_opps: int; tds: int`. These are allowed by `team` as the defense. There is exactly one row per (season, week, team, pos) for every game the team played, with zeros where none.
  - `@dataclass(frozen=True) class PlayerGame: season, week, team, player_key, player_name, pos, carries, targets, rz_carries, rz_targets, team_carries, team_targets, team_rz_opps, pos_carries, pos_targets`.
  - `def aggregate(pbp: pd.DataFrame, rosters: pd.DataFrame) -> tuple[list[TeamGame], list[PlayerGame]]`. Only REG plays with `play_type in ("run", "pass")` and `two_point_attempt != 1` count. Positions come from `rosters` by `gsis_id` (`FB → RB`). Players outside `POSITIONS` are kept out of player rows, but their plays still count toward team totals.
  - `def window_games(rows: list, season: int, week: int, n: int, key) -> list`: the last `n` rows matching `key` strictly before (season, week). Current-season rows come first. If fewer than 4 current-season rows exist, the window is filled from the previous season's latest rows, up to `n`.
  - `NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/"`.
  - `load_pbp(seasons)` reads `pbp/play_by_play_{season}.parquet` with `columns=list(PBP_COLS)`.
  - `load_rosters(seasons)` reads `rosters/roster_{season}.parquet` and returns `gsis_id, full_name, position, team, season`.

Per-position definitions, using the play's rusher or receiver position:
- **RB** `yds` = rushing yards by RB rushers + receiving yards to RB receivers.
- **WR/TE** `yds` = receiving yards.
- **QB** `yds` = passing yards on plays whose passer is a QB.
- **`rz_opps`** = red-zone carries + red-zone targets by that position.
- **`tds`** = rushing TDs by that position's rushers + receiving TDs to that position's receivers. QB counts rushing TDs only.

- [ ] **Step 1: Write the failing tests**, all on a hand-built 8–12-play frame:
  - `test_rb_yards_allowed_counts_rush_and_receiving_by_rb`;
  - `test_red_zone_is_yardline_20_or_less` (a play at 20 counts, 21 doesn't);
  - `test_two_point_tries_and_postseason_never_count`;
  - `test_fb_is_rb_and_k_is_excluded_from_players_but_counts_to_team_totals`;
  - `test_every_game_gets_four_position_rows_with_zeros`;
  - `test_player_shares_denominators` (team_carries, pos_carries, team_rz_opps);
  - `test_window_counts_games_not_weeks_and_skips_byes`: rows for weeks 1, 2, 4, 5, 6 and a request at week 7 with n=3 give weeks 4, 5, 6;
  - `test_window_week1_fills_from_last_season`: season 2024 week 1, n=8, with 17 rows in 2023, gives the last 8 of 2023;
  - `test_window_never_includes_the_requested_week`.

- [ ] **Step 2: Run** `python3 -m pytest tests/unit/test_edge_matchups.py -q`. Expected: FAIL (module missing).

- [ ] **Step 3: Implement** with pandas groupbys. Build the four-position zero rows from `(season, week, game_id, defteam)` × `POSITIONS`.

- [ ] **Step 4: Run** it again. Expected: PASS.

- [ ] **Step 5: Commit** `feat(edge): matchups aggregates from nflverse play-by-play`

### Task 5: `matchups.py` — the seven tags

**Files:**
- Modify: `src/edge/matchups.py`
- Test: `tests/unit/test_edge_matchups.py`

**Interfaces:**
- Consumes: `TeamGame`, `PlayerGame` and `window_games` from Task 4; `TD_MARKET` from Task 1.
- Produces:
  - `@dataclass(frozen=True) class Tag: key: str; direction: str; detail: str` (`direction` is `"tailwind"` or `"headwind"`, relative to the row's side).
  - `TAG_KEYS = ("def_soft", "def_tough", "def_rz", "def_td", "usage_up", "rz_role", "teammate_out", "questionable")`.
  - `POLARITY: dict[str, int]`: `+1` for all keys except `def_tough` and `questionable` (`-1`). Direction = tailwind if `POLARITY × (+1 for over, −1 for under) > 0`.
  - `POSITION_MARKETS = {"QB": ("player_pass_yds", "player_pass_completions", "player_pass_attempts"), "RB": ("player_rush_yds", "player_rush_attempts", "player_rush_reception_yds"), "WR": ("player_reception_yds", "player_receptions"), "TE": ("player_reception_yds", "player_receptions")}`.
  - `def applies(key: str, pos: str, market: str) -> bool`:

    | Key | Applies to |
    |---|---|
    | `def_soft`, `def_tough` | `POSITION_MARKETS[pos]` |
    | `def_rz` | `TD_MARKET` for RB/WR/TE; `player_rush_yds` for RB; `player_reception_yds` for WR/TE |
    | `def_td`, `rz_role` | `TD_MARKET` for RB/WR/TE |
    | `usage_up`, `teammate_out` | RB/WR/TE only: `POSITION_MARKETS[pos]` plus `TD_MARKET` |
    | `questionable` | any `player_*` market, any position |

  - `MARKETS_FOR: dict[str, frozenset[str]]`: the union of markets each key can apply to, for the report's control group.
  - `@dataclass class TagContext`, with:
    - `def_rank: dict[tuple[str, str], dict[str, tuple[float, int]]]`, keyed (team, pos), with stat `"yds" | "rz" | "td"` → (per-game value, rank 1 = most allowed). Only teams with ≥ 4 window games are ranked.
    - `players: dict[str, list[tuple[str, str]]]`: player_key → `[(team, pos)]`, the latest row per player. A list, because of duplicate keys.
    - `player_rows: dict[tuple[str, str], list[PlayerGame]]`, keyed (team, player_key), oldest first, all before the cutoff.
    - `injuries: dict[str, dict[str, str]]`: team → {player_key: `"out" | "doubtful" | "questionable"`}.
  - `def build_context(team_games, player_games, injuries, season: int, week: int) -> TagContext`. It uses only rows before (season, week), and `window_games(n=8)` for `def_rank`.
  - `def tags_for(ctx: TagContext, player: str, event_teams: tuple[str, str], market: str, side: str) -> list[Tag]`.

Detail copy (exact):
- `def_*`: `f"{opp} allows {value:.0f} {pos} {STAT_WORD} per game ({ordinal(rank)} most)"`. `STAT_WORD` is "yds" for `def_soft`/`def_tough`, "red-zone looks" for `def_rz` and "TDs" for `def_td`. For `def_tough`, the rank is shown as "(… fewest)" using `33 − rank`.
- `usage_up`: `f"{share_word} {recent:.0%} last 3 vs {earlier:.0%} before"`. `share_word` is "Carry share" for RB and "Target share" for WR/TE.
- `rz_role`: `f"{share:.0%} of team red-zone looks, last 4"`.
- `teammate_out`: `f"{name} ({status}) — {share:.0%} of {pos} {unit}, last 4"`. `unit` is "carries" for RB and "targets" for WR/TE.
- `questionable`: `"Listed Questionable"`.

- [ ] **Step 1: Write the failing tests.** Build the contexts by hand. Cover each side of every cutoff.
  - **Defense ranks:**
    - `test_def_soft_fires_at_rank_8_not_9`;
    - `test_def_tough_fires_at_rank_25_not_24`;
    - `test_def_tags_need_four_window_games`;
    - `test_def_rz_and_def_td_only_on_their_markets`.
  - **Usage:**
    - `test_usage_up_at_exactly_5_points_fires_and_4_9_does_not`;
    - `test_usage_up_needs_two_earlier_games`;
    - `test_rz_role_25_percent_fires_24_does_not`.
  - **Injuries:**
    - `test_teammate_out_requires_out_or_doubtful_and_30_percent`;
    - `test_teammate_out_not_on_the_injured_player_himself`;
    - `test_questionable_is_headwind_on_over_tailwind_on_under`.
  - **Direction:** `test_def_tough_is_headwind_for_over_and_tailwind_for_under`.
  - **Identity:**
    - `test_traded_player_whose_latest_team_is_not_in_the_event_gets_no_tags` (Review Focus 1);
    - `test_duplicate_name_key_on_event_teams_gets_no_tags` (Review Focus 2);
    - `test_kicker_and_unknown_player_get_no_tags`.
  - **Point-in-time:** `test_build_context_excludes_the_games_own_week_and_later` (Review Focus 3).
  - **Copy:** `test_detail_copy_def_soft` asserts `"JAX allows 118 RB yds per game (3rd most)"`.

- [ ] **Step 2: Run** `python3 -m pytest tests/unit/test_edge_matchups.py -q`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement `build_context` and `tags_for`.**
  - In `tags_for`, the opponent is the event team that isn't the player's team.
  - **Usage windows:**
    - `usage_up` uses current-season rows only.
    - `rz_role` uses `window_games(n=4)` over the player's rows.
    - `teammate_out` takes each injured teammate's share as Σ his carries (or targets) ÷ Σ `pos_carries` (or `pos_targets`) over his own last 4 rows.

- [ ] **Step 4: Run** it again. Expected: PASS.

- [ ] **Step 5: Commit** `feat(edge): seven point-in-time matchup, usage and injury tags`

### Task 6: Injuries — nflverse history and BallDontLie live

**Files:**
- Create: `src/edge/nfl_injuries.py`
- Modify:
  - `src/models/edge.py` (add `EdgeNflInjury`: `team String(4)`, `player_norm String(80)`, `player_name String(120)`, `pos String(4)`, `status String(16)`, `reported_at DateTime(tz)`, `loaded_at DateTime(tz)`; unique (team, player_norm));
  - `src/models/__init__.py` (export it).
- Test: `tests/unit/test_edge_nfl_injuries.py`, `tests/integration/test_edge_store_db.py`

**Interfaces:**
- Produces:
  - `def normalize_status(raw: str) -> str | None`:
    - `"Out", "IR", "IR-R", "PUP-R", "NFI-R", "Reserve-Sus"` → `"out"`;
    - `"Doubtful"` → `"doubtful"`;
    - `"Questionable"` → `"questionable"`;
    - anything else → `None`.
  - `IR_TYPES = frozenset({"IR", "IR-R", "PUP-R", "NFI-R", "Reserve-Sus"})`.
  - `STALE = timedelta(days=14)`.
  - `def history_index(frame) -> dict[tuple[int, int], dict[str, dict[str, str]]]`: (season, week) → team (BallDontLie abbreviation) → {player_key: status}. It reads REG rows of the nflverse `injuries/injuries_{season}.parquet` (`report_status`). `load_history(seasons)` downloads them.
  - `def live_rows(bdl_rows: list[dict], now: datetime) -> list[dict]`: maps BallDontLie `/nfl/v1/player_injuries` rows (`player.first_name/last_name/position_abbreviation/team.abbreviation`, `status`, `date`). It drops rows whose status normalizes to None, and drops non-IR statuses older than `STALE`.
  - `async def fetch_live(bdl: BdlNflClient) -> list[dict]`: adds a `player_injuries()` method to `BdlNflClient` that uses `_paged("player_injuries", [])`.
  - `def store_live(conn, rows, now) -> int`: deletes and inserts in one transaction.
  - `def live_index(conn) -> dict[str, dict[str, str]]`: team → {player_key: status}.

- [ ] **Step 1: Write the failing tests**
  - `test_ir_types_count_as_out`.
  - `test_unknown_status_dropped`.
  - `test_questionable_older_than_14_days_ignored_but_ir_kept` (Review Focus 4).
  - `test_history_index_maps_nflverse_team_codes_to_bdl` (e.g. `LA → LAR`, via `NFLVERSE_TO_BDL`).
  - The DB test `test_store_live_replaces_the_snapshot` checks that a second load removes players no longer listed.

- [ ] **Step 2: Run** them. Expected: FAIL.

- [ ] **Step 3: Implement** as specified.

- [ ] **Step 4: Run** the unit suite and the DB tests. Expected: all pass.

- [ ] **Step 5: Commit** `feat(edge): NFL injury status — nflverse history, BallDontLie live snapshot`

### Task 7: Live storage and worker steps

**Files:**
- Modify:
  - `src/models/edge.py`:
    - `EdgeNflTeamWeek`: the `TeamGame` columns plus `loaded_at`; unique (season, week, team, pos).
    - `EdgeNflPlayerWeek`: the `PlayerGame` columns plus `loaded_at`; unique (season, week, team, player_key).
    - `EdgeTagStatus`: `tag`, `direction`, `verdict`, `clv_lift Float`, `n Integer`, `run_id String`, `updated_at`; unique (tag, direction).
  - `src/models/__init__.py`.
  - `src/edge/worker.py`.
  - `src/edge/views.py:status`.
- Create: `src/edge/matchups_store.py`
- Test: `tests/unit/test_edge_worker.py`, `tests/integration/test_edge_store_db.py`

**Interfaces:**
- Consumes: `aggregate`, `load_pbp` and `load_rosters` (Task 4); `fetch_live`, `live_rows` and `store_live` (Task 6).
- Produces:
  - `matchups_store.upsert(conn, team_games, player_games, now) -> int`;
  - `matchups_store.load(conn, seasons) -> tuple[list[TeamGame], list[PlayerGame]]`.
  - `worker.nfl_matchups_step(conn, now, frames=None) -> int`: every 24 h, gated by state `nfl:last_matchups_step`. It loads the current and previous season. On success it sets `nfl:matchups_ok_at` and clears `matchups_error`. On an exception it rolls back and sets `matchups_error`, and never raises.
  - `worker.nfl_injuries_step(conn, nfl_bdl, now) -> int`: every 2 h from Thursday 00:00 to Monday 23:59 ET and every 6 h otherwise, gated by `nfl:last_injuries_step`. It sets `nfl:injuries_ok_at` or `injuries_error`, and never raises.
  - `views.status` adds `matchups_error`, `injuries_error`, `matchups_ok_at` and `injuries_ok_at`.

- [ ] **Step 1: Write the failing tests**
  - `test_matchups_step_runs_once_per_24h`.
  - `test_matchups_step_error_sets_state_and_does_not_raise`.
  - `test_injuries_step_cadence_2h_thursday_to_monday_6h_otherwise`: Sunday at +2 h runs, Tuesday at +2 h doesn't, Tuesday at +6 h runs.
  - `test_injuries_step_error_sets_state_and_does_not_raise`.
  - The DB test `test_matchups_upsert_round_trip_and_update_in_place`.

- [ ] **Step 2: Run** them. Expected: FAIL.

- [ ] **Step 3: Implement.** Call both steps from the NFL part of `tick`, after `nfl_snaps_step`. Run pandas work through `asyncio.to_thread`, the same way `nfl_backfill_step` handles snaps.

- [ ] **Step 4: Run** the unit suite and the DB tests. Expected: all pass.

- [ ] **Step 5: Commit** `feat(edge): store NFL matchup aggregates and injuries; daily and 2-hourly worker steps`

### Task 8: Backtest replay attaches tags

**Files:**
- Modify:
  - `src/models/edge.py:EdgeBtPick` (add `tags` JSONB, nullable);
  - `src/database.py` (column migration `("edge_bt_picks", "tags", "JSONB")`);
  - `src/edge/backtest/replay.py`;
  - `src/edge/backtest/__main__.py`.
- Test: `tests/unit/test_edge_backtest_replay.py`, `tests/integration/test_edge_backtest_db.py`

**Interfaces:**
- Consumes: `build_context` and `tags_for` (Task 5); `history_index` (Task 6); `aggregate`, `load_pbp` and `load_rosters` (Task 4).
- Produces:
  - `replay_game(..., tagger: Callable[[str, str, str], list[Tag]] | None = None)`. `tagger(player, market, side)` gives each pick `"tags": [{"key", "direction"}]`. Game lines get `[]`.
  - `run_replay(..., contexts: Callable[[Game], TagContext] | None = None)`. `PICK_COLS` gains `"tags"`, inserted as `Json(...)`.
  - The CLI `replay` loads pbp, rosters and injuries for `[min(seasons) − 1, *seasons]`, builds the aggregates once, and caches `build_context` per (season, week).

- [ ] **Step 1: Write the failing tests**
  - `test_replay_attaches_tags_from_the_tagger_for_player_picks_only`.
  - `test_replay_without_tagger_writes_empty_tags`.
  - The DB test `test_bt_pick_tags_round_trip_jsonb`.

- [ ] **Step 2: Run** them. Expected: FAIL.

- [ ] **Step 3: Implement.** The tagger closes over the game's `TagContext` and `(game.home, game.away)`.

- [ ] **Step 4: Run** the unit suite and the DB tests. Expected: all pass.

- [ ] **Step 5: Commit** `feat(edge-bt): replay tags each pick point-in-time`

### Task 9: Report — tag verdicts

**Files:**
- Modify: `src/edge/backtest/report.py`
- Test: `tests/unit/test_edge_backtest_report.py`, `tests/integration/test_edge_backtest_db.py`

**Interfaces:**
- Consumes: `MARKETS_FOR` (Task 5); the picks' `tags` (Task 8); `EdgeTagStatus` (Task 7).
- Produces:
  - `@dataclass(frozen=True) class TagVerdict: tag, direction, n, lift, lo, hi, beat_close, roi, seasons_positive, verdict`.
  - `def tag_verdicts(picks: list[dict]) -> list[TagVerdict]`. Its input is already `first_flag`-deduplicated. It counts picks with a non-null `clv`.
    - **Tagged:** picks that carry (tag, direction).
    - **Control:** picks with `market in MARKETS_FOR[tag]` and no tag with that key.
    - **Lift:** `mean(clv_t) − mean(clv_c)`, with `se = sqrt(pvar_t/n_t + pvar_c/n_c)`.
    - **Bounds:** `lo, hi = lift ∓ z·se`, with z from Global Constraints.
    - **`seasons_positive`:** the number of seasons with lift > 0, among seasons that have both tagged and control picks.
  - `build_report` adds a "## Matchup tags" table with columns `Tag | Direction | Picks | CLV lift (CI) | Beat close | ROI | Seasons + | Verdict`.
  - `run_report` upserts each verdict into `edge_tag_status` with `run_id`.

- [ ] **Step 1: Write the failing tests**, on synthetic picks:
  - `test_verdict_proven_needs_200_and_lower_bound_above_zero_and_two_seasons`;
  - `test_199_tagged_is_neutral_even_with_a_big_lift`;
  - `test_lift_positive_in_one_season_only_is_neutral`;
  - `test_negative_mirror`;
  - `test_m_counts_only_pairs_with_200_plus` (with 14 qualifying pairs, z rounds to 2.91);
  - `test_control_excludes_picks_carrying_the_same_key_in_either_direction`;
  - `test_report_has_matchup_tags_table`;
  - the DB test `test_run_report_upserts_edge_tag_status`.

- [ ] **Step 2: Run** them. Expected: FAIL.

- [ ] **Step 3: Implement** as specified.

- [ ] **Step 4: Run** the unit suite and the DB tests. Expected: all pass.

- [ ] **Step 5: Commit** `feat(edge-bt): grade tags on CLV lift with Bonferroni and season-sign rules`

### Task 10: Live API — tags on rows, ranking, staleness, tracking

**Files:**
- Create: `src/edge/tagging.py`
- Modify:
  - `src/edge/views.py` (`ev_rows`, `nfl_player`, `results`, `results_summary`);
  - `src/edge/opportunities.py`;
  - `src/edge/worker.py` (pass the tagger to `record`);
  - `src/models/edge.py:EdgeOpportunity` (add `tags` JSONB);
  - `src/database.py` (migration `("edge_opportunities", "tags", "JSONB")`).
- Test: `tests/unit/test_edge_tagging.py`, `tests/unit/test_edge_views.py`, `tests/integration/test_edge_store_db.py`

**Interfaces:**
- Consumes: `matchups_store.load` (Task 7), `live_index` (Task 6), `build_context`/`tags_for` (Task 5), `edge_tag_status` (Task 7/9).
- Produces:
  - `TAG_LABEL = {"def_soft": "Soft matchup", "def_tough": "Tough matchup", "def_rz": "Red-zone-friendly D", "def_td": "TD-friendly D", "usage_up": "Usage rising", "rz_role": "Red-zone role", "teammate_out": "Teammate out", "questionable": "Questionable"}`.
  - `def live_tagger(conn, now) -> Callable[[str, tuple[str, str], str, str], list[dict]] | None`. It returns None when `nfl:matchups_ok_at` is missing or older than 8 days. It builds the context from all stored rows (season = current NFL season, week = 99) plus `live_index`. Each tag becomes `{key, label, direction, verdict, lift, detail}`; with no status row the verdict is `"neutral"` and the lift `0.0`.
  - `def sort_key(row) -> float`: `row["ev"] + Σ t["lift"] for t in row["tags"] if t["verdict"] in ("proven", "negative")`.
  - `ev_rows` (NFL) adds `"tags"` to every row (`[]` when no tagger) and sorts by `sort_key`. NBA rows get `"tags": []` and the sort is unchanged.
  - `nfl_player` rows carry `tags` the same way.
  - `opportunities.record(conn, priced, now, tagger=None)` stores `tags` as `[{key, direction}]`.
  - `results_summary` adds `by_tag` when rows carry `tags`, keyed `f"{key}:{direction}"`. Rows without tags are left out of `by_tag`.

- [ ] **Step 1: Write the failing tests**
  - `test_rows_sort_by_ev_plus_signed_lift_of_proven_and_negative_tags_only`: a 3.0% row with a proven +0.6% tag ranks above a 3.5% row; a neutral +5% lift is ignored.
  - `test_tags_hidden_when_matchups_older_than_8_days`.
  - `test_no_tag_status_means_neutral_and_pure_ev_sort`.
  - `test_nba_rows_have_empty_tags`.
  - `test_results_by_tag_groups_by_key_and_direction`.
  - The DB test `test_opportunity_tags_round_trip`.

- [ ] **Step 2: Run** them. Expected: FAIL.

- [ ] **Step 3: Implement.** `ev_rows` builds the tagger once per call. The existing `_cached` wrapper in `api.py` already caches the response.

- [ ] **Step 4: Run** the unit suite and the DB tests. Expected: all pass.

- [ ] **Step 5: Commit** `feat(edge): live NFL tags on EV rows and player pages, proven-lift ranking, per-tag results`

### Task 11: Frontend — tag chips and per-tag results

**Files:**
- Create: `frontend/src/components/edge/TagChips.tsx`, `frontend/src/components/edge/TagChips.test.tsx`
- Modify:
  - `frontend/src/types/edge.ts` (`EvTag`, plus `EvRow.tags: EvTag[]` and `ResultsResponse.by_tag?`);
  - `frontend/src/components/edge/EvTable.tsx`;
  - `frontend/src/pages/EdgePlayer.tsx`;
  - `frontend/src/pages/EdgeResults.tsx`.

**Interfaces:**
- Consumes: the Task 10 row shape `{key, label, direction, verdict, lift, detail}`.
- Produces: `export function TagChips({ tags }: { tags: EvTag[] })`.
  - Each chip shows `label` plus `↑` for a tailwind or `↓` for a headwind.
  - Classes:
    - proven: `bg-[#a4e6ff]/15 text-[#a4e6ff] border-[#a4e6ff]/40`;
    - neutral: `bg-[#1e293b] text-[#94a3b8] border-[#334155]`;
    - negative: `bg-[#fbbf24]/10 text-[#fbbf24] border-[#fbbf24]/40`.
  - `title` text: `${detail} · ${verdictCopy}`.
    - proven: `Proven in 2023–25 backtest: ${signed lift as %} CLV`;
    - negative: `Underperformed in 2023–25 backtest: ${…} CLV`;
    - neutral: `No measurable edge in 2023–25 backtest`.

- [ ] **Step 1: Write the failing tests (Vitest)**
  - `renders one chip per tag with arrow by direction`.
  - `proven chip uses the highlight class and title shows lift`.
  - `negative chip uses the caution class`.
  - `renders nothing for an empty list`.

- [ ] **Step 2: Run** `cd ../frontend && npx vitest run src/components/edge/TagChips.test.tsx`. Expected: FAIL.

- [ ] **Step 3: Implement.**
  - **`EvTable`:** render `<TagChips tags={r.tags ?? []} />` under the bet text.
  - **`EdgePlayer`:** render it next to each EV row.
  - **`EdgeResults`:** for NFL, add a "By tag" table (n, CLV, record) from `by_tag`, using `TAG_LABEL`-equivalent copy from the row labels.

- [ ] **Step 4: Run** `npx vitest run` and `npx tsc --noEmit`. Expected: all pass, with no type errors.

- [ ] **Step 5: Commit** `feat(edge-ui): tag chips on NFL EV rows and player pages; per-tag results`

### Task 12: TD backtest pilot and hand checks

This task spends real credits (about 1–2K) and writes to prod backtest tables only.

- [ ] **Step 1: Run the pilot.** `DATABASE_URL=<Postgres DATABASE_PUBLIC_URL> ODDS_API_KEY=<edge-worker key> BT_CREDIT_CAP=80000 python3 -m src.edge.backtest fetch --kind td --season 2023 --week 10`.
  Expected: `errors: 0`, `unmatched games: 0`, and about 64 calls at 20 credits each. Record the actual credits in the ledger.

- [ ] **Step 2: Replay and report the pilot.** Run `python3 -m src.edge.backtest replay --season 2023`, then `report`, to a scratch path with `--out`. Hand-check against nflverse and the stored payloads:
  - 10 TD picks: the price is in the payload, the fair price comes from ≥ 2 sharp two-sided books, and the grade matches `rushing_tds + receiving_tds + special_teams_tds`;
  - 10 tagged picks: each tag's number matches nflverse.
  Record the results in the ledger.

- [ ] **Step 3: Stop.** Report the pilot's cost and checks to Samir, and ask him to confirm the full TD fetch (about 65K credits). The full run, the final replay and report, the merge and the push all happen only on his go-ahead.
