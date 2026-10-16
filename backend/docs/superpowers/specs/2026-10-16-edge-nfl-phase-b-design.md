# TruLine Edge for NFL, Phase B: player pages, prop grading and track record

- **Date:** 2026-10-16
- **Status:** approach and design approved in brainstorming; awaiting written-spec review
- **Launch target:** live by **Sat 2026-10-24**, before Week 7 Sunday (2026-10-25)
- **Builds on:** `2026-10-16-edge-nfl-phase-a-design.md` (NFL odds, +EV finder and odds screen; live since 2026-10-16)

## 1. Purpose

Phase A put NFL prices and +EV on the site and tracks closing-line value (CLV) for every NFL opportunity. Phase B adds the rest of the NFL product:
- player pages with hit rates;
- win/loss grading for NFL props;
- the NFL track record.

NFL is mid-season, so hit rates rest on real 2026 games plus the 2025 season.

**Success criteria:**
1. Every NFL prop opportunity whose game has finished is graded **win, loss, push or void** under sportsbook rules (§5), or is visibly waiting for data. None is guessed.
2. NFL +EV rows link to a player page showing these hit rates: last 5, last 10, this season, vs. this opponent, and home/away. Each is computed against the line the user is looking at, alongside current book prices.
3. `/nfl/edge/results` publishes the NFL record (CLV, win/loss, ROI by market and book), and it contains no NBA rows.
4. **Preseason games never enter hit rates or grading.**
5. NBA player data, pages and grading are unchanged. No NBA table is altered.
6. A spot check of 10 graded NFL props against box scores matches.

## 2. Scope

**In Phase B:**
- An NFL BallDontLie client.
- NFL player and stat storage.
- Name matching for NFL players.
- An NFL stats ingest step and a backfill (2025 season plus 2026 to date).
- nflverse snap counts as the participation source.
- NFL prop grading.
- NFL hit rates.
- `/edge/players/{id}?sport=nfl` and `/edge/results?sport=nfl` (already sport-aware from Phase A).
- The `/nfl/edge/player/:id` and `/nfl/edge/results` pages.
- A **Track record** tab in the NFL navigation.

**Not in Phase B:**
- The historical odds backtest (Phase C).
- Anytime-TD props (still excluded; see Phase A).
- Defensive or kicking props.
- NFL trends pages.
- Changes to the NFL model tracker at `/nfl` and `/nfl/performance`.

## 3. Verified data (2026-10-16)

All three sources were probed today. The responses are saved under `tests/fixtures/edge/`:

| File | Contents |
|---|---|
| `bdl_nfl_teams.json` | 32 teams, each with `id`, `abbreviation` and `full_name` |
| `bdl_nfl_games_w5.json` | Week 5, 2026: 15 games |
| `bdl_nfl_stats_game.json` | TB @ DAL, game 1392280: 63 player rows |

**BallDontLie NFL** (`https://api.balldontlie.io/nfl/v1`, ALL-STAR, `Authorization` header):
- **`/games?seasons[]=…&weeks[]=…`** returns regular-season (and postseason) games only: weeks 1–7 for 2026, with byes visible.
  - Fields: `id`, `week`, `date`, `status_state` (`final` when done), `home_team`/`visitor_team` (with `abbreviation`), and scores.
- **`/stats?game_ids[]=…&per_page=100`** returns one row per player who recorded any stat, about 60 per game in one page.
  - Fields: `passing_attempts`, `passing_completions`, `passing_yards`, `passing_touchdowns`, `passing_interceptions`, `rushing_attempts`, `rushing_yards`, `receptions`, `receiving_yards`, plus `player`, `team` and `game`.
  - Stats the player didn't record are `null`.
- **`/stats?seasons[]=…` also returns preseason games** (e.g. an Aug 7 game labelled week 1) that are absent from `/games`. Stats are therefore fetched **only by `game_ids` from `/games`**, which excludes preseason (success criterion 4).
- **`/players?team_ids[]=…`** gives rosters, paginated (`meta.next_cursor`; more than 100 per team).

**nflverse snap counts** (`nfl_data_py.import_snap_counts([season])`, already a dependency):
- 2026 data covers weeks 1–6, including Thursday's game the day after it was played.
- Columns: `game_id` (e.g. `2026_06_SEA_DEN`), `season`, `game_type`, `week`, `player`, `pfr_player_id`, `team`, `position`, and `offense_snaps`/`defense_snaps`/`st_snaps`.
- It lists every player who took a snap.

**Why snap counts are needed:** a box score has no row for a player who played but recorded nothing (in the sample, 44 of 63 rows have no offensive stats; receivers with zero catches have no row at all). Sportsbooks grade a played-but-zero receiver's Over as a loss, so "no box-score row" must not mean void. Snap counts are what show the player was active.

## 4. Architecture

**Approach 1, chosen in brainstorming:** separate NFL tables plus a small NFL players module.
- `edge_players.odds_name` is the **sole primary key** of a live NBA table, so an NFL player sharing a name with an NBA player ("Josh Allen") cannot live there. Migrating that key during NBA opener week is the riskiest option.
- Separate tables leave every NBA table untouched (success criterion 5).

Rejected alternatives:
- Generalizing the existing tables: needs a primary-key migration on live tables.
- nflverse as the primary stats source: a second dataset to reconcile, and slower to update.

**New tables** (`src/models/edge.py`, created by the existing `create_all` on startup; no ALTERs to existing tables):

- **`edge_nfl_players`**
  - Primary key `(odds_name, team)`, where `team` is the BallDontLie abbreviation.
  - Columns: `bdl_player_id` (nullable), `position`, `match_method` (`exact|normalized|fuzzy|manual|unmatched`), `confidence`, `updated_at`.
  - Keying on team as well avoids same-name collisions across NFL teams.
- **`edge_nfl_player_games`**
  - Unique on `(bdl_player_id, bdl_game_id)`.
  - Identity columns: `season`, `week`, `game_date`, `team`, `opponent`, `is_home`.
  - Stat columns: `pass_att`, `pass_cmp`, `pass_yds`, `pass_tds`, `pass_int`, `rush_att`, `rush_yds`, `receptions`, `rec_yds` (integers; NULL from BallDontLie is stored as 0).
- **`edge_nfl_snaps`**
  - Unique on `(season, week, team, player_norm)`, where `player_norm` uses the same normalization as name matching.
  - Columns: `snaps` (offense + defense + special teams), `loaded_at`.
- **`edge_nfl_games`**
  - `bdl_game_id` primary key, plus `season`, `week`, `game_date`, `home`, `away`, `status_state`, `stats_loaded_at`, `snaps_loaded_at`.
  - It records which games are complete, so ingest knows what still needs loading.

**New module `src/edge/nfl_players.py`:**
- **Reused from `src/edge/players.py`:** `normalize_name`, `match_player`, `et_date` and the rate calculation `_rate`. These are moved or shared without changing behaviour. NBA tests must pass untouched.
- **`BdlNflClient`:** `teams()`, `roster(team_id)` (follows `next_cursor`), `games(season, week)` and `stats(game_ids)`.
  - The same 1.05 s spacing and `BdlError` redaction as the NBA client. The Phase A worker's BallDontLie steps already handle 429 by retrying next tick.
- **`NFL_STAT_PARTS`:** market → columns. `player_rush_reception_yds` = `rush_yds + rec_yds`; every other market maps to one column.
- **`resolve_new_players(conn, bdl, quotes, event, now)`:** matches unseen `(player, team)` pairs from a props poll against both teams' rosters (exact → normalized → fuzzy ≥ 0.92, as for NBA). Unmatched names are stored as `unmatched` and listed on `/edge/status`.
- **`ingest_week(conn, bdl, season, week)`:** upserts `edge_nfl_games`, then loads stats for final games without `stats_loaded_at`.
- **`load_snaps(conn, season)`:** imports snap counts and upserts the weeks present.
- **`hit_rates(games, market, line, opponent, is_home, season)`:** splits for last 5, last 10, season, vs. opponent and home/away. A game counts if the player has a row. There is no minutes filter.

**nflverse vs BallDontLie abbreviations:** these differ on two teams (checked 2026-10-16): nflverse uses `LA` and `WAS` where BallDontLie uses `LAR` and `WSH`. `load_snaps` maps nflverse to BallDontLie (`NFLVERSE_TO_BDL = {"LA": "LAR", "WAS": "WSH"}`) before storing, so snap matching compares like with like. A unit test asserts that the 32 nflverse teams map onto the 32 BallDontLie abbreviations.

**Team names:** the Odds API uses full names ("Atlanta Falcons") and BallDontLie uses abbreviations. `edge/teams.py` gains `NFL_ABBR_BY_NAME`, built from `bdl_nfl_teams.json` and checked against the 24 Odds API NFL event names in `nfl_events.json`.

**Config:** `NFL.has_players` becomes `True`. With that set, the worker already calls player resolution for NFL (Phase A §4); it now dispatches to `nfl_players.resolve_new_players` by sport.

## 5. Grading NFL props (sportsbook rules)

For each ungraded NFL prop opportunity whose game kicked off more than 3 hours ago:

1. **No matched player** (`bdl_player_id` NULL): stays ungraded, listed as unmatched on `/edge/status`.
2. **The player has a stat row for that game:** graded on `NFL_STAT_PARTS` against the line (win/loss/push), with profit at the opportunity's price.
3. **No stat row, but the player is in that game's snap counts** (snaps > 0, matched on normalized name and team): played with zero stats, so the stat is **0** and the prop is graded.
4. **No stat row, and the game's snap counts are loaded, and the player is not in them:** **void** (inactive).
5. **No stat row and snap counts for that game not loaded yet:** wait. After 7 days without snap counts, the opportunity is listed as `stale_ungraded` on `/edge/status` and is still never guessed.

**When stats are complete:** a game's stats count as complete once BallDontLie shows the game `final` and its stats have been loaded at least 2 hours after the final whistle, to allow for stat corrections.

**Scope of grading:**
- NFL game lines keep their Phase A grading from final scores.
- NBA prop grading is unchanged. It still runs only for `basketball_nba` events.
- Grading covers every existing NFL opportunity, so Week 6 and Week 7 props saved since Phase A launched get results.

## 6. Worker

The BallDontLie steps become per sport. The NBA steps are unchanged; the NFL steps are new:

| Step | When | What |
|---|---|---|
| `nfl_stats_step` | hourly | `ingest_week` for the current and previous NFL week; marks `stats_loaded_at` |
| `nfl_snaps_step` | every 6 h | `load_snaps(current season)`; marks `snaps_loaded_at` for games whose snaps are present |
| `nfl_backfill_step` | each tick until done | 2025 weeks 1–18 and 2026 weeks before the current one, a few games per tick, resuming from `backfill_nfl_next` |

- **BallDontLie request budget:** the backfill is about 600 requests (about 10 minutes at 60 per minute, spread across ticks); after that, about 20 requests per hour. A BallDontLie error never blocks odds polling: it lands in `bdl_error`, as for NBA.
- **nflverse errors** (HTTP or parse) land in `snaps_error` and never block anything else.

## 7. API and site

**API:**
- **`/edge/players/{bdl_id}?sport=nfl&market=…&line=…`** returns: name, team, position, opponent and home/away for the next game; hit rates; the last 20 games, each with `week`, `opponent` and the market's value; and current prices for that player and market.
- **NFL +EV rows** carry `player_id` once the player is matched; it stays null otherwise. The Phase A rule ("player links only for sports with players") now covers NFL.
- **`/edge/results?sport=nfl`** already filters by sport (Phase A). Phase B fills it with graded NFL props.

**Site:**
- **`/nfl/edge/player/:id`:** `EdgePlayer` with a `sport` prop. On NFL pages:
  - the minutes filter is hidden;
  - the game log shows week and opponent;
  - the market switcher lists the 10 NFL prop markets.
- **`/nfl/edge/results`:** `EdgeResults` with a `sport` prop.
- **NFL navigation:** Picks · +EV · **Track record** · Results. "Results" stays the model tracker; the Edge record is called "Track record", as on NBA.
- **Notes:** the Phase A note "NFL player pages and prop results are coming next week" is removed. Each NFL player page gets: "Stats: BallDontLie. Participation: nflverse snap counts."

## 8. Error handling

- **BallDontLie 429 or 5xx:** retried next tick; the backfill cursor stays put (as for NBA).
- **An NFL roster lookup fails during name resolution:** the name stays unseen and is retried at the next props poll. It is never stored as `unmatched` because of an error.
- **A player traded mid-season:** a new `(odds_name, team)` row is matched against the new team's roster. Past games keep the team they were played for.
- **nflverse not yet published for a game:** grading waits (§5, rule 5).

## 9. Testing

**Unit tests** (fixtures from §3):
- parse BallDontLie NFL games and stats into rows (NULL → 0; preseason excluded because stats are fetched by `/games` ids only);
- every one of the 10 markets in `NFL_STAT_PARTS`, including Rush+Rec;
- `NFL_ABBR_BY_NAME` covers every team in `nfl_events.json`;
- name matching against an NFL roster, including a same-name pair on different teams;
- `hit_rates` splits;
- each grading rule in §5 (graded, zero-but-played, void, wait, unmatched).

**DB integration tests** (throwaway Postgres):
- the new tables are created by `create_all` without touching NBA tables;
- the ingest upsert is idempotent;
- an NFL "Josh Allen" and an NBA "Josh Allen" grade independently;
- `/edge/players?sport=nfl` and `/edge/results?sport=nfl` on real SQL;
- worker steps resume after errors.

**Regression:** the full NBA suite passes unchanged.

**Frontend:** vitest for the NFL player routing and markets, the type check and the build.

**Live check (by Sat 2026-10-24):**
1. The backfill completes.
2. 10 graded NFL props from Week 6 or 7 match the box scores (and snap counts for any voids).
3. A player page's hit rates match a hand count from the game log.

## 10. Rollout

1. Merge to `main`; this deploys `nba-value`, `edge-worker` and the site. The new tables are created on startup.
2. Watch the backfill progress on `/edge/status` (`backfill_nfl_next`) and the first `nfl_stats_step`/`nfl_snaps_step` runs.
3. Grading runs on its normal 10-minute cycle and picks up the Week 6 props.
4. Run the live check (§9).
5. **Kill switch:** `EDGE_NFL_PLAYERS=false` on `edge-worker` (settings `edge_nfl_players`, default `true`) skips NFL player resolution, the NFL stats, snaps and backfill steps, and NFL prop grading. NFL odds, +EV and CLV keep working.
