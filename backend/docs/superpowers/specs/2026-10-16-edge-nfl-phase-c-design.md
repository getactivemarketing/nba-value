# TruLine Edge for NFL, Phase C: historical +EV backtest (2023–2025)

- **Date:** 2026-10-16
- **Status:** approach and design approved in brainstorming; awaiting written-spec review
- **Builds on:** `2026-10-16-edge-nfl-phase-a-design.md` (lines, props, +EV, CLV) and `2026-10-16-edge-nfl-phase-b-design.md` (prop grading rules)

## 1. Purpose

Samir chose **proof plus tuning**. The backtest has two jobs:
- **Proof:** answer "does flagging +EV with the live engine beat the close and make money?"
- **Tuning:** find which markets, books, EV thresholds and lead times before kickoff actually work. Those answers decide what subscribers see and how often the worker polls.

The results are internal. This phase adds no public site page.

**Success criteria:**
1. Raw historical snapshots are stored for every 2023–2025 regular-season game: 4 snapshots per game, 10 prop markets plus game lines.
2. The replay prices them with the live engine's functions, unchanged.
3. Every flagged pick has a CLV against the close, plus a result (win, loss, push or void) graded with the live rules.
4. `backend/docs/backtests/nfl-2023-2025.md` reports CLV, win rate, ROI and pick count by season, market, book, EV bucket and lead time, and ends with concrete recommendations.
5. Total spend stays under `BT_CREDIT_CAP` (default 800,000). Live polling is never throttled by the backtest.
6. No backtest row appears in any live table or live view.

## 2. Scope

**In Phase C:**
- A CLI, `python -m src.edge.backtest`, with three commands: `fetch`, `replay` and `report`.
- Two new tables: `edge_bt_snapshots` and `edge_bt_picks`.
- Grading from nflverse weekly player stats, snap counts and schedules.
- A one-week pilot, then the full run and the report.

**Not in Phase C:**
- Playoffs, which matches Phase B.
- Anytime-TD props. They have no two-sided market, as in Phase A.
- NBA.
- Site pages.
- Changes to the live engine's rules (`MIN_BOOKS`, `SHARP_BOOKS`, `BOOK_WEIGHTS`, devig, EV threshold). The report may *recommend* changes; making them is separate work.

## 3. Verified historical data (probe, 2026-10-16)

- **Events list:** `GET /v4/historical/sports/americanfootball_nfl/events?date=<iso>` costs **1 credit**.
  - It returns `{timestamp, previous_timestamp, next_timestamp, data}`.
  - Snapshots are taken every **5 minutes**.
  - 2023-11-12 returned 25 events.
- **Props:** `GET /v4/historical/sports/americanfootball_nfl/events/{id}/odds?date=<iso>&bookmakers=<16 books>&markets=<m>` costs **10 credits per market per region-equivalent**. Our 16-book list counts as 2 region-equivalents, so **20 credits per market**.
  - **Sample:** CLE @ BAL, 2023-11-12, `player_reception_yds`.
  - **Books:** 8 quoted it: betmgm, betonlineag, betrivers, bovada, draftkings, fanduel, **pinnacle** and williamhill_us.
  - **Sharps:** Novig and ProphetX are absent in 2023, so Pinnacle is the only sharp book.
  - **Players:** 12.
  - **Fixtures:** the samples are saved as `tests/fixtures/edge/bt_events_2023.json` and `bt_props_2023.json`.
- **Game lines:** `GET /v4/historical/sports/americanfootball_nfl/odds?date=<iso>&markets=h2h,spreads,totals&bookmakers=<16>` returns every listed game in one call, at 10 × 3 markets × 2 = 60 credits.

## 4. Architecture (approach A: fetch once, replay many times)

Rejected alternatives:
- **Writing history into the live tables behind a flag:** it pollutes the track record and makes live queries depend on a filter that's easy to forget.
- **Fetching and scoring in one pass with no raw store:** every re-tune would cost credits again.

**New package `src/edge/backtest/`:**

| Module | Responsibility |
|---|---|
| `plan.py` | Pure functions. Given the season's games, return the list of `(event, snapshot label, timestamp)` jobs and their credit cost. |
| `fetch.py` | Calls the historical endpoints, stores raw JSON in `edge_bt_snapshots`, and enforces the budget stops. |
| `nflverse.py` | Loads nflverse schedules, weekly player stats and snap counts into in-memory frames, keyed for grading. |
| `replay.py` | Turns stored snapshots into quotes (`ingest.parse_event`), prices them (`engine.price_quotes`), computes CLV (`engine.clv`), grades picks, and writes `edge_bt_picks`. |
| `report.py` | Aggregates `edge_bt_picks` and writes the markdown report. |
| `__main__.py` | CLI: `fetch [--season S] [--week W] [--dry-run]`, `replay [--season S]`, `report`. |

**Reused unchanged from live:**
- `ingest.parse_event`, `engine.price_quotes`, `engine.clv` and `engine.american_to_decimal`.
- `grading.grade_over_under`, `grading.grade_game` and `grading.profit_units`.
- `nfl_players.nfl_prop_result`, `nfl_players.stat_for` and `NFL_STAT_PARTS`.
- `players.normalize_name`.
- `config.NFL_PROP_MARKETS` and the 16-book list.

The backtest imports these functions and never forks them. If the live engine changes later, a re-run of `replay` uses the new rules at no credit cost.

## 5. Snapshots

**Games:** the nflverse schedule supplies each season's regular-season games with their kickoff times. Each game is matched to its Odds API event id through one historical events call per game day, at 1 credit each. Teams are matched with `teams.NFL_ABBR_BY_NAME` and the kickoff must fall within ±3 hours.

**Snapshot times per game:**

| Label | Requested `date` | Use |
|---|---|---|
| `t24h` | kickoff − 24 h | flag picks |
| `t6h` | kickoff − 6 h | flag picks |
| `t1h` | kickoff − 1 h | flag picks |
| `close` | kickoff − 5 min | closing fair probability for CLV; never flags picks |

- The API returns the latest snapshot at or before the requested `date`. We store the API's `timestamp` as `snapshot_at`.
- If `snapshot_at` for `close` comes after kickoff, the snapshot is discarded and the game has no CLV. This can't happen when the API behaves as documented, but the check guards against it.

**Props:** each props call requests all 10 `NFL_PROP_MARKETS` at once, so 200 credits per game per snapshot. Markets the API omits for that game are simply absent.

**Game lines:**
- Games sharing a kickoff time share a slate.
- Each slate gets one game-lines call per label, timed from its kickoff (60 credits each).
- The response is split per event and stored as that event's `lines` row.

## 6. Storage

```sql
CREATE TABLE IF NOT EXISTS edge_bt_snapshots (
  id            BIGSERIAL PRIMARY KEY,
  season        INT NOT NULL,
  week          INT NOT NULL,
  event_id      TEXT NOT NULL,          -- Odds API event id
  kind          TEXT NOT NULL,          -- 'props' | 'lines'
  label         TEXT NOT NULL,          -- 't24h' | 't6h' | 't1h' | 'close'
  kickoff       TIMESTAMPTZ NOT NULL,
  requested_at  TIMESTAMPTZ NOT NULL,
  snapshot_at   TIMESTAMPTZ,            -- API's timestamp
  home          TEXT NOT NULL,          -- nflverse abbreviation
  away          TEXT NOT NULL,
  credits       INT NOT NULL,           -- x-requests-last for this call
  payload       JSONB NOT NULL,         -- the event object, as returned
  fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (event_id, kind, label)
);

CREATE TABLE IF NOT EXISTS edge_bt_picks (
  id            BIGSERIAL PRIMARY KEY,
  run_id        TEXT NOT NULL,          -- replay run, e.g. '2026-10-17T14:02Z'
  season        INT NOT NULL,
  week          INT NOT NULL,
  event_id      TEXT NOT NULL,
  label         TEXT NOT NULL,          -- lead time the pick was flagged at
  market        TEXT NOT NULL,
  player        TEXT,                   -- NULL for game lines
  side          TEXT NOT NULL,
  line          NUMERIC,
  book          TEXT NOT NULL,
  price         INT NOT NULL,
  fair_prob     DOUBLE PRECISION NOT NULL,
  ev            DOUBLE PRECISION NOT NULL,
  n_books       INT NOT NULL,
  sharps        TEXT NOT NULL,          -- comma list of sharp books in the consensus
  closing_prob  DOUBLE PRECISION,
  clv           DOUBLE PRECISION,
  result        TEXT,                   -- win | loss | push | void | NULL (ungradable)
  profit        DOUBLE PRECISION,       -- units at 1u stake; NULL unless graded win/loss/push
  UNIQUE (run_id, event_id, label, market, player, side, line, book)
);
```

- The tables are created by the existing lock-safe startup migration path in `src/database.py`, so they exist in prod after a normal deploy. The CLI also runs the same `CREATE TABLE IF NOT EXISTS` before fetching.
- `replay` writes a new `run_id` each time and never deletes older runs. `report` reads the latest run unless it is given `--run`.
- **Size:** about 3,300 props rows plus about 3,300 lines rows, roughly 150–200 MB of JSONB after TOAST compression. The Postgres volume has about 4.4 GB free.

## 7. Replay

For each game and each flagging label (`t24h`, `t6h`, `t1h`):
1. **Quotes:** `parse_event(payload, observed_at=snapshot_at)` on the props and lines payloads.
2. **Price:** `price_quotes(quotes, settings.edge_min_ev, settings.edge_verify_ev)`. Keep the rows with `is_ev` set; those are the picks.
3. **Close:** price the `close` snapshot the same way and look up the same `(market, player, line)` fair probability with `exclude_book=None`.
   - CLV is `clv(closing_prob, price)`.
   - If the same line isn't quoted at the close (the line moved), `closing_prob` and `clv` are NULL. The report shows how often that happens.
4. **Grade:**
   - **Props:** `nfl_prop_result(pick, stat, snaps_loaded=True, played)`.
     - `stat` is the nflverse weekly row for the player, matched on `normalize_name(player)` against `normalize_name(player_display_name)` plus season, week and team in the game.
     - `played` is true if the snap counts list that name key for that game.
   - **Game lines:** `grade_game` with nflverse final scores.
   - **Profit:** `profit_units(result, price)`.
   - **Ambiguous names:** if two players in the same game share a name key, the pick is left ungradable (`result` NULL) and counted in the report.

**The `sharps` column** records which sharp books were in the consensus. The report uses it for the like-for-like "Pinnacle-only consensus" cut: the pick is re-priced with Novig and ProphetX excluded, so 2023 and 2025 can be compared fairly.

## 8. Grading data (nflverse)

- **Data needed:** schedules (games, kickoffs, final scores), weekly player stats, and snap counts for 2023, 2024 and 2025.
- **Library:** `nfl_data_py`, already a dependency from Phase B.
- **Known risk:** nflverse renamed its 2025 weekly stats release (`stats_player`), and `nfl_data_py.import_weekly_data` may not read it. The plan's first task checks that all three seasons load. If 2025 fails, `nflverse.py` reads the release file directly from the nflverse GitHub release URL instead.
- **Column mapping:** a mapping from nflverse columns to `STAT_COLS` (`passing_yards → pass_yds`, `carries → rush_att`, and so on) is defined once in `nflverse.py` and covered by a test.

## 9. Credits and safety

| | Calls | Credits |
|---|---|---|
| Events (one per game day) | ~150 | ~150 |
| Props (816 games × 4) | ~3,264 | ~653,000 |
| Game lines (~3 slates per week × 4) | ~650 | ~39,000 |
| **Total** | | **~692,000** |

- **The live guard is not involved.** `fetch` never calls `store.log_api_call` and never writes `credits_used` or `credits_remaining` to edge state, so the worker's `guard_factor` projection only sees live spend. The backtest's accounting is the `credits` column in `edge_bt_snapshots`.
- **Hard stops:** before each call, `fetch` checks two limits and stops cleanly with a message if either is reached. Everything already stored is kept, and a re-run resumes from there.
  - Stored credits plus the next call's cost must stay within `BT_CREDIT_CAP` (env, default 800,000).
  - The last `x-requests-remaining` must stay at or above `BT_MIN_REMAINING` (env, default 3,600,000), so live keeps its share.
- **Errors:** 401 or 429 stops the run. Any other error is logged and that job is skipped. A re-run retries it.
- **`--dry-run`:** prints the job count and the exact planned credit cost, spends nothing, and fetches nothing.
- **API key:** read from `ODDS_API_KEY`. The run uses the worker's key, through `railway run --service edge-worker` or exported locally. The key is never logged; the process-wide redaction from `src/utils/redact.py` stays on.
- **Pacing:** at most 2 requests per second.

## 10. Running it

- The CLI runs from Samir's Mac against the production database, using the Postgres public URL; the TCP proxy still exists.
- **Pilot first:** `fetch --season 2023 --week 10` (about 16 games, about 13K credits), then `replay` and a hand spot-check:
  - 10 picks: price, line and book against the raw payload.
  - 5 grades against nflverse box scores.
- **Full run:** `fetch` for all three seasons (about 1–2 hours, resumable), then `replay` and `report`.
- **Delivery:** the report is committed to `backend/docs/backtests/` and shown to Samir.

## 11. Report

`backend/docs/backtests/nfl-2023-2025.md`, generated by `report`.

**Header:**
- Run id, snapshot count, credits spent, and pick count.
- Ungradable and no-close counts, with their reasons.

**Measures** (repeated in each table):
- Picks, win%, ROI (units won ÷ picks staked, voids excluded), average EV, average CLV, and the share of picks that beat the close (CLV > 0).

**Breakdowns:**
- Overall, by season, by market, by book, by EV bucket (2–4%, 4–6%, 6%+), and by lead time (`t24h`, `t6h`, `t1h`).
- Season × lead time.
- "Pinnacle-only consensus" by season.

**Sample sizes:**
- Rows with fewer than 50 picks are marked "small sample".
- Each ROI gets a 95% interval (normal approximation on per-pick profit).

**Recommendations:** short bullets, derived from the numbers by fixed rules so they aren't hand-picked:
- **Markets and books to show:** those with average CLV > 0 and at least 200 picks.
- **EV threshold:** the lowest bucket with average CLV > 0.
- **Lead time:** the one with the best average CLV.

## 12. Testing

**Unit tests:**
- `plan`: snapshot timestamps, slate grouping, and the credit-cost arithmetic.
- `fetch` (stubbed HTTP):
  - resume skips stored `(event_id, kind, label)`;
  - the cap stop and the minimum-remaining stop;
  - 401/429 stops;
  - nothing is written to edge state or the live call log.
- `replay` over the 2023 fixture:
  - picks equal `price_quotes` output;
  - CLV is NULL when the closing line moved;
  - ambiguous names are ungradable;
  - void, zero-stat and stat-row grades.
- `nflverse` column mapping.
- `report`: aggregation, buckets, the small-sample flag, and the recommendation rules on a synthetic pick set.

**DB integration tests** (throwaway Postgres): both tables are created; a fetch-then-replay round trip; the live tables `edge_quotes`, `edge_opportunities` and `edge_state` are unchanged after a run.

**Regression:** the full backend unit suite passes unchanged.

**Live:** the pilot spot-check (§10).

## 13. Rollout

1. Merge to `main`, with Samir's go-ahead. Startup migrations create the two tables. No service behaviour changes.
2. Run the pilot and review it with Samir.
3. Run the full fetch, replay and report.
4. Present the findings. Changes to live rules or polling based on them are a follow-up decision.
