# TruLine Edge: +EV finder, odds screen and player pages (NBA)

- **Date:** 2026-10-15
- **Status:** approved in brainstorming, awaiting written-spec review
- **Launch target:** NBA opener, Tue 2026-10-20 (first tip 19:00 UTC, BOS @ DET)

## 1. Purpose

TruLine's own models have not beaten the market:
- MLB best_bet failed its closing-line test at t = −6.64.
- NBA 2025-26 picks came from a broken feature contract: spread v2 predicted a constant −9.86 MOV.

Edge takes the opposite approach. It treats the sharp market consensus as the fair price and finds sportsbooks whose posted price beats it. This is the PropsEdge / OddsJam category of product.

**Business model:** paid subscriptions. v1 launches as a free beta. Stripe and accounts follow after the opener (see §10).

**Success criteria:**
1. Every NBA slate is ingested, priced and displayed with no manual steps.
2. Every flagged +EV opportunity is logged when it is first found, and graded against the closing line (CLV) and against the result.
3. The `/edge/results` page publishes that record honestly, including when it is bad.
4. Monthly Odds API usage stays under the configured cap (default 3.5M of 5M credits).

## 2. Scope

**In v1 (live by the opener):**
- Ingest worker.
- EV engine and opportunity log.
- Player game logs and hit rates.
- `/api/v1/edge/*` endpoints.
- The pages `/edge`, `/edge/odds/:game`, `/edge/player/:id` and `/edge/results`.
- Prerequisite fixes (§3).

**Out of v1:**
- Paywall and Stripe.
- Arbitrage and middles.
- Alerts.
- Alternate lines.
- DFS EV math.
- Comparing prices across different lines.
- NFL and MLB edge.
- Fixing or retraining the legacy NBA model.

## 3. Prerequisites (day 1, before the key is set on Railway)

1. **Key leak.** `GET /api/v1/nfl/debug/odds` returns the raw httpx error string, which includes `apiKey=<full key>`. It also returns a key prefix.
   - Remove the key from all error output (scrub `apiKey=` query values).
   - Drop the prefix field.
   - Put the debug endpoint behind admin auth.
   - Audit the other `debug` and `health` endpoints for the same pattern.
   - Deploy. Only then set the new `ODDS_API_KEY` on Railway.
2. **Legacy NBA picks become hidden tracking-only.**
   - New setting `NBA_PICKS_PUBLIC` (default `false`).
   - When it is false, NBA best_bet fields are nulled in public API responses and hidden in the UI.
   - Snapshots and grading keep running, so the record stays intact.
3. **Season rollover.** Replace the hardcoded `season=2025` and `date(2025, 10, 22)`:
   - `tasks/scheduler.py:86,95,358,937,1079,1093`
   - `services/injuries.py:269`
   - `tasks/ingestion.py`, `api/admin.py`, `services/scoring/prop_scorer.py`. The implementation plan pins the exact lines from a grep for `2025`.

   Use a `current_nba_season(today)` helper. The NBA season label is the year it starts in; a new season starts on Oct 1. Add a season filter to `get_team_quality` and to the L10 ATS query.

## 4. Architecture

A new package, `backend/src/edge/`. It runs on its own Railway service so it does not depend on the legacy scheduler.

```
src/edge/
  config.py        books, markets, cadence, thresholds, credit cap
  client.py        Odds API calls (events, event odds, game odds); key never logged
  ingest.py        normalize responses -> upsert edge_quotes, append edge_quote_history
  engine.py        pure functions: devig, consensus fair price, EV, CLV
  opportunities.py detect/log +EV, closing-line + result grading
  players.py       BallDontLie game logs, name matching, hit rates
  worker.py        scheduling loop, credit guard, heartbeat
  api.py           FastAPI router mounted at /api/v1/edge
```

- **Railway:** a new service, `edge-worker`, in project `nba`. Same repo and image as the web service, start command `python -m src.edge.worker`.
- **Web service:** keeps serving the API, so it mounts `edge/api.py`.
- **Migrations:** Alembic, run by the existing web start command.

## 5. Ingest and storage

**Books** go in a single `bookmakers=` parameter. 17 books is 2 region-equivalents, so each market costs 2 credits per call.

| Role | Books |
|---|---|
| Sharp | `pinnacle`, `novig`, `prophetx` |
| Retail | `draftkings`, `fanduel`, `betmgm`, `williamhill_us` (Caesars), `espnbet`, `fanatics`, `betrivers`, `hardrockbet`, `betonlineag`, `bovada`, `lowvig` |
| DFS (line only) | `prizepicks`, `underdog` |

**Markets:**
- Game lines (`/v4/sports/basketball_nba/odds`): `h2h`, `spreads`, `totals`.
- Props (`/v4/sports/basketball_nba/events/{id}/odds`): `player_points`, `player_rebounds`, `player_assists`, `player_threes`, `player_points_rebounds_assists`, `player_points_rebounds`, `player_points_assists`, `player_rebounds_assists`.
- Every call passes `includeLinks=true`, `oddsFormat=american`.

**Polling cadence.** Pregame only; polling for a game stops at its tip time.

| Time before tip | Game lines | Props (per event) |
|---|---|---|
| > 6 h (up to 36 h) | 10 min | 15 min |
| 1 – 6 h | 3 min | 5 min |
| < 1 h | 2 min | 2 min |

- **Event list:** refreshed every 60 min from `/events`, which is free.

**Credit guard:**
- Read `x-requests-remaining` and `x-requests-used` from every response, and store them in `edge_worker_state`.
- Project the month's use at the current burn rate. If the projection is above `EDGE_CREDIT_CAP` (default 3,500,000), double all intervals and log a warning. If it is still above, double again.
- The current factor shows on `/edge/status`.

**Tables:**
- **`edge_events`**
  - `event_id` (Odds API id, PK), `sport`, `home_team`, `away_team`, `commence_time`
  - `status` (`scheduled` | `live` | `final`), `home_score`, `away_score`
  - `last_lines_poll`, `last_props_poll`
- **`edge_quotes`**: the latest price per unique key (`event_id`, `market`, `player_name` — empty string for game lines, `side`, `line`, `book`).
  - `side` is over/under or home/away.
  - Also stores `price_american`, `price_decimal`, `link`, `updated_at`.
- **`edge_quote_history`**: append-only. A row is written only when the price or line for a key changes. It records the same key plus price, link and `observed_at`.
- **`edge_players`**
  - `odds_name` (PK), `bdl_player_id`, `team`, `match_method` (`exact` | `normalized` | `fuzzy` | `manual` | `unmatched`), `confidence`.
  - A fuzzy match is only accepted when its score is ≥ 0.92 and the team agrees. Otherwise the name stays `unmatched`.
- **`edge_player_games`**
  - `bdl_player_id`, `game_date`, `team`, `opponent`, `is_home`, `minutes`
  - `pts`, `reb`, `ast`, `fg3m`, `stl`, `blk`, `tov`
  - Unique on (player, date).
- **`edge_opportunities`**: see §6.4.
- **`edge_worker_state`**: heartbeat, credits, guard factor, last error (with the key scrubbed).

**Staleness:** a quote is stale when `now − updated_at` is more than 2× its current polling interval. Stale quotes are never priced as +EV and are shown greyed out.

## 6. EV engine (`engine.py`, pure functions)

### 6.1 Devig
- For each two-sided market at one book, use the power method: find k so that p₁ᵏ + p₂ᵏ = 1, where p is the implied probability from American odds.
- The method name is stored with each opportunity.
- Single-sided quotes (only one side posted) are not devigged and are not used in consensus.

### 6.2 Fair price
For a target (event, market, player, side, line, book):
- **Inputs:** every *other* fresh book quoting both sides at the **same line** (leave-one-out).
- **Weights:** Pinnacle 3.0, Novig 2.0, ProphetX 2.0, DraftKings 1.5, FanDuel 1.5, all others 1.0. DFS books are never inputs.
- **`fair_prob`** = the weighted mean of the devigged probabilities.
- **Minimum market:** at least 3 input books, and at least one of them sharp or an exchange. Otherwise the result is `thin` and has no EV.

### 6.3 EV
- `ev = fair_prob × decimal_price − 1`.
- **+EV:** `ev ≥ EDGE_MIN_EV` (default 0.02).
- **"Verify" flag:** `ev > EDGE_VERIFY_EV` (default 0.15). The row is still shown, but labeled.

### 6.4 Opportunity log and grading
**Logging:**
- After each pricing pass, any (event, market, player, side, line, book, price) that is +EV and not yet logged is inserted into `edge_opportunities` with `first_seen_at`, `fair_prob`, `ev`, `n_books` and `devig_method`.
- While it stays +EV, `max_ev` and `last_seen_at` are updated.
- A new price at the same book and line counts as a new opportunity.

**Closing line:**
- At tip, take the last pre-tip quotes in `edge_quote_history` for that market and line.
- Compute `closing_fair_prob` with the same consensus rule. This time the bet's own book is included, because only the bet's price is under test.
- `clv = closing_fair_prob − 1/decimal_price`.
- If the close is thin, `clv` is null and the reason is recorded.

**Result:**
- Game lines are graded from `edge_events` scores. Final scores come from the Odds API `/scores` endpoint, which costs 2 credits per call, polled every 30 min while games are live.
- Props are graded from `edge_player_games` (BallDontLie).
- Each opportunity gets `result` (win / loss / push / void) and `profit_units` at 1 unit staked.
- A player who doesn't play is `void`.

**Validation:** the CLV formula must give the same result as `services/mlb/clv.py` on the same inputs. A test covers this.

## 7. Player data and hit rates (`players.py`)

- **Nightly (09:00 UTC):** fetch BallDontLie box scores for the previous day into `edge_player_games`.
- **One-time backfill:** all of 2025-26, plus 2026-27 as it accrues.
- **Name matching** runs when a new `player_name` appears in props, in this order: exact, then normalized (accents, suffixes Jr/III, punctuation), then fuzzy with team check.
  - Unmatched names are listed on `/edge/status`. Their prices still show, but without hit rates.
- **Hit rates** for (player, market, line), computed in SQL when the page loads:
  - L5, L10, L20 and season, plus vs this opponent and home/away.
  - Optional minimum minutes.
  - Combo stats are summed from the components.
  - Push (stat equals the line) counts as neither a hit nor a miss.

## 8. API (`/api/v1/edge`, read-only, 20–30 s cache)

| Endpoint | Returns |
|---|---|
| `GET /games?sport=nba` | Slate: event, teams, tip, number of prop markets, last poll times |
| `GET /ev?market=&book=&min_ev=&event_id=` | +EV rows: bet description, book, line, price, fair odds, ev, n_books, updated_at, link, verify, stale |
| `GET /odds/{event_id}?market=` | Grid of outcome × book prices, best price per outcome, fair price per outcome |
| `GET /players/{bdl_id}?market=&line=&min_minutes=` | Hit rates, last 20 game log, current prices by book |
| `GET /results?days=&market=&book=` | n, mean CLV with 95% CI, share beating the close, win rate, ROI, broken down by market and book |
| `GET /status` (admin) | Heartbeat age, credits remaining, projected monthly use, guard factor, last error (scrubbed), unmatched names |

**Paywall-ready:** every list endpoint takes its row limits from an `access_tier` dependency. In v1 that dependency always returns `full`.

**Errors:** the API never includes `ODDS_API_KEY` or any upstream URL in a response.

## 9. Frontend

The pages use the existing React 18, React Query and Tailwind stack and the TruLine brand. The navigation adds an **Edge** section.

1. **`/edge`: the +EV table.**
   - Columns: EV %, bet, book and price, fair odds, number of books, age, **Bet** button (deep link).
   - Filters: market, book ("my books" saved in localStorage, wrapped in try/catch), minimum EV, game.
   - Verify and stale rows are styled distinctly.
   - It refetches every 30 s.
2. **`/edge/odds/:game`: the odds screen.**
   - Rows are outcomes and columns are books. The best price is highlighted, and there is a fair price column.
   - Tabs for game lines and each prop market. DFS lines appear as columns showing the line only.
3. **`/edge/player/:id`: the player page.**
   - A last-20 bar chart against the line, with hits and misses colored.
   - Hit-rate cards and a minutes filter.
   - A table of current prices by book.
   - The old `/props` route redirects to `/edge`.
4. **`/edge/results`:** CLV summary with CI, a CLV-over-time chart, a breakdown by market and book, and a plain explanation of what CLV means.
5. **Homepage (`/`):** shows the NBA slate and the top 10 +EV rows instead of the legacy best bets. MLB and NFL pages are unchanged.
6. **Banners and footer:**
   - "Odds delayed" banner when the worker heartbeat is more than 10 min old or the last poll errored.
   - Footer on every edge page: 21+, "for research only, not a guarantee", 1-800-GAMBLER.

## 10. After the opener (separate specs)

1. CLV report after the first 100 opportunities.
2. Accounts, Stripe and tiers.
3. Arbitrage and middles.
4. Alerts.
5. Comparing prices across different lines.
6. Alternate lines.
7. NFL edge.
8. The legacy NBA model: fix the feature contract, retrain without leakage, and pass a pre-registered gate before any public picks.

## 11. Error handling

| Failure | Behavior |
|---|---|
| Odds API 401 | Stop polling, set `last_error`, show the frontend banner, retry every 15 min |
| Odds API 429 / 5xx | Exponential backoff (30 s → 10 min); other events continue |
| Credit projection over cap | Guard factor ×2, up to ×8; warning in logs and on `/edge/status` |
| Malformed response | Skip that event and log it; never write partial rows for a market |
| BallDontLie failure | Hit rates come from existing data; grading retries the next night |
| Worker crash | Railway restarts it; heartbeat age shows on the frontend banner |

## 12. Testing

- **`engine.py` unit tests:**
  - power devig against hand-computed cases;
  - leave-one-out;
  - weights;
  - thin-market rule;
  - stale exclusion;
  - EV threshold and verify flag;
  - CLV matching `mlb/clv.py`.
- **Ingest tests:** saved Odds API JSON fixtures parse into the expected rows; history is appended only on change.
- **Player tests:** name matching cases (accents, Jr., duplicates across teams); hit-rate math, including combos and pushes.
- **API tests:** response shapes; no key in any error response.
- **Live dry run (Mon 2026-10-19):** against opening-day lines as books post them. Check quote counts per book and market, credits per cycle against the estimate, and a manual spot check of 10 EV rows against the books' websites.

## 13. Credit budget estimate

- **Props:** about 8 markets × 2 credits ≈ 16 credits per event poll. Over a game day that's roughly 16 × (4 polls/h × 30 h + 12 polls/h × 5 h + 30 polls/h × 1 h) ≈ 16 × 210 ≈ 3.4k credits per game.
- **NBA season pace:** about 7.5 games a day, so ≈ 25k credits a day for props.
- **Game lines:** 6 credits per call for the whole slate, ≈ 3k a day.
- **Scores:** ≈ 0.5k a day.
- **Total:** ≈ 29k a day, ≈ 870k a month. That is well under the 3.5M cap and the 5M plan.
