# TruLine Edge for NFL, Phase A: odds collection, +EV finder and odds screen

- **Date:** 2026-10-16
- **Status:** approach and design approved in brainstorming; awaiting written-spec review
- **Launch target:** Sat 2026-10-18, before the Week 6 London game (Sun 2026-10-18 13:30 UTC, HOU @ JAX)
- **Builds on:** `2026-10-15-edge-ev-finder-design.md` (NBA Edge, live)

## 1. Purpose

Samir wants NFL rolled out before NBA. NFL is mid-season, so six weeks of games, lines and player stats already exist and results can be judged against real data. NBA's early weeks would mostly be noise.

NFL is delivered in three phases, each with its own spec and plan:
- **Phase A (this spec):** the worker collects NFL lines and props; the +EV finder and odds screen work for NFL; closing-line value (CLV) is tracked from the first poll.
- **Phase B:** NFL player pages and hit rates, prop win/loss grading, and the NFL track-record page.
- **Phase C:** a historical backtest of NFL 2023–2025 through the +EV engine.

The NBA worker keeps running unchanged. Phase A must not change NBA behaviour.

**Success criteria:**
1. By Saturday, the worker polls NFL lines and props with no manual steps, alongside NBA.
2. `/nfl/edge` lists NFL +EV rows. A spot check of 10 rows against the sportsbooks' own sites matches price and line.
3. NFL data never appears in NBA views (the +EV list, games, odds, results), and NBA data never appears in NFL views.
4. Every NFL opportunity gets a CLV once its game kicks off. NFL game lines (moneyline, spread, total) also get win/loss from final scores.
5. Combined Odds API use stays under the worker's credit cap (4.5M of 5M).

## 2. Scope

**In Phase A:**
- Multi-sport worker: per-sport config, per-sport poll state, one shared credit guard and lock.
- NFL game lines and 10 NFL prop markets.
- A `sport` parameter on the public Edge endpoints.
- `/nfl/edge` (+EV list) and `/nfl/edge/odds/:eventId` (odds screen) on the site, plus a **+EV** tab in the NFL navigation.
- CLV for all NFL opportunities; win/loss for NFL game lines.

**Not in Phase A:**
- NFL player pages, hit rates and player-name matching (Phase B).
- Win/loss for NFL props (Phase B). Their `result` stays empty until then.
- An NFL track-record page (Phase B). The NBA track record must exclude NFL rows (§6).
- The backtest (Phase C).
- Anytime-TD props. Most books quote only "Yes" (in the 2026-10-16 sample: 319 Yes against 55 No), so there is no two-sided market to remove the vig from.
- Changes to the NFL model tracker at `/nfl` or its scheduler.

## 3. Verified market data (2026-10-16)

From one Week 6 game (CHI @ ATL, event `7c312fdf…`), fetched 2026-10-16 for 22 credits. It is saved as `tests/fixtures/edge/nfl_props.json`; the first three events of the week are in `nfl_events.json`.

- **Sport key:** `americanfootball_nfl`. 24 events listed (Weeks 6–7).
- **Props were already posted on Friday** for Sunday's games.
- **Books per market:** 11–14 of the 16 configured books.
- **Sharp books on props:** Pinnacle (8 markets), Novig and ProphetX (all 10), so consensus pricing is anchored by sharps on almost every market.

**NFL prop markets (Phase A):**

| Key | Label |
|---|---|
| `player_pass_yds` | Pass Yds |
| `player_pass_tds` | Pass TDs |
| `player_pass_completions` | Completions |
| `player_pass_attempts` | Pass Att |
| `player_pass_interceptions` | INTs |
| `player_rush_yds` | Rush Yds |
| `player_rush_attempts` | Rush Att |
| `player_receptions` | Receptions |
| `player_reception_yds` | Rec Yds |
| `player_rush_reception_yds` | Rush+Rec Yds |

At 2 credits per market per call, a props poll costs about 20 credits per game.

## 4. Architecture

One worker loops over sports (approach 1 in brainstorming). Rejected alternatives: a second worker service per sport, which would contend for the advisory lock and the credit accounting; and a copy of Edge for NFL.

**`src/edge/config.py`** gains a per-sport table. Existing NBA constants stay as aliases, so current imports keep working.

```python
@dataclass(frozen=True)
class SportConfig:
    key: str                     # Odds API sport key
    slug: str                    # "nba" | "nfl": state-key prefix and API value
    prop_markets: tuple[str, ...]
    lines_window_hours: int      # poll game lines from this far out
    has_players: bool            # player matching + pages (NBA only in Phase A)

SPORTS = {"nba": SportConfig("basketball_nba", "nba", NBA_PROP_MARKETS, 36, True),
          "nfl": SportConfig("americanfootball_nfl", "nfl", NFL_PROP_MARKETS, 7 * 24, False)}
```

**Enabled sports:** `EDGE_SPORTS` (settings `edge_sports`, default `"nba,nfl"`) chooses which entries of `SPORTS` the worker polls. Unknown slugs are ignored with a warning. This is the kill switch in §11.

**`src/edge/client.py`:** `get_events`, `get_game_odds`, `get_event_odds` and `get_scores` take the sport (key and prop markets) instead of reading the module constants.

**`src/edge/worker.py`:** `tick` runs the existing per-sport body once per sport.
- Poll-state keys are prefixed with the sport slug for NFL (`nfl:last_events_refresh`, `nfl:last_scores_poll`). NBA keeps its current unprefixed key names, so the running NBA state carries over untouched.
- The credit guard, heartbeat, grading, pruning and BallDontLie steps stay global and run once per tick.
- An error in one sport's polling is recorded and does not stop the other sport's polling in that tick.
- Player resolution (`players.resolve_new_players`) runs only for sports with `has_players`.

**`src/edge/schedule.py`:** game lines poll from `lines_window_hours` before kickoff. Between 36 hours and the window edge, the line interval is 60 minutes; inside 36 hours, the existing cadence table applies. Props keep the existing within-36-hours cadence for both sports.

**`src/edge/store.py`:** `upcoming_events(conn, now, sport=None)` filters on `edge_events.sport`. `edge_events` already stores the sport, so no schema change is needed.

## 5. Grading

- **CLV:** `run_grading` already computes CLV from the frozen closing quotes, with nothing sport-specific. NFL opportunities get CLV unchanged.
- **NFL game lines:** win/loss from the Odds API NFL `/scores` endpoint, polled while NFL games are live (the same 30-minute rule as NBA, per sport).
- **NFL props:** CLV only in Phase A. Grading must not mark an NFL prop void or lost for lack of a box score: `result` stays NULL until Phase B. The current prop grading looks up NBA box scores, so it must skip non-NBA events.

## 6. API

All public Edge endpoints take `sport` (`nba` | `nfl`), defaulting to `nba` so the live site keeps working: `/edge/games`, `/edge/ev`, `/edge/odds/{event_id}`, `/edge/results` and `/edge/health`. Cache keys include the sport.

- **Filtering:** every view joins `edge_events` and filters on its sport key. `/edge/results?sport=nba` must exclude NFL opportunities. That is required for success criterion 3, even though NFL results have no page until Phase B.
- **Player links:** `player_id` is set only for sports with `has_players`. NFL rows return `player_id: null`.
- **Labels:** `MARKET_LABEL` gains the NFL labels from §3, so `describe()` produces, for example, "Bijan Robinson Over 74.5 Rush Yds".
- **Health:** `/edge/health?sport=nfl` reports staleness from the NFL poll state, so the "data may be stale" banner reflects NFL's own feed.
- **Unknown sport:** a `sport` value outside `SPORTS` returns 422.

## 7. Site

- **New routes:** `/nfl/edge` (+EV list) and `/nfl/edge/odds/:eventId` (odds screen). These are the existing `EdgeFinder` and `EdgeOdds` pages with a `sport` prop. Their API calls pass `sport`, and their links stay within the sport's routes.
- **NFL navigation:** Picks · **+EV** · Results. The NBA navigation is unchanged.
- **Market filter:** the market dropdown on `/nfl/edge` lists the NFL markets from §3 with their labels (`frontend/src/lib/edgeFormat.ts`).
- **Player names** on NFL rows are plain text, not links.
- **Copy:** a one-line note on `/nfl/edge`: "NFL player pages and prop results are coming next week."

## 8. Credits

| | Estimate |
|---|---|
| NFL props | ~20 credits/poll × ~210 polls/game × ~16 games/week ≈ 67K/week |
| NFL lines | 6 credits/call; hourly for 7 days, then the cadence table ≈ 2K/week |
| **NFL total** | **≈ 290K/month** |
| NBA (existing estimate) | ≈ 870K/month |
| **Combined** | **≈ 1.2M of the 4.5M cap** |

The shared credit guard covers both sports: if projected monthly use exceeds the cap, intervals stretch for both.

## 9. Error handling

- One sport's Odds API failure (other than an account-wide 401/429) does not block the other sport. It is recorded as `last_error` with the sport prefix.
- An account-wide 401/429 triggers the existing global backoff for both sports, since the key is shared.
- If NFL has no upcoming events (bye weeks, off-season), the NFL loop does nothing and costs only the free events call.

## 10. Testing

- **Unit tests:**
  - parse the NFL fixtures (all 10 markets, both sides, links);
  - `describe()` labels for NFL markets;
  - the per-sport config;
  - the schedule's 7-day line window;
  - the worker running both sports, where an NFL failure leaves the NBA polls intact;
  - API `sport` validation and cache keys.
- **DB integration tests** (throwaway Postgres): seed NBA and NFL events, quotes and opportunities; assert each sport's `/ev`, `/games`, `/odds` and `/results` contain only that sport; assert NFL props are never graded void or lost.
- **Regression:** the full NBA test suite passes unchanged.
- **Frontend:** vitest for the NFL labels and the sport-aware links; build and type check.
- **Live check (Sat 2026-10-18):** after deploy, confirm NFL quotes per book and market in the DB; spot-check 10 `/nfl/edge` rows against the books' sites; check credits per tick against §8; confirm `/edge/ev` (NBA) is unchanged.

## 11. Rollout

1. Merge to `main`; this deploys `nba-value`, `edge-worker` and the site.
2. Watch the first NFL tick in `edge-worker` logs and on `/edge/status`.
3. Run the live check (§10).
4. If NFL polling misbehaves, the kill switch is removing `nfl` from an `EDGE_SPORTS` env var (default `nba,nfl`) on `edge-worker`. NBA continues.
