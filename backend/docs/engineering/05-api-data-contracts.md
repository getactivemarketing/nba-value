# API & Data Contract Specification

**Purpose:** what every component publishes and consumes, and which guarantees hold.
**Last verified against code:** 2026-07-31

---

## 1. Contract layers

TruLine has five, only one of which is currently enforced by tooling:

| Layer | Contract | Enforced by |
|---|---|---|
| External APIs → ingest | Third-party response shapes | Nothing — breaks surface as missing data |
| Ingest → database | Table schemas | SQLAlchemy models |
| Database → scorer | Column semantics + nullability | Convention |
| Model artifact ↔ serving code | `feature_cols` ordering | **Nothing — see §6** |
| API → frontend | Pydantic response models | Pydantic (the one enforced layer) |

Every serious incident in this repo has been a break at an *unenforced* layer.

---

## 2. HTTP API

Base prefix `settings.api_v1_prefix` = `/api/v1`. `/health` is unprefixed.

### Health
```
GET /health -> {"status": "healthy", "timestamp": iso8601,
                "checks": {"database": "ok"}}
```
**Gap:** exposes nothing about model state. Should publish `run_diff_model_version`, `trained_at`, and whether any fallback is engaged (`03-model-governance.md` §3).

### MLB — `src/api/mlb.py`
```
GET /api/v1/mlb/games                  -> MLBGamesResponse
GET /api/v1/mlb/games/{game_id}        -> MLBGameResponse
GET /api/v1/mlb/picks/top              -> TopPicksResponse
GET /api/v1/mlb/pitchers/{name}        -> PitcherInfo
GET /api/v1/mlb/evaluation/daily       -> list[DailyPerformance]
GET /api/v1/mlb/evaluation/summary     -> EvaluationSummary
GET /api/v1/mlb/stats/first-inning
GET /api/v1/mlb/debug/*                -> odds, ingest, social-config, tweet-test, trigger-posts
```

### NFL — `src/api/nfl.py`
`/api/v1/nfl/*` including `/nfl/evaluation/summary`. **Live and publicly served, but the scheduler is disabled**, so nothing new is generated. `/nfl/evaluation/summary` currently reports a meaningless 100% off 2 leftover dry-run rows until the season starts.

### NBA and shared
`markets`, `bets`, `evaluation`, `trends`, `backtest`, `admin`.

### Conventions
- Response models live in `src/schemas/`; **every endpoint declares `response_model`.**
- Debug routes are unauthenticated — treat as internal, never link from the UI.
- CORS allow-list in `config.py` (localhost:3000, localhost:5173, `*.vercel.app`).

### Frontend contract
`frontend/vercel.json` rewrites `/api/:path*` → `https://nba-value-production.up.railway.app/api/:path*`. `VITE_API_URL=/api/v1` in production, so the browser never talks to Railway directly.

**The frontend holds no betting logic.** It renders what the API returns. A display rule implemented only in the frontend is invisible to backtests and must not exist.

---

## 3. Database contracts — durability classes

The single most important distinction in the system. Confusing these has caused every serious data incident.

| Class | Tables | Guarantee |
|---|---|---|
| **IMMUTABLE** | `mlb_prediction_snapshots`, `nfl_prediction_snapshots`, `prediction_snapshots` | Append-only. Never updated except to record outcomes. **The only valid source for grading and P&L.** |
| **MUTABLE-CURRENT** | `mlb_markets`, `nfl_market`, `markets` | Overwritten in place. Current state only. **No history. Never grade from these.** |
| **MUTABLE-UPSERT** | `mlb_predictions`, `mlb_team_stats`, `mlb_pitcher_stats` | Latest run wins |
| **MUTABLE-LIFECYCLE** | `mlb_games`, `mlb_game_context` | Updated as status/scores arrive |

### Snapshot contract (`mlb_prediction_snapshots`)

One row per `game_id`. Written at T-60min by `snapshot_predictions_async`. Frozen fields:

```
identity     game_id (unique), game_date, game_time, home_team, away_team
context      venue_name, park_factor, temperature, is_dome, starters + ERA
model        predicted_run_diff, predicted_total, winner_probability, winner_confidence
per-market   best_ml_*, best_rl_*, best_total_*   (team, line, odds, value_score, edge)
decision     best_bet_type, best_bet_team, best_bet_line, best_bet_odds,
             best_bet_value_score, best_bet_edge
outcome      actual_winner, home_score, away_score, *_result, *_profit
flags        sms_alert_sent, celebration_tweet_posted
```

**Invariants:**

1. `best_bet_*` is graded **from its own frozen columns**, never re-derived from `mlb_markets` and never copied from component fields at grade time.
2. `best_bet_value_score` stores the **display** score (tanh-compressed, 50% market-regressed) — *not* `gate_score`. Consumers filtering on it (e.g. `PICK_ALERT_MIN_SCORE = 40`) are filtering on a different scale from the qualification gate. Currently non-binding: the minimum display score across all 947 picks is 42, and 0 picks have ever been silently skipped. **Fragile by construction — do not tighten either threshold without re-checking the other.**
3. `best_bet_profit` is stored in **units × 100** (a $100 stake: `+140`, `−100`), while `best_bet_edge` is a raw probability delta (`0.140`). Two different scales in adjacent columns.
4. Runline rows before `RUNLINE_SIGN_FIX_DATE` (2026-07-22) are **permanently ungradeable** and guarded by `is_legacy_runline()`.

**Incident this encodes:** `best_ml` was assigned *inside* the per-book loop, so a book with no qualifying price wiped out a good pick from an earlier book — leaving `best_ml` NULL while `best_bet` was still a moneyline. 109 snapshots became permanently ungradeable. Fixed 2026-07-28 by selecting each market's best **after** the loop.

---

## 4. Internal service contracts

### `MLBScorer.score_game(game) -> MLBGamePrediction`
Requires: game row; markets in `mlb_markets`; team/pitcher stats. Missing markets → prediction with no value results (**not** an error). Guarantees `predicted_run_diff`, `p_home_win`, and populated `best_*` where markets qualified.

### `MLBValueCalculator.calculate_value(...) -> MLBValueResult`
```
raw_edge    = model_prob - market_prob
edge_pct    = raw_edge / market_prob * 100
gate_score  = edge_pct * 4.0 * conf_mult * market_mult  (+5 bonus), clamp 0-100
sort_score  = edge_pct * conf_mult * market_mult        (unclamped)
value_score = 100 * tanh(edge_pct * 0.5 / 20) * mults   (+5 bonus), clamp 0-100
is_value    = gate_score >= 55 AND raw_edge >= 0.10 AND edge_pct <= 80
```

**Three scores with three jobs — the most confusable contract in the codebase:**

| Score | Used for | Regressed to market? |
|---|---|---|
| `gate_score` | qualification only | **No** |
| `sort_score` | selection ranking only | No |
| `value_score` | **display only** | Yes, 50% |

The number shown to users is the conservative one; the number that decides whether to bet is not. `model_confidence` always defaults to `0.5`, so `conf_mult` is permanently `1.0` — the confidence term is inert. `market_mult`: moneyline 0.95, total 0.90, runline 1.00.

In practice `raw_edge >= 0.10` is the binding constraint; for a dog priced at 40% it demands `edge_pct >= 25`, well past what `gate_score >= 55` requires. **The 55 threshold almost never decides anything.**

### `find_best_value` / `find_best_shadow` / `find_best_bet`
- `find_best_value` — qualifiers only, `max(sort_score)`, else `None`
- `find_best_shadow` — `max(sort_score)` **ignoring the gate**; feeds shadow records that would otherwise starve (`best_total` collected ~0.7/day when gate-filtered)
- `find_best_bet` — across markets, honouring `include_totals` / `include_runline`

**`None` from `find_best_value` means "nothing qualified", not "no odds available."** Any diagnostic conflating those two is wrong.

### `send_sms(body) -> bool`
Textbelt. Returns falsey on failure; caller must not mark `sms_alert_sent`. Per-row commit isolation is required — a delivered SMS whose flag rolls back re-texts on the next cycle.

---

## 5. External API contracts

| Provider | Consumed by | Failure mode |
|---|---|---|
| The Odds API | `services/data/odds_api.py`, `nfl/odds_client.py` | Missing odds → no candidates → **silent "no picks"** |
| MLB Stats API | `services/mlb/mlb_api.py` | Stale stats served without error |
| BallDontLie | `services/data/balldontlie.py` | NBA scores/schedule |
| nflverse | `services/nfl/nfl_data.py` | Backfill only |
| Weather API | `services/mlb/weather_api.py` | Writes `wind_speed`/`wind_direction`; **`wind_factor` never computed** |
| Textbelt | `notifications/sms.py` | Prepaid credits, silent stop at zero |

**Universal failure mode: every external dependency degrades into "no picks tonight,"** which is indistinguishable from a genuinely quiet slate. That ambiguity is a real operational gap — with only moneyline live, silence is now the expected output on many nights, so a broken feed hides in plain sight.

**Timezone contract:** all `game_time` values are stored UTC, timezone-aware. External feeds publish ET. NFL Phase 4 shipped a bug where ET was tagged as UTC, dropping every primetime game's odds (183/225 matched). Fixed via `ZoneInfo("America/New_York").astimezone(timezone.utc)`. **Never construct a UTC timestamp by relabeling an ET one.**

---

## 6. The unenforced contract — model artifacts

```python
{"model": ..., "feature_cols": [...], "metrics": {...},
 "trained_at": ..., "version": ...}
```

`MLBScorer._build_model_feature_vector()` builds **28 values in hardcoded positional order** and never consults `feature_cols`. `retrain_mlb_v2.py` trains on `V2_FEATURE_NAMES` (up to 32). Deploying a v2 artifact against the current builder yields a shape error at best, silently misaligned features at worst.

**Required fix — build by name, validate at load:**

```python
missing = [c for c in artifact["feature_cols"] if c not in lookup]
if missing:
    raise FeatureContractError(missing)
```

See `02-feature-engineering-playbook.md` §2. This must land in the same change as any retrain deployment.

---

## 7. Contract change protocol

| Change | Requirement |
|---|---|
| Add API field | Additive only; frontend must tolerate unknown fields |
| Remove/rename API field | Deprecate one release first; grep the frontend |
| Add snapshot column | Nullable, with a migration; old rows stay valid |
| **Change a snapshot column's meaning** | **Prohibited.** Add a new column. Historical rows carry the old semantics |
| Change model `feature_cols` | New artifact version + serving-code update in the same commit |
| Change a scoring constant | Governed change — `03-model-governance.md` §6 |
| Change durability class of a table | Full design review; this is what makes grading trustworthy |

The prohibition on redefining snapshot columns is the load-bearing rule: those rows are the only record of what was actually bet, and re-interpreting them retroactively invalidates every historical result at once.
