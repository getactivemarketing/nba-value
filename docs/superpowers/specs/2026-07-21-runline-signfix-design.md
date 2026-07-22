# Runline Sign-Pairing Fix — Design

**Date:** 2026-07-21
**Status:** Approved

## Problem

The MLB scorer's runline block computes a single `p_away_cover = P(away +1.5)` and pairs it
with `away_odds` from every book row — but on rows stored as `line = +1.5` (home +1.5 / away
−1.5), `away_odds` is the away **−1.5** (plus-money) price. The `MAX_RUNLINE_ODDS = 2.5` guard
was meant to filter these but favorites' −1.5 is often priced 2.0–2.5, so it slips through.

Result: away-favorite runline picks are stored **labeled "+1.5"** (`best_*_line = abs() = 1.5`)
but carry the **−1.5 plus-money odds**, scored with the easy +1.5 win probability and graded as
+1.5. This is a phantom: the tracker credits the +1.5 win condition at −1.5 payouts. ~82% of live
best_bet units since the 2026-07-08 baseline (+16.3u of +19.8u) are this artifact; the season
backtest (+160u, +127u runline) is inflated the same way. Structurally the model can also only
ever represent away-team +1.5 or home-team −1.5 — it cannot express a home favorite's +1.5 or an
away favorite's −1.5.

Runline is currently PAUSED out of best_bet (`57c45d9`, `runline_in_best_bet=False`). This design
is the correctness fix that must land and re-validate before un-pausing.

## Decisions (confirmed)

1. **Drop `MAX_RUNLINE_ODDS`** — it was a band-aid; the value gate + edge threshold decide.
2. **History immutable** — do NOT rewrite the ~415 bug-affected snapshots; produce a separate
   corrected report instead. Snapshots are the true record of what was shown/texted.
3. **Full re-validation** — re-measure the real runline edge before deciding to un-pause.

## Fix 1 — Scorer runline block (`backend/src/services/mlb/scorer.py`)

Compute two independent cover probabilities from `predicted_run_diff` (positive = home favored):

```
p_home_minus = _run_diff_to_cover_prob(rd, 1.5)    # P(home wins by 2+)  -> home -1.5
p_away_minus = _run_diff_to_cover_prob(-rd, 1.5)   # P(away wins by 2+)  -> away -1.5
p_home_plus  = 1 - p_away_minus                     # P(home not lose by 2+) -> home +1.5
p_away_plus  = 1 - p_home_minus                     # P(away not lose by 2+) -> away +1.5
```

(`margin` is integer so ±1.5 never pushes; the +1.5 sides are exact complements of the opposite
−1.5 side.)

For each runline market row, read the home line sign (`market.line`, always ±1.5):

- `line == -1.5`: home side = home −1.5 (`p_home_minus`, `home_odds`, line **−1.5**);
  away side = away +1.5 (`p_away_plus`, `away_odds`, line **+1.5**).
- `line == +1.5`: home side = home +1.5 (`p_home_plus`, `home_odds`, line **+1.5**);
  away side = away −1.5 (`p_away_minus`, `away_odds`, line **−1.5**).

Each side → `calculate_value(market_type="runline", bet_type=..., model_prob=<its prob>,
market_prob=<devigged from that row's two odds>, odds_decimal=<its odds>, team=<its team>,
line=<its SIGNED line>)`. Collect all sides across rows; `best_rl = find_best_value(rl_values)`.

- **Remove** the `MAX_RUNLINE_ODDS` filter.
- The chosen `MLBValueResult.line` carries the **signed** line (−1.5 or +1.5) and `.team` the bet
  team. This flows unchanged into `best_rl_line`/`best_bet_line`; grading
  (`adjusted = bet_score + line; win if adjusted > opp_score`) is already sign-correct, and the SMS
  (`pick_alerts.py`) and API/frontend formatters already render `+`/`−` from the signed value.
- `bet_type` values: keep `home_rl`/`away_rl` (team side); the line sign distinguishes ±1.5. Devig
  uses each row's `home_odds`/`away_odds` pair as today.

No change to grading, storage columns, SMS, or config. `runline_in_best_bet` stays False.

## Fix 2 — Corrected report + re-validation (`backend/scripts/runline_revalidation.py`, new)

Read-only. For every graded MLB snapshot (`best_bet_profit`/`actual_winner` not null, has
`predicted_run_diff`, `home_score`, `away_score`):

1. Re-derive the four cover probs from frozen `predicted_run_diff` (same formulas as Fix 1).
2. Pull that game's historical `mlb_markets` runline rows; for each side, use its real price.
3. Apply the **fixed** selection (best value across the four sides, gate thresholds from
   `MLBValueCalculator`) to pick the corrected runline bet.
4. Grade it against the actual margin (`away_score - home_score`) with the sign-correct rule and
   real odds; sum units at flat $100.
5. Report: corrected runline record (W-L, WR, units) vs the tracker's inflated runline figure,
   split by window (since 2026-07-08 baseline, and full snapshot history), plus how often the
   corrected pick differs from what was actually stored. State the date range covered; if snapshots
   don't reach the April season start, say so — do not fabricate the full-season number.

Snapshots are NOT mutated. Output to stdout (like `mlb_retune_tracker.py`).

## Un-pause

Out of scope for this change. Decide separately after reading Fix 2's corrected edge. If a real
edge survives, flip `runline_in_best_bet=True` (config default or env) in a follow-up.

## Testing

- Scorer: unit tests asserting correct side/prob/odds/signed-line pairing for both `line=-1.5` and
  `line=+1.5` rows; a favorite gets its +1.5 side with minus-money odds (no phantom), and a
  plus-money −1.5 lay is evaluated at its true (low) cover prob → not a phantom win. Regression:
  the old bug's input (favorite, line=+1.5 row) no longer yields a high-scoring +1.5-labeled pick
  at −1.5 odds.
- Re-validation script: a small fixture-style unit test on the per-game re-derivation + grading
  helper (pure function), independent of the DB.
- Full unit suite green.

## Out of scope

- Un-pausing runline (separate follow-up gated on Fix 2 results).
- Rewriting/regrading historical snapshots.
- Any NBA/NFL runline logic.
