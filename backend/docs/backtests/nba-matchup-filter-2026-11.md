# NBA matchup filter validation (Blind Plays gate)

Generated 2026-11-17, NBA season 2026-27. **Verdict: PASS**

## Pre-registered rule

Pass iff favorable mean CLV >= rest mean CLV - 0.0025, AND the favorable 95% CI lower bound > 0, AND at least 200 favorable bets. Fixed before looking at results.

## Method

- Bets: graded NBA player-prop +EV opportunities with a closing line, one per (event, market, player, side, line), earliest sighting kept.
- Favorable is decided with ranks and usage from games before each bet's game date (no look-ahead).
- Bets that cannot be classified (no position, no prior games, unranked opponent) count as not favorable.
- CI95 is mean +/- 1.96 standard errors across bets. Win rate and units are reported, not gated.

## Results

| Group | Bets | Mean CLV | CLV 95% CI | Win rate | Units | ROI |
|---|---|---|---|---|---|---|
| Favorable | 1269 | 1.34% | [1.18%, 1.50%] | 53.80% | +129.79 | 10.23% |
| Rest | 13347 | 1.03% | [0.98%, 1.08%] | 48.90% | -2.26 | -0.02% |
| All | 14616 | 1.06% | [1.01%, 1.10%] | 49.30% | +127.53 | 0.87% |

Favorable minus rest: 0.31%

## Data quality

- Opportunity rows: 47946; unique bets: 14616.
- Position coverage: 93.22% of player-minutes in the season to date (games before the last bet date) belong to a player with a non-NULL position.
- Why bets were not favorable: no prior games 1133, not favorable 12173, opponent unranked 41

## Why it passed

- All three conditions held.
