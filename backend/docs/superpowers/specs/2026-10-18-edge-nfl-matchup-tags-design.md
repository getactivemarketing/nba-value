# TruLine Edge for NFL: matchup tags and anytime TD

- **Date:** 2026-10-18
- **Status:** approach and all four design sections approved in brainstorming; awaiting written-spec review
- **Builds on:** Phase A (lines, props, +EV, CLV), Phase B (NFL players, prop grading), Phase C (historical backtest, `docs/backtests/nfl-2023-2025.md`), and the hold filter (`MAX_HOLD`, main 23d5ba1)

## 1. Purpose

Samir wants injury news and matchup context (for example "Montgomery is out, so Marks gets the carries", "Jacksonville gives up yards to running backs", red-zone usage) to inform the NFL board. He also wants the anytime TD market added.

Edge stays a market-consensus product. The sharp books already price public matchup and injury information, so our own adjustment to the fair price would mostly double-count it. Instead:

- **Tags, not a model.** Matchup, usage and injury facts are **tags** on +EV rows the engine already finds. They never change the fair price or the EV.
- **Tested before trusted.** Each tag is graded on the 2023–25 backtest. Only **proven** tags affect ranking, and only by the CLV advantage the backtest measured.
- **Anytime TD becomes a market**, live and in the backtest.

**Success criteria:**
1. The backtest report has a "Matchup tags" table: each tag's count, CLV advantage with its margin of error, beat-close rate, ROI, direction per season, and verdict.
2. The live NFL board and player pages show tag chips: proven chips highlighted, neutral ones grey, negative ones as a caution. The default sort is EV plus the advantage of the row's proven tags.
3. Anytime TD rows appear on the live board, priced by rule (A) in §5, graded and tracked with CLV. They are also backtested on 2023–25.

## 2. Scope

**In:** the seven tags in §3; live and historical injury status; the anytime TD market (live, grading, backtest fetch and replay); tag verdicts; ranking; frontend chips.

**Out (v1):**
- opponent defensive injuries (for example a top cornerback out);
- weather;
- Vegas team totals;
- NBA tags;
- any change to the fair price or EV from tags;
- tuning tag cutoffs on backtest results.

## 3. Tags

All tags are **point-in-time**: a tag for a game in week *w* uses only games before week *w*. Each tag has a **direction**: tailwind for over/Yes and headwind for under/No, except tag 7, which is a caution on over/Yes. A row shows a tag only when the tag applies to its player's position and its market.

Ranks are among 32 teams. **"Top 8"** means the 8 teams allowing the most.

| # | Key | Rule | Markets |
|---|---|---|---|
| 1 | `def_soft` / `def_tough` | Opponent's yards allowed per game to the player's position over its last 8 games: top 8 is soft, bottom 8 is tough | RB: rush yds, rush attempts. WR/TE: rec yds, receptions. QB: pass yds, completions, attempts |
| 2 | `def_rz` | Opponent ranks top 8 in red-zone carries plus targets allowed to the position, over its last 8 games | Anytime TD; RB rush yds; WR/TE rec yds |
| 3 | `def_td` | Opponent ranks top 8 in TDs (rushing plus receiving) allowed to the position, over its last 8 games | Anytime TD |
| 4 | `usage_up` | Player's share over his last 3 games is at least 5 percentage points above his earlier share that season (at least 2 earlier games). RB: carry share. WR/TE: target share | Over on the position's stat markets; Yes on anytime TD |
| 5 | `rz_role` | Player has at least 25% of his team's red-zone carries plus targets over his last 4 games | Anytime TD |
| 6 | `teammate_out` | A same-position teammate is listed Out or Doubtful for this game, and had at least 30% of the position group's share (carries for RB, targets for WR/TE) over the last 4 games | Over and Yes for the other players at that position |
| 7 | `questionable` | The player is listed Questionable for this game | Caution on over and Yes |

**Definitions:**
- **Red zone:** plays with `yardline_100 <= 20`.
- **Position:** the nflverse roster position for that season. FB counts as RB. Players outside QB/RB/WR/TE get no matchup tags.
- **Yards allowed by position:** RB rush yds + RB rec yds; WR rec yds; TE rec yds; QB pass yds.
- **Window filling:** if a team or player has fewer than 4 games in the current season, the window is filled from the end of last season. With no earlier data, the tag does not fire.
- **Fixed cutoffs:** the numbers above are fixed by this spec. They are never tuned against backtest results.

## 4. Data

| Data | Backtest source | Live source |
|---|---|---|
| Play-by-play (red zone, yards allowed by position) | nflverse `pbp` parquet per season | the same, refreshed daily |
| Weekly player stats (carries, targets, TDs incl. `special_teams_tds`) | nflverse `player_stats` (already loaded in Phase C) | the same, refreshed daily |
| Rosters (position) | nflverse `rosters` per season | the same, refreshed daily |
| Injury status | nflverse `injuries` per season (final pre-game `report_status`) | BallDontLie NFL injuries if our tier serves them (verified in the plan's first task). Otherwise nflverse's in-season `injuries` file, refreshed |

Player identity uses the existing name-key matching (`normalize_name`). An unmatched name gets no player tags. It is never a guessed match.

## 5. Anytime TD

- **Live polling:** add `player_anytime_td` to `NFL_PROP_MARKETS`. Measured cost: 2 credits per event per poll, taking a props poll from 20 to 22 credits.
- **Parsing:** a "Yes" outcome becomes side `over`, line 0.5. A "No" becomes `under`, line 0.5. With several outcomes for one book, player and side (exchange ladders), keep the best price.
- **Fair price, rule (A):**
  - For `player_anytime_td` only, the consensus inputs are Pinnacle, Novig and ProphetX, and only where they quote both Yes and No.
  - At least **2** such books are needed. `MAX_HOLD` still applies.
  - Retail books (Yes only) are priced as bets and never feed the consensus.
  - This is a per-market override (`MARKET_RULES`). Every other market keeps `MIN_BOOKS = 3` with a sharp book required.
- **Grading:**
  - **Win:** rushing TDs + receiving TDs + return TDs ≥ 1.
  - **Loss:** the player played and scored none.
  - **Void:** under the existing Phase B rule for players who did not play.
  - Passing TDs never count.
  - Live grading uses BallDontLie box-score fields; the backtest uses nflverse `rushing_tds + receiving_tds + special_teams_tds`.
- **Display:** the board shows "Anytime TD — Yes", not "over 0.5".

## 6. Backtest additions

- **Fetch:** `python -m src.edge.backtest fetch --kind td`.
  - It stores snapshots with `kind = 'td'`, with the same labels, slates, resume and stub rules as `props`.
  - Already-stored `props` snapshots are never re-bought.
  - Cap: `BT_CREDIT_CAP` 80,000 for this run. The floor is unchanged.
  - Order: a one-week pilot (2023 week 10) first to measure the per-call cost, then the full run **only after Samir confirms the spend** (estimate about 65K credits).
- **Replay:**
  - Reads `props`, `lines` and `td` snapshots, and attaches tags to each pick.
  - Picks gain a JSONB `tags` column: `[{key, direction}]`, computed from data before the pick's week.
  - The `td` picks follow the same flag, CLV and grade path as other props.
- **Grading tags:** in `report`, on distinct picks (first flag).
  - For each tag and direction, compare tagged picks with untagged picks in the same set of markets the tag applies to.
  - **Main measure:** average CLV difference and beat-close rate. ROI and win % are reported too.
  - **Proven:** all three of the following.
    1. At least 200 tagged picks.
    2. The CLV difference is above 0 with a Bonferroni-corrected confidence interval across all tag-and-direction tests (with 14 tests at α = 0.05, z ≈ 2.91).
    3. The difference has the same sign in at least 2 of the 3 seasons.
  - **Negative:** the same three conditions, with the difference below 0.
  - **Neutral:** everything else.
  - `report` writes each verdict and its CLV advantage to `edge_tag_status` (`tag, direction, verdict, clv_lift, n, run_id, updated_at`) and to the "Matchup tags" table in the report.

## 7. Live architecture

- **`src/edge/matchups.py`** (pure, no I/O):
  - builders that turn nflverse frames into per-team-week and per-player-week aggregates;
  - `tags_for(...)` → `list[Tag]`.
  - The live code and the backtest both use these functions.
- **`src/edge/nfl_injuries.py`:** loads injury status into `edge_nfl_injuries (season, week, team, player_key, status, updated_at)`.
- **Tables:**
  - `edge_nfl_team_week` and `edge_nfl_player_week` (aggregates);
  - `edge_nfl_injuries`;
  - `edge_tag_status`.
  - All are created on startup in the existing lock-safe migration style.
- **Worker steps:**
  - `nfl_matchups_step` runs every 24 h.
  - `nfl_injuries_step` runs every 2 h from Thursday to Monday ET and every 6 h otherwise.
  - Both follow the `nfl_snaps_step` state and error pattern: `last_*` and `*_error` keys, and an error never blocks the tick.
- **API:**
  - `/edge/ev?sport=nfl` and the NFL player page add `tags: [{key, label, direction, verdict, detail}]` to each row.
  - `detail` is the human stat, for example "JAX allows 118 RB rush yds/game (3rd most)".
  - Default sort key: `ev + Σ clv_lift of the row's proven tailwind tags − Σ clv_lift of its proven headwind tags`. With no `edge_tag_status` rows, every verdict is neutral and the sort is by EV alone.
- **Frontend:**
  - Chips in `EvTable.tsx` and `EdgePlayer.tsx`: proven = highlighted, neutral = grey, negative = caution.
  - Hovering a chip shows `detail` and the verdict ("Proven in 2023–25 backtest: +0.6% CLV").
  - The results page gains a per-tag CLV breakdown of live picks.

## 8. Failure handling

- **No tags never means no board.** If nflverse or BallDontLie fails, tags go stale or are omitted, and the EV rows still serve.
- `/edge/status` reports `matchups_error`, `injuries_error` and the age of the last refresh. Tags older than 8 days are not shown.
- **Missing injury data:** tags 6 and 7 do not fire.
- **Unmatched player names:** no player tags.

## 9. Testing

- **Unit tests:**
  - every tag rule on hand-built frames, on both sides of each cutoff;
  - point-in-time: week *w* data never reaches a week-*w* tag;
  - window filling across seasons;
  - position mapping;
  - TD parsing (Yes/No, ladders keep the best price);
  - the 2-of-3 sharp rule and `MAX_HOLD` on TD;
  - TD grading: rushing, receiving and return TDs count, passing TDs don't, voids;
  - the verdict logic: the minimum count, Bonferroni z, the season-sign rule, negative and neutral;
  - the sort key;
  - the API row shape;
  - frontend chip rendering.
- **DB tests:** the new tables round-trip on the throwaway Postgres (`EDGE_TEST_DB_URL`); `edge_tag_status` read and write; the replay `tags` column.
- **Backtest pilot:** 10 TD picks and 10 tagged picks checked by hand against nflverse.

## 10. Rollout

1. Build on a branch, test-first, with a whole-branch review at the end.
2. TD fetch pilot (about 1–2K credits). **Samir confirms** the full TD spend, then the full fetch runs.
3. Replay and report with tags. Samir reviews the verdict table.
4. Merge and push **on Samir's go-ahead**. The worker starts polling anytime TD and refreshing tags, and the board shows chips.
5. Live check: TD rows priced against the exchanges; 10 spot-checks of tags against nflverse; credit pace still under 1.2M a month.
