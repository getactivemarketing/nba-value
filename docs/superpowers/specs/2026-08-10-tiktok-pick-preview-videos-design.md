# TikTok Pick-Preview Videos — Design

**Date:** 2026-08-10
**Status:** Approved, ready for implementation planning

## Problem

TruLine has a working automated video pipeline (`truline-videos/`) that renders
8-second post-game *celebration* clips and auto-posts them to TikTok and
Instagram via Blotato. It has been dormant since 2026-04-16.

Celebration-only content is structurally cherry-picking: it publishes wins and
is silent on losses. Against a moneyline record that is +1.06 SE from zero
lifetime and 9-16 over the last 28 measured picks, that is both indefensible in
a betting-TikTok comment section and not the most interesting thing TruLine has.

This spec covers a second, *pre-game* format: a 30-45s narrated pick preview
that argues against itself before making its case.

## Approved decisions

| Decision | Choice | Rationale |
|---|---|---|
| Footage | Generic unbranded stock b-roll, low opacity | League/Getty footage is editorial-only, unlicensable for betting promo, and Content-ID muted. AI footage needs synthetic-media disclosure and reads as scam signal. |
| Format | Pre-game pick preview, case-against-then-turn | Argues against itself first; reads as analysis, not a pitch. Keeps distance from TikTok's gambling-promotion line. |
| Narration | Voiceover, per-beat clips | 30-45s of silent text underperforms. Per-beat clips make desync structurally impossible. |
| Script source | Deterministic template | An LLM inventing a stat published under a model's name with odds attached is a real liability. |
| Publishing | Live auto-post | User decision, 2026-08-10. |
| Markets | Moneyline only | Runline is paused (no real edge), totals suppressed (52.1% over 125). Publishing them would advertise markets we've disabled. |

## Reference script

```
Model likes White Sox ML at +155 (underdog). They're 5-5 in their last 10,
on a 2-game losing streak. Castillo (5.06 ERA) has held opponents scoreless
in the 1st in 10 of 17 starts.

48% model vs 39% market — 9.3% edge.
```

## Data

All fields verified available against prod on 2026-08-10 — 1,726 final games,
`2026-03-25` to date, 100% coverage on first-inning runs and starter IDs.

| Field | Source | Status |
|---|---|---|
| Team, market, odds | `mlb_prediction_snapshots.best_ml_team`, `best_ml_odds` | stored |
| Last-10 record | `mlb_team_stats.last_10_record` | stored |
| Win/loss streak | `mlb_games` | **derive** |
| Starter name + ERA | `mlb_pitcher_stats.era`, `mlb_games.home/away_starter_id` | stored |
| First-inning scoreless split | `mlb_games` | **derive** |
| Model probability | `mlb_prediction_snapshots.winner_probability` | stored |
| Market probability | breakeven from `best_ml_odds` | computed |

### Derivation 1 — current streak

From `mlb_games` where `status = 'final'`, per team, ordered by `game_date`
descending: count consecutive games with the same result until the result
flips. Returns direction (`won`/`lost`) and length.

### Derivation 2 — starter first-inning scoreless split

A starter's "opponents scoreless in the 1st" is the *opposing* side's
first-inning runs. Union both halves:

- home starts: `home_starter_id = X` → count `away_first_inning_runs = 0`
- away starts: `away_starter_id = X` → count `home_first_inning_runs = 0`

Yields `scoreless / total_starts`. Verified returning real values (one starter
is 13-of-15 on home starts alone before the union).

### Point-in-time correctness

Both derivations filter `game_date < target_game_date` strictly. Consistent
with `docs/engineering/02-feature-engineering-playbook.md`.

### Missing data degrades, never fabricates

If the probable starter is unset, or a stat's denominator is empty, **drop that
beat** and let the video run shorter. Never substitute a league average or a
placeholder. Same rule as CLV: absent is not neutral.

## Narration and on-screen copy

Deterministic template, filled from the values above. Six beats, each its own
audio clip and its own on-screen graphic.

| # | Beat | Content |
|---|---|---|
| 1 | Hook | The contrarian line — the strongest reason *not* to bet |
| 2 | Pick | Team logo, ML, odds, underdog badge |
| 3 | Case against | Last-10, streak, starter ERA |
| 4 | The turn | The first-inning split; counter animates |
| 5 | Numbers | Model projection vs breakeven |
| 6 | Close | `truline.app`, disclaimer |

Beat 4 is the retention anchor — the point the video stops being a pick and
becomes an argument.

### Required copy guardrail

The reference script's `9.3% edge` **must not ship as written.**

`winner_probability` is a model output, not a measured result. Across the 28
picks with measured closing lines the market moved toward our side by +0.49
points on average — real (25/28 positive, t = +3.43) but roughly one
nineteenth of the claimed edge, and realized CLV remains negative.

On-screen and in narration this is labelled **"model projection"**, never
"edge". The market figure is the raw breakeven implied by the offered price
(39.2% at +155), which is the conservative choice — the devigged fair price is
nearer 38% and would make the gap look larger.

Beat 6 carries `Not betting advice. 21+.`

## Architecture

```
mlb_scheduler writes snapshot
  → select picks where best_bet_type = 'moneyline' and not yet posted
  → derivations + template
  → TTS: 6 clips
  → measure clips (getAudioDurationInSeconds)
  → Remotion render (duration from calculateMetadata)
  → Blotato upload → TikTok + Instagram
```

### Components

**`backend` — pick payload endpoint.** Assembles one JSON payload per
publishable pick: pick, derivations, template-rendered beat text. Owns both SQL
derivations. The video project stays a pure renderer with no database access,
matching how `render-celebrations.ts` already consumes
`/mlb/evaluation/underdogs`.

**`truline-videos/src/tts/` — provider abstraction.** One interface,
`synthesize(text, outPath) -> void`. Adapters: `elevenlabs`, `openai`, `say`
(offline dev fallback, never published). Selected by env var. No key present →
`say`, and the orchestrator refuses to upload a `say`-narrated render.

**`truline-videos/src/compositions/PickPreview.tsx`** — new composition
alongside `ModelHit`, which is untouched. Consumes an array of beats each
carrying its audio path and measured duration. `calculateMetadata` sums them
for total duration. Stock b-roll loops underneath at low opacity; existing
`musicFile` support retained, ducked under narration.

**`truline-videos/scripts/render-pick-previews.ts`** — orchestrator, modelled
on `render-celebrations.ts`. Reuses the Blotato upload path and `rendered.json`
dedupe verbatim.

### Timing

Fires after snapshot generation. A pick is skipped unless first pitch is **at
least 45 minutes away** at upload time, so a pick is never published after the
game has started. Blotato's `useNextFreeSlot` scheduling means the gate is
measured against upload time, not render time, and the two can differ by
minutes.

Selection is restricted to `best_bet_type = 'moneyline'`, which by construction
excludes the paused runline and suppressed totals markets.

## Testing

- Both SQL derivations against known prod rows, including the point-in-time
  boundary (a game on the target date must not appear in its own inputs)
- Template rendering with each field absent in turn — asserts beats drop rather
  than emitting placeholders
- Assert the string `edge` never appears in generated narration or overlay copy
- TTS adapter selection, including the refusal to publish a `say` render
- `calculateMetadata` duration equals the sum of beat durations
- Orchestrator dedupe and the lead-time gate

## Out of scope

- Word-level karaoke captions. Per-beat graphics are already synced to
  narration by construction; revisit only if retention data demands it.
- CLV-explainer format. Genuinely the more differentiated angle and still worth
  building, but it is a separate content line, not this pipeline.
- Reviving or altering the dormant celebration flow.
- NFL. Every NFL market, alert and public recommendation stays disabled.

## Known consequence

Publishing pre-game picks creates a permanent public timestamped record. That
is the strongest credibility asset available — a receipt no capper account can
fake — and it also means a bad stretch stays visible. At 9-16 over the last 28,
the early public record should be expected to look rough before CLV accumulates
enough sample to defend it.
