# Social Post Redesign — Cards, Copy, and Stats Fix

## Problem

TruLine's social media posts have three issues:
1. Team stats (W-L records, division rank, ATS, O/U) don't show on card images — `get_team_card_stats()` returns empty dicts on Railway
2. Card images are too small/generic — text unreadable at Twitter's 50% preview scale, flat visual design
3. Tweet copy is formulaic and shallow — stat dumps with no narrative, robotic tone

## Goals

- Fix stats pipeline so cards always show team data
- Redesign PIL cards to be readable and visually distinctive on social feeds
- Rewrite tweet copy with data-backed analysis and natural voice
- Priority: daily picks + NRFI pick cards first, then final recaps + results

---

## 1. Fix Team Stats on Cards

### Root Cause Investigation

`get_team_card_stats()` in `content.py:112-171` does two queries:
1. `MLBTeamStats` by `team_abbr` — gets W-L, L10, ATS, O/U
2. `MLBTeam` by `team_abbr` — gets league + division for rank calculation

If either query returns None, parts of the stats silently disappear. The division rank calculation loops through all teams in the same division and compares `win_pct`, which requires both tables populated.

### Fix

- Add `structlog` debug logging to `get_team_card_stats()` showing what was found/missing
- Make the function resilient: if `MLBTeam` lookup fails, still return W-L/L10/ATS/O-U from `MLBTeamStats`
- Add a `last_10_wins`/`last_10_losses` fallback if `last_10_record` string is None (build it from the int columns)
- Log warnings when stats are empty so we can see this in Railway logs

### Files Changed
- `backend/src/services/social/content.py` — `get_team_card_stats()`

---

## 2. Card Image Redesign

### Design Principles
- **Readable at 50% scale** — minimum font 42px, headlines 180px+
- **Square aspect ratio** — 1200x1200, Twitter shows square cards larger than landscape
- **Team color accents** — primary color bars/gradients per team
- **Visual hierarchy** — clear sections with spacing, not wall of text
- **Fewer data points, bigger** — only show what matters for the card type

### Team Colors

Add a `TEAM_COLORS` dict mapping team abbreviations to `(primary_rgb, secondary_rgb)`. Examples:
```python
TEAM_COLORS = {
    "NYY": ((0, 48, 135), (255, 255, 255)),      # Navy / White
    "LAD": ((0, 90, 156), (239, 62, 66)),         # Blue / Red
    "BOS": ((189, 48, 57), (12, 35, 64)),         # Red / Navy
    "HOU": ((0, 45, 98), (235, 110, 31)),         # Navy / Orange
    # ... all 30 teams
}
```

### Card Layout: NRFI Pick (1200x1200)

```
┌──────────────────────────────────────┐
│ [Team color gradient bar - 8px]      │
│                                      │
│  NRFI PICK              7:05 PM ET  │  (42px, accent/muted)
│                                      │
│  [Away Logo]    vs    [Home Logo]    │  (220px logos)
│   YANKEES              DODGERS       │  (72px, white)
│   12-5 · 1st AL East  14-3 · 1st NL │  (36px, muted)
│                                      │
│ ┌────────────────────────────────┐   │
│ │           72%                  │   │  (180px, tier color)
│ │       STRONG NRFI              │   │  (48px, tier color)
│ └────────────────────────────────┘   │
│                                      │
│  STARTING PITCHERS                   │  (36px, dim)
│  ┌──────────────────────────────┐   │
│  │  Cole          Yamamoto      │   │  (56px, white)
│  │  3.12 ERA      2.89 ERA      │   │  (42px, muted)
│  └──────────────────────────────┘   │
│                                      │
│  L10  7-3 / 8-2    ATS  10-7 / 12-5│  (36px inline stats)
│                                      │
│  truline.app            @trulineapp  │  (30px, accent)
│ [Team color gradient bar - 8px]      │
└──────────────────────────────────────┘
```

Key changes from current:
- 1200x1200 (was 1200x900)
- NRFI % at 180px (was 140px)
- Team names at 72px (was 64px)
- Logos at 220px (was 180px)
- Stats bar simplified to single-line format instead of 2-column grid
- Team color gradient bars at top and bottom edges
- Background: radial gradient from center (dark-dark to slightly lighter) instead of grid

### Card Layout: Final Score (1200x1200)

```
┌──────────────────────────────────────┐
│ [Winner team color gradient - 8px]   │
│                                      │
│  FINAL                               │  (42px, accent)
│                                      │
│  [Away Logo]    —    [Home Logo]     │  (220px logos)
│   YANKEES              DODGERS       │  (72px, white)
│   12-5                 14-3          │  (36px, muted)
│                                      │
│         4  —  7                       │  (200px, white)
│                                      │
│    1st Inning: 0-1 (YRFI)           │  (48px, amber/green)
│                                      │
│  ┌────────────────────────────────┐  │
│  │  Our Pick: LAD ML — W          │  │  (48px, green/red)
│  └────────────────────────────────┘  │
│                                      │
│  L10  7-3 / 8-2    ATS  10-7 / 12-5│  (36px inline)
│                                      │
│  truline.app            @trulineapp  │
│ [Winner team color gradient - 8px]   │
└──────────────────────────────────────┘
```

### Card Layout: 1st Inning Recap (1200x1200)

```
┌──────────────────────────────────────┐
│ [Color bar]                          │
│                                      │
│  1ST INNING RECAP                    │  (42px, accent)
│                                      │
│  [Away Logo]    vs    [Home Logo]    │  (220px)
│   YANKEES              DODGERS       │  (72px)
│                                      │
│         0  —  1                       │  (200px)
│       1ST INNING                     │  (42px, muted)
│                                      │
│         YRFI                          │  (120px, amber)
│                                      │
│   Model predicted: 72% NRFI          │  (42px, muted)
│                                      │
│  truline.app            @trulineapp  │
│ [Color bar]                          │
└──────────────────────────────────────┘
```

### Visual Effects

- **Background gradient**: Replace grid with radial gradient — center slightly lighter (#0F1525) fading to edges (#0A0E17)
- **Team color bars**: 8px horizontal bars at very top and bottom of card, using the away team's primary color on left half, home team's on right half
- **Card surfaces**: Rounded-corner rectangles with 1px border, slightly lighter fill (#161D2E)
- **Glow effect**: Subtle color glow behind the main number (NRFI % or score) using the tier color at low opacity

### Implementation Notes

- All cards change from 1200x900 to 1200x1200
- New helper: `_draw_gradient_bg()` for radial background
- New helper: `_draw_team_bars()` for color accent bars
- New helper: `_draw_rounded_rect()` for card surfaces
- New dict: `TEAM_COLORS` with all 30 MLB + 30 NBA teams
- Font size minimums: nothing below 36px (was 30px)

### Files Changed
- `backend/src/services/social/image_generator.py` — full redesign of all card generators

---

## 3. Tweet Copy Redesign

### Principles (from Humanizer)
- No robotic headers ("Today's value plays from the model")
- No emoji-prefixed lists
- Varied sentence rhythm — mix short punchy lines with context
- Opinions and edge — "The model loves X here" not "Model: X (58%)"
- Specific data woven into narrative, not dumped as labels
- No sycophantic filler, no "let's dive in"

### Daily Picks Thread — Header Tweet

**Before:**
```
MLB Picks 04/13

Today's value plays from the model.
15 games analyzed.

Full card: truline.app

#MLB #SportsBetting #GamblingX
```

**After:**
```
MLB picks for Sunday, 15 games on the board.

3 value plays cleared the model today — two underdogs and a divisional favorite.

truline.app

#MLB #GamblingX
```

Changes: Natural date format, preview of what's in the thread (how many picks, what kind), drop "value plays from the model" robot-speak. Dynamic summary based on actual picks (underdog count, favorite count).

### Daily Picks Thread — Per-Game Tweet

**Before:**
```
Red Sox @ Yankees  7:05 PM ET

Model: Yankees ML (58%)
Market: Yankees +120 (55%)
Edge: +3.1%

Rodon 3.12 vs Severino 3.45

#RedSox #MLB
```

**After:**
```
Red Sox @ Yankees — 7:05 PM

Model likes the Yankees ML here at +120. They're 8-2 in their last 10 and Severino (3.45 ERA) has held opponents scoreless in the 1st in 5 of his last 7 starts.

58% model vs 55% market — 3.1% edge.

#Yankees #MLB
```

Changes: Narrative lead with the pick and why, weave in L10 record and pitcher context, stat line as a clean closer. Requires querying additional context (L10, pitcher 1st-inning stats).

### Pregame NRFI Tweet

**Before:**
```
⚾ Red Sox @ Yankees
🕐 7:05 PM ET

NRFI Chance: 71%
✅ LEAN NRFI

Rodon 3.12 vs Severino 3.45

@RedSox vs @Yankees

truline.app

#NRFI #MLB
```

**After:**
```
Red Sox @ Yankees, 7:05 PM

NRFI at 71% — both pitchers have been stingy in the 1st. Severino has blanked the top of the order in 5 of 7 starts, Rodon in 4 of 6.

@Yankees vs @RedSox

#NRFI #MLB
```

Changes: Drop emoji prefixes, add pitcher 1st-inning context, tighter format.

### Results Recap

**Before:**
```
RESULTS 04/12

Record: 3-2 (60.0%)
P/L: +4.32u

NRFI: 7/15 (46%)

Best: LAD ML W
Worst: SF RL L

Season tracking at truline.app

#MLB #SportsBetting
```

**After:**
```
Yesterday: 3-2, +4.32u

The model's been hot on moneyline underdogs — both wins came from dogs at +130 or better. NRFI went 7 for 15, right around league average.

Best: Dodgers ML W
Worst: Giants runline L

truline.app

#MLB #SportsBetting
```

Changes: Conversational lead, add context about what's working (which bet types, underdog vs favorite), use full team names.

### Additional Context Queries Needed

To support richer copy, we need to pull more data in the content generators:
- **Pitcher 1st-inning scoreless rate**: query `mlb_games` for games where this pitcher started and check `home_first_inning_runs` / `away_first_inning_runs`
- **Team streak**: compute from recent `mlb_games` results (consecutive W or L)
- **Underdog/favorite classification**: compare to market odds
- **Bet type performance**: aggregate `best_bet_type` results from `mlb_prediction_snapshots`

New helper functions in `content.py`:
- `_get_pitcher_first_inning_record(session, pitcher_id)` → `(scoreless_games, total_starts)`
- `_get_team_streak(session, team_abbr)` → `"W4"` or `"L2"` etc.
- `_get_bet_type_summary(session, days=7)` → `{"moneyline": {"wins": 5, "losses": 2}, ...}`

### Files Changed
- `backend/src/services/social/content.py` — all tweet generators rewritten + new helper functions

---

## 4. Implementation Phases

### Phase 1 (This Session)
1. Fix `get_team_card_stats()` — add logging, make resilient
2. Redesign NRFI pick card — 1200x1200, team colors, bigger fonts
3. Redesign final score card — same treatment
4. Redesign 1st inning recap card — same treatment
5. Add `TEAM_COLORS` dict
6. Add helper functions for visual effects (gradient, team bars, rounded rects)
7. Rewrite daily picks thread copy (header + per-game)
8. Rewrite pregame NRFI copy
9. Rewrite results recap copy
10. Add new context helper functions (pitcher 1st inning, streaks, bet type summary)
11. Apply humanizer pass to all new copy

### Phase 2 (Follow-up)
- NRFI results recap copy
- 1st inning recap copy
- Final recap copy
- NBA pick cards + copy
- NBA results copy

### Phase 3 (Future)
- NBA card redesign
- Season-long performance tracking in copy
- A/B test different copy styles

---

## 5. Testing

- Generate all card types locally with test data and verify readability at 50% scale
- Generate cards with missing stats to verify fallback behavior
- Run tweet generators with mock data and verify under 280 chars
- Verify all cards render correctly with long team names (Timberwolves, Trail Blazers, Diamondbacks)
- Deploy to Railway and verify via Blotato dry-run mode before enabling live posting
