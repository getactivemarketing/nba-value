# TikTok Pick-Preview Videos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically render and publish 30-45s narrated pre-game MLB pick-preview videos to TikTok and Instagram.

**Architecture:** The backend owns all data derivation and narration text, exposing one JSON endpoint of publishable picks. The `truline-videos/` Remotion project stays a pure renderer with no database access, exactly as the existing celebration flow already consumes `/mlb/evaluation/underdogs`. Narration is synthesised as six per-beat audio clips, measured, and laid end-to-end so beats cannot desync.

**Tech Stack:** Python 3.13 / FastAPI / SQLAlchemy / pytest (backend); TypeScript / Remotion 4 / React 18 / vitest (video); ElevenLabs or OpenAI TTS; Pexels API; Blotato.

**Spec:** `docs/superpowers/specs/2026-08-10-tiktok-pick-preview-videos-design.md`

## Global Constraints

- The word **`edge`** must never appear in generated narration or overlay copy. The model-vs-market gap ships as **"model projection"**.
- Beat 6 must carry the exact string `Not betting advice. 21+.`
- Only picks where `best_bet_type = 'moneyline'` are eligible. Runline is paused and totals are suppressed.
- A pick is skipped unless first pitch is **at least 45 minutes away at upload time**.
- Missing data **drops its beat**. Never substitute a league average or placeholder.
- Point-in-time: all derivations filter `game_date < target_game_date` strictly.
- A `say`-narrated render must never be uploaded.
- Every NFL market, alert and public recommendation stays disabled. No task touches NFL.
- Never `git add -A` or `git add .` — stage named files only.
- Do not push. Commits stay local unless the user asks.
- Python is `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`.
- Prod DB access: `export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)` and `export DEBUG=false`. Mask passwords in any reported output.

---

## File Structure

**Backend (`backend/`)**

| File | Responsibility |
|---|---|
| `src/services/mlb/team_form.py` | Current win/loss streak from game results (pure) |
| `src/services/mlb/first_inning.py` | Starter first-inning-scoreless split (pure) |
| `src/services/mlb/pick_script.py` | Narration + overlay beats from a pick payload (pure) |
| `src/api/mlb.py` | New `/mlb/video/pick-previews` endpoint |
| `tests/unit/test_mlb_team_form.py` | Streak tests |
| `tests/unit/test_mlb_first_inning.py` | First-inning split tests |
| `tests/unit/test_mlb_pick_script.py` | Beat construction + copy guardrail tests |

**Video (`truline-videos/`)**

| File | Responsibility |
|---|---|
| `src/tts/types.ts` | `TtsAdapter` interface |
| `src/tts/elevenlabs.ts` / `openai.ts` / `say.ts` | Provider adapters |
| `src/tts/index.ts` | Adapter selection from env |
| `src/broll.ts` | Pexels clip fetch + local cache |
| `src/compositions/PickPreview.tsx` | The composition + `calculatePickPreviewMetadata` |
| `scripts/render-pick-previews.ts` | Orchestrator: fetch → TTS → render → upload |
| `src/tts/*.test.ts`, `src/*.test.ts` | vitest unit tests |

`src/compositions/ModelHit.tsx` is **not modified**.

`scripts/render-celebrations.ts` is modified in Task 7 only to import the two
modules extracted from it (`src/blotato.ts`, `src/teams.ts`). Its behaviour is
unchanged. This overrides the spec's "not modified" line, by decision of
2026-08-11: duplicating ~40 lines of upload logic into a second script leaves
two copies that must change together, and the review rubric would flag it
correctly.

---

### Task 1: Current streak derivation

**Files:**
- Create: `backend/src/services/mlb/team_form.py`
- Test: `backend/tests/unit/test_mlb_team_form.py`

**Interfaces:**
- Consumes: nothing
- Produces: `GameResult(date: str, won: bool)`, `Streak(direction: str, length: int)`, `current_streak(results: Iterable[GameResult], as_of: str) -> Streak | None`. `direction` is `"won"` or `"lost"`.

- [ ] **Step 1: Write the failing test**

```python
"""Current win/loss streak, for the 'case against' beat."""
from src.services.mlb.team_form import GameResult, Streak, current_streak


class TestCurrentStreak:
    def test_counts_consecutive_losses_from_most_recent(self):
        results = [
            GameResult("2026-08-05", won=True),
            GameResult("2026-08-06", won=False),
            GameResult("2026-08-07", won=False),
        ]
        assert current_streak(results, as_of="2026-08-08") == Streak("lost", 2)

    def test_counts_consecutive_wins(self):
        results = [
            GameResult("2026-08-05", won=False),
            GameResult("2026-08-06", won=True),
            GameResult("2026-08-07", won=True),
        ]
        assert current_streak(results, as_of="2026-08-08") == Streak("won", 2)

    def test_input_order_does_not_matter(self):
        """Callers must not be able to get this wrong by passing rows unsorted."""
        results = [
            GameResult("2026-08-07", won=False),
            GameResult("2026-08-05", won=True),
            GameResult("2026-08-06", won=False),
        ]
        assert current_streak(results, as_of="2026-08-08") == Streak("lost", 2)

    def test_excludes_games_on_or_after_as_of(self):
        """Point-in-time: a game must never appear in its own inputs."""
        results = [
            GameResult("2026-08-06", won=False),
            GameResult("2026-08-07", won=False),
            GameResult("2026-08-08", won=True),
        ]
        assert current_streak(results, as_of="2026-08-08") == Streak("lost", 2)

    def test_no_prior_games_is_unmeasured(self):
        assert current_streak([], as_of="2026-08-08") is None

    def test_single_game_is_a_streak_of_one(self):
        results = [GameResult("2026-08-07", won=True)]
        assert current_streak(results, as_of="2026-08-08") == Streak("won", 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/unit/test_mlb_team_form.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.mlb.team_form'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Current win/loss streak for a team.

Used by the 'case against' beat of the pick-preview video, which states the
strongest reason NOT to take the pick before making its case.

Pure functions only — no database, no network. See
`docs/superpowers/specs/2026-08-10-tiktok-pick-preview-videos-design.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GameResult:
    """One completed game from a single team's perspective."""

    date: str  # ISO, YYYY-MM-DD
    won: bool


@dataclass(frozen=True)
class Streak:
    direction: str  # "won" | "lost"
    length: int


def current_streak(results: Iterable[GameResult], as_of: str) -> Streak | None:
    """Consecutive same-result games ending most recently before `as_of`.

    Sorting happens here rather than being required of the caller: a caller
    that passes rows in query order would silently get a wrong streak, and a
    wrong streak is narrated aloud in a published video.

    Returns None when there are no prior games — an unmeasured streak drops
    its clause rather than being narrated as zero.
    """
    prior = sorted((r for r in results if r.date < as_of), key=lambda r: r.date)
    if not prior:
        return None

    latest = prior[-1].won
    length = 0
    for result in reversed(prior):
        if result.won != latest:
            break
        length += 1

    return Streak("won" if latest else "lost", length)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/unit/test_mlb_team_form.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/mlb/team_form.py backend/tests/unit/test_mlb_team_form.py
git commit -m "feat(mlb): current win/loss streak derivation"
```

---

### Task 2: Starter first-inning-scoreless split

**Files:**
- Create: `backend/src/services/mlb/first_inning.py`
- Test: `backend/tests/unit/test_mlb_first_inning.py`

**Interfaces:**
- Consumes: nothing
- Produces: `StarterAppearance(date: str, opponent_first_inning_runs: int | None)`, `FirstInningSplit(scoreless: int, starts: int)`, `starter_first_inning_split(appearances: Iterable[StarterAppearance], as_of: str) -> FirstInningSplit | None`

- [ ] **Step 1: Write the failing test**

```python
"""A starter's 'opponents scoreless in the 1st' record.

The stat is the OPPOSING side's first-inning runs: for a home start that is
`away_first_inning_runs`, for an away start `home_first_inning_runs`. Getting
this backwards would narrate a pitcher's own offence as his run prevention.
"""
from src.services.mlb.first_inning import (
    FirstInningSplit,
    StarterAppearance,
    starter_first_inning_split,
)


class TestStarterFirstInningSplit:
    def test_counts_scoreless_first_innings(self):
        apps = [
            StarterAppearance("2026-07-01", opponent_first_inning_runs=0),
            StarterAppearance("2026-07-06", opponent_first_inning_runs=2),
            StarterAppearance("2026-07-11", opponent_first_inning_runs=0),
        ]
        assert starter_first_inning_split(apps, as_of="2026-08-08") == FirstInningSplit(2, 3)

    def test_excludes_games_on_or_after_as_of(self):
        apps = [
            StarterAppearance("2026-07-01", opponent_first_inning_runs=0),
            StarterAppearance("2026-08-08", opponent_first_inning_runs=0),
        ]
        assert starter_first_inning_split(apps, as_of="2026-08-08") == FirstInningSplit(1, 1)

    def test_appearances_with_unknown_runs_are_dropped_not_counted_scoreless(self):
        """None means 'not recorded'. Counting it as zero invents a stat."""
        apps = [
            StarterAppearance("2026-07-01", opponent_first_inning_runs=0),
            StarterAppearance("2026-07-06", opponent_first_inning_runs=None),
        ]
        assert starter_first_inning_split(apps, as_of="2026-08-08") == FirstInningSplit(1, 1)

    def test_no_usable_appearances_is_unmeasured(self):
        apps = [StarterAppearance("2026-07-01", opponent_first_inning_runs=None)]
        assert starter_first_inning_split(apps, as_of="2026-08-08") is None

    def test_empty_is_unmeasured(self):
        assert starter_first_inning_split([], as_of="2026-08-08") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/unit/test_mlb_first_inning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.mlb.first_inning'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Starter first-inning run prevention.

'Held opponents scoreless in the 1st in N of M starts' is the turn beat of the
pick-preview video — the moment it stops being a pick and becomes an argument.

The stat is always the OPPOSING side's first-inning runs. Callers are
responsible for selecting the correct column per appearance:

    home start -> mlb_games.away_first_inning_runs
    away start -> mlb_games.home_first_inning_runs

Pure functions only — no database, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StarterAppearance:
    """One start, with the runs the OPPOSING side scored in the 1st."""

    date: str  # ISO, YYYY-MM-DD
    opponent_first_inning_runs: int | None


@dataclass(frozen=True)
class FirstInningSplit:
    scoreless: int
    starts: int


def starter_first_inning_split(
    appearances: Iterable[StarterAppearance],
    as_of: str,
) -> FirstInningSplit | None:
    """Scoreless-first-inning count over starts strictly before `as_of`.

    Appearances with unrecorded first-inning runs are dropped from BOTH the
    numerator and the denominator. Counting them as scoreless would inflate the
    stat; counting them as scoring would deflate it. Neither is measured.

    Returns None when nothing is measurable — the beat is dropped rather than
    narrated as '0 of 0'.
    """
    usable = [
        a for a in appearances
        if a.date < as_of and a.opponent_first_inning_runs is not None
    ]
    if not usable:
        return None

    return FirstInningSplit(
        scoreless=sum(1 for a in usable if a.opponent_first_inning_runs == 0),
        starts=len(usable),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/unit/test_mlb_first_inning.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/mlb/first_inning.py backend/tests/unit/test_mlb_first_inning.py
git commit -m "feat(mlb): starter first-inning-scoreless split derivation"
```

---

### Task 3: Narration and overlay beats

**Files:**
- Create: `backend/src/services/mlb/pick_script.py`
- Test: `backend/tests/unit/test_mlb_pick_script.py`

**Interfaces:**
- Consumes: `Streak` (Task 1), `FirstInningSplit` (Task 2), `MLBValueCalculator.american_to_decimal` (`src/services/mlb/value_calculator.py:312`)
- Produces: `Beat(key: str, narration: str, overlay: dict[str, str])`, `PickPayload` (fields below), `breakeven_prob(odds_american: int) -> float`, `build_beats(payload: PickPayload) -> list[Beat]`. Beat keys in order: `hook`, `pick`, `case_against`, `turn`, `numbers`, `close`.

- [ ] **Step 1: Write the failing test**

```python
"""Beat construction for the pick-preview video.

Two rules carry real consequence and are asserted hard:

  1. The word 'edge' never ships. winner_probability is a model OUTPUT, not a
     measured result — across 28 picks with closing lines the market moved
     +0.49 points toward us, roughly a nineteenth of the claimed gap, and
     realized CLV is still negative. It ships as 'model projection'.
  2. Missing data drops its beat. A league-average substitute would be narrated
     aloud as fact in a published video.
"""
import pytest

from src.services.mlb.first_inning import FirstInningSplit
from src.services.mlb.pick_script import (
    Beat, PickPayload, breakeven_prob, build_beats,
)
from src.services.mlb.team_form import Streak


def payload(**overrides) -> PickPayload:
    base = dict(
        team_abbr="CWS", team_name="White Sox",
        odds_american=155, model_prob=0.48,
        last_10_record="5-5", streak=Streak("lost", 2),
        starter_name="Castillo", starter_era=5.06,
        first_inning=FirstInningSplit(10, 17),
    )
    base.update(overrides)
    return PickPayload(**base)


class TestBreakevenProb:
    def test_underdog(self):
        assert breakeven_prob(155) == pytest.approx(0.3922, abs=1e-4)

    def test_favourite(self):
        assert breakeven_prob(-175) == pytest.approx(0.6364, abs=1e-4)


class TestCopyGuardrail:
    def test_the_word_edge_never_appears(self):
        for beat in build_beats(payload()):
            assert "edge" not in beat.narration.lower()
            assert "edge" not in " ".join(beat.overlay.values()).lower()

    def test_numbers_beat_says_projection(self):
        numbers = next(b for b in build_beats(payload()) if b.key == "numbers")
        assert "projection" in numbers.narration.lower()

    def test_close_beat_carries_the_disclaimer(self):
        close = next(b for b in build_beats(payload()) if b.key == "close")
        assert "Not betting advice. 21+." in close.overlay.values()


class TestBeatsDropRatherThanFabricate:
    def test_all_six_beats_when_data_is_complete(self):
        assert [b.key for b in build_beats(payload())] == [
            "hook", "pick", "case_against", "turn", "numbers", "close",
        ]

    def test_turn_beat_drops_without_a_first_inning_split(self):
        keys = [b.key for b in build_beats(payload(first_inning=None))]
        assert "turn" not in keys
        assert "numbers" in keys  # the rest of the video survives

    def test_case_against_drops_when_form_and_starter_are_both_absent(self):
        keys = [b.key for b in build_beats(
            payload(last_10_record=None, streak=None, starter_name=None, starter_era=None)
        )]
        assert "case_against" not in keys

    def test_case_against_survives_on_partial_data(self):
        beats = build_beats(payload(starter_name=None, starter_era=None))
        case = next(b for b in beats if b.key == "case_against")
        assert "5-5" in case.narration
        assert "ERA" not in case.narration

    def test_no_placeholder_tokens_anywhere(self):
        for beat in build_beats(payload(streak=None, first_inning=None)):
            text = beat.narration + " ".join(beat.overlay.values())
            for token in ("None", "N/A", "null", "{", "}"):
                assert token not in text


class TestContent:
    def test_hook_leads_with_the_losing_streak(self):
        hook = next(b for b in build_beats(payload()) if b.key == "hook")
        assert "2" in hook.narration and "lost" in hook.narration.lower()

    def test_hook_falls_back_to_underdog_framing_without_a_streak(self):
        hook = next(b for b in build_beats(payload(streak=None)) if b.key == "hook")
        assert "+155" in hook.narration

    def test_turn_beat_states_the_split(self):
        turn = next(b for b in build_beats(payload()) if b.key == "turn")
        assert "10 of 17" in turn.narration
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/unit/test_mlb_pick_script.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.mlb.pick_script'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Narration and overlay copy for the pick-preview video.

Deterministic templates, never an LLM: a model inventing a stat that ships
under TruLine's name with odds attached is a liability, and the variation the
feed wants comes from the data changing daily rather than the phrasing.

Structure — the video argues AGAINST itself before making its case, which is
what makes it read as analysis rather than a pitch:

    hook          the strongest reason not to take it
    pick          team, market, price
    case_against  form and starter ERA
    turn          the first-inning split — the retention anchor
    numbers       model projection vs breakeven
    close         CTA + disclaimer

Pure functions only — no database, no network.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.services.mlb.first_inning import FirstInningSplit
from src.services.mlb.team_form import Streak
from src.services.mlb.value_calculator import MLBValueCalculator

DISCLAIMER = "Not betting advice. 21+."


@dataclass(frozen=True)
class Beat:
    """One narrated segment and the graphic shown over it."""

    key: str
    narration: str
    overlay: dict[str, str]


@dataclass(frozen=True)
class PickPayload:
    team_abbr: str
    team_name: str
    odds_american: int
    model_prob: float
    last_10_record: str | None = None
    streak: Streak | None = None
    starter_name: str | None = None
    starter_era: float | None = None
    first_inning: FirstInningSplit | None = None


def breakeven_prob(odds_american: int) -> float:
    """Probability the offered price must beat to break even.

    Deliberately the RAW breakeven, not the devigged consensus: at +155 that is
    39.2% where the devigged fair price is nearer 38%, so quoting breakeven
    makes the model-vs-market gap look SMALLER. Between two defensible numbers,
    publish the conservative one.
    """
    return 1.0 / MLBValueCalculator.american_to_decimal(odds_american)


def _price(odds_american: int) -> str:
    return f"+{odds_american}" if odds_american > 0 else str(odds_american)


def build_beats(payload: PickPayload) -> list[Beat]:
    """Beats for one pick, in order. Unmeasurable beats are omitted entirely."""
    price = _price(payload.odds_american)
    market = breakeven_prob(payload.odds_american)
    beats: list[Beat] = []

    # -- hook: lead with the reason to walk away ------------------------------
    if payload.streak is not None and payload.streak.direction == "lost":
        hook = (
            f"Backing a team that's lost {payload.streak.length} straight."
            if payload.streak.length > 1
            else "Backing a team coming off a loss."
        )
    else:
        hook = f"The model's taking a {price} underdog tonight."
    beats.append(Beat("hook", hook, {"line": hook}))

    # -- pick ----------------------------------------------------------------
    beats.append(Beat(
        "pick",
        f"{payload.team_name}, moneyline, {price}.",
        {"team": payload.team_name.upper(), "market": "MONEYLINE", "price": price},
    ))

    # -- case against --------------------------------------------------------
    clauses: list[str] = []
    chips: list[str] = []
    if payload.last_10_record:
        clauses.append(f"They're {payload.last_10_record} in their last ten")
        chips.append(f"{payload.last_10_record} L10")
    if payload.streak is not None and payload.streak.length > 1:
        verb = "losing" if payload.streak.direction == "lost" else "winning"
        clauses.append(f"on a {payload.streak.length}-game {verb} streak")
        chips.append(f"{payload.streak.length}-game {verb} streak")
    if payload.starter_name and payload.starter_era is not None:
        clauses.append(
            f"and {payload.starter_name} carries a {payload.starter_era:.2f} ERA"
        )
        chips.append(f"{payload.starter_name} {payload.starter_era:.2f} ERA")

    if clauses:
        beats.append(Beat(
            "case_against",
            ", ".join(clauses) + ".",
            {"chips": " · ".join(chips)},
        ))

    # -- turn: the retention anchor ------------------------------------------
    if payload.first_inning is not None and payload.starter_name:
        split = payload.first_inning
        beats.append(Beat(
            "turn",
            f"But {payload.starter_name} has held opponents scoreless in the "
            f"first in {split.scoreless} of {split.starts} starts.",
            {
                "stat": f"{split.scoreless} of {split.starts}",
                "label": "SCORELESS 1ST",
            },
        ))

    # -- numbers -------------------------------------------------------------
    beats.append(Beat(
        "numbers",
        f"Model projection: {payload.model_prob:.0%}. "
        f"The price needs {market:.0%}.",
        {
            "model": f"{payload.model_prob:.0%}",
            "model_label": "MODEL PROJECTION",
            "market": f"{market:.0%}",
            "market_label": "BREAKEVEN",
        },
    ))

    # -- close ---------------------------------------------------------------
    beats.append(Beat(
        "close",
        "Full model at truline dot app.",
        {"cta": "truline.app", "disclaimer": DISCLAIMER},
    ))

    return beats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/unit/test_mlb_pick_script.py -v`
Expected: PASS, 14 tests

- [ ] **Step 5: Run the full backend suite for regressions**

Run: `cd backend && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/unit -q`
Expected: PASS, no failures introduced

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/mlb/pick_script.py backend/tests/unit/test_mlb_pick_script.py
git commit -m "feat(mlb): narration beats for pick-preview video

Ships the model-vs-market gap as 'model projection', never 'edge' —
winner_probability is a model output, and measured market movement is
~1/19th of the claimed gap with CLV still negative. Asserted in tests."
```

---

### Task 4: Pick-preview payload endpoint

**Files:**
- Modify: `backend/src/api/mlb.py` (append endpoint + response models)
- Test: `backend/tests/unit/test_mlb_pick_preview_api.py`

**Interfaces:**
- Consumes: `build_beats`, `PickPayload` (Task 3); `current_streak`, `GameResult` (Task 1); `starter_first_inning_split`, `StarterAppearance` (Task 2)
- Produces: `GET /api/v1/mlb/video/pick-previews` returning `PickPreviewList`. Each item: `game_id`, `game_date`, `game_time` (ISO), `team_abbr`, `team_name`, `logo_url`, `odds_american`, `beats[]` where each beat is `{key, narration, overlay}`.
- Produces: `_build_pick_preview(session, snapshot) -> PickPreviewItem | None` — importable for tests.

- [ ] **Step 1: Write the failing test**

```python
"""The pick-preview endpoint's selection rules.

Selection is where the disabled markets are mechanically enforced. Runline is
paused and totals suppressed; publishing them would advertise markets we have
deliberately turned off.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.api.mlb import eligible_for_preview


class _Snap:
    def __init__(self, **kw):
        self.best_bet_type = kw.get("best_bet_type", "moneyline")
        self.best_ml_team = kw.get("best_ml_team", "CWS")
        self.best_ml_odds = kw.get("best_ml_odds", 2.55)
        self.game_time = kw.get("game_time")


def _in(minutes: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


class TestEligibility:
    def test_moneyline_with_enough_lead_time_is_eligible(self):
        assert eligible_for_preview(_Snap(game_time=_in(120))) is True

    def test_runline_is_never_eligible(self):
        assert eligible_for_preview(_Snap(best_bet_type="runline", game_time=_in(120))) is False

    def test_total_is_never_eligible(self):
        assert eligible_for_preview(_Snap(best_bet_type="total", game_time=_in(120))) is False

    def test_inside_the_lead_time_gate_is_rejected(self):
        assert eligible_for_preview(_Snap(game_time=_in(30))) is False

    def test_already_started_is_rejected(self):
        assert eligible_for_preview(_Snap(game_time=_in(-10))) is False

    def test_exactly_at_the_gate_is_rejected(self):
        """Boundary is strict — Blotato can hold a post for minutes."""
        assert eligible_for_preview(_Snap(game_time=_in(45)), min_lead_minutes=45) is False

    def test_missing_price_is_rejected(self):
        assert eligible_for_preview(_Snap(best_ml_odds=None, game_time=_in(120))) is False

    def test_missing_game_time_is_rejected(self):
        assert eligible_for_preview(_Snap(game_time=None)) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/unit/test_mlb_pick_preview_api.py -v`
Expected: FAIL with `ImportError: cannot import name 'eligible_for_preview'`

- [ ] **Step 3: Add the eligibility gate and response models to `src/api/mlb.py`**

Append near the other response models:

```python
class PickPreviewBeat(BaseModel):
    key: str
    narration: str
    overlay: dict[str, str]


class PickPreviewItem(BaseModel):
    game_id: str
    game_date: str
    game_time: str
    team_abbr: str
    team_name: str
    logo_url: str
    odds_american: int
    beats: list[PickPreviewBeat]


class PickPreviewList(BaseModel):
    generated_at: str
    previews: list[PickPreviewItem]


# Minutes of clearance required before first pitch. Measured at UPLOAD time,
# not render time — Blotato's useNextFreeSlot can hold a post for minutes, and
# a pick published after first pitch is worthless as a receipt.
PREVIEW_MIN_LEAD_MINUTES = 45


def eligible_for_preview(snapshot, min_lead_minutes: int = PREVIEW_MIN_LEAD_MINUTES) -> bool:
    """Whether a snapshot may be published as a pre-game video.

    Restricting to moneyline is what mechanically keeps the paused runline and
    suppressed totals out of published video.
    """
    if (snapshot.best_bet_type or "").lower() != "moneyline":
        return False
    if not snapshot.best_ml_team or snapshot.best_ml_odds is None:
        return False
    if snapshot.game_time is None:
        return False

    from datetime import datetime as _dt, timezone as _tz
    kickoff = snapshot.game_time
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=_tz.utc)
    lead = (kickoff - _dt.now(_tz.utc)).total_seconds() / 60.0
    return lead > min_lead_minutes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/unit/test_mlb_pick_preview_api.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Add the endpoint**

Append to `src/api/mlb.py`.

`TEAM_NAMES` already exists at `src/services/social/content.py:63` — import it
rather than redefining. Team colours are **not** returned by the API: they live
only in the video project today, and duplicating a 50-team colour table into
the backend to serve a renderer would be the wrong home for it. Task 7 gives
the video project `src/teams.ts` for that.

```python
@router.get("/video/pick-previews", response_model=PickPreviewList)
async def get_pick_previews(
    days: int = Query(1, ge=1, le=3, description="Slate look-ahead in days"),
) -> PickPreviewList:
    """Publishable pre-game pick previews, fully rendered to narration beats.

    The video project holds no database access, so everything it needs —
    derivations included — is assembled here. Same split the celebration flow
    already uses against /mlb/evaluation/underdogs.
    """
    from datetime import date as _date, timedelta as _timedelta
    from src.services.mlb.first_inning import StarterAppearance, starter_first_inning_split
    from src.services.mlb.pick_script import PickPayload, build_beats
    from src.services.mlb.team_form import GameResult, current_streak
    from src.services.mlb.value_calculator import MLBValueCalculator
    from src.services.social.content import TEAM_NAMES

    today = _date.today()
    horizon = today + _timedelta(days=days)

    async with async_session() as session:
        result = await session.execute(
            select(MLBPredictionSnapshot).where(
                and_(
                    MLBPredictionSnapshot.game_date >= today,
                    MLBPredictionSnapshot.game_date <= horizon,
                )
            )
        )
        snapshots = [s for s in result.scalars().all() if eligible_for_preview(s)]

        previews: list[PickPreviewItem] = []
        for snap in snapshots:
            team = snap.best_ml_team
            as_of = snap.game_date.isoformat()

            games = (await session.execute(
                select(MLBGame).where(
                    and_(
                        MLBGame.status == "final",
                        or_(MLBGame.home_team == team, MLBGame.away_team == team),
                    )
                )
            )).scalars().all()

            results = []
            for g in games:
                if g.home_score is None or g.away_score is None:
                    continue
                is_home = g.home_team == team
                mine = g.home_score if is_home else g.away_score
                theirs = g.away_score if is_home else g.home_score
                results.append(GameResult(g.game_date.isoformat(), won=mine > theirs))
            streak = current_streak(results, as_of=as_of)

            is_home_pick = team == snap.home_team
            starter_name = snap.home_starter_name if is_home_pick else snap.away_starter_name
            starter_era = snap.home_starter_era if is_home_pick else snap.away_starter_era

            starter_id = None
            game_row = next((g for g in games if g.game_id == snap.game_id), None)
            if game_row is not None:
                starter_id = game_row.home_starter_id if is_home_pick else game_row.away_starter_id

            split = None
            if starter_id is not None:
                starts = (await session.execute(
                    select(MLBGame).where(
                        or_(
                            MLBGame.home_starter_id == starter_id,
                            MLBGame.away_starter_id == starter_id,
                        )
                    )
                )).scalars().all()
                apps = [
                    StarterAppearance(
                        g.game_date.isoformat(),
                        g.away_first_inning_runs if g.home_starter_id == starter_id
                        else g.home_first_inning_runs,
                    )
                    for g in starts
                ]
                split = starter_first_inning_split(apps, as_of=as_of)

            stats = (await session.execute(
                select(MLBTeamStats)
                .where(and_(MLBTeamStats.team_abbr == team,
                            MLBTeamStats.stat_date < snap.game_date))
                .order_by(desc(MLBTeamStats.stat_date))
                .limit(1)
            )).scalars().first()

            odds_american = MLBValueCalculator.decimal_to_american(float(snap.best_ml_odds))
            beats = build_beats(PickPayload(
                team_abbr=team,
                team_name=TEAM_NAMES.get(team, team),
                odds_american=odds_american,
                model_prob=float(snap.winner_probability),
                last_10_record=stats.last_10_record if stats else None,
                streak=streak,
                starter_name=starter_name,
                starter_era=float(starter_era) if starter_era is not None else None,
                first_inning=split,
            ))

            previews.append(PickPreviewItem(
                game_id=snap.game_id,
                game_date=as_of,
                game_time=snap.game_time.isoformat(),
                team_abbr=team,
                team_name=TEAM_NAMES.get(team, team),
                logo_url=f"https://a.espncdn.com/i/teamlogos/mlb/500/{team.lower()}.png",
                odds_american=odds_american,
                beats=[PickPreviewBeat(**b.__dict__) for b in beats],
            ))

    return PickPreviewList(
        generated_at=datetime.now(timezone.utc).isoformat(),
        previews=previews,
    )
```

Add `or_` to the existing `from sqlalchemy import ...` line at the top of the file.

- [ ] **Step 6: Verify the endpoint against prod data**

```bash
cd backend
export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)
export DEBUG=false
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m uvicorn src.main:app --port 8899 &
sleep 8
curl -s "http://localhost:8899/api/v1/mlb/video/pick-previews?days=2" | head -c 2000
kill %1
```

Expected: JSON with a `previews` array. Confirm by eye that no beat text contains `None`, `N/A`, or the word `edge`, and that any `turn` beat reads as a plausible "N of M".

- [ ] **Step 7: Commit**

```bash
git add backend/src/api/mlb.py backend/tests/unit/test_mlb_pick_preview_api.py
git commit -m "feat(mlb): pick-preview payload endpoint

Moneyline-only selection plus a 45-minute pre-first-pitch gate; the
market filter is what keeps paused runline and suppressed totals out
of published video."
```

---

### Task 5: TTS provider abstraction

**Files:**
- Create: `truline-videos/src/tts/types.ts`, `elevenlabs.ts`, `openai.ts`, `say.ts`, `index.ts`
- Create: `truline-videos/src/tts/index.test.ts`
- Modify: `truline-videos/package.json` (add vitest + `test` script)
- Create: `truline-videos/vitest.config.ts`

**Interfaces:**
- Consumes: nothing
- Produces: `TtsAdapter { id: 'elevenlabs' | 'openai' | 'say'; publishable: boolean; synthesize(text: string, outPath: string): Promise<void> }`, `selectAdapter(env: NodeJS.ProcessEnv): TtsAdapter`

The video project currently has **no test runner**. This task introduces vitest.

- [ ] **Step 1: Add vitest**

```bash
cd truline-videos
npm install --save-dev vitest@^2
```

Add to `package.json` `"scripts"`: `"test": "vitest run"`.

Create `vitest.config.ts`:

```ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: { include: ['src/**/*.test.ts'], environment: 'node' },
});
```

- [ ] **Step 2: Write the failing test**

```ts
// src/tts/index.test.ts
import { describe, expect, it } from 'vitest';
import { selectAdapter } from './index';

describe('selectAdapter', () => {
  it('prefers elevenlabs when its key is present', () => {
    expect(selectAdapter({ ELEVENLABS_API_KEY: 'k' }).id).toBe('elevenlabs');
  });

  it('falls back to openai when only that key is present', () => {
    expect(selectAdapter({ OPENAI_API_KEY: 'k' }).id).toBe('openai');
  });

  it('honours an explicit TTS_PROVIDER override', () => {
    const env = { TTS_PROVIDER: 'openai', ELEVENLABS_API_KEY: 'k', OPENAI_API_KEY: 'k' };
    expect(selectAdapter(env).id).toBe('openai');
  });

  it('falls back to say when no key is configured', () => {
    expect(selectAdapter({}).id).toBe('say');
  });

  it('marks say as not publishable', () => {
    expect(selectAdapter({}).publishable).toBe(false);
  });

  it('marks real providers as publishable', () => {
    expect(selectAdapter({ ELEVENLABS_API_KEY: 'k' }).publishable).toBe(true);
    expect(selectAdapter({ OPENAI_API_KEY: 'k' }).publishable).toBe(true);
  });

  it('throws when an explicit provider has no key', () => {
    expect(() => selectAdapter({ TTS_PROVIDER: 'elevenlabs' })).toThrow(/ELEVENLABS_API_KEY/);
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd truline-videos && npm test`
Expected: FAIL — cannot resolve `./index`

- [ ] **Step 4: Write the implementation**

`src/tts/types.ts`:

```ts
export interface TtsAdapter {
  readonly id: 'elevenlabs' | 'openai' | 'say';
  /** False means renders using this adapter must never be uploaded. */
  readonly publishable: boolean;
  synthesize(text: string, outPath: string): Promise<void>;
}
```

`src/tts/elevenlabs.ts`:

```ts
import axios from 'axios';
import { writeFileSync } from 'fs';
import type { TtsAdapter } from './types';

const VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'JBFqnCBsd6RMkjVDRZzb';

export function elevenLabsAdapter(apiKey: string): TtsAdapter {
  return {
    id: 'elevenlabs',
    publishable: true,
    async synthesize(text, outPath) {
      const resp = await axios.post(
        `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}`,
        { text, model_id: 'eleven_turbo_v2_5' },
        {
          headers: { 'xi-api-key': apiKey, 'Content-Type': 'application/json' },
          responseType: 'arraybuffer',
          timeout: 60000,
        },
      );
      writeFileSync(outPath, Buffer.from(resp.data));
    },
  };
}
```

`src/tts/openai.ts`:

```ts
import axios from 'axios';
import { writeFileSync } from 'fs';
import type { TtsAdapter } from './types';

export function openAiAdapter(apiKey: string): TtsAdapter {
  return {
    id: 'openai',
    publishable: true,
    async synthesize(text, outPath) {
      const resp = await axios.post(
        'https://api.openai.com/v1/audio/speech',
        { model: 'gpt-4o-mini-tts', voice: 'onyx', input: text, response_format: 'mp3' },
        {
          headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
          responseType: 'arraybuffer',
          timeout: 60000,
        },
      );
      writeFileSync(outPath, Buffer.from(resp.data));
    },
  };
}
```

`src/tts/say.ts`:

```ts
import { execFileSync } from 'child_process';
import type { TtsAdapter } from './types';

/**
 * macOS `say`. Offline dev fallback so the pipeline is testable without a key.
 * publishable=false — it is too robotic to publish, and the orchestrator
 * refuses to upload renders narrated with it.
 */
export function sayAdapter(): TtsAdapter {
  return {
    id: 'say',
    publishable: false,
    async synthesize(text, outPath) {
      execFileSync('say', ['-v', 'Samantha', '-o', outPath, '--data-format=LEF32@22050', text]);
    },
  };
}
```

`src/tts/index.ts`:

```ts
import { elevenLabsAdapter } from './elevenlabs';
import { openAiAdapter } from './openai';
import { sayAdapter } from './say';
import type { TtsAdapter } from './types';

export type { TtsAdapter };

export function selectAdapter(env: NodeJS.ProcessEnv = process.env): TtsAdapter {
  const explicit = env.TTS_PROVIDER;

  if (explicit === 'elevenlabs') {
    if (!env.ELEVENLABS_API_KEY) throw new Error('TTS_PROVIDER=elevenlabs but ELEVENLABS_API_KEY is unset');
    return elevenLabsAdapter(env.ELEVENLABS_API_KEY);
  }
  if (explicit === 'openai') {
    if (!env.OPENAI_API_KEY) throw new Error('TTS_PROVIDER=openai but OPENAI_API_KEY is unset');
    return openAiAdapter(env.OPENAI_API_KEY);
  }
  if (explicit === 'say') return sayAdapter();

  if (env.ELEVENLABS_API_KEY) return elevenLabsAdapter(env.ELEVENLABS_API_KEY);
  if (env.OPENAI_API_KEY) return openAiAdapter(env.OPENAI_API_KEY);
  return sayAdapter();
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd truline-videos && npm test`
Expected: PASS, 7 tests

- [ ] **Step 6: Commit**

```bash
git add truline-videos/src/tts truline-videos/vitest.config.ts truline-videos/package.json truline-videos/package-lock.json
git commit -m "feat(video): pluggable TTS with a non-publishable say fallback"
```

---

### Task 6: PickPreview composition

**Files:**
- Create: `truline-videos/src/compositions/PickPreview.tsx`
- Create: `truline-videos/src/compositions/PickPreview.test.ts`
- Modify: `truline-videos/src/Root.tsx` (register the composition)

**Interfaces:**
- Consumes: nothing
- Produces: `BeatClip { key: string; overlay: Record<string, string>; audioSrc: string; durationInFrames: number }`, `PickPreviewProps { beats: BeatClip[]; teamColor: string; logoUrl: string; brollSrc?: string; musicFile?: string }`, `calculatePickPreviewMetadata({ props }): { durationInFrames: number }`

- [ ] **Step 1: Write the failing test**

```ts
// src/compositions/PickPreview.test.ts
import { describe, expect, it } from 'vitest';
import { calculatePickPreviewMetadata, type BeatClip } from './PickPreview';

const beat = (durationInFrames: number): BeatClip => ({
  key: 'x', overlay: {}, audioSrc: 'a.mp3', durationInFrames,
});

describe('calculatePickPreviewMetadata', () => {
  it('total duration is the sum of beat durations', () => {
    const props = { beats: [beat(60), beat(90), beat(30)], teamColor: '#000', logoUrl: '' };
    expect(calculatePickPreviewMetadata({ props }).durationInFrames).toBe(180);
  });

  it('a longer narration lengthens only its own beat', () => {
    const base = { beats: [beat(60), beat(90)], teamColor: '#000', logoUrl: '' };
    const longer = { beats: [beat(60), beat(150)], teamColor: '#000', logoUrl: '' };
    expect(calculatePickPreviewMetadata({ props: longer }).durationInFrames
      - calculatePickPreviewMetadata({ props: base }).durationInFrames).toBe(60);
  });

  it('never returns zero frames, which would fail the render', () => {
    const props = { beats: [], teamColor: '#000', logoUrl: '' };
    expect(calculatePickPreviewMetadata({ props }).durationInFrames).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd truline-videos && npm test`
Expected: FAIL — cannot resolve `./PickPreview`

- [ ] **Step 3: Write the composition**

```tsx
import React from 'react';
import {
  AbsoluteFill, Audio, Img, OffthreadVideo, Sequence,
  interpolate, spring, staticFile, useCurrentFrame, useVideoConfig,
} from 'remotion';
import { COLORS, FONTS, FPS } from '../constants';

export interface BeatClip {
  key: string;
  overlay: Record<string, string>;
  audioSrc: string;
  durationInFrames: number;
}

export interface PickPreviewProps {
  beats: BeatClip[];
  teamColor: string;
  logoUrl: string;
  brollSrc?: string;
  musicFile?: string;
}

/**
 * Duration is derived from the narration, never the reverse. Hardcoding beat
 * lengths and dropping audio in afterwards desyncs the moment a team name or a
 * stat reads longer than the template assumed.
 */
export const calculatePickPreviewMetadata = ({ props }: { props: PickPreviewProps }) => ({
  durationInFrames: Math.max(
    1,
    props.beats.reduce((sum, b) => sum + b.durationInFrames, 0),
  ),
});

const BeatText: React.FC<{ overlay: Record<string, string>; teamColor: string; logoUrl: string }> = ({
  overlay, teamColor, logoUrl,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const progress = spring({ frame, fps, config: { damping: 18, stiffness: 220, mass: 0.5 } });
  const opacity = interpolate(progress, [0, 1], [0, 1]);
  const scale = interpolate(progress, [0, 1], [0.6, 1]);

  const entries = Object.entries(overlay);

  return (
    <AbsoluteFill style={{
      justifyContent: 'center', alignItems: 'center', padding: 80,
      opacity, transform: `scale(${scale})`,
    }}>
      {overlay.team && (
        <Img src={logoUrl} width={280} height={280}
             style={{ filter: `drop-shadow(0 0 60px ${teamColor})`, marginBottom: 40 }} />
      )}
      {entries.filter(([k]) => k !== 'team').map(([key, value]) => (
        <div key={key} style={{
          textAlign: 'center',
          fontFamily: key === 'price' || key === 'stat' ? FONTS.mono : FONTS.display,
          fontWeight: key.endsWith('label') || key === 'disclaimer' ? 500 : 800,
          fontSize: key === 'price' || key === 'stat' ? 150
            : key.endsWith('label') || key === 'disclaimer' ? 36 : 64,
          color: key === 'price' || key === 'stat' ? COLORS.accent
            : key.endsWith('label') || key === 'disclaimer' ? COLORS.muted : COLORS.text,
          letterSpacing: '-0.02em', lineHeight: 1.15, marginBottom: 18,
        }}>
          {value}
        </div>
      ))}
    </AbsoluteFill>
  );
};

export const PickPreview: React.FC<PickPreviewProps> = ({
  beats, teamColor, logoUrl, brollSrc, musicFile,
}) => {
  let cursor = 0;

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.bg }}>
      {brollSrc && (
        <AbsoluteFill style={{ opacity: 0.15 }}>
          <OffthreadVideo src={brollSrc} muted loop
                          style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        </AbsoluteFill>
      )}

      <AbsoluteFill style={{
        background: `radial-gradient(circle at 50% 35%, ${teamColor}55 0%, transparent 65%)`,
      }} />

      {musicFile && <Audio src={staticFile(musicFile)} volume={0.15} />}

      {beats.map((beat) => {
        const from = cursor;
        cursor += beat.durationInFrames;
        return (
          <Sequence key={beat.key} from={from} durationInFrames={beat.durationInFrames}>
            <Audio src={beat.audioSrc} />
            <BeatText overlay={beat.overlay} teamColor={teamColor} logoUrl={logoUrl} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

export const PICK_PREVIEW_FPS = FPS;
```

- [ ] **Step 4: Register in `src/Root.tsx`**

Add alongside the existing `model-hit` composition, leaving it untouched:

```tsx
import { PickPreview, calculatePickPreviewMetadata, type PickPreviewProps } from './compositions/PickPreview';

// ...inside <> ... </> alongside the existing <Composition>:
<Composition
  id="pick-preview"
  component={PickPreview as unknown as React.ComponentType<Record<string, unknown>>}
  durationInFrames={900}
  fps={FPS}
  width={WIDTH}
  height={HEIGHT}
  defaultProps={{ beats: [], teamColor: '#27251F', logoUrl: '' } as unknown as Record<string, unknown>}
  calculateMetadata={calculatePickPreviewMetadata as never}
/>
```

Wrap the two `<Composition>` elements in a fragment if `Root.tsx` currently returns a single element.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd truline-videos && npm test`
Expected: PASS, 10 tests total

- [ ] **Step 6: Commit**

```bash
git add truline-videos/src/compositions/PickPreview.tsx truline-videos/src/compositions/PickPreview.test.ts truline-videos/src/Root.tsx
git commit -m "feat(video): PickPreview composition with narration-driven duration"
```

---

### Task 7: Pexels b-roll fetch, plus shared team colours and Blotato upload

**Files:**
- Create: `truline-videos/src/broll.ts`, `truline-videos/src/broll.test.ts`
- Create: `truline-videos/src/teams.ts`, `truline-videos/src/blotato.ts`
- Modify: `truline-videos/scripts/render-celebrations.ts` (import the extracted modules)

**Interfaces:**
- Consumes: nothing
- Produces: `pickBrollQuery(sport: string): string`, `fetchBroll(query: string, cacheDir: string, deps: { get: (url: string, cfg: unknown) => Promise<{ data: unknown }>; exists: (p: string) => boolean; write: (p: string, b: Buffer) => void; apiKey?: string }): Promise<string | undefined>`

Returns the cached local path, or `undefined` when no key is set or the fetch fails — b-roll is decoration, and its absence must never block a render.

- [ ] **Step 1: Write the failing test**

```ts
// src/broll.test.ts
import { describe, expect, it, vi } from 'vitest';
import { fetchBroll, pickBrollQuery } from './broll';

const deps = (over: Partial<Parameters<typeof fetchBroll>[2]> = {}) => ({
  get: vi.fn(), exists: () => false, write: vi.fn(), apiKey: 'k', ...over,
});

describe('pickBrollQuery', () => {
  it('returns an unbranded query — never a team or player name', () => {
    const q = pickBrollQuery('mlb');
    expect(q).toMatch(/baseball|stadium|crowd/i);
    expect(q).not.toMatch(/yankees|dodgers|white sox/i);
  });
});

describe('fetchBroll', () => {
  it('returns undefined without an api key rather than throwing', async () => {
    const d = deps({ apiKey: undefined });
    await expect(fetchBroll('baseball', '/tmp', d)).resolves.toBeUndefined();
    expect(d.get).not.toHaveBeenCalled();
  });

  it('returns undefined when the request fails — b-roll never blocks a render', async () => {
    const d = deps({ get: vi.fn().mockRejectedValue(new Error('429')) });
    await expect(fetchBroll('baseball', '/tmp', d)).resolves.toBeUndefined();
  });

  it('skips the network entirely when the clip is already cached', async () => {
    const d = deps({ exists: () => true });
    await expect(fetchBroll('baseball', '/tmp', d)).resolves.toContain('/tmp');
    expect(d.get).not.toHaveBeenCalled();
  });

  it('downloads and writes the first returned video file', async () => {
    const d = deps({
      get: vi.fn()
        .mockResolvedValueOnce({ data: { videos: [{ video_files: [{ link: 'http://v/1.mp4', width: 1080 }] }] } })
        .mockResolvedValueOnce({ data: Buffer.from('bytes') }),
    });
    const out = await fetchBroll('baseball', '/tmp', d);
    expect(out).toContain('/tmp');
    expect(d.write).toHaveBeenCalled();
  });

  it('returns undefined when the search yields no clips', async () => {
    const d = deps({ get: vi.fn().mockResolvedValue({ data: { videos: [] } }) });
    await expect(fetchBroll('baseball', '/tmp', d)).resolves.toBeUndefined();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd truline-videos && npm test`
Expected: FAIL — cannot resolve `./broll`

- [ ] **Step 3: Write the implementation**

```ts
import { createHash } from 'crypto';
import { resolve } from 'path';

/**
 * Unbranded stock b-roll only.
 *
 * League game footage and Getty/AP editorial clips cannot be licensed for
 * betting promotion, and TikTok Content-ID mutes them. Queries here must never
 * name a team or player.
 */
const QUERIES: Record<string, string[]> = {
  mlb: ['baseball stadium night', 'baseball crowd', 'stadium floodlights'],
  nba: ['basketball court', 'arena crowd', 'basketball hoop night'],
};

export function pickBrollQuery(sport: string): string {
  const pool = QUERIES[sport.toLowerCase()] || QUERIES.mlb;
  return pool[0];
}

export interface BrollDeps {
  get: (url: string, cfg: unknown) => Promise<{ data: unknown }>;
  exists: (path: string) => boolean;
  write: (path: string, body: Buffer) => void;
  apiKey?: string;
}

export async function fetchBroll(
  query: string,
  cacheDir: string,
  deps: BrollDeps,
): Promise<string | undefined> {
  const slug = createHash('sha1').update(query).digest('hex').slice(0, 12);
  const out = resolve(cacheDir, `broll_${slug}.mp4`);

  if (deps.exists(out)) return out;
  if (!deps.apiKey) return undefined;

  try {
    const search = await deps.get('https://api.pexels.com/videos/search', {
      headers: { Authorization: deps.apiKey },
      params: { query, orientation: 'portrait', per_page: 1 },
      timeout: 30000,
    }) as { data: { videos?: { video_files: { link: string }[] }[] } };

    const link = search.data.videos?.[0]?.video_files?.[0]?.link;
    if (!link) return undefined;

    const clip = await deps.get(link, { responseType: 'arraybuffer', timeout: 60000 });
    deps.write(out, Buffer.from(clip.data as ArrayBuffer));
    return out;
  } catch {
    return undefined;
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd truline-videos && npm test`
Expected: PASS, 16 tests total

- [ ] **Step 5: Extract the team tables**

**Move** (do not copy) `TEAM_COLORS` and `TEAM_NAMES` out of
`scripts/render-celebrations.ts` into a new `truline-videos/src/teams.ts`:

```ts
/** MLB and NBA team colours and display names, shared by the video scripts. */
export const TEAM_COLORS: Record<string, string> = {
  // ...moved verbatim from render-celebrations.ts
};

export const TEAM_NAMES: Record<string, string> = {
  // ...moved verbatim from render-celebrations.ts
};

export const teamColor = (abbr: string): string => TEAM_COLORS[abbr] || '#059669';
export const teamName = (abbr: string): string => TEAM_NAMES[abbr] || abbr;
```

Delete both dict literals from `render-celebrations.ts` and add
`import { TEAM_COLORS, TEAM_NAMES } from '../src/teams';`. Its existing uses
of both names then resolve unchanged.

- [ ] **Step 6: Extract the Blotato upload**

**Move** `uploadToBlotato` out of `scripts/render-celebrations.ts` into a new
`truline-videos/src/blotato.ts`, taking its module-level config as parameters
so it has no hidden environment dependency:

```ts
import axios from 'axios';
import { readFileSync } from 'fs';

const BLOTATO_API = 'https://backend.blotato.com/v2';

export interface BlotatoConfig {
  apiKey: string;
  tiktokAccountId?: string;
  instagramAccountId?: string;
}

export function blotatoConfigFromEnv(env: NodeJS.ProcessEnv = process.env): BlotatoConfig {
  return {
    apiKey: env.BLOTATO_API_KEY || '',
    tiktokAccountId: env.BLOTATO_TIKTOK_ACCOUNT_ID,
    instagramAccountId: env.BLOTATO_INSTAGRAM_ACCOUNT_ID,
  };
}

/** Uploads and schedules. A missing apiKey is a DRY RUN, never an error —
 *  that is how both scripts are exercised without posting. */
export async function uploadToBlotato(
  videoPath: string,
  caption: string,
  cfg: BlotatoConfig,
): Promise<void> {
  if (!cfg.apiKey) {
    console.log('[DRY-RUN] Would upload:', videoPath);
    console.log('[DRY-RUN] Caption:', caption);
    return;
  }

  const headers = { 'blotato-api-key': cfg.apiKey, 'Content-Type': 'application/json' };
  const filename = videoPath.split('/').pop() || 'video.mp4';

  try {
    const { data: { presignedUrl, publicUrl } } =
      await axios.post(`${BLOTATO_API}/media/uploads`, { filename }, { headers, timeout: 60000 });
    await axios.put(presignedUrl, readFileSync(videoPath), {
      headers: { 'Content-Type': 'video/mp4' }, timeout: 120000, maxBodyLength: Infinity,
    });
    console.log(`Uploaded: ${publicUrl}`);

    for (const [platform, accountId] of [
      ['tiktok', cfg.tiktokAccountId],
      ['instagram', cfg.instagramAccountId],
    ] as const) {
      if (!accountId) continue;
      try {
        const { data } = await axios.post(`${BLOTATO_API}/posts`, {
          post: {
            accountId,
            content: { text: caption, mediaUrls: [publicUrl], platform },
            target: { targetType: platform },
          },
          useNextFreeSlot: true,
        }, { headers, timeout: 30000 });
        console.log(`Posted to ${platform}:`, data.postSubmissionId);
      } catch (err: any) {
        console.error(`Failed to post to ${platform}:`, err?.response?.data || err.message);
      }
    }
  } catch (err: any) {
    console.error('Upload failed:', err?.response?.data || err.message);
  }
}
```

In `render-celebrations.ts`: delete its local `uploadToBlotato` and the three
`BLOTATO_*` module constants, add
`import { blotatoConfigFromEnv, uploadToBlotato } from '../src/blotato';`, and
change its one call site to
`await uploadToBlotato(outputPath, caption, blotatoConfigFromEnv());`.

- [ ] **Step 7: Verify the celebration script still type-checks and dry-runs**

```bash
cd truline-videos
npx tsc --noEmit
npx tsx scripts/render-celebrations.ts
```

Expected: `tsc` clean. The script fetches underdogs and either finds none
(prints `Done. 0 new video(s) rendered.`) or renders and prints
`[DRY-RUN] Would upload` — the same behaviour as before the extraction. Run
with `BLOTATO_API_KEY` unset so nothing posts.

- [ ] **Step 8: Commit**

```bash
git add truline-videos/src/broll.ts truline-videos/src/broll.test.ts \
        truline-videos/src/teams.ts truline-videos/src/blotato.ts \
        truline-videos/scripts/render-celebrations.ts
git commit -m "feat(video): b-roll fetch; extract shared team tables and Blotato upload

Both video scripts now share one upload path. Duplicating it would have
left two copies to change together."
```

---

### Task 8: Orchestrator

**Files:**
- Create: `truline-videos/scripts/render-pick-previews.ts`
- Create: `truline-videos/src/publish-guard.ts`, `truline-videos/src/publish-guard.test.ts`
- Modify: `truline-videos/package.json` (add `previews` script)

**Interfaces:**
- Consumes: `selectAdapter` (Task 5), `BeatClip` (Task 6), `fetchBroll` / `pickBrollQuery` / `teamColor` / `uploadToBlotato` / `blotatoConfigFromEnv` (Task 7), `GET /api/v1/mlb/video/pick-previews` (Task 4)
- Produces: `mayPublish(adapterPublishable: boolean, minutesToFirstPitch: number, minLeadMinutes?: number): boolean`

- [ ] **Step 1: Write the failing test**

```ts
// src/publish-guard.test.ts
import { describe, expect, it } from 'vitest';
import { mayPublish } from './publish-guard';

describe('mayPublish', () => {
  it('allows a publishable adapter with clearance', () => {
    expect(mayPublish(true, 120)).toBe(true);
  });

  it('refuses a say-narrated render however much time is left', () => {
    expect(mayPublish(false, 600)).toBe(false);
  });

  it('refuses inside the lead-time gate', () => {
    expect(mayPublish(true, 30)).toBe(false);
  });

  it('refuses after first pitch', () => {
    expect(mayPublish(true, -5)).toBe(false);
  });

  it('re-checks at upload time, not render time', () => {
    // A render that took 20 minutes leaves 40 — under the gate, so refuse.
    expect(mayPublish(true, 40)).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd truline-videos && npm test`
Expected: FAIL — cannot resolve `./publish-guard`

- [ ] **Step 3: Write the guard**

```ts
/** Minutes of clearance required before first pitch. Mirrors the backend's
 *  PREVIEW_MIN_LEAD_MINUTES — the backend gates selection, this gates upload,
 *  and rendering between them can take minutes. */
export const MIN_LEAD_MINUTES = 45;

export function mayPublish(
  adapterPublishable: boolean,
  minutesToFirstPitch: number,
  minLeadMinutes: number = MIN_LEAD_MINUTES,
): boolean {
  if (!adapterPublishable) return false;
  return minutesToFirstPitch > minLeadMinutes;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd truline-videos && npm test`
Expected: PASS, 21 tests total

- [ ] **Step 5: Write the orchestrator**

```ts
/**
 * Render and publish pre-game pick previews.
 *
 * Run manually: npx tsx scripts/render-pick-previews.ts
 *
 * Mirrors render-celebrations.ts, which is left untouched. Reuses its Blotato
 * upload path and rendered.json-style dedupe.
 */

import { config } from 'dotenv';
import axios from 'axios';
import { execSync } from 'child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { resolve } from 'path';

import { selectAdapter } from '../src/tts';
import { fetchBroll, pickBrollQuery } from '../src/broll';
import { teamColor } from '../src/teams';
import { blotatoConfigFromEnv, uploadToBlotato } from '../src/blotato';
import { mayPublish } from '../src/publish-guard';
import { FPS } from '../src/constants';
import type { BeatClip } from '../src/compositions/PickPreview';

config({ path: resolve(__dirname, '..', '.env') });

const API_BASE = 'https://nba-value-production.up.railway.app/api/v1';

const OUT_DIR = resolve(__dirname, '..', 'rendered', 'previews');
const AUDIO_DIR = resolve(OUT_DIR, 'audio');
const POSTED_FILE = resolve(__dirname, '..', 'previews-posted.json');

interface ApiBeat { key: string; narration: string; overlay: Record<string, string>; }
interface ApiPreview {
  game_id: string; game_date: string; game_time: string;
  team_abbr: string; team_name: string;
  logo_url: string; odds_american: number; beats: ApiBeat[];
}

const loadPosted = (): string[] =>
  existsSync(POSTED_FILE) ? JSON.parse(readFileSync(POSTED_FILE, 'utf-8')) : [];
const savePosted = (ids: string[]) =>
  writeFileSync(POSTED_FILE, JSON.stringify(ids, null, 2));

async function measure(path: string): Promise<number> {
  const { getAudioDurationInSeconds } = await import('@remotion/media-utils');
  return getAudioDurationInSeconds(path);
}

async function main() {
  mkdirSync(AUDIO_DIR, { recursive: true });
  const tts = selectAdapter(process.env);
  console.log(`TTS provider: ${tts.id} (publishable: ${tts.publishable})`);

  const posted = loadPosted();
  const resp = await axios.get(`${API_BASE}/mlb/video/pick-previews?days=1`, { timeout: 20000 });
  const previews: ApiPreview[] = resp.data.previews || [];
  console.log(`${previews.length} eligible pick(s)`);

  for (const preview of previews) {
    if (posted.includes(preview.game_id)) continue;

    // 1. narration, one clip per beat
    const beats: BeatClip[] = [];
    for (const [i, beat] of preview.beats.entries()) {
      const audioPath = resolve(AUDIO_DIR, `${preview.game_id}_${i}_${beat.key}.mp3`);
      if (!existsSync(audioPath)) await tts.synthesize(beat.narration, audioPath);
      const seconds = await measure(audioPath);
      beats.push({
        key: beat.key,
        overlay: beat.overlay,
        audioSrc: audioPath,
        // Half-second of air after each beat so it does not clip into the next.
        durationInFrames: Math.round((seconds + 0.5) * FPS),
      });
    }

    // 2. b-roll (optional — absence never blocks a render)
    const brollSrc = await fetchBroll(pickBrollQuery('mlb'), OUT_DIR, {
      get: (url, cfg) => axios.get(url, cfg as never),
      exists: existsSync,
      write: (p, b) => writeFileSync(p, b),
      apiKey: process.env.PEXELS_API_KEY,
    });

    // 3. render
    const outPath = resolve(OUT_DIR, `${preview.game_id}.mp4`);
    const propsFile = resolve(OUT_DIR, `${preview.game_id}_props.json`);
    writeFileSync(propsFile, JSON.stringify({
      beats, teamColor: teamColor(preview.team_abbr), logoUrl: preview.logo_url, brollSrc,
    }));
    execSync(
      `npx remotion render src/index.ts pick-preview "${outPath}" --props="${propsFile}"`,
      { cwd: resolve(__dirname, '..'), stdio: 'inherit', timeout: 300000 },
    );

    // 4. re-check the gate at UPLOAD time — rendering just consumed minutes
    const minutesLeft = (new Date(preview.game_time).getTime() - Date.now()) / 60000;
    if (!mayPublish(tts.publishable, minutesLeft)) {
      console.log(
        `SKIP upload ${preview.game_id}: publishable=${tts.publishable}, ` +
        `${minutesLeft.toFixed(0)}min to first pitch. Render kept at ${outPath}`,
      );
      continue;
    }

    const caption = [
      `${preview.team_name} ML ${preview.odds_american > 0 ? '+' : ''}${preview.odds_american}.`,
      '',
      preview.beats.find((b) => b.key === 'turn')?.narration || '',
      '',
      'Not betting advice. 21+.',
      '#MLB #SportsAnalytics',
    ].filter(Boolean).join('\n');

    await uploadToBlotato(outPath, caption, blotatoConfigFromEnv());
    posted.push(preview.game_id);
    savePosted(posted);
    console.log(`Posted: ${preview.game_id}`);
  }
}

main().catch(console.error);
```

Add to `package.json` `"scripts"`: `"previews": "tsx scripts/render-pick-previews.ts"`.

- [ ] **Step 6: End-to-end dry run**

With no `BLOTATO_API_KEY` exported, so upload is a dry run:

```bash
cd truline-videos && npx tsx scripts/render-pick-previews.ts
```

Expected: prints the TTS provider, renders at least one MP4 into
`rendered/previews/`, and logs `[DRY-RUN] Would upload`. Open the MP4 and
confirm: narration matches the on-screen text beat for beat, total runtime lands
in 30-45s, and the word "edge" appears nowhere.

- [ ] **Step 7: Commit**

```bash
git add truline-videos/scripts/render-pick-previews.ts truline-videos/src/publish-guard.ts truline-videos/src/publish-guard.test.ts truline-videos/package.json
git commit -m "feat(video): pick-preview orchestrator with upload-time lead gate"
```

---

## Deferred to a follow-up

- **Scheduling.** This plan produces a script run by hand. Wiring it to `mlb_scheduler` or cron is a separate change, and worth making only after a human has watched several renders end to end.
- **Word-level captions.** Per-beat graphics are already synced by construction.
- **CLV-explainer format.** The more differentiated content line, but its own pipeline.
