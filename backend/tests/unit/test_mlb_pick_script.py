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
