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
    MIN_TURN_BEAT_STARTS, Beat, PickPayload, breakeven_prob, build_beats,
    NarrationContractError,
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


class TestCaseAgainstGrammar:
    """All 7 non-empty subsets of optional case_against fields.

    The first included clause must always stand on its own as a complete
    sentence with a subject, capitalized and capable of being spoken aloud.
    """

    def test_form_only(self):
        """last_10_record present, streak and starter absent."""
        case = next(b for b in build_beats(
            payload(streak=None, starter_name=None, starter_era=None)
        ) if b.key == "case_against")
        assert case.narration == "They're 5-5 in their last ten."
        assert case.narration[0].isupper()

    def test_streak_only(self):
        """Streak present, form and starter absent."""
        case = next(b for b in build_beats(
            payload(last_10_record=None, starter_name=None, starter_era=None)
        ) if b.key == "case_against")
        assert case.narration == "They're on a 2-game losing streak."
        assert case.narration[0].isupper()

    def test_starter_only(self):
        """Starter present, form and streak absent."""
        case = next(b for b in build_beats(
            payload(last_10_record=None, streak=None)
        ) if b.key == "case_against")
        assert case.narration == "Castillo carries a 5.06 ERA."
        assert case.narration[0].isupper()

    def test_form_and_streak(self):
        """Form and streak present, starter absent."""
        case = next(b for b in build_beats(
            payload(starter_name=None, starter_era=None)
        ) if b.key == "case_against")
        assert case.narration == "They're 5-5 in their last ten, on a 2-game losing streak."
        assert case.narration[0].isupper()

    def test_form_and_starter(self):
        """Form and starter present, streak absent."""
        case = next(b for b in build_beats(
            payload(streak=None)
        ) if b.key == "case_against")
        assert case.narration == "They're 5-5 in their last ten, Castillo carries a 5.06 ERA."
        assert case.narration[0].isupper()

    def test_streak_and_starter(self):
        """Streak and starter present, form absent."""
        case = next(b for b in build_beats(
            payload(last_10_record=None)
        ) if b.key == "case_against")
        assert case.narration == "They're on a 2-game losing streak, Castillo carries a 5.06 ERA."
        assert case.narration[0].isupper()

    def test_all_three_present(self):
        """Form, streak, and starter all present."""
        case = next(b for b in build_beats(payload()) if b.key == "case_against")
        assert case.narration == "They're 5-5 in their last ten, on a 2-game losing streak, Castillo carries a 5.06 ERA."
        assert case.narration[0].isupper()


class TestCaseAgainstAdmitsOnlyLosingStreaks:
    """The case-against beat must never argue FOR the pick.

    Its whole job is the strongest reason not to bet, and the turn beat that
    follows opens with "But" — which only parses after a negative. A winning
    streak is an argument for the pick, so it is excluded outright rather
    than reworded. These mirror the subsets in TestCaseAgainstGrammar with
    the streak flipped to "won".
    """

    def test_form_and_winning_streak_narrates_form_only(self):
        case = next(b for b in build_beats(payload(
            streak=Streak("won", 4), starter_name=None, starter_era=None,
        )) if b.key == "case_against")
        assert case.narration == "They're 5-5 in their last ten."
        assert "winning" not in case.narration.lower()
        assert "streak" not in case.narration.lower()

    def test_winning_streak_only_drops_the_beat_entirely(self):
        """Nothing else to say against the pick, so there is no beat at all —
        rather than a beat that reads as a reason to take it."""
        keys = [b.key for b in build_beats(payload(
            last_10_record=None, streak=Streak("won", 4),
            starter_name=None, starter_era=None,
        ))]
        assert "case_against" not in keys
        assert "numbers" in keys  # the rest of the video survives

    def test_winning_streak_and_starter_narrates_starter_only(self):
        case = next(b for b in build_beats(payload(
            last_10_record=None, streak=Streak("won", 4),
        )) if b.key == "case_against")
        assert case.narration == "Castillo carries a 5.06 ERA."
        assert case.narration[0].isupper()

    def test_all_three_with_a_winning_streak_omits_the_streak(self):
        case = next(b for b in build_beats(payload(streak=Streak("won", 4))
                                           ) if b.key == "case_against")
        assert case.narration == "They're 5-5 in their last ten, Castillo carries a 5.06 ERA."
        assert "streak" not in case.narration.lower()

    def test_winning_streak_never_reaches_the_overlay_chips(self):
        case = next(b for b in build_beats(payload(streak=Streak("won", 4))
                                           ) if b.key == "case_against")
        assert "winning" not in case.overlay["chips"].lower()

    def test_a_losing_streak_is_still_admitted(self):
        """Guard against the fix over-correcting into dropping all streaks."""
        case = next(b for b in build_beats(payload(streak=Streak("lost", 4))
                                           ) if b.key == "case_against")
        assert "4-game losing streak" in case.narration


class TestTurnBeatSampleFloor:
    """The turn beat is the video's centrepiece evidence, so it needs a real
    sample behind it. One-of-one is a coin flip published with odds attached.
    """

    def test_the_floor_is_five_starts(self):
        assert MIN_TURN_BEAT_STARTS == 5

    def test_one_of_one_is_dropped(self):
        keys = [b.key for b in build_beats(payload(first_inning=FirstInningSplit(1, 1)))]
        assert "turn" not in keys

    def test_four_starts_is_dropped(self):
        keys = [b.key for b in build_beats(payload(first_inning=FirstInningSplit(3, 4)))]
        assert "turn" not in keys
        assert "numbers" in keys  # the rest of the video survives

    def test_five_starts_is_kept(self):
        beats = build_beats(payload(first_inning=FirstInningSplit(3, 5)))
        turn = next(b for b in beats if b.key == "turn")
        assert "3 of 5" in turn.narration
        assert turn.overlay["stat"] == "3 of 5"

    def test_a_perfect_but_tiny_sample_is_still_dropped(self):
        """2-of-2 is the most tempting number in the dataset and the least
        supported. It must not ship."""
        keys = [b.key for b in build_beats(payload(first_inning=FirstInningSplit(2, 2)))]
        assert "turn" not in keys


class TestNarrationContractGuard:
    """The 'edge' ban is enforced at the guard, not just in templates."""

    def test_normal_payload_does_not_raise(self):
        """A payload with clean fields should not raise."""
        beats = build_beats(payload())
        assert len(beats) == 6

    def test_team_name_with_edge_raises(self):
        """Payload with 'Edge' in team_name should raise NarrationContractError."""
        with pytest.raises(NarrationContractError):
            build_beats(payload(team_name="Edge City"))

    def test_starter_name_with_edge_raises(self):
        """Payload with 'edge' in starter_name should raise NarrationContractError."""
        with pytest.raises(NarrationContractError):
            build_beats(payload(starter_name="Wedge"))
