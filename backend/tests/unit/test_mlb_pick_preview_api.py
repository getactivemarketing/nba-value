"""The pick-preview endpoint's selection rules and per-pick assembly.

Selection is where the disabled markets are mechanically enforced. Runline is
paused and totals suppressed; publishing them would advertise markets we have
deliberately turned off.

`_build_pick_preview` is tested here with a stubbed session because two
production bugs lived in it with zero coverage: the turn beat depended on the
previewed game already being final (it never is — eligible_for_preview only
admits games ≥45 minutes out, so the game is always still "scheduled"), and
the model probability was used unattributed (winner_probability is
P(predicted_winner), not P(best_ml_team), and the two differ whenever the
value pick isn't the model's favourite).
"""
from datetime import date, datetime, timedelta, timezone

import pytest

from src.api.mlb import _build_pick_preview, eligible_for_preview


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


# ---------------------------------------------------------------------------
# _build_pick_preview — stubbed session
# ---------------------------------------------------------------------------

class _SnapRow:
    """Duck-typed stand-in for an MLBPredictionSnapshot row."""

    def __init__(self, **kw):
        self.game_id = kw.get("game_id", "G1")
        self.game_date = kw.get("game_date", date(2026, 6, 15))
        self.game_time = kw.get(
            "game_time", datetime(2026, 6, 15, 23, 5, tzinfo=timezone.utc)
        )
        self.home_team = kw.get("home_team", "NYY")
        self.away_team = kw.get("away_team", "BOS")
        self.home_starter_name = kw.get("home_starter_name", "Gerrit Cole")
        self.away_starter_name = kw.get("away_starter_name", "Brayan Bello")
        self.home_starter_era = kw.get("home_starter_era", 3.10)
        self.away_starter_era = kw.get("away_starter_era", 4.20)
        self.predicted_winner = kw.get("predicted_winner", "NYY")
        self.winner_probability = kw.get("winner_probability", 0.55)
        self.best_ml_team = kw.get("best_ml_team", "NYY")
        self.best_ml_odds = kw.get("best_ml_odds", 1.91)


class _GameRow:
    """Duck-typed stand-in for an MLBGame row."""

    def __init__(self, **kw):
        self.game_id = kw.get("game_id", "G1")
        self.game_date = kw.get("game_date", date(2026, 6, 15))
        self.status = kw.get("status", "scheduled")
        self.home_team = kw.get("home_team", "NYY")
        self.away_team = kw.get("away_team", "BOS")
        self.home_starter_id = kw.get("home_starter_id")
        self.away_starter_id = kw.get("away_starter_id")
        self.home_score = kw.get("home_score")
        self.away_score = kw.get("away_score")
        self.home_first_inning_runs = kw.get("home_first_inning_runs")
        self.away_first_inning_runs = kw.get("away_first_inning_runs")


class _StatsRow:
    def __init__(self, last_10_record=None):
        self.last_10_record = last_10_record


class _Scalars:
    def __init__(self, items):
        self._items = list(items)

    def all(self):
        return list(self._items)

    def first(self):
        return self._items[0] if self._items else None


class _Result:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return _Scalars(self._items)


class _FakeSession:
    """Returns queued result sets in the exact order `_build_pick_preview`
    issues its queries: games (final, season-bounded) -> game_row (by
    game_id, unconditional on status) -> [starts, only if a starter_id was
    found] -> team stats.
    """

    def __init__(self, *result_sets):
        self._queue = list(result_sets)

    async def execute(self, _stmt):
        return _Result(self._queue.pop(0))


class TestBuildPickPreview:
    async def test_scheduled_current_game_still_produces_a_turn_beat(self):
        """CRITICAL 1 regression: the previewed game is always still
        'scheduled' (eligible_for_preview only admits games ≥45 minutes from
        first pitch), so the starter id must be resolved from a direct
        game_id lookup, not from the final-games-only list used for streak.
        """
        snap = _SnapRow(home_team="NYY", away_team="BOS", best_ml_team="NYY",
                         predicted_winner="NYY", winner_probability=0.55)
        game_row = _GameRow(game_id="G1", status="scheduled",
                             home_team="NYY", away_team="BOS",
                             home_starter_id=101)
        starts = [
            _GameRow(game_id="S1", game_date=date(2026, 6, 1),
                     home_starter_id=101, away_first_inning_runs=0),
            _GameRow(game_id="S2", game_date=date(2026, 6, 6),
                     home_starter_id=101, away_first_inning_runs=1),
            _GameRow(game_id="S3", game_date=date(2026, 6, 11),
                     home_starter_id=101, away_first_inning_runs=0),
        ]
        session = _FakeSession(
            [],          # games (final) for streak — none needed
            [game_row],  # game_row lookup by game_id
            starts,      # starter appearances
            [],          # team stats — none
        )

        preview = await _build_pick_preview(session, snap)

        assert preview is not None
        turn_beats = [b for b in preview.beats if b.key == "turn"]
        assert len(turn_beats) == 1
        assert turn_beats[0].overlay["stat"] == "2 of 3"
        assert "2 of 3" in turn_beats[0].narration

    async def test_model_prob_is_inverted_when_backed_team_is_not_the_favourite(self):
        """CRITICAL 2 regression: winner_probability is P(predicted_winner).
        When the value pick is the OTHER team, the narrated projection must
        be 1 - winner_probability, not winner_probability verbatim.
        """
        snap = _SnapRow(home_team="PHI", away_team="WSH",
                         best_ml_team="WSH", predicted_winner="PHI",
                         winner_probability=0.528)
        session = _FakeSession(
            [],  # games (final) for streak
            [],  # game_row lookup — none found, so no starter/turn beat
            [],  # team stats
        )

        preview = await _build_pick_preview(session, snap)

        assert preview is not None
        numbers = next(b for b in preview.beats if b.key == "numbers")
        # 1 - 0.528 = 0.472 -> "47%"
        assert numbers.overlay["model"] == "47%"
        assert "47%" in numbers.narration

    async def test_model_prob_is_unchanged_when_backed_team_is_the_favourite(self):
        """Counterpart to the inversion test: when best_ml_team IS the
        predicted winner, winner_probability is used as-is.
        """
        snap = _SnapRow(home_team="PHI", away_team="WSH",
                         best_ml_team="PHI", predicted_winner="PHI",
                         winner_probability=0.528)
        session = _FakeSession(
            [],  # games (final) for streak
            [],  # game_row lookup — none found, so no starter/turn beat
            [],  # team stats
        )

        preview = await _build_pick_preview(session, snap)

        assert preview is not None
        numbers = next(b for b in preview.beats if b.key == "numbers")
        assert numbers.overlay["model"] == "53%"
        assert "53%" in numbers.narration

    async def test_null_predicted_winner_is_unmeasurable_not_guessed(self):
        snap = _SnapRow(predicted_winner=None)
        session = _FakeSession()  # no queries should even be issued

        preview = await _build_pick_preview(session, snap)

        assert preview is None
