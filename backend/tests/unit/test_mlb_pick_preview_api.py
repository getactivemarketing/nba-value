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

import src.api.mlb as mlb_api
from src.api.mlb import (
    PickPreviewItem, _build_pick_preview, _usable_last_10,
    eligible_for_preview, get_pick_previews,
)
from src.services.mlb.pick_script import NarrationContractError


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
        # Only read by eligible_for_preview, which the slate tests below
        # exercise for real rather than stubbing out.
        self.best_bet_type = kw.get("best_bet_type", "moneyline")


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
        # Five starts, the MIN_TURN_BEAT_STARTS floor: below it the turn beat
        # is dropped as an unsupported sample regardless of this lookup.
        starts = [
            _GameRow(game_id="S1", game_date=date(2026, 6, 1),
                     home_starter_id=101, away_first_inning_runs=0),
            _GameRow(game_id="S2", game_date=date(2026, 6, 3),
                     home_starter_id=101, away_first_inning_runs=1),
            _GameRow(game_id="S3", game_date=date(2026, 6, 6),
                     home_starter_id=101, away_first_inning_runs=0),
            _GameRow(game_id="S4", game_date=date(2026, 6, 9),
                     home_starter_id=101, away_first_inning_runs=2),
            _GameRow(game_id="S5", game_date=date(2026, 6, 11),
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
        assert turn_beats[0].overlay["stat"] == "3 of 5"
        assert "3 of 5" in turn_beats[0].narration

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


# ---------------------------------------------------------------------------
# get_pick_previews — slate assembly
# ---------------------------------------------------------------------------

class _WindowSession:
    """Applies the endpoint's OWN compiled date window to a snapshot list.

    Reading the bind params off the statement rather than hardcoding a window
    is what makes the late-game test real: narrow the floor back to `today`
    and the yesterday-dated snapshot stops being returned, and the test fails.
    """

    def __init__(self, snapshots):
        self._snapshots = snapshots

    async def execute(self, stmt):
        params = stmt.compile().params
        low, high = params["game_date_1"], params["game_date_2"]
        return _Result([s for s in self._snapshots if low <= s.game_date <= high])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


def _install_slate(monkeypatch, snapshots, builder):
    monkeypatch.setattr(mlb_api, "async_session", lambda: _WindowSession(snapshots))
    monkeypatch.setattr(mlb_api, "_build_pick_preview", builder)


def _stub_item(snap) -> PickPreviewItem:
    return PickPreviewItem(
        game_id=snap.game_id, game_date=snap.game_date.isoformat(),
        game_time=snap.game_time.isoformat(), team_abbr="NYY", team_name="Yankees",
        logo_url="https://example/nyy.png", odds_american=145, beats=[],
    )


class TestLateGamesAreVisible:
    """CRITICAL 2 regression.

    game_date is MLB's officialDate (US LOCAL). date.today() on Railway is
    the UTC date. A 10:10pm ET first pitch is 02:10Z, so by the time its
    snapshot exists the UTC date has already rolled past its game_date and a
    floor at `today` deletes it — every West Coast game, every day.
    """

    async def test_a_yesterday_dated_game_two_hours_out_is_still_returned(self, monkeypatch):
        yesterday = date.today() - timedelta(days=1)
        snap = _SnapRow(game_id="LATE", game_date=yesterday, game_time=_in(120))

        _install_slate(monkeypatch, [snap],
                       lambda _s, sn: _async_value(_stub_item(sn)))
        result = await get_pick_previews(days=1)

        assert [p.game_id for p in result.previews] == ["LATE"]

    async def test_yesterdays_completed_game_is_still_excluded(self, monkeypatch):
        """The wider floor is only safe because eligible_for_preview rejects
        anything inside the lead-time gate. A game that already started is in
        the past and fails it, whatever its game_date says."""
        yesterday = date.today() - timedelta(days=1)
        snap = _SnapRow(game_id="DONE", game_date=yesterday, game_time=_in(-300))

        _install_slate(monkeypatch, [snap],
                       lambda _s, sn: _async_value(_stub_item(sn)))
        result = await get_pick_previews(days=1)

        assert result.previews == []

    async def test_beyond_the_horizon_is_excluded(self, monkeypatch):
        far = date.today() + timedelta(days=5)
        snap = _SnapRow(game_id="FAR", game_date=far, game_time=_in(120))

        _install_slate(monkeypatch, [snap],
                       lambda _s, sn: _async_value(_stub_item(sn)))
        result = await get_pick_previews(days=1)

        assert result.previews == []


async def _async_value(value):
    return value


class TestOneBadPickDoesNotZeroTheSlate:
    """CRITICAL 1 regression.

    build_beats fails CLOSED on banned copy, and the guard is a naive
    substring check — a starter named "Wedge" raises. Unguarded, that 500s
    the whole request and the orchestrator reports an empty slate,
    indistinguishable from "no picks today".
    """

    async def test_the_other_picks_still_ship(self, monkeypatch):
        today = date.today()
        snaps = [
            _SnapRow(game_id="A", game_date=today, game_time=_in(120)),
            _SnapRow(game_id="BAD", game_date=today, game_time=_in(130)),
            _SnapRow(game_id="C", game_date=today, game_time=_in(140)),
        ]

        async def builder(_session, snap):
            if snap.game_id == "BAD":
                raise NarrationContractError("Narration contains 'edge': Wedge carries...")
            return _stub_item(snap)

        _install_slate(monkeypatch, snaps, builder)
        result = await get_pick_previews(days=1)

        assert [p.game_id for p in result.previews] == ["A", "C"]

    async def test_the_refused_pick_never_appears_in_the_payload(self, monkeypatch):
        """Fail-closed is preserved: it is skipped, not degraded into a
        publishable preview with the banned copy stripped."""
        today = date.today()
        snaps = [_SnapRow(game_id="BAD", game_date=today, game_time=_in(120))]

        async def builder(_session, _snap):
            raise NarrationContractError("banned")

        _install_slate(monkeypatch, snaps, builder)
        result = await get_pick_previews(days=1)

        assert result.previews == []
        assert result.skipped == 1

    async def test_skipped_is_counted_and_reported(self, monkeypatch):
        today = date.today()
        snaps = [
            _SnapRow(game_id="A", game_date=today, game_time=_in(120)),
            _SnapRow(game_id="BAD", game_date=today, game_time=_in(130)),
        ]

        async def builder(_session, snap):
            if snap.game_id == "BAD":
                raise ValueError("a missing pitcher row, a None Decimal, anything")
            return _stub_item(snap)

        _install_slate(monkeypatch, snaps, builder)
        result = await get_pick_previews(days=1)

        assert len(result.previews) == 1
        assert result.skipped == 1

    async def test_an_unmeasurable_pick_counts_as_skipped_not_silence(self, monkeypatch):
        """_build_pick_preview returning None (no predicted_winner) is also a
        pick that was eligible and produced nothing. It must be visible."""
        today = date.today()
        snaps = [_SnapRow(game_id="A", game_date=today, game_time=_in(120))]

        _install_slate(monkeypatch, snaps, lambda _s, _sn: _async_value(None))
        result = await get_pick_previews(days=1)

        assert result.previews == []
        assert result.skipped == 1

    async def test_a_clean_slate_reports_zero_skipped(self, monkeypatch):
        today = date.today()
        snaps = [_SnapRow(game_id="A", game_date=today, game_time=_in(120))]

        _install_slate(monkeypatch, snaps,
                       lambda _s, sn: _async_value(_stub_item(sn)))
        result = await get_pick_previews(days=1)

        assert len(result.previews) == 1
        assert result.skipped == 0


class TestUsableLast10:
    """MINOR 10d. ingest writes f"{wins}-{losses}" with both defaulting to 0,
    so absent standings are stored as the truthy string "0-0" — which
    build_beats happily narrates as "They're 0-0 in their last ten."
    """

    def test_a_real_record_passes_through(self):
        assert _usable_last_10("6-4") == "6-4"

    def test_a_ten_zero_record_passes_through(self):
        assert _usable_last_10("10-0") == "10-0"

    def test_zero_zero_is_rejected_as_missing_standings(self):
        assert _usable_last_10("0-0") is None

    def test_none_is_rejected(self):
        assert _usable_last_10(None) is None

    def test_empty_string_is_rejected(self):
        assert _usable_last_10("") is None

    def test_a_malformed_value_is_rejected(self):
        for bad in ("None-None", "6", "6-", "-4", "6-4-2", "six-four", "6 - 4"):
            assert _usable_last_10(bad) is None, bad
