"""Pick-alert job hardening: per-row commit + per-row exception isolation.

A mid-batch failure must neither lose already-confirmed sends (flag must be
committed per row, or a crash re-texts the founder next cycle) nor stop the
remaining rows from being attempted.
"""

import asyncio
from types import SimpleNamespace

import src.tasks.social_scheduler as ss


def _snap(sid):
    return SimpleNamespace(
        id=sid, away_team="DET", home_team="LAA", best_bet_type="moneyline",
        best_bet_team="DET", best_bet_line=None, best_bet_odds=None,
        best_bet_value_score=80, best_bet_edge=None, game_time=None,
        away_starter_name=None, home_starter_name=None,
    )


class FakeResult:
    def __init__(self, snaps):
        self._snaps = snaps

    def scalars(self):
        return self

    def all(self):
        return self._snaps


class FakeSession:
    def __init__(self, snaps):
        self._snaps = snaps
        self.commits = 0
        self.updated_ids = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def execute(self, stmt, params=None):
        if params and "sid" in params:
            self.updated_ids.append(params["sid"])
            return None
        return FakeResult(self._snaps)

    async def commit(self):
        self.commits += 1


def _run_job_with(monkeypatch, snaps, fmt=None, send=None):
    session = FakeSession(snaps)
    monkeypatch.setattr(ss, "_social_session_factory", lambda: session)
    import src.services.notifications.pick_alerts as pa
    import src.services.notifications.sms as sms_mod
    monkeypatch.setattr(pa, "format_pick_alert", fmt or (lambda **kw: "msg"))
    monkeypatch.setattr(sms_mod, "send_sms", send or (lambda body: True))
    result = asyncio.run(ss._post_pick_alerts_async())
    return session, result


def test_flag_committed_per_successful_send(monkeypatch):
    session, result = _run_job_with(monkeypatch, [_snap(1), _snap(2)])
    assert result == {"sent": 2, "skipped": 0, "type": "pick_alerts"}
    assert session.updated_ids == [1, 2]
    # One commit per confirmed send — not a single end-of-batch commit
    assert session.commits == 2


def test_one_bad_row_does_not_stop_the_batch(monkeypatch):
    def fmt(**kw):
        if kw["value_score"] == 66:
            raise ValueError("boom")
        return "msg"

    bad = _snap(2)
    bad.best_bet_value_score = 66
    session, result = _run_job_with(monkeypatch, [_snap(1), bad, _snap(3)], fmt=fmt)
    assert result == {"sent": 2, "skipped": 1, "type": "pick_alerts"}
    assert session.updated_ids == [1, 3]
    assert session.commits == 2


def test_failed_send_neither_flags_nor_commits(monkeypatch):
    session, result = _run_job_with(monkeypatch, [_snap(1)], send=lambda body: False)
    assert result == {"sent": 0, "skipped": 1, "type": "pick_alerts"}
    assert session.updated_ids == []
    assert session.commits == 0


class FlakyCommitSession(FakeSession):
    """commit() raises once (simulated transient DB failure), then recovers."""

    def __init__(self, snaps):
        super().__init__(snaps)
        self.rollbacks = 0
        self._fail_next_commit = True

    async def commit(self):
        if self._fail_next_commit:
            self._fail_next_commit = False
            raise RuntimeError("transient db failure")
        await super().commit()

    async def rollback(self):
        self.rollbacks += 1


def test_db_failure_rolls_back_and_batch_continues(monkeypatch):
    session = FlakyCommitSession([_snap(1), _snap(2)])
    monkeypatch.setattr(ss, "_social_session_factory", lambda: session)
    import src.services.notifications.pick_alerts as pa
    import src.services.notifications.sms as sms_mod
    monkeypatch.setattr(pa, "format_pick_alert", lambda **kw: "msg")
    monkeypatch.setattr(sms_mod, "send_sms", lambda body: True)
    result = asyncio.run(ss._post_pick_alerts_async())
    # Row 1's commit blew up -> rolled back, counted skipped, row 2 unaffected
    assert result == {"sent": 1, "skipped": 1, "type": "pick_alerts"}
    assert session.rollbacks == 1
    assert session.commits == 1
