from datetime import datetime, timedelta, timezone

import pytest

from src.edge import matchups, nfl_injuries, store, worker


class _Conn:
    rolled = 0

    def rollback(self):
        self.rolled += 1


@pytest.fixture
def state(monkeypatch):
    st = {}
    monkeypatch.setattr(store, "get_state", lambda conn: dict(st))
    monkeypatch.setattr(store, "set_state", lambda conn, k, v, now: st.__setitem__(k, v))
    return st


def _iso(dt):
    return dt.isoformat()


NOW = datetime(2026, 10, 18, 15, 0, tzinfo=timezone.utc)  # Sunday


async def test_matchups_step_runs_once_per_24h(monkeypatch, state):
    calls = []
    monkeypatch.setattr(matchups, "aggregate", lambda pbp, rosters: ([], []))
    monkeypatch.setattr(worker.matchups_store, "upsert", lambda c, t, p, now: calls.append(now) or 0)
    await worker.nfl_matchups_step(_Conn(), NOW, frames=(None, None))
    assert state["nfl:matchups_ok_at"] == _iso(NOW) and state["matchups_error"] == ""
    await worker.nfl_matchups_step(_Conn(), NOW + timedelta(hours=23), frames=(None, None))
    assert len(calls) == 1
    await worker.nfl_matchups_step(_Conn(), NOW + timedelta(hours=24), frames=(None, None))
    assert len(calls) == 2


async def test_matchups_step_error_sets_state_and_does_not_raise(monkeypatch, state):
    def boom(pbp, rosters):
        raise RuntimeError("bad parquet")
    monkeypatch.setattr(matchups, "aggregate", boom)
    conn = _Conn()
    assert await worker.nfl_matchups_step(conn, NOW, frames=(None, None)) == 0
    assert "bad parquet" in state["matchups_error"] and conn.rolled == 1
    assert "nfl:matchups_ok_at" not in state


class _Bdl:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    async def player_injuries(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("bdl down")
        return []


@pytest.fixture
def stored(monkeypatch):
    calls = []
    monkeypatch.setattr(nfl_injuries, "store_live", lambda c, rows, now: calls.append(now) or 0)
    return calls


async def test_injuries_step_cadence_2h_thursday_to_monday_6h_otherwise(state, stored):
    bdl = _Bdl()
    await worker.nfl_injuries_step(_Conn(), bdl, NOW)           # Sunday
    assert bdl.calls == 1 and state["nfl:injuries_ok_at"] == _iso(NOW)
    await worker.nfl_injuries_step(_Conn(), bdl, NOW + timedelta(hours=1, minutes=59))
    assert bdl.calls == 1
    await worker.nfl_injuries_step(_Conn(), bdl, NOW + timedelta(hours=2))
    assert bdl.calls == 2
    tue = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)
    state["nfl:last_injuries_step"] = _iso(tue)
    await worker.nfl_injuries_step(_Conn(), bdl, tue + timedelta(hours=2))
    assert bdl.calls == 2
    await worker.nfl_injuries_step(_Conn(), bdl, tue + timedelta(hours=6))
    assert bdl.calls == 3


async def test_injuries_step_error_sets_state_and_does_not_raise(state, stored):
    conn = _Conn()
    assert await worker.nfl_injuries_step(conn, _Bdl(fail=True), NOW) == 0
    assert "bdl down" in state["injuries_error"] and conn.rolled == 1
    assert "nfl:injuries_ok_at" not in state


async def test_matchups_step_timeout_sets_error_and_does_not_raise(monkeypatch, state):
    import time
    monkeypatch.setattr(worker, "MATCHUPS_TIMEOUT_S", 0.05, raising=False)
    monkeypatch.setattr(matchups, "aggregate", lambda pbp, rosters: time.sleep(0.5) or ([], []))
    conn = _Conn()
    assert await worker.nfl_matchups_step(conn, NOW, frames=(None, None)) == 0
    assert "Timeout" in state["matchups_error"] and "nfl:matchups_ok_at" not in state


async def test_matchups_failure_retries_after_an_hour_not_a_day(monkeypatch, state):
    calls = []

    def boom(pbp, rosters):
        calls.append(1)
        raise RuntimeError("x")
    monkeypatch.setattr(matchups, "aggregate", boom)
    await worker.nfl_matchups_step(_Conn(), NOW, frames=(None, None))
    await worker.nfl_matchups_step(_Conn(), NOW + timedelta(minutes=30), frames=(None, None))
    assert len(calls) == 1
    await worker.nfl_matchups_step(_Conn(), NOW + timedelta(minutes=61), frames=(None, None))
    assert len(calls) == 2


async def test_success_clears_previous_errors(monkeypatch, state, stored):
    state["matchups_error"] = state["injuries_error"] = "x"
    monkeypatch.setattr(matchups, "aggregate", lambda pbp, rosters: ([], []))
    monkeypatch.setattr(worker.matchups_store, "upsert", lambda c, t, p, now: 0)
    await worker.nfl_matchups_step(_Conn(), NOW, frames=(None, None))
    await worker.nfl_injuries_step(_Conn(), _Bdl(), NOW)
    assert state["matchups_error"] == "" and state["injuries_error"] == ""
