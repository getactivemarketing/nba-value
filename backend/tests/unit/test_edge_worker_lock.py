import asyncio

import pytest

from src.edge import store, worker


class _Stop(Exception):
    pass


class _Conn:
    closed = 0

    def rollback(self):
        pass


def _run_ticks(monkeypatch, lock_results):
    """Run the worker loop for len(lock_results) iterations; return tick count."""
    ticks, locks = [], iter(lock_results)
    monkeypatch.setattr(worker, "EdgeOddsClient", lambda: object())
    monkeypatch.setattr(worker.players, "BdlClient", lambda: object())
    monkeypatch.setattr(store, "connect", lambda: _Conn())
    monkeypatch.setattr(store, "try_worker_lock", lambda conn: next(locks))
    monkeypatch.setattr(store, "set_state", lambda *a, **k: None)

    async def tick(conn, client, bdl, now):
        ticks.append(now)
        return {}

    sleeps = {"n": 0}

    async def sleep(_):
        sleeps["n"] += 1
        if sleeps["n"] >= len(lock_results):
            raise _Stop

    monkeypatch.setattr(worker, "tick", tick)
    monkeypatch.setattr(worker.asyncio, "sleep", sleep)
    with pytest.raises(_Stop):
        asyncio.run(worker._run())
    return len(ticks)


def test_standby_worker_never_ticks(monkeypatch):
    assert _run_ticks(monkeypatch, [False, False, False]) == 0


def test_worker_ticks_once_it_takes_the_lock(monkeypatch):
    # The lock is held for the session, so it is only requested until granted.
    assert _run_ticks(monkeypatch, [False, True]) == 1


def test_reconnect_closes_the_old_session_so_its_lock_is_released(monkeypatch):
    conns = []

    class _BadConn(_Conn):
        def __init__(self):
            self.closed_by_worker = False
            conns.append(self)

        def rollback(self):
            raise RuntimeError("connection is wedged")

        def close(self):
            self.closed_by_worker = True

    monkeypatch.setattr(worker, "EdgeOddsClient", lambda: object())
    monkeypatch.setattr(worker.players, "BdlClient", lambda: object())
    monkeypatch.setattr(store, "connect", _BadConn)
    monkeypatch.setattr(store, "try_worker_lock", lambda conn: True)
    monkeypatch.setattr(store, "set_state", lambda *a, **k: None)

    async def tick(*a):
        raise ValueError("tick blew up")

    async def sleep(_):
        raise _Stop

    monkeypatch.setattr(worker, "tick", tick)
    monkeypatch.setattr(worker.asyncio, "sleep", sleep)
    with pytest.raises(_Stop):
        asyncio.run(worker._run())
    assert len(conns) == 2 and conns[0].closed_by_worker
