from datetime import datetime, timezone

from src.edge.backtest import __main__ as cli
from src.edge.backtest import nflverse
from src.edge.backtest.nflverse import Game

UTC = timezone.utc
GAMES = [Game("a", 2023, 10, "BAL", "CLE", datetime(2023, 11, 12, 18, 0, tzinfo=UTC), 31, 33),
         Game("b", 2023, 11, "NE", "IND", datetime(2023, 11, 19, 18, 0, tzinfo=UTC), 10, 6)]


def test_default_report_path_is_backend_docs():
    assert cli.DEFAULT_OUT.as_posix().endswith("backend/docs/backtests/nfl-2023-2025.md")


def test_dry_run_prints_cost_and_never_connects(monkeypatch, capsys):
    monkeypatch.setattr(nflverse, "load_schedule", lambda seasons: GAMES)
    monkeypatch.setattr(cli.store, "connect", lambda: (_ for _ in ()).throw(AssertionError("connected")))
    assert cli.main(["fetch", "--season", "2023", "--week", "10", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "games: 1" in out and "credits: 1,041" in out


def test_fetch_reconnects_and_resumes_after_a_dropped_db_connection(monkeypatch):
    import psycopg2

    from src.edge.backtest import fetch
    from src.edge.backtest.fetch import Summary

    class Conn:
        def close(self):
            pass

    calls = []

    async def flaky(repo, client, games, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise psycopg2.OperationalError("server closed the connection unexpectedly")
        return Summary(calls=3, credits=600)

    monkeypatch.setattr(nflverse, "load_schedule", lambda seasons: GAMES)
    monkeypatch.setattr(cli, "get_sync_database_url", lambda: "postgresql://x")
    monkeypatch.setattr(fetch, "ensure_tables", lambda url: None)
    monkeypatch.setattr(fetch, "run_fetch", flaky)
    monkeypatch.setattr(cli.store, "connect", lambda: Conn())
    monkeypatch.setattr(cli, "RETRY_WAIT", 0)
    assert cli.main(["fetch", "--season", "2023"]) == 0
    assert len(calls) == 2
