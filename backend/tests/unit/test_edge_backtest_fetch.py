import asyncio
from datetime import datetime, timezone

import pytest

from src.edge import store
from src.edge.backtest import fetch
from src.edge.backtest.nflverse import Game
from src.edge.backtest import plan
from src.edge.client import OddsApiError, Usage
from src.edge.config import TD_MARKET

UTC = timezone.utc
GAME = Game("2023_10_CLE_BAL", 2023, 10, "BAL", "CLE", datetime(2023, 11, 12, 18, 0, tzinfo=UTC), 31, 33)
EVENT = {"id": "e1", "sport_key": "americanfootball_nfl", "home_team": "Baltimore Ravens",
         "away_team": "Cleveland Browns", "commence_time": "2023-11-12T18:03:00Z"}


class FakeRepo:
    def __init__(self):
        self.rows = []

    def done_keys(self):
        return {(r["event_id"], r["kind"], r["label"]) for r in self.rows}

    def spent(self):
        return sum(r["credits"] for r in self.rows)

    def events_payload(self, day_key):
        return next((r["payload"] for r in self.rows if r["event_id"] == day_key), None)

    def insert(self, rows):
        have = self.done_keys()
        self.rows += [r for r in rows if (r["event_id"], r["kind"], r["label"]) not in have]


class FakeClient:
    def __init__(self, events=(EVENT,), fail=None, remaining=4_000_000, lines_events=("e1",)):
        self.events, self.fail, self.remaining, self.lines_events = list(events), fail or {}, remaining, lines_events
        self.calls = []
        self.markets = []

    def _usage(self, cost):
        self.remaining -= cost
        return Usage(self.remaining, None, cost)

    async def get_historical_events(self, sport, date):
        self.calls.append(("events", date))
        return {"timestamp": date, "data": self.events}, self._usage(1)

    async def get_historical_game_odds(self, sport, date):
        self.calls.append(("lines", date))
        if ("lines", date) in self.fail:
            raise self.fail[("lines", date)]
        data = [{**EVENT, "id": e, "bookmakers": [{"key": "pinnacle", "markets": []}]} for e in self.lines_events]
        return {"timestamp": date, "data": data}, self._usage(60)

    async def get_historical_event_odds(self, event_id, sport, date, markets=None):
        self.markets.append(markets)
        kind = "td" if markets == (TD_MARKET,) else "props"
        self.calls.append((kind, date))
        if (kind, date) in self.fail:
            raise self.fail[(kind, date)]
        return {"timestamp": date, "data": {**EVENT, "bookmakers": [{"key": "pinnacle", "markets": []}]}}, \
            self._usage(20 if kind == "td" else 200)


async def _nosleep(_):
    return None


def run(repo, client, cap=800_000, min_remaining=3_600_000, **kw):
    return asyncio.run(fetch.run_fetch(repo, client, [GAME], cap=cap, min_remaining=min_remaining,
                                       sleep=_nosleep, **kw))


def test_fetch_stores_every_snapshot_once_and_resume_spends_nothing():
    repo, client = FakeRepo(), FakeClient()
    s = run(repo, client)
    kinds = sorted((r["kind"], r["label"]) for r in repo.rows)
    assert kinds.count(("events", "list")) == 1
    assert sorted(l for k, l in kinds if k == "props") == ["close", "t1h", "t24h", "t6h"]
    assert sorted(l for k, l in kinds if k == "lines") == ["close", "t1h", "t24h", "t6h"]
    assert s.credits == repo.spent() == 1 + 4 * 200 + 4 * 60 and s.stopped is None
    props_close = next(r for r in repo.rows if r["kind"] == "props" and r["label"] == "close")
    assert props_close["requested_at"] == datetime(2023, 11, 12, 17, 58, tzinfo=UTC)  # event kickoff 18:03
    assert props_close["game_id"] == "2023_10_CLE_BAL" and props_close["home"] == "BAL"
    assert props_close["snapshot_at"] == datetime(2023, 11, 12, 17, 58, tzinfo=UTC)
    client2 = FakeClient()
    s2 = run(repo, client2)
    assert client2.calls == [] and s2.credits == 0


def test_cap_stops_before_the_call_that_would_exceed_it():
    repo, client = FakeRepo(), FakeClient()
    s = run(repo, client, cap=1 + 200 + 60 + 100)
    assert s.stopped.startswith("credit cap")
    assert repo.spent() <= 1 + 200 + 60 + 100


def test_min_remaining_floor_stops_the_run():
    repo, client = FakeRepo(), FakeClient(remaining=3_600_050)
    s = run(repo, client)
    assert s.stopped.startswith("credits remaining")
    assert len(client.calls) == 2  # events leaves 3,600,049; the 60-credit lines call drops below the floor


def test_429_stops_and_stores_nothing_for_that_job():
    t24 = "2023-11-11T18:03:00Z"
    repo, client = FakeRepo(), FakeClient(fail={("props", t24): OddsApiError(429, "slow down")})
    s = run(repo, client)
    assert s.stopped == "odds_api 429"
    assert ("e1", "props", "t24h") not in repo.done_keys()


def test_other_errors_skip_the_job_and_the_next_run_retries_it():
    t6 = "2023-11-12T12:03:00Z"
    repo = FakeRepo()
    s = run(repo, FakeClient(fail={("props", t6): OddsApiError(500, "boom")}))
    assert s.errors == 1 and s.stopped is None
    assert ("e1", "props", "t6h") not in repo.done_keys()
    client = FakeClient()
    run(repo, client)
    assert client.calls == [("props", t6)]


def test_404_stores_an_empty_stub_so_it_is_not_bought_again():
    t1 = "2023-11-12T17:03:00Z"
    repo = FakeRepo()
    run(repo, FakeClient(fail={("props", t1): OddsApiError(404, "not found")}))
    stub = next(r for r in repo.rows if r["kind"] == "props" and r["label"] == "t1h")
    assert stub["credits"] == 0 and stub["payload"]["bookmakers"] == [] and stub["snapshot_at"] is None


def test_event_missing_from_a_lines_response_gets_a_stub_and_credits_land_once():
    other = Game("2023_10_IND_NE", 2023, 10, "NE", "IND", datetime(2023, 11, 12, 18, 5, tzinfo=UTC), 10, 6)
    ev2 = {**EVENT, "id": "e2", "home_team": "New England Patriots", "away_team": "Indianapolis Colts",
           "commence_time": "2023-11-12T18:05:00Z"}
    repo, client = FakeRepo(), FakeClient(events=(EVENT, ev2), lines_events=("e1",))
    asyncio.run(fetch.run_fetch(repo, client, [GAME, other], cap=800_000, min_remaining=0, sleep=_nosleep))
    lines = [r for r in repo.rows if r["kind"] == "lines" and r["label"] == "close"]
    assert {r["event_id"] for r in lines} == {"e1", "e2"}
    assert sorted(r["credits"] for r in lines) == [0, 60]
    assert next(r for r in lines if r["event_id"] == "e2")["payload"]["bookmakers"] == []


def test_unmatched_games_are_reported_and_not_fetched():
    repo, client = FakeRepo(), FakeClient(events=())
    s = run(repo, client)
    assert s.unmatched == ["2023_10_CLE_BAL"]
    assert [c[0] for c in client.calls] == ["events"]


def test_fetch_never_writes_live_worker_state(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("live state touched")
    monkeypatch.setattr(store, "log_api_call", boom)
    monkeypatch.setattr(store, "set_state", boom)
    s = run(FakeRepo(), FakeClient())
    assert s.stopped is None


class MalformedClient(FakeClient):
    async def get_historical_event_odds(self, event_id, sport, date, markets=None):
        if date == "2023-11-12T12:03:00Z":
            self.calls.append(("props", date))
            return {"timestamp": date, "data": None}, self._usage(200)
        return await super().get_historical_event_odds(event_id, sport, date, markets)


def test_malformed_paid_response_is_stored_as_a_stub_and_never_rebought():
    repo = FakeRepo()
    s = run(repo, MalformedClient())
    assert s.stopped is None and s.errors == 1
    stub = next(r for r in repo.rows if r["kind"] == "props" and r["label"] == "t6h")
    assert stub["credits"] == 200 and stub["payload"]["bookmakers"] == [] and stub["snapshot_at"] is None
    assert len([r for r in repo.rows if r["kind"] == "props"]) == 4
    client = FakeClient()
    run(repo, client)
    assert client.calls == []


def test_kickoff_comes_from_the_games_own_day_list_not_an_earlier_stale_one():
    thu = Game("2023_10_PIT_CIN", 2023, 10, "CIN", "PIT", datetime(2023, 11, 10, 1, 15, tzinfo=UTC), 1, 2)
    stale = {**EVENT, "commence_time": "2023-11-12T20:25:00Z"}  # Thursday's list: before a move to 18:03
    other = {**EVENT, "id": "t1", "home_team": "Cincinnati Bengals", "away_team": "Pittsburgh Steelers",
             "commence_time": "2023-11-10T01:15:00Z"}

    class DayClient(FakeClient):
        async def get_historical_events(self, sport, date):
            self.calls.append(("events", date))
            data = [other, stale] if date.startswith("2023-11-09") else [EVENT]
            return {"timestamp": date, "data": data}, self._usage(1)

    repo = FakeRepo()
    asyncio.run(fetch.run_fetch(repo, DayClient(), [thu, GAME], cap=800_000, min_remaining=0, sleep=_nosleep))
    close = next(r for r in repo.rows if r["event_id"] == "e1" and r["kind"] == "props" and r["label"] == "close")
    assert close["kickoff"] == datetime(2023, 11, 12, 18, 3, tzinfo=UTC)


def test_fetch_td_kind_requests_only_the_td_market_and_never_rebuys_props():
    repo, client = FakeRepo(), FakeClient()
    run(repo, client)
    assert set(client.markets) == {plan.BT_PROP_MARKETS}  # props never buy TD
    props_before = [r for r in repo.rows if r["kind"] == "props"]
    client2 = FakeClient()
    s = run(repo, client2, kinds=("td",))
    assert client2.markets == [(TD_MARKET,)] * 4 and [c[0] for c in client2.calls if c[0] != "events"] == ["td"] * 4
    td = [r for r in repo.rows if r["kind"] == "td"]
    assert sorted(r["label"] for r in td) == ["close", "t1h", "t24h", "t6h"]
    assert all(r["credits"] == plan.TD_COST and r["game_id"] == GAME.game_id for r in td)
    assert [r for r in repo.rows if r["kind"] == "props"] == props_before
    assert s.credits == 4 * plan.TD_COST
    client3 = FakeClient()
    assert run(repo, client3, kinds=("td",)).credits == 0 and client3.calls == []


def test_stored_td_rows_do_not_block_props_and_a_td_404_is_a_stub():
    repo = FakeRepo()
    client = FakeClient(fail={("td", "2023-11-12T17:58:00Z"): OddsApiError(404, "gone")})
    run(repo, client, kinds=("td",))
    stub = next(r for r in repo.rows if r["kind"] == "td" and r["label"] == "close")
    assert stub["payload"]["missing"] == 404
    client2 = FakeClient()
    run(repo, client2)
    assert sum(1 for c in client2.calls if c[0] == "props") == 4
