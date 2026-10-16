# backend/tests/unit/test_edge_ingest.py
import json
from datetime import datetime, timezone

import httpx
import pytest

from src.edge import client as client_mod
from src.edge.ingest import event_row, parse_event

NOW = datetime(2026, 10, 16, 2, 10, tzinfo=timezone.utc)
LINES = json.load(open("tests/fixtures/edge/lines.json"))
PROPS = json.load(open("tests/fixtures/edge/props.json"))


def test_event_row():
    row = event_row(LINES[0])
    assert row == {"event_id": "26b036ff107f3c658258eaf4a6f26228", "sport": "basketball_nba",
                   "home_team": "Detroit Pistons", "away_team": "Boston Celtics",
                   "commence_time": datetime(2026, 10, 20, 19, 0, tzinfo=timezone.utc)}


def test_h2h_maps_team_names_to_home_away():
    quotes = [x for x in parse_event(LINES[0], NOW) if x.market == "h2h" and x.book == "fanduel"]
    by_side = {x.side: x for x in quotes}
    assert by_side["away"].price == 100 and by_side["home"].price == -118
    assert by_side["home"].line == 0.0 and by_side["home"].player == ""
    assert by_side["home"].link.startswith("https://sportsbook.fanduel.com/addToBetslip")


def test_spreads_store_home_perspective_line_on_both_sides():
    quotes = [x for x in parse_event(LINES[0], NOW) if x.market == "spreads" and x.book == "fanduel"]
    assert len(quotes) == 2
    assert quotes[0].line == quotes[1].line  # same market key for both sides


def test_props_parse_player_side_and_point():
    quotes = parse_event(PROPS, NOW)
    tatum = [x for x in quotes if x.player == "Jayson Tatum" and x.market == "player_assists"
             and x.book == "fanduel"]
    assert {(x.side, x.line, x.price) for x in tatum} >= {("over", 5.5, -138), ("under", 5.5, 104)}
    assert all(x.observed_at == NOW for x in quotes)


def test_parse_skips_malformed_outcomes():
    bad = {**PROPS, "bookmakers": [{"key": "fanduel", "markets": [
        {"key": "player_points", "outcomes": [{"name": "Over", "price": -110}]}]}]}  # no description/point
    assert parse_event(bad, NOW) == []


@pytest.mark.asyncio
async def test_client_errors_are_redacted(monkeypatch):
    def handler(request):
        return httpx.Response(401, json={"message": "bad key"})

    transport = httpx.MockTransport(handler)
    c = client_mod.EdgeOddsClient(api_key="secretsecret1234", transport=transport)
    with pytest.raises(client_mod.OddsApiError) as exc:
        await c.get_events()
    assert exc.value.status == 401
    assert "secretsecret1234" not in str(exc.value)


@pytest.mark.asyncio
async def test_client_reads_usage_headers():
    def handler(request):
        assert request.url.params["bookmakers"].startswith("pinnacle,novig")
        return httpx.Response(200, json=[], headers={"x-requests-remaining": "4999000",
                                                     "x-requests-used": "1000", "x-requests-last": "6"})

    c = client_mod.EdgeOddsClient(api_key="k" * 16, transport=httpx.MockTransport(handler))
    data, usage = await c.get_game_odds()
    assert data == [] and usage == client_mod.Usage(4999000, 1000, 6)
