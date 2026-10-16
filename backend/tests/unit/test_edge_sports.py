# backend/tests/unit/test_edge_sports.py
import json
from datetime import datetime, timezone

from src.edge import views
from src.edge.config import (NBA, NFL, NFL_PROP_MARKETS, PROP_MARKETS, SPORTS, enabled_sports,
                             sport_by_key, state_key)
from src.edge.ingest import parse_event

FIX = "tests/fixtures/edge"


def test_sports_table():
    assert SPORTS == {"nba": NBA, "nfl": NFL}
    assert NBA.key == "basketball_nba" and NBA.lines_window_hours == 36 and NBA.has_players
    assert NFL.key == "americanfootball_nfl" and NFL.lines_window_hours == 168 and NFL.has_players
    assert PROP_MARKETS == NBA.prop_markets  # legacy alias unchanged
    assert len(NFL_PROP_MARKETS) == 10 and "player_anytime_td" not in NFL_PROP_MARKETS
    assert sport_by_key("americanfootball_nfl") is NFL and sport_by_key("icehockey_nhl") is None


def test_enabled_sports_ignores_unknown_and_blank():
    assert enabled_sports("nba,nfl") == [NBA, NFL]
    assert enabled_sports(" nfl , bogus,,") == [NFL]


def test_state_keys_keep_nba_names():
    assert state_key(NBA, "last_events_refresh") == "last_events_refresh"
    assert state_key(NFL, "last_events_refresh") == "nfl:last_events_refresh"


def test_every_nfl_market_has_a_label_and_parses_from_the_real_fixture():
    ev = json.load(open(f"{FIX}/nfl_props.json"))
    quotes = parse_event(ev, observed_at=datetime(2026, 10, 16, tzinfo=timezone.utc))
    markets = {q.market for q in quotes}
    assert set(NFL_PROP_MARKETS) <= markets  # all 10 present in the CHI @ ATL sample
    for m in NFL_PROP_MARKETS:
        assert m in views.MARKET_LABEL
    q = next(q for q in quotes if q.market == "player_rush_yds" and q.side == "over")
    assert views.describe(q.market, q.player, q.side, q.line, "Atlanta Falcons", "Chicago Bears") == \
        f"{q.player} Over {q.line:g} Rush Yds"


def test_enabled_sports_is_case_insensitive_and_never_empty(caplog):
    assert enabled_sports("NBA, Nfl") == [NBA, NFL]
    # A config typo must not silently stop all polling: fall back to NBA.
    assert enabled_sports("nbaa") == [NBA]
    assert enabled_sports("") == [NBA]
    assert "falling back" in caplog.text
