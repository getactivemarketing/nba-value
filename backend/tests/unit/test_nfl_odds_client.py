from src.services.nfl.odds_client import parse_nfl_odds_to_markets, NFL_TEAM_NAME_TO_ABBR


def _event():
    # Minimal Odds API v4 shape: one game, one book, three markets.
    return [{
        "id": "evt1", "commence_time": "2026-09-13T17:00:00Z",
        "home_team": "Kansas City Chiefs", "away_team": "Cincinnati Bengals",
        "bookmakers": [{"key": "draftkings", "markets": [
            {"key": "h2h", "outcomes": [
                {"name": "Kansas City Chiefs", "price": 1.62},
                {"name": "Cincinnati Bengals", "price": 2.40}]},
            {"key": "spreads", "outcomes": [
                {"name": "Kansas City Chiefs", "price": 1.91, "point": -3.5},
                {"name": "Cincinnati Bengals", "price": 1.91, "point": 3.5}]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": 1.91, "point": 48.5},
                {"name": "Under", "price": 1.91, "point": 48.5}]},
        ]}]}]


def test_all_32_teams_mapped():
    assert len(NFL_TEAM_NAME_TO_ABBR) == 32
    assert NFL_TEAM_NAME_TO_ABBR["Kansas City Chiefs"] == "KC"
    assert NFL_TEAM_NAME_TO_ABBR["Las Vegas Raiders"] == "LV"


def test_parse_yields_three_markets_with_correct_convention():
    rows = parse_nfl_odds_to_markets(_event(), NFL_TEAM_NAME_TO_ABBR)
    by_type = {r["market_type"]: r for r in rows}
    assert set(by_type) == {"spread", "moneyline", "total"}
    # home spread stored as HOME-FAVORED POSITIVE: KC is -3.5, so home favored by 3.5 -> line = 3.5
    assert by_type["spread"]["line"] == 3.5
    assert round(by_type["spread"]["home_odds"], 2) == 1.91
    assert by_type["total"]["line"] == 48.5
    assert round(by_type["total"]["over_odds"], 2) == 1.91
    assert round(by_type["moneyline"]["home_odds"], 2) == 1.62
    assert by_type["spread"]["home_team_abbr"] == "KC"
    assert by_type["spread"]["away_team_abbr"] == "CIN"
