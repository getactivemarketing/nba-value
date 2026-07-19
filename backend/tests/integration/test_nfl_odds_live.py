import pytest
from src.services.nfl.odds_client import NFLOddsClient, parse_nfl_odds_to_markets, NFL_TEAM_NAME_TO_ABBR


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_nfl_odds_shape():
    events = await NFLOddsClient().get_nfl_odds()
    # Out of season this is legitimately empty — assert it parses without error either way.
    rows = parse_nfl_odds_to_markets(events, NFL_TEAM_NAME_TO_ABBR)
    for r in rows:
        assert r["market_type"] in {"spread", "moneyline", "total"}
        assert "home_team_abbr" in r
