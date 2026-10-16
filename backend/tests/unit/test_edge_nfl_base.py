# backend/tests/unit/test_edge_nfl_base.py
import json
from datetime import date

from src.database import Base
import src.models  # noqa: F401
from src.edge.teams import NFL_ABBR_BY_NAME, NFLVERSE_TO_BDL
from src.utils.season import current_nfl_season

FIX = "tests/fixtures/edge"


def test_nfl_team_names_cover_bdl_and_the_odds_api():
    bdl = {t["full_name"]: t["abbreviation"] for t in json.load(open(f"{FIX}/bdl_nfl_teams.json"))["data"]}
    assert NFL_ABBR_BY_NAME == bdl and len(bdl) == 32
    for e in json.load(open(f"{FIX}/nfl_events.json")):
        assert e["home_team"] in NFL_ABBR_BY_NAME and e["away_team"] in NFL_ABBR_BY_NAME


def test_nflverse_codes_map_onto_bdl():
    bdl = set(NFL_ABBR_BY_NAME.values())
    nflverse = (bdl - {"LAR", "WSH"}) | {"LA", "WAS"}
    assert {NFLVERSE_TO_BDL.get(t, t) for t in nflverse} == bdl


def test_current_nfl_season():
    assert current_nfl_season(date(2026, 10, 16)) == 2026
    assert current_nfl_season(date(2027, 1, 20)) == 2026   # playoffs
    assert current_nfl_season(date(2027, 3, 1)) == 2027


def test_nfl_tables_are_registered():
    names = set(Base.metadata.tables)
    assert {"edge_nfl_players", "edge_nfl_player_games", "edge_nfl_snaps", "edge_nfl_games"} <= names
    pk = [c.name for c in Base.metadata.tables["edge_nfl_players"].primary_key.columns]
    assert pk == ["odds_name", "team"]
