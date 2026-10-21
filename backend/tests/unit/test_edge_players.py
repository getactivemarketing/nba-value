# backend/tests/unit/test_edge_players.py
from datetime import date, datetime, timezone

import pytest

from src.edge import players as p

CANDS = [
    {"id": 246, "first_name": "Nikola", "last_name": "Jokić", "team_abbr": "DEN"},
    {"id": 3547269, "first_name": "Jaren", "last_name": "Jackson Jr.", "team_abbr": "MEM"},
    {"id": 1, "first_name": "Jalen", "last_name": "Williams", "team_abbr": "OKC"},
    {"id": 2, "first_name": "Jalen", "last_name": "Williams", "team_abbr": "DEN"},  # different player
]


def test_normalize_name_strips_accents_suffixes_and_punctuation():
    assert p.normalize_name("Nikola Jokić") == "nikola jokic"
    assert p.normalize_name("Jaren Jackson Jr.") == "jaren jackson"
    assert p.normalize_name("De'Aaron Fox") == "deaaron fox"
    assert p.normalize_name("Shai Gilgeous-Alexander") == "shai gilgeous alexander"


def test_exact_and_normalized_matches():
    m = p.match_player("Nikola Jokic", CANDS, {"DEN", "LAL"})
    assert m and m.bdl_player_id == 246 and m.method == "normalized"
    m = p.match_player("Jaren Jackson Jr", CANDS, {"MEM", "BOS"})
    assert m and m.bdl_player_id == 3547269


def test_same_name_resolved_by_event_team():
    assert p.match_player("Jalen Williams", CANDS, {"OKC", "SAS"}).bdl_player_id == 1
    assert p.match_player("Jalen Williams", CANDS, {"DEN", "UTA"}).bdl_player_id == 2


def test_no_match_when_team_disagrees_or_too_far():
    assert p.match_player("Nikola Jokic", CANDS, {"BOS", "DET"}) is None
    assert p.match_player("Nikolai Jokinen", CANDS, {"DEN", "LAL"}) is None


def test_known_nickname_aliases_match_the_bdl_legal_name():
    cands = [{"id": 1028028405, "first_name": "Alexandre", "last_name": "Sarr", "team_abbr": "WAS"},
             {"id": 17896024, "first_name": "Herbert", "last_name": "Jones", "team_abbr": "NOP"},
             {"id": 1028026508, "first_name": "Ronald", "last_name": "Holland II", "team_abbr": "DET"},
             {"id": 1057260888, "first_name": "Airious", "last_name": "Bailey", "team_abbr": "UTA"}]
    teams = {"WAS", "NOP", "DET", "UTA"}
    for odds_name, pid in [("Alex Sarr", 1028028405), ("Herb Jones", 17896024),
                           ("Ron Holland", 1028026508), ("Ace Bailey", 1057260888)]:
        m = p.match_player(odds_name, cands, teams)
        assert m and m.bdl_player_id == pid and m.method == "manual", odds_name
    assert p.match_player("Ace Bailey", cands, {"BOS", "NYK"}) is None  # alias still needs a team in the event


def test_parse_minutes():
    assert p.parse_minutes("34:30") == pytest.approx(34.5)
    assert p.parse_minutes("28") == 28.0
    assert p.parse_minutes(None) == 0.0 and p.parse_minutes("") == 0.0


def test_box_row():
    stat = {"player": {"id": 434}, "team": {"id": 2, "abbreviation": "BOS"},
            "game": {"id": 99, "date": "2026-04-10", "season": 2025, "home_team_id": 2, "visitor_team_id": 9},
            "min": "36", "pts": 31, "reb": 8, "ast": 6, "fg3m": 4, "stl": 1, "blk": 0, "turnover": 3}
    row = p.box_row(stat, {2: "BOS", 9: "DET"})
    assert row == {"bdl_player_id": 434, "bdl_game_id": 99, "season": 2025, "game_date": date(2026, 4, 10),
                   "team": "BOS", "opponent": "DET", "is_home": True, "minutes": 36.0,
                   "pts": 31, "reb": 8, "ast": 6, "fg3m": 4, "stl": 1, "blk": 0, "tov": 3}


def g(d, pts, reb=5, ast=5, mins=30.0, opp="DET", home=True, season=2026):
    return {"game_date": d, "pts": pts, "reb": reb, "ast": ast, "fg3m": 2, "minutes": mins,
            "opponent": opp, "is_home": home, "season": season}


def test_stat_for_combos():
    row = g(date(2026, 11, 1), 20, reb=7, ast=3)
    assert p.stat_for("player_points_rebounds_assists", row) == 30
    assert p.stat_for("player_rebounds_assists", row) == 10
    assert p.stat_for("player_threes", row) == 2


def test_hit_rates_windows_pushes_and_dnp():
    games = [g(date(2026, 11, 10 - i), pts) for i, pts in enumerate([30, 25, 20, 28, 10, 40])]
    games.append(g(date(2026, 11, 3), 0, mins=0.0))  # DNP: excluded
    r = p.hit_rates(games, "player_points", 25.0, opponent="DET", is_home=True, season=2026)
    assert r["l5"] == {"hits": 2, "misses": 2, "pushes": 1, "games": 5, "rate": 0.5}
    assert r["l10"]["games"] == 6
    assert r["vs_opponent"]["games"] == 6 and r["home_away"]["games"] == 6


def test_min_minutes_filter():
    games = [g(date(2026, 11, 2), 30, mins=12.0), g(date(2026, 11, 1), 30, mins=34.0)]
    assert p.hit_rates(games, "player_points", 25.0, None, None, 2026, min_minutes=20)["l5"]["games"] == 1


def test_et_date():
    assert p.et_date(datetime(2026, 10, 21, 2, 30, tzinfo=timezone.utc)) == date(2026, 10, 20)


# --- Roster-based resolution (BallDontLie rate limits: free 5/min, ALL-STAR 60/min) ---

import httpx  # noqa: E402

from src.edge.quotes import Quote  # noqa: E402


class _Cur:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.sql.append(sql)

    def fetchall(self):
        return []


class _Conn:
    def __init__(self):
        self.sql = []

    def cursor(self, *a, **k):
        return _Cur(self)

    def commit(self):
        pass


EVENT = {"event_id": "e1", "home_team": "Detroit Pistons", "away_team": "Boston Celtics"}
NOW = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)


def _q(name):
    return Quote("e1", "player_points", name, "over", 20.5, "fanduel", -110, None, NOW)


@pytest.mark.asyncio
async def test_roster_fetch_is_two_calls_per_event_and_cached():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.path.endswith("/teams"):
            return httpx.Response(200, json={"data": [{"id": 2, "abbreviation": "BOS"},
                                                      {"id": 9, "abbreviation": "DET"}]})
        tid = request.url.params["team_ids[]"]
        abbr = {"2": "BOS", "9": "DET"}[tid]
        return httpx.Response(200, json={"data": [{"id": int(tid) * 100, "first_name": "P", "last_name": tid,
                                                   "team": {"abbreviation": abbr}}], "meta": {}})

    bdl = p.BdlClient(api_key="k", transport=httpx.MockTransport(handler))
    first = await bdl.roster("BOS")
    again = await bdl.roster("BOS")
    assert first == again == [{"id": 200, "first_name": "P", "last_name": "2", "team_abbr": "BOS"}]
    assert sum("/players" in c for c in calls) == 1  # cached


@pytest.mark.asyncio
async def test_roster_is_the_active_roster_not_every_player_in_franchise_history():
    """BallDontLie's /players?team_ids[] returns everyone whose last team was the franchise (260 names
    for BOS, Bill Russell included); a fuzzy match could land on a retired player. /players/active is
    the real roster (2026-10-18)."""
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/teams"):
            return httpx.Response(200, json={"data": [{"id": 2, "abbreviation": "BOS"}]})
        everyone = [{"id": 1, "first_name": "Bill", "last_name": "Russell", "team": {"abbreviation": "BOS"}}]
        active = [{"id": 434, "first_name": "Jayson", "last_name": "Tatum", "team": {"abbreviation": "BOS"}}]
        data = active if request.url.path.endswith("/players/active") else everyone + active
        return httpx.Response(200, json={"data": data, "meta": {}})

    bdl = p.BdlClient(api_key="k", transport=httpx.MockTransport(handler))
    roster = await bdl.roster("BOS")
    assert [r["last_name"] for r in roster] == ["Tatum"]
    assert any(path.endswith("/players/active") for path in paths)


@pytest.mark.asyncio
async def test_resolve_survives_rate_limits():
    def handler(request):
        return httpx.Response(429, json={"error": "rate limited"})

    bdl = p.BdlClient(api_key="k", transport=httpx.MockTransport(handler))
    conn = _Conn()
    n = await p.resolve_new_players(conn, bdl, [_q("Jayson Tatum")], EVENT, NOW)
    assert n == 0  # nothing written, nothing raised; names retry on the next poll
    assert not any("INSERT" in s for s in conn.sql)


@pytest.mark.asyncio
async def test_bdl_network_errors_become_bdl_errors():
    def handler(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    bdl = p.BdlClient(api_key="k", transport=httpx.MockTransport(handler), min_interval=0)
    with pytest.raises(p.BdlError) as exc:
        await bdl.teams()
    assert exc.value.status == 0
