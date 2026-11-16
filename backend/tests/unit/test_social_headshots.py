import asyncio

import httpx

from src.edge.social import headshots

ROSTER = [{"displayName": "Pascal Siakam", "id": "3149673", "position": {"abbreviation": "PF"}},
          {"displayName": "Tyrese Haliburton", "id": "4396993"}]
TWO_JALENS = [{"displayName": "Jalen Williams", "id": "1"}, {"displayName": "Jalen Williams", "id": "2"}]


def test_exact_name_and_team_match():
    assert headshots.match("Pascal Siakam", "IND", ROSTER) == 3149673


def test_ambiguous_name_gets_no_headshot():
    assert headshots.match("Jalen Williams", "OKC", TWO_JALENS) is None


def test_no_match_returns_none():
    assert headshots.match("Nobody Here", "IND", ROSTER) is None


def test_alias_and_accents():
    assert headshots.match("Alex Sarr", "WAS", [{"displayName": "Alexandre Sarr", "id": "9"}]) == 9
    assert headshots.match("Nikola Jokić", "DEN", [{"displayName": "Nikola Jokic", "id": "5"}]) == 5


def test_nfl_grouped_roster_shape():
    payload = {"athletes": [{"position": "offense", "items": [{"displayName": "Josh Allen", "id": "3918298"}]},
                            {"position": "defense", "items": []}]}
    assert headshots.match("Josh Allen", "BUF", headshots._athletes(payload)) == 3918298


def test_espn_abbr_mapping():
    assert headshots.espn_abbr("nba", "GSW") == "gs" and headshots.espn_abbr("nba", "WAS") == "wsh"
    assert headshots.espn_abbr("nfl", "WAS") == "wsh" and headshots.espn_abbr("nfl", "JAX") == "jax"


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_fetch_logo_caches_and_swallows_errors():
    headshots._logo_cache.clear()
    calls = []

    def handler(req):
        calls.append(str(req.url))
        return httpx.Response(200, content=b"PNG") if "/gs.png" in str(req.url) else httpx.Response(404)

    async def run():
        async with _client(handler) as c:
            return (await headshots.fetch_logo("nba", "GSW", c), await headshots.fetch_logo("nba", "GSW", c),
                    await headshots.fetch_logo("nba", "BOS", c))

    a, b, c3 = asyncio.run(run())
    assert a == b == b"PNG" and c3 is None and len(calls) == 2


def test_fetch_logo_network_error_is_none():
    headshots._logo_cache.clear()

    def handler(req):
        raise httpx.ConnectError("down")

    async def run():
        async with _client(handler) as c:
            return await headshots.fetch_logo("nfl", "BUF", c)

    assert asyncio.run(run()) is None


def test_athletes_malformed_payloads_are_empty():
    for bad in ([], "x", None, {"athletes": "x"}, {"athletes": ["x", 3, None]}):
        assert headshots._athletes(bad) == []


class _Cur:
    def __init__(self, row): self.row = row
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, *a): pass
    def fetchone(self): return self.row


class _Conn:
    def __init__(self, row): self.row = row
    def cursor(self): return _Cur(self.row)


def test_fetch_png_network_error_and_non_png_and_team_mismatch():
    def boom(req): raise httpx.ConnectError("down")
    html = lambda req: httpx.Response(200, content=b"<html>nope</html>")
    good = lambda req: httpx.Response(200, content=headshots.PNG_MAGIC + b"x")

    async def run(handler, team="IND"):
        async with _client(handler) as c:
            return await headshots.fetch_png(_Conn((7, "IND")), "nba", "X Y", team, c)

    assert asyncio.run(run(boom)) is None
    assert asyncio.run(run(html)) is None
    assert asyncio.run(run(good, team="BOS")) is None
    assert asyncio.run(run(good)) == headshots.PNG_MAGIC + b"x"
