# backend/src/edge/social/headshots.py
"""ESPN player headshots and team logos for social cards. Table: edge_headshots.

A headshot is used only when exactly one roster entry on the player's team matches the odds name;
anything ambiguous or missing stores espn_id NULL so the card draws a silhouette (never a wrong face).
All network failures are logged and swallowed.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx

from src.edge.players import NAME_ALIASES, normalize_name
from src.services.social.image_generator import NBA_ESPN_MAP

ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/{path}/teams/{abbr}/roster"
HEADSHOT_URL = "https://a.espncdn.com/i/headshots/{league}/players/full/{espn_id}.png"
LOGO_URL = "https://a.espncdn.com/i/teamlogos/{league}/500/{abbr}.png"
TIMEOUT = 10.0
RECHECK = timedelta(days=7)
PATHS = {"nba": "basketball/nba", "nfl": "football/nfl"}
NFL_ESPN_EXCEPTIONS = {"WAS": "wsh"}  # every other BallDontLie NFL abbreviation is ESPN's, lowercased
log = logging.getLogger(__name__)
_logo_cache: dict[tuple[str, str], bytes] = {}


def espn_abbr(sport: str, abbr: str) -> str | None:
    a = abbr.upper()
    if sport == "nba":
        return NBA_ESPN_MAP.get(a)
    if sport == "nfl":
        return NFL_ESPN_EXCEPTIONS.get(a, a.lower())
    return None


def _athletes(payload: dict) -> list[dict]:
    """Flat list (NBA) or groups with an `items` list (NFL)."""
    out: list[dict] = []
    for a in payload.get("athletes") or []:
        if isinstance(a, dict) and isinstance(a.get("items"), list):
            out.extend(x for x in a["items"] if isinstance(x, dict))
        elif isinstance(a, dict):
            out.append(a)
    return out


def match(odds_name: str, team: str, roster: list[dict]) -> int | None:
    """ESPN id when exactly one roster entry has this name, else None. `roster` is already the team's."""
    want = normalize_name(odds_name)
    want = NAME_ALIASES.get(want, want)
    ids = set()
    for a in roster:
        got = normalize_name(a.get("displayName") or a.get("fullName") or "")
        if got and got == want and a.get("id") is not None:
            try:
                ids.add(int(a["id"]))
            except (TypeError, ValueError):
                continue
    # two entries with the same name, even with different ids, is ambiguous
    count = sum(1 for a in roster if normalize_name(a.get("displayName") or a.get("fullName") or "") == want)
    return next(iter(ids)) if len(ids) == 1 and count == 1 else None


async def _get(http: httpx.AsyncClient, url: str) -> httpx.Response | None:
    try:
        r = await http.get(url, timeout=TIMEOUT)
    except httpx.HTTPError as e:
        log.warning("ESPN fetch failed %s: %s", url, type(e).__name__)
        return None
    if r.status_code != 200:
        log.warning("ESPN %s on %s", r.status_code, url)
        return None
    return r


def _odds_names(conn, sport: str, teams: set[str]) -> dict[str, set[str]]:
    sql = ("SELECT odds_name, team FROM edge_players WHERE team = ANY(%s)" if sport == "nba"
           else "SELECT odds_name, team FROM edge_nfl_players WHERE team = ANY(%s)")
    with conn.cursor() as cur:
        cur.execute(sql, (sorted(teams),))
        rows = cur.fetchall()
    out: dict[str, set[str]] = {}
    for name, team in rows:
        out.setdefault(team, set()).add(name)
    return out


async def refresh(conn, sport: str, teams: set[str], now: datetime, http: httpx.AsyncClient) -> int:
    """Re-resolve ESPN ids for teams whose rows are missing or older than 7 days. Returns rows written."""
    path = PATHS.get(sport)
    if path is None or not teams:
        return 0
    with conn.cursor() as cur:
        cur.execute("SELECT team FROM edge_headshots WHERE sport = %s AND team = ANY(%s) GROUP BY team "
                    "HAVING MIN(checked_at) >= %s", (sport, sorted(teams), now - RECHECK))
        fresh = {r[0] for r in cur.fetchall()}
    todo = teams - fresh
    names = _odds_names(conn, sport, todo) if todo else {}
    written = 0
    for team in sorted(todo):
        if not names.get(team):
            continue
        abbr = espn_abbr(sport, team)
        r = await _get(http, ROSTER_URL.format(path=path, abbr=abbr)) if abbr else None
        if r is None:
            continue
        try:
            roster = _athletes(r.json())
        except ValueError:
            log.warning("ESPN roster for %s %s is not JSON", sport, team)
            continue
        if not roster:
            continue
        rows = [(sport, n, team, match(n, team, roster), now) for n in sorted(names[team])]
        with conn.cursor() as cur:
            cur.executemany("""INSERT INTO edge_headshots (sport, odds_name, team, espn_id, checked_at)
                               VALUES (%s, %s, %s, %s, %s)
                               ON CONFLICT (sport, odds_name) DO UPDATE
                                 SET team = EXCLUDED.team, espn_id = EXCLUDED.espn_id, checked_at = EXCLUDED.checked_at""",
                            rows)
        conn.commit()
        written += len(rows)
    return written


async def fetch_png(conn, sport: str, odds_name: str, http: httpx.AsyncClient) -> bytes | None:
    """Headshot bytes, or None (no id, non-200, network error) so the card falls back to a silhouette."""
    with conn.cursor() as cur:
        cur.execute("SELECT espn_id FROM edge_headshots WHERE sport = %s AND odds_name = %s", (sport, odds_name))
        row = cur.fetchone()
    if not row or row[0] is None or sport not in PATHS:
        return None
    r = await _get(http, HEADSHOT_URL.format(league=sport, espn_id=row[0]))
    return r.content if r is not None else None


async def fetch_logo(sport: str, abbr: str, http: httpx.AsyncClient) -> bytes | None:
    key = (sport, abbr.upper())
    if key in _logo_cache:
        return _logo_cache[key]
    e = espn_abbr(sport, abbr)
    if e is None or sport not in PATHS:
        return None
    r = await _get(http, LOGO_URL.format(league=sport, abbr=e))
    if r is None:
        return None
    _logo_cache[key] = r.content
    return r.content
