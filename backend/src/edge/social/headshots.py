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


def _athletes(payload) -> list[dict]:
    """Flat list (NBA) or groups with an `items` list (NFL). Anything malformed yields fewer/no entries."""
    if not isinstance(payload, dict) or not isinstance(payload.get("athletes"), list):
        return []
    out: list[dict] = []
    for a in payload["athletes"]:
        if not isinstance(a, dict):
            continue
        if isinstance(a.get("items"), list):
            out.extend(x for x in a["items"] if isinstance(x, dict))
        else:
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


def _odds_names(conn, sport: str, teams: set[str]) -> tuple[dict[str, set[str]], set[str]]:
    """({team: odds names}, names that appear on more than one team in the whole league table)."""
    table = "edge_players" if sport == "nba" else "edge_nfl_players"
    with conn.cursor() as cur:
        cur.execute(f"SELECT odds_name, team FROM {table} WHERE team = ANY(%s)", (sorted(teams),))
        rows = cur.fetchall()
        cur.execute(f"SELECT odds_name FROM {table} GROUP BY odds_name HAVING COUNT(DISTINCT team) > 1")
        dupes = {r[0] for r in cur.fetchall()}
    out: dict[str, set[str]] = {}
    for name, team in rows:
        out.setdefault(team, set()).add(name)
    return out, dupes


async def refresh(conn, sport: str, teams: set[str], now: datetime, http: httpx.AsyncClient) -> int:
    """Re-resolve ESPN ids for teams with a missing or >7-day-old row. Returns rows written.

    Names shared by players on two teams are stored with espn_id NULL (never risk a wrong face).
    """
    path = PATHS.get(sport)
    if path is None or not teams:
        return 0
    names, dupes = _odds_names(conn, sport, teams)
    with conn.cursor() as cur:
        cur.execute("SELECT odds_name, team, checked_at FROM edge_headshots WHERE sport = %s AND team = ANY(%s)",
                    (sport, sorted(teams)))
        have = {(n, t): c for n, t, c in cur.fetchall()}
    written = 0
    for team in sorted(names):
        if all(have.get((n, team)) is not None and have[(n, team)] >= now - RECHECK for n in names[team]):
            continue
        abbr = espn_abbr(sport, team)
        r = await _get(http, ROSTER_URL.format(path=path, abbr=abbr)) if abbr else None
        if r is None:
            continue
        try:
            roster = _athletes(r.json())
        except (ValueError, AttributeError, TypeError):
            log.warning("ESPN roster for %s %s is malformed", sport, team)
            continue
        if not roster:
            continue
        rows = [(sport, n, team, None if n in dupes else match(n, team, roster), now) for n in sorted(names[team])]
        with conn.cursor() as cur:
            cur.executemany("""INSERT INTO edge_headshots (sport, odds_name, team, espn_id, checked_at)
                               VALUES (%s, %s, %s, %s, %s)
                               ON CONFLICT (sport, odds_name) DO UPDATE
                                 SET team = EXCLUDED.team, espn_id = EXCLUDED.espn_id, checked_at = EXCLUDED.checked_at""",
                            rows)
        conn.commit()
        written += len(rows)
    return written


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


async def fetch_png(conn, sport: str, odds_name: str, team: str, http: httpx.AsyncClient) -> bytes | None:
    """Headshot bytes, or None (no id, team mismatch, non-200, non-PNG, network error) -> silhouette."""
    with conn.cursor() as cur:
        cur.execute("SELECT espn_id, team FROM edge_headshots WHERE sport = %s AND odds_name = %s", (sport, odds_name))
        row = cur.fetchone()
    if not row or row[0] is None or row[1] != team or sport not in PATHS:
        return None
    r = await _get(http, HEADSHOT_URL.format(league=sport, espn_id=row[0]))
    if r is None or not r.content.startswith(PNG_MAGIC):
        return None
    return r.content


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
