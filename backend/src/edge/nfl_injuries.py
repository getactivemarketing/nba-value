# backend/src/edge/nfl_injuries.py
"""NFL injury status: nflverse weekly history (backtest) and a BallDontLie live snapshot.

Both sides produce team (BallDontLie abbreviation) -> {player_key: "out"|"doubtful"|"questionable"}.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from psycopg2.extras import execute_values

from src.edge.nfl_players import BdlNflClient
from src.edge.players import normalize_name
from src.edge.teams import NFLVERSE_TO_BDL

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/"
IR_TYPES = frozenset({"IR", "IR-R", "PUP-R", "NFI-R", "Reserve-Sus"})
STALE = timedelta(days=14)
_STATUS = {"Out": "out", "Doubtful": "doubtful", "Questionable": "questionable", **{s: "out" for s in IR_TYPES}}
_SEVERITY = {"questionable": 1, "doubtful": 2, "out": 3}


def normalize_status(raw: str) -> str | None:
    return _STATUS.get(raw.strip()) if isinstance(raw, str) else None


def _worse(a: str | None, b: str) -> str:
    return b if a is None or _SEVERITY[b] > _SEVERITY[a] else a


def load_history(seasons: list[int]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f"{NFLVERSE}injuries/injuries_{s}.parquet") for s in seasons],
                     ignore_index=True)


def history_index(frame) -> dict[tuple[int, int], dict[str, dict[str, str]]]:
    out: dict[tuple[int, int], dict[str, dict[str, str]]] = {}
    for r in frame.itertuples(index=False):
        if r.game_type != "REG" or not isinstance(r.full_name, str):
            continue
        status = normalize_status(r.report_status)
        if status is None:
            continue
        team = NFLVERSE_TO_BDL.get(r.team, r.team)
        players = out.setdefault((int(r.season), int(r.week)), {}).setdefault(team, {})
        key = normalize_name(r.full_name)
        players[key] = _worse(players.get(key), status)
    return out


def _parse_date(raw) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        d = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def live_rows(bdl_rows: list[dict], now: datetime) -> list[dict]:
    best: dict[tuple[str, str], dict] = {}
    for r in bdl_rows:
        raw = (r.get("status") or "").strip()
        status = normalize_status(raw)
        player = r.get("player") or {}
        team = (player.get("team") or {}).get("abbreviation")
        first, last = player.get("first_name") or "", player.get("last_name") or ""
        if status is None or not team or not (first or last):
            continue
        reported = _parse_date(r.get("date"))
        if raw not in IR_TYPES and (reported is None or now - reported > STALE):
            continue
        name = f"{first} {last}".strip()
        key = (team, normalize_name(name))
        cur = best.get(key)
        if cur is None or _SEVERITY[status] > _SEVERITY[cur["status"]]:
            best[key] = {"team": team, "player_norm": key[1], "player_name": name,
                         "pos": player.get("position_abbreviation"), "status": status,
                         "reported_at": reported, "loaded_at": now}
    return list(best.values())


async def fetch_live(bdl: BdlNflClient) -> list[dict]:
    return await bdl.player_injuries()


def store_live(conn, rows: list[dict], now: datetime) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM edge_nfl_injuries")
        if rows:
            execute_values(cur, """INSERT INTO edge_nfl_injuries
                                   (team, player_norm, player_name, pos, status, reported_at, loaded_at) VALUES %s""",
                           [(r["team"], r["player_norm"], r["player_name"], r["pos"], r["status"],
                             r["reported_at"], r.get("loaded_at") or now) for r in rows])
    conn.commit()
    return len(rows)


def live_index(conn) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT team, player_norm, status FROM edge_nfl_injuries")
        for team, player, status in cur.fetchall():
            out.setdefault(team, {})[player] = status
    return out
