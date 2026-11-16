"""One-off: reload NBA box scores so old rows gain FGM/FGA/FTA/OREB, and set G/F/C positions.

    python -m src.edge.scripts.backfill_box_fields --since 2025-10-01 [--dry-run]

Box scores reuse players.backfill (upsert, so existing rows get the new fields). Positions cost
one roster call per team. --dry-run makes no BallDontLie calls and writes nothing; it only
reports the days and teams a real run would cover.
"""
import argparse
import asyncio
import logging
from datetime import date

from src.edge import players, store

log = logging.getLogger("edge.backfill_box_fields")


def position_updates(rosters: dict[str, list[dict]]) -> dict[int, str]:
    """{bdl_player_id: G|F|C} from rosters keyed by team; players without a usable position are left out."""
    out: dict[int, str] = {}
    for roster in rosters.values():
        for x in roster:
            grp = players.position_group(x.get("position"))
            if grp:
                out[x["id"]] = grp
    return out


def matched_teams(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT team FROM edge_players WHERE bdl_player_id IS NOT NULL AND team IS NOT NULL "
                    "ORDER BY team")
        return [r[0] for r in cur.fetchall()]


def write_positions(conn, positions: dict[int, str]) -> int:
    with conn.cursor() as cur:
        cur.executemany("UPDATE edge_players SET position = %s WHERE bdl_player_id = %s",
                        [(grp, pid) for pid, grp in positions.items()])
    conn.commit()
    return len(positions)


async def run(conn, bdl, since: date, end: date, dry_run: bool) -> dict:
    teams = matched_teams(conn)
    days = (end - since).days + 1
    if dry_run:
        log.info("dry run: would reload %d days (%s..%s) and fetch %d rosters", days, since, end, len(teams))
        return {"days": days, "teams": len(teams), "box_rows": 0, "positions": 0}
    box_rows = await players.backfill(conn, bdl, since, end)
    rosters = {}
    for t in teams:
        rosters[t] = await bdl.roster(t)
    positions = write_positions(conn, position_updates(rosters))
    log.info("reloaded %d box-score rows over %d days; set %d positions across %d teams",
             box_rows, days, positions, len(teams))
    return {"days": days, "teams": len(teams), "box_rows": box_rows, "positions": positions}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", required=True, type=date.fromisoformat, help="first game date, YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    conn = store.connect()
    try:
        asyncio.run(run(conn, players.BdlClient(), args.since, date.today(), args.dry_run))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
