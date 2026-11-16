"""Matchup gate end to end on a small synthetic season in a throwaway Postgres.

Skipped unless EDGE_TEST_DB_URL points at a disposable database (see test_edge_store_db.py).
"""
import os
from datetime import date, datetime, timedelta, timezone

import psycopg2
import pytest
from sqlalchemy import create_engine

from src.database import Base
import src.models  # noqa: F401  registers the edge tables
from src.edge import store
from src.edge.backtest import matchup_gate as mg
from src.edge.teams import ABBR_BY_NAME

URL = os.environ.get("EDGE_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not URL, reason="EDGE_TEST_DB_URL not set")
EDGE_TABLES = [t for name, t in Base.metadata.tables.items() if name.startswith("edge_")]
ABBRS = sorted(ABBR_BY_NAME.values())
NAME = {v: k for k, v in ABBR_BY_NAME.items()}
T0 = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    engine = create_engine(URL)
    Base.metadata.drop_all(engine, tables=EDGE_TABLES)
    Base.metadata.create_all(engine, tables=EDGE_TABLES)
    engine.dispose()
    c = psycopg2.connect(URL)
    yield c
    c.close()


def _seed_league(conn):
    """15 pairs of teams, each pair plays 5 games Nov 1-5. ABBRS[28] scores 50 with its F, so its partner
    ABBRS[29] allows the most to forwards and has the worst defense. Players 1000+i (F) and 2000+i (G)."""
    with conn.cursor() as cur:
        gid = 0
        for i, team in enumerate(ABBRS):
            cur.execute("""INSERT INTO edge_players (odds_name, bdl_player_id, team, position, match_method,
                           confidence, updated_at) VALUES (%s, %s, %s, 'F', 'exact', 1, now()),
                           (%s, %s, %s, 'G', 'exact', 1, now())""",
                        (f"F {team}", 1000 + i, team, f"G {team}", 2000 + i, team))
        for k in range(15):
            for day in range(5):
                gid += 1
                for i, opp in ((2 * k, 2 * k + 1), (2 * k + 1, 2 * k)):
                    for pid, pts in ((1000 + i, 50 if i == 28 else 10 + i), (2000 + i, 5)):
                        cur.execute("""INSERT INTO edge_player_games (bdl_player_id, bdl_game_id, season, game_date,
                                       team, opponent, is_home, minutes, pts, reb, ast, fg3m, stl, blk, tov,
                                       fga, fta, oreb) VALUES (%s,%s,2026,%s,%s,%s,%s,30,%s,0,0,0,0,0,1,10,2,1)""",
                                    (pid, gid, date(2026, 11, 1 + day), ABBRS[i], ABBRS[opp], i % 2 == 0, pts))
    conn.commit()


def _bet(conn, eid, player, side, clv, book="fanduel", seen=T0, home=0, away=29, day=20, result="win",
         closed=True):
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_events (event_id, sport, home_team, away_team, commence_time, status, completed)
                       VALUES (%s, 'basketball_nba', %s, %s, %s, 'scheduled', TRUE) ON CONFLICT DO NOTHING""",
                    (eid, NAME[ABBRS[home]], NAME[ABBRS[away]], datetime(2026, 11, day, 23, tzinfo=timezone.utc)))
        cur.execute("""INSERT INTO edge_opportunities (event_id, market, player, side, line, book, price,
                       first_seen_at, last_seen_at, fair_prob, ev, max_ev, n_books, devig_method, closing_done,
                       clv, result, profit_units)
                       VALUES (%s,'player_points',%s,%s,20.5,%s,-110,%s,%s,0.5,0.05,0.05,5,'power',%s,%s,%s,0.9)""",
                    (eid, player, side, book, seen, seen, closed, clv, result))
    conn.commit()


def test_gate_end_to_end_small_dataset(conn, tmp_path, monkeypatch):
    _seed_league(conn)
    hero = f"F {ABBRS[0]}"
    for n in range(4):  # favorable: hero Over vs the worst defender, on four different days
        _bet(conn, f"fav{n}", hero, "over", 0.03 + 0.001 * n, day=20 + n)
    # the same first bet seen again later at another book: must count once, at the earliest sighting
    _bet(conn, "fav0", hero, "over", 0.50, book="betmgm", seen=T0 + timedelta(hours=3), day=20)
    for n in range(4):  # rest: Under is not favorable against that defender; a G has no vs-position edge
        _bet(conn, f"rest{n}", hero, "under", 0.00 + 0.001 * n, day=20 + n)
    _bet(conn, "nopos", "Nobody Known", "over", -0.01)
    # not part of the gate's input: ungraded, void, and not yet closed
    _bet(conn, "ungraded", hero, "over", 0.40, result=None, day=21)
    _bet(conn, "void", hero, "over", 0.40, result="void", day=22)
    _bet(conn, "open", hero, "over", 0.40, closed=False, day=23)

    out = mg.evaluate(conn, 2026, min_n=3)
    g = out["gate"]
    assert g["n_fav"] == 4 and g["n_rest"] == 5 and out["raw_rows"] == 10 and out["bets"] == 9
    assert g["fav_mean"] == pytest.approx(0.0315)  # 0.03..0.033, the 0.50 duplicate ignored
    assert g["pass"], g["failed"]
    assert out["position_coverage"] == pytest.approx(1.0)
    assert out["why_not"] == {"no position": 1, "not favorable": 4}

    # the production threshold still rejects four bets
    assert not mg.evaluate(conn, 2026)["gate"]["pass"]

    report = tmp_path / "report.md"
    monkeypatch.setattr(mg, "evaluate", lambda c, s: out)
    monkeypatch.setattr(store, "connect", lambda: psycopg2.connect(URL))
    rc = mg.main(["--sport", "nba", "--season", "2026", "--out", str(report), "--write-state"])
    assert rc == 0
    assert "Position coverage: 100.00%" in report.read_text() and "Verdict: PASS" in report.read_text()
    assert store.get_state(conn)["social_blind_gate:nba"] == "pass"
