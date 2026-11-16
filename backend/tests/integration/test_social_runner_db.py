"""runner.social_step end to end in dry run against a throwaway Postgres. Needs EDGE_TEST_DB_URL.

Only the live board (views.ev_rows, covered elsewhere) is faked; everything else is real SQL:
ledger, blind plays, Top-5 lock, grading join, box-score second source, team lookup, cards.
"""
import os
from datetime import date, datetime, timedelta, timezone

import psycopg2
import pytest
from sqlalchemy import create_engine

from src.database import Base
import src.models  # noqa: F401
from src.edge import store
from src.edge.social import ledger, runner
from src.edge.social.poster import DryRunPoster

URL = os.environ.get("EDGE_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not URL, reason="EDGE_TEST_DB_URL not set")
EDGE_TABLES = [t for name, t in Base.metadata.tables.items() if name.startswith("edge_")]
NOW = datetime(2026, 11, 17, 20, 0, tzinfo=timezone.utc)      # 3:00 PM ET Tue 11/17
DONE_TIP = datetime(2026, 11, 17, 0, 30, tzinfo=timezone.utc)  # 7:30 PM ET 11/16, final
NEXT_TIP = datetime(2026, 11, 18, 0, 0, tzinfo=timezone.utc)   # 7:00 PM ET 11/17


def _bet(player, event, line=20.5):
    return {"event_id": event, "market": "player_points", "player": player, "side": "over", "line": line}


@pytest.fixture
def conn():
    engine = create_engine(URL)
    Base.metadata.drop_all(engine, tables=EDGE_TABLES)
    Base.metadata.create_all(engine, tables=EDGE_TABLES)
    engine.dispose()
    c = psycopg2.connect(URL)
    with c.cursor() as cur:
        cur.execute("""INSERT INTO edge_events (event_id, sport, home_team, away_team, commence_time, status, completed,
                       home_score, away_score) VALUES
                       ('done','basketball_nba','Boston Celtics','Detroit Pistons',%s,'final',true,110,104),
                       ('next','basketball_nba','Miami Heat','Atlanta Hawks',%s,'scheduled',false,NULL,NULL)""",
                    (DONE_TIP, NEXT_TIP))
        for name, team, bdl in (("A One", "BOS", 1), ("B Two", "MIA", 2), ("C Three", "ATL", 3)):
            cur.execute("""INSERT INTO edge_players (odds_name, bdl_player_id, team, match_method, confidence,
                           updated_at) VALUES (%s,%s,%s,'exact',1,%s)""", (name, bdl, team, NOW))
        # box scores of the final: BOS 30+80 = 110, DET 104 (the odds feed agrees)
        for pid, team, opp, pts, home in ((1, "BOS", "DET", 30, True), (11, "BOS", "DET", 80, True),
                                          (12, "DET", "BOS", 104, False)):
            cur.execute("""INSERT INTO edge_player_games (bdl_player_id, bdl_game_id, season, game_date, team, opponent,
                           is_home, minutes, pts, reb, ast, fg3m, stl, blk, tov)
                           VALUES (%s,7,2026,%s,%s,%s,%s,34,%s,6,4,0,0,0,0)""",
                        (pid, date(2026, 11, 16), team, opp, home, pts))
        # blind play #1 on the finished game, already posted (dry run) yesterday, graded a win
        cur.execute("""INSERT INTO edge_blind_plays (sport, n, slate_date, event_id, market, market_label, player, side,
                       line, book, price, fair_prob, ev, n_books, position, commence_time, created_at)
                       VALUES ('nba',1,%s,'done','player_points','Points','A One','over',20.5,'fanduel',116,0.49,
                               0.08,9,'F',%s,%s),
                              ('nba',2,%s,'next','player_points','Points','B Two','over',20.5,'fanduel',110,0.5,
                               0.05,9,'G',%s,%s)""",
                    (date(2026, 11, 16), DONE_TIP, DONE_TIP - timedelta(hours=5),
                     date(2026, 11, 17), NEXT_TIP, NOW - timedelta(minutes=1)))
        cur.execute("""INSERT INTO edge_opportunities (event_id, market, player, side, line, book, price,
                       first_seen_at, last_seen_at, fair_prob, ev, max_ev, n_books, devig_method, closing_done,
                       result, profit_units)
                       VALUES ('done','player_points','A One','over',20.5,'fanduel',116,%s,%s,0.49,0.08,0.08,9,
                               'power',true,'win',1.16)""", (DONE_TIP - timedelta(hours=5), DONE_TIP))
        cur.execute("""INSERT INTO edge_social_posts (sport, kind, key, status, attempts, text, image, created_at,
                       updated_at) VALUES ('nba','blind','blind:nba:1','dry_run',1,'old',%s,%s,%s)""",
                    (psycopg2.Binary(b"original-card"), DONE_TIP - timedelta(hours=5), DONE_TIP - timedelta(hours=5)))
        # tonight's Top 5, locked
        cur.execute("""INSERT INTO edge_nightly_legs (sport, slate_date, kind, rank, locked_at, event_id, matchup,
                       commence_time, market, market_label, player, side, line, bet, book, price, fair_prob, ev,
                       n_books) VALUES ('nba',%s,'top',0,%s,'next','Atlanta Hawks @ Miami Heat',%s,'player_points',
                       'Points','C Three','over',18.5,'C Three Over 18.5 Points','draftkings',105,0.5,0.03,8)""",
                    (date(2026, 11, 17), NOW - timedelta(minutes=10), NEXT_TIP))
    c.commit()
    store.set_state(c, "social_blind_gate:nba", "pass", NOW)
    yield c
    c.close()


def _statuses(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT key, status FROM edge_social_posts ORDER BY key")
        return dict(cur.fetchall())


async def test_social_step_end_to_end_dry_run(conn, monkeypatch):
    live = {**_bet("B Two", "next"), "book": "fanduel", "price": 110, "ev": 0.05, "n_books": 9, "verify": False,
            "form_flag": False, "commence_time": NEXT_TIP.isoformat(), "market_label": "Points",
            "matchup": "Atlanta Hawks @ Miami Heat", "bdl_player_id": 2, "fair_prob": 0.5, "form": None}
    monkeypatch.setattr(runner, "_live_rows", lambda c, sport, now: [dict(live)])
    p = DryRunPoster()
    await runner.social_step(conn, NOW, poster=p)
    assert _statuses(conn) == {
        "blind:nba:1": "dry_run",                  # seeded
        "blind:nba:2": "dry_run",                  # posted from edge_blind_plays, still +EV, 4 h to tip
        "cheatsheet:nba:2026-11-17": "skipped",    # one +EV prop is too thin for a sheet
        "final:done": "dry_run",                   # the final with a posted, graded play; scores cross-checked
        "receipt:blind:nba:1": "dry_run",          # the win
        "value:nba:2026-11-17": "dry_run",         # the locked Top 5, game not started
    }
    final = ledger.find(conn, "final", "final:done")
    assert "Detroit Pistons 104, Boston Celtics 110" in final["text"] and "1-0" in final["text"]
    assert bytes(final["image"]).startswith(b"\x89PNG")
    receipt = ledger.find(conn, "receipt", "receipt:blind:nba:1")
    assert "Blind Play #1 was A One" in receipt["text"]
    receipt_call = next(imgs for texts, imgs in p.calls if "CASHED" in texts[0])
    assert receipt_call[0][1] == b"original-card"  # no post URL: the original card rides along
    blind2 = ledger.find(conn, "blind", "blind:nba:2")
    assert "Blind Play #2" in blind2["text"] and "B Two" not in blind2["text"]
    n_calls = len(p.calls)
    assert n_calls == 4

    await runner.social_step(conn, NOW + timedelta(minutes=1), poster=p)
    assert len(p.calls) == n_calls  # nothing posts twice
    assert len(_statuses(conn)) == 6
