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


_CLOCK = {}


@pytest.fixture(autouse=True)
def _wall_clock(monkeypatch):
    """runner rechecks tips against the wall clock right before posting: pin it to the pass's now."""
    monkeypatch.setattr(runner, "_utcnow", lambda: _CLOCK["now"])


async def run(conn, now, poster):
    _CLOCK["now"] = now
    return await runner.social_step(conn, now, poster=poster)


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
    await run(conn, NOW, p)
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

    await run(conn, NOW + timedelta(minutes=1), p)
    assert len(p.calls) == n_calls  # nothing posts twice
    assert len(_statuses(conn)) == 6


class _LivePoster:
    """A live (non-dry-run) poster that never touches the network."""
    dry_run = False

    def __init__(self, published_url=None):
        self.calls, self.published_url = [], published_url

    def post(self, texts, images):
        from src.edge.social.poster import PostResult
        self.calls.append(texts)
        return PostResult(True, f"sub-{len(self.calls)}", [], None)

    def lookup(self, sid):
        from src.edge.social.poster import UrlLookup
        return UrlLookup("published", self.published_url) if self.published_url else UrlLookup("pending")


async def test_crash_after_blotato_accepted_never_posts_again_and_reconciles(conn, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "social_kinds", "blind")
    live = {**_bet("B Two", "next"), "book": "fanduel", "price": 110, "ev": 0.05, "n_books": 9, "verify": False,
            "form_flag": False, "commence_time": NEXT_TIP.isoformat(), "market_label": "Points",
            "matchup": "Atlanta Hawks @ Miami Heat", "bdl_player_id": 2, "fair_prob": 0.5, "form": None}
    monkeypatch.setattr(runner, "_live_rows", lambda c, sport, now: [dict(live)])
    # a live post of blind #2 that Blotato accepted; the worker died before finish()
    crashed_at = NOW - timedelta(minutes=20)
    c = ledger.claim(conn, "nba", "blind", "blind:nba:2", crashed_at)
    ledger.note_submission(conn, c, "sub-lost")
    p = _LivePoster()
    await run(conn, NOW, p)
    assert p.calls == []  # never re-posted
    assert ledger.find(conn, "blind", "blind:nba:2")["status"] == "unknown"
    assert store.get_state(conn).get("social_unknown") == "0"  # it has a submission id: reconcilable
    await run(conn, NOW + timedelta(minutes=1), p)  # Blotato: still in progress
    assert ledger.find(conn, "blind", "blind:nba:2")["status"] == "unknown"
    p.published_url = "https://x.com/trulineapp/status/2"
    await run(conn, NOW + timedelta(minutes=2), p)
    row = ledger.find(conn, "blind", "blind:nba:2")
    assert row["status"] == "posted" and row["post_url"] == "https://x.com/trulineapp/status/2"
    assert p.calls == []


async def test_live_post_stores_submission_and_counts_for_brakes(conn, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "social_kinds", "blind")
    live = {**_bet("B Two", "next"), "book": "fanduel", "price": 110, "ev": 0.05, "n_books": 9, "verify": False,
            "form_flag": False, "commence_time": NEXT_TIP.isoformat(), "market_label": "Points",
            "matchup": "Atlanta Hawks @ Miami Heat", "bdl_player_id": 2, "fair_prob": 0.5, "form": None}
    monkeypatch.setattr(runner, "_live_rows", lambda c, sport, now: [dict(live)])
    p = _LivePoster()
    await run(conn, NOW, p)
    row = ledger.find(conn, "blind", "blind:nba:2")
    assert len(p.calls) == 1 and row["status"] == "posted" and row["submission_id"] == "sub-1"


NFL_SUNDAY_1030 = datetime(2026, 11, 22, 15, 30, tzinfo=timezone.utc)  # 10:30 AM EST
NFL_TIP = datetime(2026, 11, 22, 18, 0, tzinfo=timezone.utc)


async def test_nfl_cheatsheet_posts_from_nfl_box_scores(conn, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "social_sports", "nfl")
    monkeypatch.setattr(settings, "social_kinds", "cheatsheet")
    rows = []
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_events (event_id, sport, home_team, away_team, commence_time, status, completed)
                       VALUES ('nflg','americanfootball_nfl','Philadelphia Eagles','Dallas Cowboys',%s,'scheduled',false)""",
                    (NFL_TIP,))
        for i in range(7):
            team, pid = ("PHI" if i % 2 else "DAL"), 500 + i
            cur.execute("""INSERT INTO edge_nfl_players (odds_name, team, bdl_player_id, position, match_method,
                           confidence, updated_at) VALUES (%s,%s,%s,'WR','exact',1,%s)""",
                        (f"Receiver {i}", team, pid, NOW))
            for w in range(10):  # weeks 1-10: hits 9-i of the last 10 at 4.5 receptions
                cur.execute("""INSERT INTO edge_nfl_player_games (bdl_player_id, bdl_game_id, season, week, game_date,
                               team, opponent, is_home, receptions, rec_yds) VALUES (%s,%s,2026,%s,%s,%s,'NYG',true,%s,%s)""",
                            (pid, 9000 + 100 * i + w, w + 1, date(2026, 9, 10) + timedelta(days=7 * w), team,
                             6 if w >= 1 + i else 3, 60))
            rows.append({"event_id": "nflg", "matchup": "Dallas Cowboys @ Philadelphia Eagles",
                         "commence_time": NFL_TIP.isoformat(), "market": "player_receptions",
                         "market_label": "Receptions", "player": f"Receiver {i}", "side": "over", "line": 4.5,
                         "book": "fanduel", "price": 105, "fair_prob": 0.52, "ev": 0.03, "n_books": 8,
                         "verify": False, "form_flag": False, "bdl_player_id": pid, "tags": [], "form": None})
    conn.commit()
    monkeypatch.setattr(runner, "_live_rows", lambda c, sport, now: [dict(r) for r in rows])
    p = DryRunPoster()
    await run(conn, NFL_SUNDAY_1030, p)
    row = ledger.find(conn, "cheatsheet", "cheatsheet:nfl:2026-11-22")
    assert row["status"] == "dry_run" and row["detail"]["rows"] == 7, row["detail"]
    assert "NFL Cheat Sheet" in row["text"] and bytes(row["image"]).startswith(b"\x89PNG")
    assert len(p.calls) == 1


async def test_unknown_rows_without_a_submission_id_show_in_worker_state(conn):
    for i in range(2):
        ledger.claim(conn, "nba", "final", f"final:lost{i}", NOW - timedelta(minutes=30))
        ledger.claim(conn, "nba", "final", f"final:lost{i}", NOW)  # stale live pending -> unknown
    assert ledger.unknown_without_submission(conn) == 2
    await run(conn, NOW, _LivePoster())
    assert store.get_state(conn)["social_unknown"] == "2"


def test_last_blind_post_counts_posted_and_dry_run_blind_rows_of_the_sport(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM edge_social_posts")
    conn.commit()
    assert runner._last_blind_post(conn, "nba") is None
    for kind, key, status, at in (("blind", "blind:nba:1", "dry_run", NOW - timedelta(minutes=30)),
                                  ("blind", "blind:nba:2", "skipped", NOW - timedelta(minutes=5)),
                                  ("cheatsheet", "cheatsheet:nba:x", "posted", NOW - timedelta(minutes=1)),
                                  ("blind", "blind:nfl:1", "posted", NOW)):
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO edge_social_posts (sport, kind, key, status, attempts, created_at, updated_at)
                           VALUES (%s,%s,%s,%s,1,%s,%s)""", (key.split(":")[1] if kind == "blind" else "nba",
                                                             kind, key, status, at, at))
        conn.commit()
    assert runner._last_blind_post(conn, "nba") == NOW - timedelta(minutes=30)
