"""blind.record / listing / detail against a throwaway Postgres. Needs EDGE_TEST_DB_URL."""
import os
from datetime import datetime, timedelta, timezone

import psycopg2
import pytest
from sqlalchemy import create_engine

from src.database import Base
import src.models  # noqa: F401
from src.edge.social import blind
from src.edge.social.nba_matchup import Panel

URL = os.environ.get("EDGE_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not URL, reason="EDGE_TEST_DB_URL not set")
EDGE_TABLES = [t for name, t in Base.metadata.tables.items() if name.startswith("edge_")]
NOW = datetime(2026, 11, 16, 20, tzinfo=timezone.utc)
TIP = NOW + timedelta(hours=3)


@pytest.fixture
def conn():
    engine = create_engine(URL)
    Base.metadata.drop_all(engine, tables=EDGE_TABLES)
    Base.metadata.create_all(engine, tables=EDGE_TABLES)
    engine.dispose()
    c = psycopg2.connect(URL)
    with c.cursor() as cur:
        cur.execute("INSERT INTO edge_events (event_id, sport, home_team, away_team, commence_time, status, completed) "
                    "VALUES ('e1','basketball_nba','Boston Celtics','Detroit Pistons',%s,'scheduled',false), "
                    "('e2','basketball_nba','Miami Heat','Atlanta Hawks',%s,'scheduled',false)", (TIP, TIP))
        for name, team in (("A One", "BOS"), ("B Two", "MIA")):
            cur.execute("INSERT INTO edge_players (odds_name, team, match_method, confidence, updated_at) "
                        "VALUES (%s,%s,'exact',1,%s)", (name, team, NOW))
        # A One has a headshot row for the right team; B Two's row is for another team (traded)
        cur.execute("INSERT INTO edge_headshots (sport, odds_name, team, espn_id, checked_at) VALUES "
                    "('nba','A One','BOS',111,%s), ('nba','B Two','ATL',222,%s)", (NOW, NOW))
    c.commit()
    yield c
    c.close()


def pick(player="A One", event="e1", book="fanduel", line=20.5):
    return {"event_id": event, "market": "player_points", "market_label": "Points", "player": player,
            "side": "over", "line": line, "book": book, "price": 116, "fair_prob": 0.49, "ev": 0.1,
            "n_books": 9, "form": {"hits": 8, "games": 10, "rate": 0.8}, "position": "F",
            "commence_time": TIP.isoformat()}


def panels(*picks):
    return {blind.panel_key(p): Panel(True, 25, 20, 10, 25.0, 1.0, 34.0) for p in picks}


def test_record_numbers_per_sport_and_never_duplicates_a_bet(conn):
    a, b = pick("A One", "e1"), pick("B Two", "e2")
    assert blind.record(conn, "nba", [a, b], panels(a, b), NOW) == [1, 2]
    again = {**a, "book": "betmgm", "price": 120}  # same bet at another book
    c = pick("A One", "e1", line=21.5)
    assert blind.record(conn, "nba", [again, c], panels(c), NOW) == [3]
    assert blind.record(conn, "nba", [a, b, c], panels(a, b, c), NOW) == []
    assert blind.record(conn, "nfl", [a], panels(a), NOW) == [1]  # numbering is per sport


def test_detail_reveals_name_and_never_a_wrong_face(conn):
    a, b = pick("A One", "e1"), pick("B Two", "e2")
    blind.record(conn, "nba", [a, b], panels(a, b), NOW)
    d1 = blind.detail(conn, "nba", 1)
    assert d1["player"] == "A One" and d1["team"] == "BOS" and "111.png" in d1["headshot_url"]
    assert d1["home_logo_url"].endswith("/nba/500/bos.png") and d1["away_logo_url"].endswith("/det.png")
    assert d1["panel"]["opp_vs_pos_rank"] == 25 and "9 books" in d1["explanation"]
    d2 = blind.detail(conn, "nba", 2)
    assert d2["headshot_url"] is None  # headshot row says ATL, the player is a Heat player
    assert blind.detail(conn, "nba", 99) is None


def test_result_joins_grading_from_any_book(conn):
    a = pick()
    blind.record(conn, "nba", [a], panels(a), NOW)
    assert blind.result(conn, "nba", 1) == {"result": None, "profit_units": None}
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_opportunities (event_id, market, player, side, line, book, price,
                       first_seen_at, last_seen_at, fair_prob, ev, max_ev, n_books, devig_method, closing_done, result)
                       VALUES ('e1','player_points','A One','over',20.5,'betmgm',110,%s,%s,0.5,0.1,0.1,9,'power',false,'win')""",
                    (NOW, NOW))
    conn.commit()
    r = blind.result(conn, "nba", 1)
    assert r["result"] == "win" and r["profit_units"] == pytest.approx(1.16)
    out = blind.listing(conn, "nba")
    assert out["record"] == {"n": 1, "wins": 1, "losses": 0, "pushes": 0, "units": 1.16}
    assert out["plays"][0]["result"] == "win"


def test_build_panels_fails_closed_without_season_data(conn):
    r = {**pick(), "matchup": "Detroit Pistons @ Boston Celtics", "bdl_player_id": 5}
    assert blind.build_panels(conn, "nba", [r], NOW) == {}
    assert r["home_team"] == "Boston Celtics" and r["away_team"] == "Detroit Pistons"
    assert blind.build_panels(conn, "nfl", [r], NOW) == {}
