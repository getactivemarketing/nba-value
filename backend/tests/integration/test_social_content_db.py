"""content.second_source_* and posted_plays against a throwaway Postgres. Needs EDGE_TEST_DB_URL."""
import os
from datetime import date, datetime, timedelta, timezone

import psycopg2
import pytest
from sqlalchemy import create_engine

from src.database import EDGE_COLUMN_MIGRATIONS, Base
import src.models  # noqa: F401
from src.edge.social import content

URL = os.environ.get("EDGE_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not URL, reason="EDGE_TEST_DB_URL not set")
EDGE_TABLES = [t for name, t in Base.metadata.tables.items() if name.startswith("edge_")]
NOW = datetime(2026, 11, 16, 20, tzinfo=timezone.utc)
TIP = datetime(2026, 11, 17, 0, 30, tzinfo=timezone.utc)  # 7:30 PM ET 11/16
DAY = date(2026, 11, 16)


@pytest.fixture
def conn():
    engine = create_engine(URL)
    Base.metadata.drop_all(engine, tables=EDGE_TABLES)
    Base.metadata.create_all(engine, tables=EDGE_TABLES)
    engine.dispose()
    c = psycopg2.connect(URL)
    yield c
    c.close()


def pg(cur, pid, team, opp, pts, day=DAY, home=True):
    cur.execute("""INSERT INTO edge_player_games (bdl_player_id, bdl_game_id, season, game_date, team, opponent,
                   is_home, minutes, pts, reb, ast, fg3m, stl, blk, tov)
                   VALUES (%s,1,2026,%s,%s,%s,%s,30,%s,0,0,0,0,0,0)""", (pid, day, team, opp, home, pts))


def nba_event():
    return {"event_id": "e1", "home_team": "Boston Celtics", "away_team": "Detroit Pistons", "commence_time": TIP}


def test_second_source_nba_sums_both_teams(conn):
    with conn.cursor() as cur:
        pg(cur, 1, "BOS", "DET", 60), pg(cur, 2, "BOS", "DET", 50)
        pg(cur, 3, "DET", "BOS", 104, home=False)
        pg(cur, 4, "BOS", "MIA", 99, day=DAY - timedelta(days=1))  # other date: ignored
    conn.commit()
    assert content.second_source_nba(conn, nba_event()) == (110, 104)


def test_second_source_nba_none_when_a_team_missing(conn):
    with conn.cursor() as cur:
        pg(cur, 1, "BOS", "DET", 110)
    conn.commit()
    assert content.second_source_nba(conn, nba_event()) is None
    assert content.second_source_nba(conn, {**nba_event(), "home_team": "Nowhere Stars"}) is None


def nfl_row(cur, gid, state, hs, as_, day):
    cur.execute("""INSERT INTO edge_nfl_games (bdl_game_id, season, week, game_date, home, away, postseason,
                   status_state, home_score, away_score) VALUES (%s,2026,10,%s,'DAL','TB',false,%s,%s,%s)""",
                (gid, day, state, hs, as_))


def test_second_source_nfl_final_only_and_utc_date(conn):
    ev = {"home_team": "Dallas Cowboys", "away_team": "Tampa Bay Buccaneers", "commence_time": TIP}
    with conn.cursor() as cur:
        nfl_row(cur, 1, "in", 7, 3, DAY)
    conn.commit()
    assert content.second_source_nfl(conn, ev) is None
    with conn.cursor() as cur:
        cur.execute("DELETE FROM edge_nfl_games")
        nfl_row(cur, 2, "final", 16, 24, DAY + timedelta(days=1))  # BDL's UTC date
    conn.commit()
    assert content.second_source_nfl(conn, ev) == (16, 24)


def test_scores_columns_are_a_registered_migration():
    assert ("edge_nfl_games", "home_score", "INTEGER") in EDGE_COLUMN_MIGRATIONS
    assert ("edge_nfl_games", "away_score", "INTEGER") in EDGE_COLUMN_MIGRATIONS


def test_posted_plays_blind_and_value_with_results(conn):
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_blind_plays (sport, n, slate_date, event_id, market, market_label, player,
                       side, line, book, price, fair_prob, ev, n_books, commence_time, created_at)
                       VALUES ('nba',4,%s,'e1','player_points','Points','A One','over',20.5,'fanduel',116,0.49,0.1,9,%s,%s),
                              ('nba',5,%s,'e2','player_points','Points','Z Other','over',9.5,'fanduel',110,0.49,0.1,9,%s,%s)""",
                    (DAY, TIP, NOW, DAY, TIP, NOW))
        for kind, rank, player, bet in (("top", 0, "B Two", "B Two Over 5.5 Rebounds"), ("parlay", 0, "C Three", "x")):
            cur.execute("""INSERT INTO edge_nightly_legs (sport, slate_date, kind, rank, locked_at, event_id, matchup,
                           commence_time, market, market_label, player, side, line, bet, book, price, fair_prob, ev, n_books)
                           VALUES ('nba',%s,%s,%s,%s,'e1','D @ B',%s,'player_rebounds','Rebounds',%s,'over',5.5,%s,
                                   'fanduel',-110,0.55,0.05,8)""", (DAY, kind, rank, NOW, TIP, player, bet))
        cur.execute("""INSERT INTO edge_opportunities (event_id, market, player, side, line, book, price, fair_prob,
                       ev, max_ev, n_books, devig_method, closing_done, result, first_seen_at, last_seen_at) VALUES ('e1','player_points','A One',
                       'over',20.5,'betmgm',116,0.49,0.1,0.1,9,'power',false,'win',%s,%s)""", (NOW, NOW))
    conn.commit()
    plays = content.posted_plays(conn, "e1")
    assert [(p["kind"], p["player"], p["result"]) for p in plays] == [
        ("blind", "A One", "win"), ("value", "B Two", None)]  # no parlay leg, no other game's play
    assert plays[0]["bet"] == "A One Over 20.5 Points" and plays[0]["profit_units"] == pytest.approx(1.16)
    assert plays[1]["profit_units"] is None


def test_posted_plays_dedupes_blind_that_is_also_a_top_leg(conn):
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_blind_plays (sport, n, slate_date, event_id, market, market_label, player,
                       side, line, book, price, fair_prob, ev, n_books, commence_time, created_at)
                       VALUES ('nba',4,%s,'e1','player_points','Points','A One','over',20.5,'fanduel',116,0.49,0.1,9,%s,%s)""",
                    (DAY, TIP, NOW))
        cur.execute("""INSERT INTO edge_nightly_legs (sport, slate_date, kind, rank, locked_at, event_id, matchup,
                       commence_time, market, market_label, player, side, line, bet, book, price, fair_prob, ev, n_books)
                       VALUES ('nba',%s,'top',0,%s,'e1','D @ B',%s,'player_points','Points','A One','over',20.5,
                               'x','fanduel',116,0.55,0.05,8)""", (DAY, NOW, TIP))
    conn.commit()
    plays = content.posted_plays(conn, "e1")
    assert [(p["kind"], p["player"]) for p in plays] == [("blind", "A One")]


def ev_row(cur, eid, sport, home, away, hs, as_, completed=True, status="final", day=date(2026, 11, 1)):
    cur.execute("""INSERT INTO edge_events (event_id, sport, home_team, away_team, commence_time, status, completed,
                   home_score, away_score) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (eid, sport, home, away, datetime(day.year, day.month, day.day, 23, tzinfo=timezone.utc),
                 status, completed, hs, as_))


def test_team_records(conn):
    with conn.cursor() as cur:
        n = "basketball_nba"
        ev_row(cur, "a", n, "Boston Celtics", "Detroit Pistons", 110, 100)
        ev_row(cur, "b", n, "Detroit Pistons", "Boston Celtics", 90, 95)
        ev_row(cur, "c", n, "Boston Celtics", "Miami Heat", 100, 101)
        ev_row(cur, "d", n, "Boston Celtics", "Miami Heat", None, None, completed=False, status="scheduled")
        ev_row(cur, "e", n, "Boston Celtics", "Miami Heat", 0, 0, status="postponed")
        ev_row(cur, "f", n, "Boston Celtics", "Miami Heat", 100, 80, day=date(2026, 10, 1))  # preseason
        ev_row(cur, "g", "americanfootball_nfl", "Dallas Cowboys", "Tampa Bay Buccaneers", 27, 27,
               day=date(2026, 10, 11))
    conn.commit()
    nba = content.team_records(conn, "nba", content.SEASON_START["nba"])
    assert nba["BOS"] == {"w": 2, "l": 1, "t": 0, "games": 3, "label": "2-1"}
    assert nba["DET"]["label"] == "0-2" and nba["MIA"]["label"] == "1-0"
    nfl = content.team_records(conn, "nfl", content.SEASON_START["nfl"])
    assert nfl["DAL"] == {"w": 0, "l": 0, "t": 1, "games": 1, "label": "0-0-1"}


def test_posted_plays_carry_market_label_onto_the_final_card(conn, monkeypatch):
    """Real posted_plays output drawn by render_final: the stat label ('Pts') must be on the card."""
    from PIL import ImageDraw

    from src.edge.social import cards
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_blind_plays (sport, n, slate_date, event_id, market, market_label, player,
                       side, line, book, price, fair_prob, ev, n_books, commence_time, created_at)
                       VALUES ('nba',4,%s,'e1','player_points','Points','A One','over',20.5,'fanduel',116,0.49,0.1,9,%s,%s)""",
                    (DAY, TIP, NOW))
        cur.execute("""INSERT INTO edge_nightly_legs (sport, slate_date, kind, rank, locked_at, event_id, matchup,
                       commence_time, market, market_label, player, side, line, bet, book, price, fair_prob, ev, n_books)
                       VALUES ('nba',%s,'top',0,%s,'e1','D @ B',%s,'player_rebounds','Rebounds','B Two','over',5.5,
                               'B Two Over 5.5 Rebounds','fanduel',-110,0.55,0.05,8)""", (DAY, NOW, TIP))
    conn.commit()
    plays = content.posted_plays(conn, "e1")
    assert [p["market_label"] for p in plays] == ["Points", "Rebounds"]
    drawn: list[str] = []
    orig = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, **kw):
        drawn.append(str(text))
        return orig(self, xy, text, *a, **kw)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    event = {"sport": "nba", "away_team": "Detroit Pistons", "home_team": "Boston Celtics", "away_score": 104,
             "home_score": 110}
    cards.render_final(event, {}, {}, plays, {})
    text = "\n".join(drawn)
    assert "One O 20.5 Pts +116" in text and "Two O 5.5 Reb -110" in text
