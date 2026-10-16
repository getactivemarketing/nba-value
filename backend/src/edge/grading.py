# backend/src/edge/grading.py
"""Closing-line value and results for logged opportunities (spec §6.4)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psycopg2.extras import RealDictCursor

from src.config import settings
from src.edge import store
from src.edge.config import CLOSING_WINDOW_MINUTES
from src.edge.engine import american_to_decimal, clv, fair_prob, group, latest_before
from src.edge.players import et_date, stat_for
from src.edge.quotes import Quote
from src.edge.teams import ABBR_BY_NAME


def grade_over_under(actual: float, line: float, side: str) -> str:
    if actual == line:
        return "push"
    return "win" if (actual > line) == (side == "over") else "loss"


def grade_game(market: str, side: str, line: float, home: int, away: int) -> str:
    if market == "totals":
        return grade_over_under(home + away, line, side)
    margin = home - away + (line if market == "spreads" else 0.0)
    if margin == 0:
        return "push"
    return "win" if (margin > 0) == (side == "home") else "loss"


def profit_units(result: str, price: int) -> float:
    if result == "win":
        return american_to_decimal(price) - 1
    return -1.0 if result == "loss" else 0.0


def closing_for(quotes: list[Quote], opp: dict, tip: datetime) -> tuple[float | None, float | None, str | None]:
    """quotes = the event's edge_quotes rows (frozen at tip). Own book included: only our price is on trial."""
    closing = latest_before(quotes, tip, timedelta(minutes=CLOSING_WINDOW_MINUTES))
    key = (opp["event_id"], opp["market"], opp["player"], float(opp["line"]))
    books = group(closing).get(key, {})
    f = fair_prob(books, opp["market"], opp["side"], exclude_book=None)
    if f.thin:
        return None, None, f"thin close ({f.n_books} books, sharp={f.has_sharp})"
    return f.prob, clv(f.prob, int(opp["price"])), None


def prop_result(opp: dict, player_game: dict | None, team_has_box: bool) -> str | None:
    if player_game is None:
        return "void" if team_has_box else None
    if (player_game.get("minutes") or 0) <= 0:
        return "void"
    return grade_over_under(stat_for(opp["market"], player_game), float(opp["line"]), opp["side"])


def box_scores_missing(conn, d) -> bool:
    """True when a team that finished a game on ET date `d` has no box-score rows."""
    with conn.cursor() as cur:
        cur.execute("""SELECT home_team, away_team, commence_time FROM edge_events
                       WHERE completed AND sport = 'basketball_nba' AND commence_time BETWEEN %s AND %s""",
                    (datetime.combine(d, datetime.min.time(), timezone.utc) - timedelta(hours=12),
                     datetime.combine(d, datetime.min.time(), timezone.utc) + timedelta(hours=36)))
        teams = {ABBR_BY_NAME.get(t) for h, a, c in cur.fetchall() if et_date(c) == d for t in (h, a)} - {None}
        if not teams:
            return False
        cur.execute("SELECT DISTINCT team FROM edge_player_games WHERE game_date = %s AND team = ANY(%s)",
                    (d, sorted(teams)))
        return len({r[0] for r in cur.fetchall()}) < len(teams)


def games_in_progress(conn, now: datetime, sport: str | None = None, hours: int = 4) -> bool:
    """A game started within `hours` and not yet final."""
    sql = """SELECT 1 FROM edge_events WHERE NOT completed
             AND commence_time <= %s AND commence_time > %s"""
    params: list = [now, now - timedelta(hours=hours)]
    if sport:
        sql += " AND sport = %s"
        params.append(sport)
    with conn.cursor() as cur:
        cur.execute(sql + " LIMIT 1", params)
        return cur.fetchone() is not None


def update_scores(conn, scores_json: list[dict]) -> int:
    n = 0
    with conn.cursor() as cur:
        for ev in scores_json:
            if not ev.get("completed") or not ev.get("scores"):
                continue
            by_name = {s["name"]: int(s["score"]) for s in ev["scores"]}
            home, away = by_name.get(ev["home_team"]), by_name.get(ev["away_team"])
            if home is None or away is None:
                continue
            cur.execute("""UPDATE edge_events SET home_score = %s, away_score = %s, completed = TRUE,
                           status = 'final' WHERE event_id = %s""", (home, away, ev["id"]))
            n += cur.rowcount
    conn.commit()
    return n


def _set(cur, opp_id: int, **cols) -> None:
    sets = ", ".join(f"{k} = %s" for k in cols)
    cur.execute(f"UPDATE edge_opportunities SET {sets} WHERE id = %s", (*cols.values(), opp_id))


def run_grading(conn, now: datetime) -> dict:
    closed = graded = 0
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # 1. Closing line for every opportunity whose game has tipped.
        cur.execute("""SELECT o.*, e.commence_time FROM edge_opportunities o
                       JOIN edge_events e USING (event_id)
                       WHERE NOT o.closing_done AND e.commence_time <= %s""", (now,))
        rows = cur.fetchall()
        quote_cache: dict[str, list[Quote]] = {}
        for o in rows:
            tip = o["commence_time"]
            if o["event_id"] not in quote_cache:
                quote_cache[o["event_id"]] = store.load_quotes(conn, [o["event_id"]])
            prob, value, note = closing_for(quote_cache[o["event_id"]], o, tip)
            _set(cur, o["id"], closing_done=True, closing_fair_prob=prob, clv=value, clv_note=note)
            closed += 1

        # 2. Game-line results from final scores.
        cur.execute("""SELECT o.id, o.market, o.side, o.line, o.price, e.home_score, e.away_score
                       FROM edge_opportunities o JOIN edge_events e USING (event_id)
                       WHERE o.result IS NULL AND o.player = '' AND e.completed""")
        for o in cur.fetchall():
            r = grade_game(o["market"], o["side"], float(o["line"]), o["home_score"], o["away_score"])
            _set(cur, o["id"], result=r, profit_units=profit_units(r, o["price"]))
            graded += 1

        # 3. Prop results from box scores.
        cur.execute("""SELECT o.id, o.market, o.side, o.line, o.price, e.commence_time, e.home_team,
                              e.away_team, p.bdl_player_id
                       FROM edge_opportunities o JOIN edge_events e USING (event_id)
                       LEFT JOIN edge_players p ON p.odds_name = o.player
                       WHERE o.result IS NULL AND o.player <> '' AND e.commence_time <= %s
                         -- prop grading needs NBA box scores; NFL props get CLV only until Phase B
                         AND e.sport = 'basketball_nba'""",
                    (now - timedelta(hours=3),))
        for o in cur.fetchall():
            if o["bdl_player_id"] is None:
                continue  # unmatched name: stays ungraded, listed on /edge/status
            d = et_date(o["commence_time"])
            teams = [ABBR_BY_NAME.get(o["home_team"]), ABBR_BY_NAME.get(o["away_team"])]
            cur.execute("SELECT * FROM edge_player_games WHERE bdl_player_id = %s AND game_date = %s",
                        (o["bdl_player_id"], d))
            pg = cur.fetchone()
            cur.execute("SELECT 1 FROM edge_player_games WHERE game_date = %s AND team = ANY(%s) LIMIT 1",
                        (d, teams))
            r = prop_result(o, dict(pg) if pg else None, cur.fetchone() is not None)
            if r is not None:
                _set(cur, o["id"], result=r, profit_units=profit_units(r, o["price"]))
                graded += 1
    conn.commit()
    if settings.edge_nfl_players:
        from src.edge import nfl_players  # NFL props: box scores + snap counts (Phase B)
        graded += nfl_players.grade_props(conn, now)
    return {"closed": closed, "graded": graded}
