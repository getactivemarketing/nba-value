# backend/src/edge/views.py
"""Response builders for /api/v1/edge. Pure helpers first, DB readers after."""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

from psycopg2.extras import RealDictCursor

from src.config import settings
from src.edge import players, schedule, store
from src.edge.engine import fair_prob, group, prob_to_american, price_quotes
from src.edge.quotes import Quote, sides_for
from src.edge.teams import ABBR_BY_NAME
from src.services.mlb.clv import summarize_clv
from src.utils.season import current_nba_season

MARKET_LABEL = {
    "player_points": "Points", "player_rebounds": "Rebounds", "player_assists": "Assists",
    "player_threes": "Threes", "player_points_rebounds_assists": "Pts+Reb+Ast",
    "player_points_rebounds": "Pts+Reb", "player_points_assists": "Pts+Ast",
    "player_rebounds_assists": "Reb+Ast", "h2h": "Moneyline", "spreads": "Spread", "totals": "Total",
}


def _signed(x: float) -> str:
    return f"+{x:g}" if x > 0 else f"{x:g}"


def describe(market: str, player: str, side: str, line: float, home_team: str, away_team: str) -> str:
    if market.startswith("player_"):
        return f"{player} {side.title()} {line:g} {MARKET_LABEL[market]}"
    if market == "totals":
        return f"{side.title()} {line:g}"
    team = home_team if side == "home" else away_team
    if market == "h2h":
        return f"{team} ML"
    return f"{team} {_signed(line if side == 'home' else -line)}"


def odds_grid(quotes: list[Quote], fair_by_outcome: dict[tuple, float]) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for q in quotes:
        k = (q.player, q.line, q.side)
        r = rows.setdefault(k, {"player": q.player, "line": q.line, "side": q.side, "prices": {},
                                "fair_prob": fair_by_outcome.get(k)})
        r["prices"][q.book] = {"price": q.price, "link": q.link, "observed_at": q.observed_at.isoformat()}
    for r in rows.values():
        if r["prices"]:
            r["best_book"] = max(r["prices"], key=lambda b: r["prices"][b]["price"])
        r["fair_price"] = prob_to_american(r["fair_prob"]) if r["fair_prob"] else None
    return sorted(rows.values(), key=lambda r: (r["player"], r["line"], r["side"]))


def results_summary(rows: list[dict]) -> dict:
    def block(rs: list[dict]) -> dict:
        clv = summarize_clv([float(r["clv"]) if r["clv"] is not None else None for r in rs])
        if clv["mean_clv"] is not None and clv["std_error"] is not None:
            clv["ci95"] = [round(clv["mean_clv"] - 1.96 * clv["std_error"], 5),
                           round(clv["mean_clv"] + 1.96 * clv["std_error"], 5)]
        else:
            clv["ci95"] = None
        graded = [r for r in rs if r["result"] in ("win", "loss", "push")]
        wins = sum(r["result"] == "win" for r in graded)
        losses = sum(r["result"] == "loss" for r in graded)
        units = round(sum(float(r["profit_units"] or 0) for r in graded), 4)
        return {"n": len(rs), "clv": clv, "record": {
            "wins": wins, "losses": losses, "pushes": len(graded) - wins - losses,
            "voids": sum(r["result"] == "void" for r in rs),
            "win_rate": round(wins / (wins + losses), 3) if wins + losses else None,
            "units": units, "roi": round(units / len(graded), 4) if graded else None}}

    out = block(rows)
    for dim in ("market", "book"):
        keys = sorted({r[dim] for r in rows})
        out[f"by_{dim}"] = {k: block([r for r in rows if r[dim] == k]) for k in keys}
    return out


def _factor(conn) -> int:
    return int(store.get_state(conn).get("guard_factor", "1"))


def games(conn, now: datetime) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT e.event_id, e.home_team, e.away_team, e.commence_time, e.last_lines_poll,
                              e.last_props_poll, COUNT(DISTINCT (q.player, q.market)) FILTER
                              (WHERE q.player <> '') AS prop_markets
                       FROM edge_events e LEFT JOIN edge_quotes q USING (event_id)
                       WHERE e.commence_time > %s AND e.commence_time <= %s
                       GROUP BY e.event_id ORDER BY e.commence_time""", (now, now + timedelta(hours=36)))
        return [dict(r) for r in cur.fetchall()]


def ev_rows(conn, now: datetime, market: str | None = None, book: str | None = None,
            min_ev: float | None = None, event_id: str | None = None) -> list[dict]:
    events = {e["event_id"]: e for e in store.upcoming_events(conn, now)}
    ids = [event_id] if event_id else list(events)
    if not ids:
        return []
    commence = {k: v["commence_time"] for k, v in events.items()}
    quotes = schedule.fresh(store.load_quotes(conn, ids, market), commence, now, _factor(conn))
    threshold = settings.edge_min_ev if min_ev is None else min_ev
    player_ids = _player_ids(conn)
    out = []
    for p in price_quotes(quotes, threshold, settings.edge_verify_ev):
        q = p.quote
        if not p.is_ev or (book and q.book != book):
            continue
        e = events[q.event_id]
        out.append({
            "event_id": q.event_id, "matchup": f"{e['away_team']} @ {e['home_team']}",
            "commence_time": e["commence_time"].isoformat(), "market": q.market,
            "market_label": MARKET_LABEL[q.market], "player": q.player, "side": q.side, "line": q.line,
            "bet": describe(q.market, q.player, q.side, q.line, e["home_team"], e["away_team"]),
            "book": q.book, "price": q.price, "link": q.link, "fair_prob": round(p.fair.prob, 4),
            "fair_price": prob_to_american(p.fair.prob), "ev": round(p.ev, 4), "n_books": p.fair.n_books,
            "observed_at": q.observed_at.isoformat(), "verify": p.verify,
            "bdl_player_id": player_ids.get(q.player),
        })
    return sorted(out, key=lambda r: r["ev"], reverse=True)


def _player_ids(conn) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT odds_name, bdl_player_id FROM edge_players WHERE bdl_player_id IS NOT NULL")
        return {name: pid for name, pid in cur.fetchall()}


def odds(conn, now: datetime, event_id: str, market: str) -> dict:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM edge_events WHERE event_id = %s", (event_id,))
        e = cur.fetchone()
    if not e:
        return {"event": None, "rows": []}
    quotes = store.load_quotes(conn, [event_id], market)
    fresh_q = schedule.fresh(quotes, {event_id: e["commence_time"]}, now, _factor(conn))
    fair = {}
    for (_, mkt, player, line), books in group(fresh_q).items():
        for side in sides_for(mkt):
            f = fair_prob(books, mkt, side, exclude_book=None)
            if not f.thin:
                fair[(player, line, side)] = round(f.prob, 4)
    fresh_keys = {q.key for q in fresh_q}
    rows = odds_grid(quotes, fair)
    for r in rows:
        for b, cell in r["prices"].items():
            cell["stale"] = (e["event_id"], market, r["player"], r["line"], r["side"], b) not in fresh_keys
    return {"event": {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in dict(e).items()},
            "market": market, "rows": rows}


def player(conn, now: datetime, bdl_id: int, market: str, line: float | None, min_minutes: float) -> dict:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT odds_name, team FROM edge_players WHERE bdl_player_id = %s", (bdl_id,))
        names = [dict(r) for r in cur.fetchall()]
    games_ = players.load_games(conn, bdl_id)
    events = store.upcoming_events(conn, now)
    odds_names = {n["odds_name"] for n in names}
    team = names[0]["team"] if names else None
    event = next((e for e in events if team in (ABBR_BY_NAME.get(e["home_team"]), ABBR_BY_NAME.get(e["away_team"]))), None)
    prices = []
    if event:
        prices = [q for q in store.load_quotes(conn, [event["event_id"]], market) if q.player in odds_names]
        if line is None and prices:
            line = statistics.median(q.line for q in prices)
    opponent = is_home = None
    if event and team:
        is_home = ABBR_BY_NAME.get(event["home_team"]) == team
        opponent = ABBR_BY_NAME.get(event["away_team"] if is_home else event["home_team"])
    return {
        "bdl_player_id": bdl_id, "name": names[0]["odds_name"] if names else None, "team": team,
        "market": market, "line": line, "opponent": opponent, "is_home": is_home,
        "hit_rates": players.hit_rates(games_, market, line, opponent, is_home, current_nba_season(),
                                       min_minutes) if line is not None else None,
        "games": [{**g, "game_date": g["game_date"].isoformat(), "value": players.stat_for(market, g)}
                  for g in games_[:20]],
        "prices": [{"book": q.book, "side": q.side, "line": q.line, "price": q.price, "link": q.link}
                   for q in prices],
    }


def results(conn, now: datetime, days: int, market: str | None, book: str | None) -> dict:
    sql = """SELECT market, book, clv, result, profit_units, first_seen_at FROM edge_opportunities
             WHERE closing_done AND first_seen_at >= %s"""
    params: list = [now - timedelta(days=days)]
    if market:
        sql += " AND market = %s"
        params.append(market)
    if book:
        sql += " AND book = %s"
        params.append(book)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
    out = results_summary(rows)
    daily: dict[str, list[float]] = {}
    for r in rows:
        if r["clv"] is not None:
            daily.setdefault(r["first_seen_at"].date().isoformat(), []).append(float(r["clv"]))
    out["daily_clv"] = [{"date": d, "mean_clv": round(statistics.fmean(v), 5), "n": len(v)}
                        for d, v in sorted(daily.items())]
    return out


def status(conn, now: datetime) -> dict:
    state = store.get_state(conn)
    hb = state.get("heartbeat_at")
    with conn.cursor() as cur:
        cur.execute("SELECT odds_name FROM edge_players WHERE match_method = 'unmatched' ORDER BY odds_name")
        unmatched = [r[0] for r in cur.fetchall()]
    cost = store.cost_last_24h(conn, now)
    return {
        "heartbeat_age_s": (now - datetime.fromisoformat(hb)).total_seconds() if hb else None,
        "credits_remaining": state.get("credits_remaining"), "credits_last_24h": cost,
        "projected_monthly": cost * 30, "guard_factor": int(state.get("guard_factor", "1")),
        "last_error": state.get("last_error") or None, "bdl_error": state.get("bdl_error") or None,
        "unmatched_players": unmatched, "last_box_scores_date": state.get("last_box_scores_date"),
        "backfills": {k: v for k, v in state.items() if k.startswith("backfill_")},
    }


def public_status(conn, now: datetime) -> dict:
    """The banner's view: no credits, no errors, nothing an outsider can use."""
    s = status(conn, now)
    age = s["heartbeat_age_s"]
    return {"delayed": age is None or age > 600 or bool(s["last_error"])}
