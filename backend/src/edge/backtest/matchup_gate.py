# backend/src/edge/backtest/matchup_gate.py
"""Pre-registered validation gate for the Blind Plays "favorable matchup" filter.

    python -m src.edge.backtest.matchup_gate --sport nba [--write-state] [--out PATH]

Blind plays may only use favorable bets, so before launch we check that favorable
bets did not do worse on closing-line value than the rest. The rule is fixed here,
before looking at results:

    pass  iff  fav_mean >= rest_mean - 0.0025  AND  fav_ci95[0] > 0  AND  n_fav >= 200

Each bet counts once (earliest sighting). Whether a bet was favorable is decided
with ranks and usage built only from games BEFORE the bet's game date, so the
backtest never sees the future. A bet that cannot be classified (unknown team,
too little history, no position) is not favorable and goes into "rest".
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from src.edge import players, store
from src.edge.social import nba_matchup
from src.edge.teams import ABBR_BY_NAME
from src.edge.views import results_summary
from src.utils.season import current_nba_season

TOLERANCE = 0.0025
MIN_FAV_BETS = 200
NFL_MIN_MEAN = 0.0
DEFAULT_OUT = "docs/backtests/nba-matchup-filter-2026-11.md"
BET_KEY = ("event_id", "market", "player", "side", "line")


def dedupe(rows: list[dict]) -> list[dict]:
    """One row per (event_id, market, player, side, line): the earliest `first_seen` sighting."""
    best: dict[tuple, dict] = {}
    for r in rows:
        k = tuple(r[f] for f in BET_KEY)
        if k not in best or r["first_seen"] < best[k]["first_seen"]:
            best[k] = r
    return list(best.values())


def _rows(clv: list[float]) -> list[dict]:
    # results_summary wants bet rows; bare CLV values are each their own bet.
    return [{"clv": v, "result": None, "profit_units": None, "market": "", "book": ""} for v in clv]


def _mean_ci(clv: list[float]) -> tuple[float | None, list[float] | None]:
    if not clv:
        return None, None
    ci = results_summary(_rows(clv))["clv"]["ci95"]
    return statistics.fmean(clv), ci


def gate(fav_clv: list[float], rest_clv: list[float], min_n: int = MIN_FAV_BETS) -> dict:
    fav_mean, fav_ci = _mean_ci(fav_clv)
    rest_mean, rest_ci = _mean_ci(rest_clv)
    n_fav = len(fav_clv)
    reasons = []
    if fav_mean is None or rest_mean is None:
        reasons.append("an empty group")
    else:
        if fav_mean < rest_mean - TOLERANCE:
            reasons.append(f"favorable trails the rest by more than {TOLERANCE:.4f}")
    if fav_ci is None or fav_ci[0] <= 0:
        reasons.append("favorable CLV 95% CI does not exclude 0")
    if n_fav < min_n:
        reasons.append(f"only {n_fav} favorable bets (need {min_n})")
    return {"pass": not reasons, "n_fav": n_fav, "n_rest": len(rest_clv), "fav_mean": fav_mean,
            "rest_mean": rest_mean, "fav_ci95": fav_ci, "rest_ci95": rest_ci, "failed": reasons}


def position_coverage(games: list[dict]) -> float | None:
    """Share of player-minutes whose player has a non-NULL position (None with no minutes)."""
    total = sum(g["minutes"] or 0 for g in games)
    if total <= 0:
        return None
    return sum(g["minutes"] or 0 for g in games if g.get("position")) / total


def load_opportunities(conn, season: int) -> list[dict]:
    """Closed, graded (win/loss/push) NBA player-prop opportunities with CLV, joined to the event and player mapping."""
    from src.utils.season import nba_season_start
    with conn.cursor() as cur:
        cur.execute("""SELECT o.event_id, o.market, o.player, o.side, o.line, o.book, o.clv, o.result,
                              o.profit_units, o.first_seen_at AS first_seen, e.home_team, e.away_team,
                              e.commence_time, p.bdl_player_id, p.position
                       FROM edge_opportunities o
                       JOIN edge_events e ON e.event_id = o.event_id
                       LEFT JOIN edge_players p ON p.odds_name = o.player
                       WHERE e.sport = 'basketball_nba' AND o.closing_done AND o.clv IS NOT NULL
                         AND o.result IN ('win', 'loss', 'push') AND o.player <> ''
                         AND e.commence_time >= %s AND o.market = ANY(%s)""",
                    (datetime.combine(nba_season_start(season), datetime.min.time(), tzinfo=timezone.utc),
                     list(players.STAT_PARTS)))
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.commit()
    for r in rows:
        r["line"], r["clv"] = float(r["line"]), float(r["clv"])
        r["profit_units"] = float(r["profit_units"]) if r["profit_units"] is not None else None
    return rows


def classify(opps: list[dict], games: list[dict]) -> list[dict]:
    """Add `favorable` (bool) and `why_not` (reason when not) to each bet, using only games before its date."""
    by_date: dict[date, list[dict]] = defaultdict(list)
    for o in opps:
        by_date[players.et_date(o["commence_time"])].append(o)
    out: list[dict] = []
    for d in sorted(by_date):
        prior = [g for g in games if g["game_date"] < d]
        ranks = nba_matchup.build_ranks(prior)
        totals = nba_matchup.team_totals(prior)
        by_player: dict[int, list[dict]] = defaultdict(list)
        for g in prior:
            by_player[g["bdl_player_id"]].append(g)
        for o in by_date[d]:
            fav, why = _favorable(o, ranks, totals, by_player)
            out.append({**o, "favorable": fav, "why_not": why})
    return out


def _favorable(o, ranks, totals, by_player) -> tuple[bool, str | None]:
    if not o.get("position"):
        return False, "no position"
    rows = by_player.get(o["bdl_player_id"]) if o.get("bdl_player_id") is not None else None
    if not rows:
        return False, "no prior games"
    team = max(rows, key=lambda g: g["game_date"])["team"]
    home, away = ABBR_BY_NAME.get(o["home_team"]), ABBR_BY_NAME.get(o["away_team"])
    if team == home:
        opp = away
    elif team == away:
        opp = home
    else:
        return False, "team not in event"
    opp_ranks = ranks.get(opp) if opp else None
    if opp_ranks is None:
        return False, "opponent unranked"
    usage = nba_matchup.player_usage(rows, totals)
    p = nba_matchup.panel(o, o["position"], opp_ranks, usage)
    return (True, None) if p.favorable else (False, "not favorable")


def evaluate(conn, season: int, min_n: int = MIN_FAV_BETS) -> dict:
    """Run the whole gate on the database. `min_n` exists only so tests can use a small dataset."""
    games = nba_matchup.load_games(conn, season)
    raw = load_opportunities(conn, season)
    bets = classify(dedupe(raw), games)
    fav = [b for b in bets if b["favorable"]]
    rest = [b for b in bets if not b["favorable"]]
    res = gate([b["clv"] for b in fav], [b["clv"] for b in rest], min_n)
    last = max((players.et_date(b["commence_time"]) for b in bets), default=None)
    window = [g for g in games if last is None or g["game_date"] < last]
    return {"gate": res, "fav": fav, "rest": rest, "raw_rows": len(raw), "bets": len(bets),
            "position_coverage": position_coverage(window),
            "why_not": {k: sum(1 for b in rest if b["why_not"] == k) for k in sorted({b["why_not"] for b in rest})}}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.2f}%"


def _group_row(name: str, bets: list[dict]) -> str:
    s = results_summary(bets)
    clv, rec = s["clv"], s["record"]
    ci = clv["ci95"]
    ci_txt = f"[{_pct(ci[0])}, {_pct(ci[1])}]" if ci else "n/a"
    return (f"| {name} | {len(bets)} | {_pct(clv['mean_clv'])} | {ci_txt} | {_pct(rec['win_rate'])} "
            f"| {rec['units']:+.2f} | {_pct(rec['roi'])} |")


def render_report(result: dict, season: int, today: date) -> str:
    g = result["gate"]
    verdict = "PASS" if g["pass"] else "FAIL"
    cov = result["position_coverage"]
    lines = [
        "# NBA matchup filter validation (Blind Plays gate)", "",
        f"Generated {today.isoformat()}, NBA season {season}-{str(season + 1)[-2:]}. **Verdict: {verdict}**", "",
        "## Pre-registered rule", "",
        f"Pass iff favorable mean CLV >= rest mean CLV - {TOLERANCE}, AND the favorable 95% CI lower bound > 0, "
        f"AND at least {MIN_FAV_BETS} favorable bets. Fixed before looking at results.", "",
        "## Method", "",
        "- Bets: graded NBA player-prop +EV opportunities with a closing line, one per "
        "(event, market, player, side, line), earliest sighting kept.",
        "- Favorable is decided with ranks and usage from games before each bet's game date (no look-ahead).",
        "- Bets that cannot be classified (no position, no prior games, unranked opponent) count as not favorable.",
        "- CI95 is mean +/- 1.96 standard errors across bets. Win rate and units are reported, not gated.", "",
        "## Results", "",
        "| Group | Bets | Mean CLV | CLV 95% CI | Win rate | Units | ROI |", "|---|---|---|---|---|---|---|",
        _group_row("Favorable", result["fav"]), _group_row("Rest", result["rest"]),
        _group_row("All", result["fav"] + result["rest"]), "",
        f"Favorable minus rest: {_pct(g['fav_mean'] - g['rest_mean']) if g['rest_mean'] is not None and g['fav_mean'] is not None else 'n/a'}",
        "",
        "## Data quality", "",
        f"- Opportunity rows: {result['raw_rows']}; unique bets: {result['bets']}.",
        f"- Position coverage: {_pct(cov)} of player-minutes in the season to date (games before the last bet "
        f"date) belong to a player with a non-NULL position.",
        "- Why bets were not favorable: " + (", ".join(f"{k} {v}" for k, v in result["why_not"].items()) or "n/a"), "",
        "## Why it " + ("passed" if g["pass"] else "failed"), "",
    ]
    lines += [f"- {x}" for x in g["failed"]] if g["failed"] else ["- All three conditions held."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sport", choices=["nba", "nfl"], required=True)
    ap.add_argument("--write-state", action="store_true", help="record pass/fail as social_blind_gate:<sport>")
    ap.add_argument("--out", default=DEFAULT_OUT, help="report path (relative to backend/)")
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args(argv)
    if args.sport == "nfl":
        print("not implemented: the NFL gate arrives with the NFL matchup filter", file=sys.stderr)
        return 2
    season = args.season or current_nba_season()
    conn = store.connect()
    try:
        result = evaluate(conn, season)
        now = datetime.now(timezone.utc)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_report(result, season, now.date()))
        if args.write_state:
            store.set_state(conn, "social_blind_gate:nba", "pass" if result["gate"]["pass"] else "fail", now)
    finally:
        conn.close()
    g = result["gate"]
    print(f"nba matchup gate: {'PASS' if g['pass'] else 'FAIL'} (n_fav={g['n_fav']}); report {out}")
    return 0 if g["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
