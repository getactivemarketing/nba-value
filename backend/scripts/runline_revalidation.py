#!/usr/bin/env python3
"""Corrected runline re-validation (read-only).

Re-derives the correctly-paired runline pick for each graded MLB snapshot from
the frozen predicted_run_diff + historical mlb_markets prices (via the fixed
MLBScorer._runline_side_values), grades it against the actual margin, and reports
the real runline edge vs the tracker's inflated figure. Does NOT mutate the DB.

Run from backend/: python3 scripts/runline_revalidation.py
"""
import os
import re
import sys
from pathlib import Path

BASELINE = "2026-07-08"


def grade_runline(bet_team, signed_line, home_team, home_score, away_score, odds_decimal):
    if bet_team == home_team:
        bet_score, opp_score = home_score, away_score
    else:
        bet_score, opp_score = away_score, home_score
    adjusted = bet_score + signed_line
    if adjusted > opp_score:
        return "win", odds_decimal - 1.0
    if adjusted == opp_score:
        return "push", 0.0
    return "loss", -1.0


def _db_url():
    tracker = Path(__file__).resolve().parent.parent / "src" / "tasks" / "prediction_tracker.py"
    m = re.search(r"postgresql://[^\"'\s]+", tracker.read_text())
    if not m:
        sys.exit("prod DB URL not found")
    return m.group(0)


def corrected_pick(predicted_run_diff, market_rows):
    """market_rows: list of (line, home_odds, away_odds, home_team, away_team).
    Returns the best correctly-paired runline MLBValueResult or None."""
    from src.services.mlb.scorer import MLBScorer
    from src.services.mlb.value_calculator import MLBValueCalculator
    vals = []
    for line, ho, ao, h, a in market_rows:
        if line is None or abs(float(line)) != 1.5 or ho is None or ao is None:
            continue
        vals += MLBScorer._runline_side_values(
            float(predicted_run_diff), h, a, float(line), float(ho), float(ao)
        )
    return MLBValueCalculator.find_best_value(vals)


def main():
    import psycopg2
    conn = psycopg2.connect(os.environ.get("PGURL") or _db_url())
    cur = conn.cursor()
    cur.execute(
        """
        SELECT game_id, game_date, home_team, away_team, predicted_run_diff,
               home_score, away_score,
               best_bet_type, best_bet_team, best_bet_value_score
        FROM mlb_prediction_snapshots
        WHERE predicted_run_diff IS NOT NULL AND actual_winner IS NOT NULL
          AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY game_date
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("no graded snapshots found")
        return

    def market_rows(gid):
        cur.execute(
            """SELECT line, home_odds, away_odds FROM mlb_markets
               WHERE game_id=%s AND market_type='runline'
                 AND home_odds IS NOT NULL AND away_odds IS NOT NULL""",
            (gid,),
        )
        return cur.fetchall()

    windows = {"since_baseline": [], "all_history": []}
    dates = []
    for (gid, d, home, away, rd, hs, as_, bt, bteam, bscore) in rows:
        dates.append(d)
        mrows = [(l, ho, ao, home, away) for (l, ho, ao) in market_rows(gid)]
        pick = corrected_pick(rd, mrows)
        if pick is None:
            continue
        res, profit = grade_runline(pick.team, float(pick.line), home, hs, as_, float(pick.odds_decimal))
        rec = (res, profit)
        windows["all_history"].append(rec)
        if str(d) >= BASELINE:
            windows["since_baseline"].append(rec)

    def summarize(name, recs):
        n = len(recs)
        w = sum(1 for r, _ in recs if r == "win")
        l = sum(1 for r, _ in recs if r == "loss")
        u = sum(p for _, p in recs)
        wr = w / (w + l) * 100 if (w + l) else 0.0
        print(f"  {name}: {w}-{l} ({wr:.1f}% WR over {n}) -> {u:+.1f}u (flat)")

    print(f"=== corrected runline re-validation | snapshots {min(dates)}..{max(dates)} ===")
    print("(re-derived from frozen predicted_run_diff + historical odds; snapshots NOT modified)")
    summarize("since 2026-07-08 baseline", windows["since_baseline"])
    summarize("all snapshot history", windows["all_history"])
    if min(str(d) for d in dates) > "2026-04-05":
        print(f"  NOTE: snapshots start {min(dates)} — earlier season (Apr start) not covered; "
              f"full-season backtest number cannot be reproduced from snapshots.")


if __name__ == "__main__":
    main()
