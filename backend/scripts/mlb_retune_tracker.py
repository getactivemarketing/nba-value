#!/usr/bin/env python3
"""MLB value-retune performance tracker.

Reports cumulative graded best_bet results for the retuned model since the
clean-deploy baseline (2026-07-08), framed against the Apr 3-Jul 5 backtest.

Pre-2026-07-08 season data is contaminated by the dynamic-bravery duplicate
service (old-formula snapshots on the shared DB), so attribution to the retune
is only valid from the baseline forward. See truline-session-jul6-mlb-retune.

Usage: run from backend/ (needs the hardcoded prod DB_URL in
src/tasks/prediction_tracker.py). Prints a report to stdout.
"""
import os
import re
import sys
from pathlib import Path

BASELINE = "2026-07-08"  # first clean new-code day (SUPPRESS_TOTALS=false, dynamic-bravery down)

# Backtest baseline, best_bet, Apr 3-Jul 5 2026, $100 flat units.
BT_UNITS = 160.0
BT_WR = 50.9
BT_PICKS = 889

# Totals re-entry gate thresholds.
GATE_MIN_N = 100
GATE_MIN_WR = 53.0


def get_db_url() -> str:
    tracker = Path(__file__).resolve().parent.parent / "src" / "tasks" / "prediction_tracker.py"
    text = tracker.read_text()
    m = re.search(r"postgresql://[^\"'\s]+", text)
    if not m:
        sys.exit("Could not find prod DB URL in prediction_tracker.py")
    return m.group(0)


def main() -> None:
    import psycopg2

    conn = psycopg2.connect(os.environ.get("PGURL") or get_db_url())
    cur = conn.cursor()

    # --- Cumulative best_bet since baseline ---
    cur.execute(
        """
        SELECT COUNT(*),
               SUM(CASE WHEN best_bet_profit > 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN best_bet_profit < 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN best_bet_profit = 0 THEN 1 ELSE 0 END),
               ROUND((SUM(best_bet_profit) / 100.0)::numeric, 2)
        FROM mlb_prediction_snapshots
        WHERE game_date >= %s AND best_bet_profit IS NOT NULL AND actual_winner IS NOT NULL
        """,
        (BASELINE,),
    )
    n, w, l, p, units = cur.fetchone()
    n = n or 0
    w, l, p = w or 0, l or 0, p or 0
    units = float(units or 0)
    decided = w + l
    wr = (w / decided * 100) if decided else 0.0

    print(f"=== MLB retune tracker  |  baseline {BASELINE} -> now ===")
    print(f"BEST_BET: {w}-{l}-{p}  ({wr:.1f}% WR, {n} graded)  {units:+.2f}u")
    print(f"  backtest baseline (Apr 3-Jul 5): {BT_WR}% WR, {BT_UNITS:+.1f}u over {BT_PICKS} picks")
    if n:
        pace = units / n
        print(f"  clean-window pace: {pace:+.3f}u/pick")

    # --- By market type ---
    print("\nby market:")
    cur.execute(
        """
        SELECT best_bet_type, COUNT(*),
               SUM(CASE WHEN best_bet_profit > 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN best_bet_profit < 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN best_bet_profit = 0 THEN 1 ELSE 0 END),
               ROUND((SUM(best_bet_profit) / 100.0)::numeric, 2)
        FROM mlb_prediction_snapshots
        WHERE game_date >= %s AND best_bet_profit IS NOT NULL AND actual_winner IS NOT NULL
        GROUP BY best_bet_type ORDER BY 6 DESC
        """,
        (BASELINE,),
    )
    for t, tn, tw, tl, tp, tu in cur.fetchall():
        print(f"  {t:<10} {tw}-{tl}-{tp}  ({tn})  {float(tu or 0):+.2f}u")

    # --- Last night ---
    cur.execute(
        """
        SELECT game_date, COUNT(*),
               SUM(CASE WHEN best_bet_profit > 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN best_bet_profit < 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN best_bet_profit = 0 THEN 1 ELSE 0 END),
               ROUND((SUM(best_bet_profit) / 100.0)::numeric, 2)
        FROM mlb_prediction_snapshots
        WHERE game_date >= %s AND best_bet_profit IS NOT NULL AND actual_winner IS NOT NULL
        GROUP BY game_date ORDER BY game_date DESC LIMIT 1
        """,
        (BASELINE,),
    )
    row = cur.fetchone()
    if row:
        d, dn, dw, dl, dp, du = row
        print(f"\nlast graded night {d}: {dw}-{dl}-{dp}  {float(du or 0):+.2f}u")

    # --- Saturation health check ---
    cur.execute(
        """
        SELECT MIN(best_bet_value_score), MAX(best_bet_value_score),
               ROUND(AVG(best_bet_value_score)::numeric, 1),
               SUM(CASE WHEN best_bet_value_score >= 99.5 THEN 1 ELSE 0 END), COUNT(*)
        FROM mlb_prediction_snapshots
        WHERE game_date >= %s AND best_bet_value_score IS NOT NULL
        """,
        (BASELINE,),
    )
    smin, smax, savg, sat, sn = cur.fetchone()
    flag = "" if (sat or 0) == 0 else "  <-- SATURATION REGRESSION"
    print(f"\nscore health: min={smin} max={smax} avg={savg}  saturated={sat}/{sn}{flag}")

    # --- Totals re-entry gate progress ---
    cur.execute(
        """
        SELECT COUNT(*),
               SUM(CASE WHEN best_total_profit > 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN best_total_profit < 0 THEN 1 ELSE 0 END),
               ROUND((SUM(best_total_profit) / 100.0)::numeric, 2)
        FROM mlb_prediction_snapshots
        WHERE game_date >= %s AND best_total_profit IS NOT NULL AND actual_winner IS NOT NULL
        """,
        (BASELINE,),
    )
    tn, tw, tl, tu = cur.fetchone()
    tn = tn or 0
    tw, tl = tw or 0, tl or 0
    twr = (tw / (tw + tl) * 100) if (tw + tl) else 0.0
    tu = float(tu or 0)
    gate_ok = tn >= GATE_MIN_N and twr >= GATE_MIN_WR and tu > 0
    status = "OPEN" if gate_ok else f"not yet ({tn}/{GATE_MIN_N} graded)"
    print(f"\ntotals re-entry gate [{status}]: shadow {tw}-{tl} ({twr:.1f}% WR) {tu:+.2f}u")
    print(f"  need >={GATE_MIN_N} graded AND WR>={GATE_MIN_WR}% AND units>0 to flip totals_in_best_bet")

    # --- Data-integrity guard: warn if snapshot count spikes (duplicate service redux) ---
    cur.execute(
        """
        SELECT game_date, COUNT(*)
        FROM mlb_prediction_snapshots
        WHERE game_date >= %s
        GROUP BY game_date HAVING COUNT(*) > 20 ORDER BY game_date
        """,
        (BASELINE,),
    )
    spikes = cur.fetchall()
    if spikes:
        print("\n!! snapshot-count spike (possible duplicate scheduler / dynamic-bravery redux):")
        for d, c in spikes:
            print(f"   {d}: {c} snapshots (>20 = suspicious for a ~15-game slate)")

    conn.close()


if __name__ == "__main__":
    main()
