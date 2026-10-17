"""Backtest report (spec §11). Pure aggregation plus one DB entry point.

A pick flagged at several lead times counts once (its earliest label) in every
table except the lead-time ones, which treat each label on its own.
"""
from __future__ import annotations

from math import sqrt
from pathlib import Path
from statistics import mean, pstdev

from psycopg2.extras import RealDictCursor

from src.config import settings
from src.edge.backtest.plan import FLAG_LABELS

BUCKETS = ("2–4%", "4–6%", "6%+")
BUCKET_FLOOR = {"2–4%": 2, "4–6%": 4, "6%+": 6}
SMALL = 50
MIN_RECOMMEND = 200
_ORDER = {label: i for i, label in enumerate(FLAG_LABELS)}


def bucket(ev: float) -> str:
    return "2–4%" if ev < 0.04 else "4–6%" if ev < 0.06 else "6%+"


def first_flag(picks: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for p in picks:
        k = (p["event_id"], p["market"], p["player"], p["side"], float(p["line"]), p["book"])
        if k not in best or _ORDER[p["label"]] < _ORDER[best[k]["label"]]:
            best[k] = p
    return list(best.values())


def metrics(rows: list[dict]) -> dict:
    graded = [r for r in rows if r["result"] in ("win", "loss", "push")]
    wl = [r for r in graded if r["result"] in ("win", "loss")]
    profits = [float(r["profit"]) for r in graded]
    clvs = [float(r["clv"]) for r in rows if r["clv"] is not None]
    return {
        "picks": len(rows), "graded": len(graded),
        "win_pct": sum(r["result"] == "win" for r in wl) / len(wl) if wl else None,
        "roi": mean(profits) if profits else None,
        "roi_ci": 1.96 * pstdev(profits) / sqrt(len(profits)) if len(profits) > 1 else None,
        "avg_ev": mean(float(r["ev"]) for r in rows) if rows else None,
        "avg_clv": mean(clvs) if clvs else None,
        "beat_close": sum(c > 0 for c in clvs) / len(clvs) if clvs else None,
        "small": len(rows) < SMALL,
    }


def _pct(v, signed=False) -> str:
    if v is None:
        return "–"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def _table(groups: list[tuple[str, list[dict]]], first_col: str) -> str:
    out = [f"| {first_col} | Picks | Win% | ROI (95% CI) | Avg EV | Avg CLV | Beat close | |",
           "|---|---|---|---|---|---|---|---|"]
    for name, rows in groups:
        m = metrics(rows)
        roi = "–" if m["roi"] is None else f"{_pct(m['roi'], True)} ± {_pct(m['roi_ci'])}"
        out.append(f"| {name} | {m['picks']:,} | {_pct(m['win_pct'])} | {roi} | {_pct(m['avg_ev'])} | "
                   f"{_pct(m['avg_clv'], True)} | {_pct(m['beat_close'])} | "
                   f"{'small sample' if m['small'] else ''} |")
    return "\n".join(out)


def _by(rows: list[dict], key) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(key(r)), []).append(r)
    return sorted(groups.items())


def recommend(picks: list[dict]) -> list[str]:
    once = first_flag(picks)

    def winners(key) -> list[str]:
        return [name for name, rows in _by(once, key)
                if len(rows) >= MIN_RECOMMEND and (metrics(rows)["avg_clv"] or 0) > 0]

    out = [f"- **Markets to show:** {', '.join(winners(lambda r: r['market'])) or 'none qualify'} "
           f"(average CLV > 0, at least {MIN_RECOMMEND} picks)",
           f"- **Books to show:** {', '.join(winners(lambda r: r['book'])) or 'none qualify'}"]
    by_bucket = dict(_by(once, lambda r: bucket(float(r["ev"]))))
    floor = next((b for b in BUCKETS if b in by_bucket and (metrics(by_bucket[b])["avg_clv"] or 0) > 0), None)
    out.append(f"- **EV threshold:** EV ≥ {BUCKET_FLOOR[floor]}% (lowest bucket with average CLV > 0)" if floor
               else "- **EV threshold:** no EV bucket beats the close on average")
    leads = [(metrics(rows)["avg_clv"], name) for name, rows in _by(picks, lambda r: r["label"])
             if metrics(rows)["avg_clv"] is not None]
    out.append(f"- **Lead time:** {max(leads)[1]} has the best average CLV" if leads
               else "- **Lead time:** no lead time has CLV data")
    return out


def build_report(picks: list[dict], meta: dict) -> str:
    once = first_flag(picks)
    pin = [r for r in once if r["pin_ev"] is not None and float(r["pin_ev"]) >= settings.edge_min_ev]
    sections = [
        "# NFL +EV backtest, 2023–2025 regular seasons",
        f"- Run `{meta['run_id']}` · {meta['snapshots']:,} snapshots · {meta['credits']:,} credits spent",
        f"- {len(picks):,} flagged rows; {len(once):,} distinct picks (counted once, at the earliest lead time)",
        f"- Ungradable: {sum(r['result'] is None for r in once):,} · Void: "
        f"{sum(r['result'] == 'void' for r in once):,} · No close (line moved or thin): "
        f"{sum(r['clv'] is None for r in once):,} · Verify-flagged (EV > 15%): "
        f"{sum(bool(r['verify']) for r in once):,}",
        "", "## Overall", _table([("All", once)], "Scope"),
        "", "## By season", _table(_by(once, lambda r: r["season"]), "Season"),
        "", "## By market", _table(_by(once, lambda r: r["market"]), "Market"),
        "", "## By book", _table(_by(once, lambda r: r["book"]), "Book"),
        "", "## By EV bucket", _table(_by(once, lambda r: bucket(float(r["ev"]))), "EV"),
        "", "## By lead time", _table(_by(picks, lambda r: r["label"]), "Lead time"),
        "", "## Season × lead time", _table(_by(picks, lambda r: f"{r['season']} {r['label']}"), "Season, lead"),
        "", "## Pinnacle-only consensus",
        "Picks that stay +EV with Novig and ProphetX left out of the consensus, for a like-for-like 2023 vs 2025 view.",
        _table(_by(pin, lambda r: r["season"]), "Season"),
        "", "## Recommendations", *recommend(picks), "",
    ]
    return "\n".join(sections)


def run_report(conn, run_id: str | None, out_path) -> str:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if run_id is None:
            cur.execute("SELECT run_id FROM edge_bt_picks ORDER BY run_id DESC LIMIT 1")
            row = cur.fetchone()
            if row is None:
                raise SystemExit("no replay run found; run `replay` first")
            run_id = row["run_id"]
        cur.execute("SELECT * FROM edge_bt_picks WHERE run_id = %s", (run_id,))
        picks = cur.fetchall()
        cur.execute("SELECT COUNT(*) AS n, COALESCE(SUM(credits), 0) AS c FROM edge_bt_snapshots")
        snap = cur.fetchone()
    md = build_report(picks, {"run_id": run_id, "snapshots": int(snap["n"]), "credits": int(snap["c"])})
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    return run_id
