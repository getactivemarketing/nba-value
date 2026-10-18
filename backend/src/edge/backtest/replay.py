"""Replays stored snapshots through the live engine (spec §7).

`replay_game` is pure; `run_replay` reads edge_bt_snapshots one event at a time
(the payloads are large) and writes edge_bt_picks under a fresh run_id.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable

from psycopg2.extras import Json, execute_values

from src.edge.backtest.nflverse import Game
from src.edge.backtest.plan import FLAG_LABELS
from src.edge.config import DFS_BOOKS, SHARP_BOOKS
from src.edge.engine import american_to_decimal, fair_prob, group, price_quotes
from src.edge.engine import clv as clv_of
from src.edge.grading import grade_game, profit_units
from src.edge.ingest import parse_event
from src.edge.matchups import Tag, TagContext, build_context, tags_for
from src.edge.nfl_players import nfl_prop_result
from src.edge.players import normalize_name
from src.edge.quotes import Quote, sides_for

log = logging.getLogger("edge.backtest.replay")
PIN_EXCLUDE = frozenset({"novig", "prophetx"})
PICK_COLS = ("run_id", "season", "week", "event_id", "label", "market", "player", "side", "line", "book",
             "price", "fair_prob", "ev", "verify", "pin_ev", "n_books", "sharps", "closing_prob", "clv",
             "result", "profit", "tags")


@dataclass(frozen=True)
class Snap:
    payload: dict
    snapshot_at: datetime | None


def _quotes(by_kind: dict[str, Snap], before: datetime, inclusive: bool) -> list[Quote]:
    out: list[Quote] = []
    for s in by_kind.values():
        if s.snapshot_at is None or not s.payload.get("bookmakers"):
            continue
        if s.snapshot_at > before or (not inclusive and s.snapshot_at == before):
            continue
        out += parse_event(s.payload, s.snapshot_at)
    return out


def _sharps(books: dict, market: str, exclude_book: str) -> str:
    a, b = sides_for(market)
    return ",".join(sorted(bk for bk, sides in books.items()
                           if bk in SHARP_BOOKS and bk != exclude_book and bk not in DFS_BOOKS
                           and a in sides and b in sides))


def grade(market: str, player: str, side: str, line: float, game: Game, stats: list[dict],
          snap_names: set[str] | None, known: set[str] | None = None) -> str | None:
    """`known` = every name key nflverse has for the season; a key outside it is an
    unmatched name (ungradable), never an inactive player (void)."""
    if not player:
        if game.home_score is None or game.away_score is None:
            return None
        return grade_game(market, side, line, game.home_score, game.away_score)
    key = normalize_name(player)
    rows = [r for r in stats if r["name"] == key]
    if len(rows) > 1:
        return None  # two players in this game share the name key: never guess
    if not rows and known is not None and key not in known:
        return None  # nickname or variant neither nflverse source uses: never a silent void
    return nfl_prop_result({"market": market, "line": line, "side": side}, rows[0] if rows else None,
                           snaps_loaded=snap_names is not None,
                           played=snap_names is not None and key in snap_names)


def replay_game(game: Game, snaps: dict[str, dict[str, Snap]], stats: list[dict], snap_names: set[str] | None,
                min_ev: float, verify_ev: float, known: set[str] | None = None,
                tagger: Callable[[str, str, str], list[Tag]] | None = None) -> list[dict]:
    # The close may sit at kickoff; a snapshot after kickoff holds in-game prices and is never the close.
    closing = group(_quotes(snaps.get("close", {}), game.kickoff, inclusive=True))
    picks = []
    for label in FLAG_LABELS:
        quotes = _quotes(snaps.get(label, {}), game.kickoff, inclusive=False)
        if not quotes:
            continue
        grouped = group(quotes)
        for p in price_quotes(quotes, min_ev, verify_ev):
            if not p.is_ev:
                continue
            q = p.quote
            books = grouped[q.market_key]
            pin = fair_prob({b: v for b, v in books.items() if b not in PIN_EXCLUDE}, q.market, q.side,
                            exclude_book=q.book)
            close = fair_prob(closing.get(q.market_key, {}), q.market, q.side, exclude_book=None)
            closing_prob = None if close.thin else close.prob
            result = grade(q.market, q.player, q.side, q.line, game, stats, snap_names, known)
            picks.append({
                "event_id": q.event_id, "label": label, "market": q.market, "player": q.player, "side": q.side,
                "line": q.line, "book": q.book, "price": q.price, "fair_prob": p.fair.prob, "ev": p.ev,
                "verify": p.verify,
                "pin_ev": None if pin.thin else pin.prob * american_to_decimal(q.price) - 1,
                "n_books": p.fair.n_books, "sharps": _sharps(books, q.market, q.book),
                "closing_prob": closing_prob,
                "clv": None if closing_prob is None else clv_of(closing_prob, q.price),
                "result": result,
                "profit": profit_units(result, q.price) if result in ("win", "loss", "push") else None,
                # Additive only: tags never touch the price, EV or grade above.
                "tags": [{"key": t.key, "direction": t.direction} for t in tagger(q.player, q.market, q.side)]
                        if tagger and q.player else [],
            })
    return picks


def make_contexts(team_games, player_games, injury_index) -> Callable[[Game], TagContext]:
    """One TagContext per (season, week), built from rows strictly before it. Injuries are that week's
    own pre-game report only; another week's report is never used."""
    cache: dict[tuple[int, int], TagContext] = {}

    def contexts(game: Game) -> TagContext:
        k = (game.season, game.week)
        if k not in cache:
            cache[k] = build_context(team_games, player_games, injury_index.get(k, {}), *k)
        return cache[k]
    return contexts


def run_replay(conn, run_id: str, games: dict[str, Game], stats: dict[str, list[dict]],
               snaps_idx: dict[str, set[str]], min_ev: float, verify_ev: float,
               contexts: Callable[[Game], TagContext] | None = None) -> int:
    with conn.cursor() as cur:
        cur.execute("""SELECT DISTINCT event_id, game_id, kickoff FROM edge_bt_snapshots
                       WHERE kind IN ('props', 'lines', 'td') AND game_id = ANY(%s) ORDER BY game_id""",
                    (sorted(games),))
        targets = cur.fetchall()
    known: dict[int, set[str]] = defaultdict(set)
    for gid, gm in games.items():
        known[gm.season] |= {r["name"] for r in stats.get(gid, [])} | snaps_idx.get(gid, set())
    total = 0
    for event_id, game_id, kickoff in targets:
        game = replace(games[game_id], kickoff=kickoff)
        with conn.cursor() as cur:
            cur.execute("""SELECT kind, label, payload, snapshot_at FROM edge_bt_snapshots
                           WHERE event_id = %s AND kind IN ('props', 'lines', 'td')""", (event_id,))
            by_label: dict[str, dict[str, Snap]] = defaultdict(dict)
            for kind, label, payload, at in cur.fetchall():
                by_label[label][kind] = Snap(payload, at)
        tagger = None
        if contexts is not None:
            ctx = contexts(game)
            teams = (game.home, game.away)
            tagger = lambda player, market, side: tags_for(ctx, player, teams, market, side)  # noqa: E731
        picks = replay_game(game, by_label, stats.get(game_id, []), snaps_idx.get(game_id), min_ev, verify_ev,
                             known[game.season], tagger=tagger)
        rows = [tuple(Json(v) if c == "tags" else v
                      for c, v in ((c, {**p, "run_id": run_id, "season": game.season, "week": game.week}[c])
                                   for c in PICK_COLS)) for p in picks]
        if rows:
            with conn.cursor() as cur:
                execute_values(cur, f"INSERT INTO edge_bt_picks ({', '.join(PICK_COLS)}) VALUES %s "
                                    "ON CONFLICT DO NOTHING", rows)
        conn.commit()
        total += len(rows)
    log.info("backtest replay %s: %d picks from %d events", run_id, total, len(targets))
    return total
