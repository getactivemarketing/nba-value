"""Which historical snapshots to buy, and what they cost (spec §5, §9). Pure."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta, timezone

from src.edge.backtest.nflverse import ET, Game
from src.edge.config import LINE_MARKETS, NFL_PROP_MARKETS, TD_MARKET
from src.edge.ingest import _ts
from src.edge.teams import NFL_ABBR_BY_NAME

LABELS = {"t24h": timedelta(hours=24), "t6h": timedelta(hours=6),
          "t1h": timedelta(hours=1), "close": timedelta(minutes=5)}
FLAG_LABELS = ("t24h", "t6h", "t1h")
REGION_EQUIVALENTS = 2          # 16 bookmakers; every 10 count as one region
EVENTS_COST = 1
BT_PROP_MARKETS = tuple(m for m in NFL_PROP_MARKETS if m != TD_MARKET)
PROPS_COST = 10 * REGION_EQUIVALENTS * len(BT_PROP_MARKETS)
LINES_COST = 10 * REGION_EQUIVALENTS * len(LINE_MARKETS)
SLATE_GAP = timedelta(minutes=15)
MATCH_TOLERANCE = timedelta(hours=3)
EVENTS_AT = time(10, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Job:
    kind: str                       # events | props | lines
    label: str                      # LABELS key, or 'list' for events
    requested_at: datetime
    games: tuple[Game, ...]
    event_ids: tuple[str, ...]      # parallel to games; ('day:YYYY-MM-DD',) for events

    @property
    def cost(self) -> int:
        return {"events": EVENTS_COST, "props": PROPS_COST, "lines": LINES_COST}[self.kind]

    @property
    def keys(self) -> list[tuple[str, str, str]]:
        return [(e, self.kind, self.label) for e in self.event_ids]


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def day_jobs(games: list[Game]) -> list[Job]:
    days: dict = {}
    for gm in games:
        days.setdefault(gm.kickoff.astimezone(ET).date(), []).append(gm)
    return [Job("events", "list", datetime.combine(d, EVENTS_AT), tuple(gs), (f"day:{d.isoformat()}",))
            for d, gs in sorted(days.items())]


def match_events(games: list[Game], events: list[dict]) -> tuple[list[tuple[Game, str]], list[str]]:
    pairs, unmatched = [], []
    for gm in games:
        hit = None
        for e in events:
            if (NFL_ABBR_BY_NAME.get(e.get("home_team")) == gm.home
                    and NFL_ABBR_BY_NAME.get(e.get("away_team")) == gm.away
                    and abs(_ts(e["commence_time"]) - gm.kickoff) <= MATCH_TOLERANCE):
                hit = e
                break
        if hit is None:
            unmatched.append(gm.game_id)
        else:
            pairs.append((replace(gm, kickoff=_ts(hit["commence_time"])), hit["id"]))
    return pairs, unmatched


def _slates(pairs: list[tuple[Game, str]]) -> list[list[tuple[Game, str]]]:
    out: list[list[tuple[Game, str]]] = []
    for p in sorted(pairs, key=lambda p: p[0].kickoff):
        if out and p[0].kickoff - out[-1][0][0].kickoff <= SLATE_GAP:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def snapshot_jobs(pairs: list[tuple[Game, str]]) -> list[Job]:
    jobs = [Job("props", label, gm.kickoff - off, (gm,), (eid,))
            for gm, eid in pairs for label, off in LABELS.items()]
    for slate in _slates(pairs):
        first = slate[0][0].kickoff
        jobs += [Job("lines", label, first - off, tuple(p[0] for p in slate), tuple(p[1] for p in slate))
                 for label, off in LABELS.items()]
    return sorted(jobs, key=lambda j: (j.requested_at, j.kind, j.event_ids))


def dry_run(games: list[Game]) -> dict:
    days = day_jobs(games)
    snaps = snapshot_jobs([(gm, gm.game_id) for gm in games])
    props = sum(1 for j in snaps if j.kind == "props")
    lines = sum(1 for j in snaps if j.kind == "lines")
    return {"games": len(games), "events_calls": len(days), "props_calls": props, "lines_calls": lines,
            "credits": sum(j.cost for j in days) + sum(j.cost for j in snaps)}
