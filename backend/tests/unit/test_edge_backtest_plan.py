from datetime import datetime, timedelta, timezone

from src.edge.backtest import plan
from src.edge.backtest.nflverse import Game

UTC = timezone.utc


def g(gid="2023_10_CLE_BAL", home="BAL", away="CLE", kick=datetime(2023, 11, 12, 18, 0, tzinfo=UTC)):
    return Game(gid, 2023, 10, home, away, kick, 31, 33)


def ev(eid, home, away, commence):
    return {"id": eid, "home_team": home, "away_team": away, "commence_time": commence}


def test_costs():
    assert (plan.EVENTS_COST, plan.PROPS_COST, plan.LINES_COST) == (1, 200, 60)


def test_props_jobs_use_the_four_offsets():
    jobs = [j for j in plan.snapshot_jobs([(g(), "e1")]) if j.kind == "props"]
    got = {j.label: j.requested_at for j in jobs}
    k = datetime(2023, 11, 12, 18, 0, tzinfo=UTC)
    assert got == {"t24h": k - timedelta(hours=24), "t6h": k - timedelta(hours=6),
                   "t1h": k - timedelta(hours=1), "close": k - timedelta(minutes=5)}
    assert jobs[0].keys == [("e1", "props", jobs[0].label)]


def test_lines_jobs_group_kickoffs_within_15_minutes_into_one_slate():
    a = g("a", kick=datetime(2023, 11, 12, 18, 0, tzinfo=UTC))
    b = g("b", "NE", "IND", kick=datetime(2023, 11, 12, 18, 5, tzinfo=UTC))
    c = g("c", "SF", "JAX", kick=datetime(2023, 11, 12, 21, 25, tzinfo=UTC))
    lines = [j for j in plan.snapshot_jobs([(a, "ea"), (b, "eb"), (c, "ec")]) if j.kind == "lines"]
    assert len(lines) == 8
    close_ab = next(j for j in lines if j.label == "close" and "ea" in j.event_ids)
    assert close_ab.event_ids == ("ea", "eb")
    assert close_ab.requested_at == datetime(2023, 11, 12, 17, 55, tzinfo=UTC)
    assert close_ab.keys == [("ea", "lines", "close"), ("eb", "lines", "close")]


def test_snapshot_jobs_sorted_by_request_time():
    jobs = plan.snapshot_jobs([(g(), "e1")])
    assert [j.requested_at for j in jobs] == sorted(j.requested_at for j in jobs)


def test_day_jobs_use_the_eastern_game_day_at_10_utc():
    mnf = g("m", "LAR", "WSH", kick=datetime(2023, 11, 14, 1, 15, tzinfo=UTC))  # Monday 20:15 ET
    jobs = plan.day_jobs([g(), mnf])
    assert [(j.event_ids, j.requested_at) for j in jobs] == [
        (("day:2023-11-12",), datetime(2023, 11, 12, 10, 0, tzinfo=UTC)),
        (("day:2023-11-13",), datetime(2023, 11, 13, 10, 0, tzinfo=UTC))]
    assert jobs[0].keys == [("day:2023-11-12", "events", "list")]


def test_match_events_by_teams_and_time_including_renamed_abbreviations():
    games = [g(), g("2023_10_WAS_LA", "LAR", "WSH", datetime(2023, 11, 14, 1, 15, tzinfo=UTC)),
             g("2023_10_X", "KC", "MIA", datetime(2023, 11, 5, 14, 30, tzinfo=UTC))]
    events = [ev("e1", "Baltimore Ravens", "Cleveland Browns", "2023-11-12T18:03:00Z"),
              ev("e2", "Los Angeles Rams", "Washington Commanders", "2023-11-14T01:20:00Z"),
              ev("e3", "Miami Dolphins", "Kansas City Chiefs", "2023-11-05T14:30:00Z")]  # home/away swapped
    pairs, unmatched = plan.match_events(games, events)
    assert [(p[0].game_id, p[1]) for p in pairs] == [("2023_10_CLE_BAL", "e1"), ("2023_10_WAS_LA", "e2")]
    assert pairs[0][0].kickoff == datetime(2023, 11, 12, 18, 3, tzinfo=UTC)  # event time wins
    assert unmatched == ["2023_10_X"]


def test_match_events_rejects_kickoff_more_than_3_hours_off():
    pairs, unmatched = plan.match_events([g()], [ev("e1", "Baltimore Ravens", "Cleveland Browns",
                                                    "2023-11-12T22:30:00Z")])
    assert pairs == [] and unmatched == ["2023_10_CLE_BAL"]


def test_dry_run_counts_calls_and_credits():
    a = g("a")
    b = g("b", "NE", "IND", kick=datetime(2023, 11, 12, 18, 5, tzinfo=UTC))
    assert plan.dry_run([a, b]) == {"games": 2, "events_calls": 1, "props_calls": 8, "lines_calls": 4,
                                    "credits": 1 + 8 * 200 + 4 * 60}


def test_iso():
    assert plan.iso(datetime(2023, 11, 12, 17, 55, tzinfo=UTC)) == "2023-11-12T17:55:00Z"
