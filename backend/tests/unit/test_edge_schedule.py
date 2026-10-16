# backend/tests/unit/test_edge_schedule.py
from datetime import datetime, timedelta, timezone

from src.edge import schedule as s
from src.edge.quotes import Quote
from src.edge.worker import Backoff

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def ev(eid, hours_to_tip, last_lines=None, last_props=None):
    return {"event_id": eid, "commence_time": NOW + timedelta(hours=hours_to_tip),
            "last_lines_poll": last_lines, "last_props_poll": last_props}


def test_base_intervals_follow_the_spec_table():
    assert s.base_intervals(30 * 60) == (10, 15)
    assert s.base_intervals(6 * 60) == (3, 5)
    assert s.base_intervals(61) == (3, 5)
    assert s.base_intervals(59) == (2, 2)
    assert s.base_intervals(0) is None          # tipped
    assert s.base_intervals(37 * 60) is None    # too far out


def test_guard_factor_scales_intervals():
    assert s.interval_for("player_points", 120, 1) == timedelta(minutes=5)
    assert s.interval_for("player_points", 120, 4) == timedelta(minutes=20)
    assert s.interval_for("h2h", 120, 1) == timedelta(minutes=3)


def test_is_due():
    assert s.is_due(None, timedelta(minutes=5), NOW)
    assert not s.is_due(NOW - timedelta(minutes=4), timedelta(minutes=5), NOW)
    assert s.is_due(NOW - timedelta(minutes=5), timedelta(minutes=5), NOW)


def test_stale_after_two_intervals_and_after_tip():
    assert not s.is_stale(NOW - timedelta(minutes=9), timedelta(minutes=5), NOW)
    assert s.is_stale(NOW - timedelta(minutes=11), timedelta(minutes=5), NOW)
    assert s.is_stale(NOW, None, NOW)


def test_fresh_filters_by_each_event_and_market():
    commence = {"e1": NOW + timedelta(hours=2)}  # props 5 min, lines 3 min
    old_prop = Quote("e1", "player_points", "A", "over", 20.5, "fanduel", -110, None, NOW - timedelta(minutes=8))
    old_line = Quote("e1", "h2h", "", "home", 0.0, "fanduel", -110, None, NOW - timedelta(minutes=8))
    assert s.fresh([old_prop, old_line], commence, NOW, 1) == [old_prop]


def test_guard_factor():
    cap, H = 3_500_000, timedelta(hours=1)
    # 130k/day projects 3.9M > cap: one notch up, not a ratchet to 8
    assert s.guard_factor(130_000, 1, cap, 4_000_000, None, NOW) == 2
    # within 6h of the last change: hold, whatever the window says
    assert s.guard_factor(130_000, 2, cap, 4_000_000, NOW - 2 * H, NOW) == 2
    # after a day at x2 the spend is 65k/day (1.95M): halving would project 3.9M > 0.9*cap, so stay
    assert s.guard_factor(65_000, 2, cap, 4_000_000, NOW - 24 * H, NOW) == 2
    # genuinely quiet: step back down
    assert s.guard_factor(30_000, 2, cap, 4_000_000, NOW - 24 * H, NOW) == 1
    # never above 8; nearly out of credits forces 8 immediately
    assert s.guard_factor(10_000_000, 8, cap, 4_000_000, NOW - 24 * H, NOW) == 8
    assert s.guard_factor(1_000, 1, cap, 40_000, NOW - H, NOW) == 8
    assert s.guard_factor(30_000, 1, cap, 4_000_000, None, NOW) == 1



def test_plan_polls():
    events = [ev("near", 2, last_lines=NOW - timedelta(minutes=4), last_props=NOW - timedelta(minutes=1)),
              ev("far", 20, last_lines=NOW, last_props=None),
              ev("done", -1)]
    plan = s.plan_polls(events, NOW, 1)
    assert plan.lines is True                  # "near" lines interval (3m) elapsed
    assert plan.prop_events == ["far"]         # never polled; "near" polled 1m ago



def test_backoff_rules():
    b = Backoff()
    assert b.failure(401, NOW) == NOW + timedelta(minutes=15)
    assert b.failure(429, NOW) == NOW + timedelta(seconds=30)
    assert b.failure(503, NOW) == NOW + timedelta(seconds=60)
    for _ in range(10):
        until = b.failure(500, NOW)
    assert until == NOW + timedelta(minutes=10)
    b.success()
    assert b.failure(429, NOW) == NOW + timedelta(seconds=30)
