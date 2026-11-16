from datetime import datetime, timedelta, timezone

from src.edge.social import safety

NOW = datetime(2026, 11, 16, 20, tzinfo=timezone.utc)


def row(minutes):
    return {"commence_time": (NOW + timedelta(minutes=minutes)).isoformat()}


def test_blind_pick_accepted_at_exactly_60_minutes():
    assert safety.pick_ok(row(60), NOW, 0.05)


def test_blind_pick_rejected_59_minutes_before_tip():
    assert not safety.pick_ok(row(59), NOW, 0.05)


def test_blind_pick_rejected_when_ev_gone():
    assert not safety.pick_ok(row(120), NOW, 0.0)
    assert not safety.pick_ok(row(120), NOW, -0.01)
    assert not safety.pick_ok(row(120), NOW, None)


def test_value_post_skipped_if_any_leg_started():
    legs = [row(30), row(-1)]
    assert not safety.value_ok(legs, NOW)
    assert not safety.value_ok([row(0)], NOW)
    assert safety.value_ok([row(30), row(90)], NOW)
    assert not safety.value_ok([], NOW)


def ev(**kw):
    return {"completed": True, "status": "final", "home_score": 110, "away_score": 104, **kw}


def test_final_ok_when_scores_agree():
    assert safety.final_ok(ev(), (110, 104))


def test_score_mismatch_blocks_recap():
    assert not safety.final_ok(ev(), (110, 105))
    assert not safety.final_ok(ev(), (104, 110))  # swapped sides


def test_no_second_source_or_incomplete_blocks():
    assert not safety.final_ok(ev(), None)
    assert not safety.final_ok(ev(completed=False), (110, 104))
    assert not safety.final_ok(ev(home_score=None), (110, 104))


def test_postponed_game_never_recapped_or_receipted():
    for status in ("postponed", "Postponed", "CANCELLED", "suspended", "Game Suspended"):
        assert not safety.final_ok(ev(status=status), (110, 104))
