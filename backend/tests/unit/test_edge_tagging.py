from datetime import datetime, timedelta, timezone

from src.edge import matchups, tagging

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def _patch(monkeypatch, ok_at, statuses=None, tags=None, boom=False):
    monkeypatch.setattr(tagging.store, "get_state", lambda conn: {"nfl:matchups_ok_at": ok_at} if ok_at else {})
    monkeypatch.setattr(tagging.matchups_store, "load", lambda conn, seasons: ([], []))
    monkeypatch.setattr(tagging.nfl_injuries, "live_index", lambda conn: {})
    monkeypatch.setattr(tagging, "_statuses", lambda conn: statuses or {})
    if boom:
        monkeypatch.setattr(tagging.matchups, "build_context", lambda *a: 1 / 0)
    else:
        monkeypatch.setattr(tagging.matchups, "build_context", lambda *a: "ctx")
    monkeypatch.setattr(tagging.matchups, "tags_for",
                        lambda ctx, player, teams, market, side: tags or [])


def test_tags_hidden_when_matchups_older_than_8_days(monkeypatch):
    _patch(monkeypatch, (NOW - timedelta(days=8, minutes=1)).isoformat())
    assert tagging.live_tagger(object(), NOW) is None
    _patch(monkeypatch, (NOW - timedelta(days=7, hours=23)).isoformat())
    assert tagging.live_tagger(object(), NOW) is not None


def test_tags_hidden_when_refresh_missing_or_garbage(monkeypatch):
    _patch(monkeypatch, None)
    assert tagging.live_tagger(object(), NOW) is None
    _patch(monkeypatch, "not-a-date")
    assert tagging.live_tagger(object(), NOW) is None


def test_context_failure_never_raises(monkeypatch):
    _patch(monkeypatch, NOW.isoformat(), boom=True)
    assert tagging.live_tagger(object(), NOW) is None


def test_no_tag_status_means_neutral(monkeypatch):
    t = matchups.Tag("def_soft", "tailwind", "JAX allows a lot")
    _patch(monkeypatch, NOW.isoformat(), tags=[t])
    out = tagging.live_tagger(object(), NOW)("A B", ("JAX", "KC"), "player_rush_yds", "over")
    assert out == [{"key": "def_soft", "label": "Soft matchup", "direction": "tailwind",
                    "verdict": "neutral", "lift": 0.0, "detail": "JAX allows a lot"}]


def test_status_row_sets_verdict_and_lift(monkeypatch):
    t = matchups.Tag("def_soft", "tailwind", "d")
    _patch(monkeypatch, NOW.isoformat(), tags=[t], statuses={("def_soft", "tailwind"): ("proven", 0.006)})
    out = tagging.live_tagger(object(), NOW)("A B", ("JAX", "KC"), "player_rush_yds", "over")
    assert out[0]["verdict"] == "proven" and out[0]["lift"] == 0.006


def test_tagger_error_gives_empty_tags(monkeypatch):
    _patch(monkeypatch, NOW.isoformat())
    monkeypatch.setattr(tagging.matchups, "tags_for", lambda *a: 1 / 0)
    assert tagging.live_tagger(object(), NOW)("A B", ("JAX", "KC"), "player_rush_yds", "over") == []


def _row(ev, *tags):
    return {"ev": ev, "tags": [{"verdict": v, "lift": l} for v, l in tags]}


def test_rows_sort_by_ev_plus_signed_lift_of_proven_and_negative_tags_only():
    boosted = _row(0.030, ("proven", 0.006))
    plain = _row(0.035)
    ignored = _row(0.020, ("neutral", 0.05))
    hurt = _row(0.040, ("negative", -0.01))
    rows = sorted([plain, ignored, boosted, hurt], key=tagging.sort_key, reverse=True)
    # 3.6% > 3.5% > 3.0% > 2.0%: the proven row passes the plain 3.5% one; the neutral +5% lift is ignored.
    assert rows == [boosted, plain, hurt, ignored]
    assert tagging.sort_key(ignored) == 0.020


class _BadConn:
    def __init__(self):
        self.rolled_back = 0

    def rollback(self):
        self.rolled_back += 1


def test_db_error_rolls_back_and_returns_none(monkeypatch):
    _patch(monkeypatch, NOW.isoformat())
    monkeypatch.setattr(tagging.matchups_store, "load", lambda conn, seasons: 1 / 0)
    c = _BadConn()
    assert tagging.live_tagger(c, NOW) is None
    assert c.rolled_back == 1


def test_rollback_failure_is_swallowed(monkeypatch):
    _patch(monkeypatch, NOW.isoformat(), boom=True)
    c = _BadConn()
    c.rollback = lambda: 1 / 0
    assert tagging.live_tagger(c, NOW) is None


def test_unknown_tag_key_does_not_raise(monkeypatch):
    _patch(monkeypatch, NOW.isoformat(), tags=[matchups.Tag("brand_new", "tailwind", "d")])
    out = tagging.live_tagger(object(), NOW)("A B", ("JAX", "KC"), "player_rush_yds", "over")
    assert out[0]["label"] == "brand_new"


def test_naive_matchups_timestamp_is_treated_as_utc(monkeypatch):
    _patch(monkeypatch, "2026-10-17T12:00:00")
    assert tagging.live_tagger(object(), NOW) is not None
    _patch(monkeypatch, "2026-10-01T12:00:00")
    assert tagging.live_tagger(object(), NOW) is None


def _injuries_seen(monkeypatch, injuries_ok_at):
    """The injuries dict live_tagger hands build_context for a given nfl:injuries_ok_at."""
    _patch(monkeypatch, NOW.isoformat())
    state = {"nfl:matchups_ok_at": NOW.isoformat()}
    if injuries_ok_at is not None:
        state["nfl:injuries_ok_at"] = injuries_ok_at
    monkeypatch.setattr(tagging.store, "get_state", lambda conn: state)
    monkeypatch.setattr(tagging.nfl_injuries, "live_index", lambda conn: {"KC": {"rudy runner": "questionable"}})
    seen = []
    monkeypatch.setattr(tagging.matchups, "build_context", lambda tg, pg, inj, s, w: seen.append(inj) or "ctx")
    assert tagging.live_tagger(object(), NOW) is not None  # matchup tags still serve
    return seen[0]


def test_live_injuries_used_only_when_refreshed_within_24_hours(monkeypatch):
    fresh = {"KC": {"rudy runner": "questionable"}}
    assert _injuries_seen(monkeypatch, (NOW - timedelta(hours=23, minutes=59)).isoformat()) == fresh
    assert _injuries_seen(monkeypatch, "2026-10-18T00:00:00") == fresh  # naive = UTC, 12 h old
    assert _injuries_seen(monkeypatch, (NOW - timedelta(hours=24, minutes=1)).isoformat()) == {}
    assert _injuries_seen(monkeypatch, "2026-10-17T11:00:00") == {}  # naive = UTC, 25 h old
    assert _injuries_seen(monkeypatch, None) == {}
    assert _injuries_seen(monkeypatch, "not-a-date") == {}
