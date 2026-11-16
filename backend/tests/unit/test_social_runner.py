"""runner.social_step with every DB helper faked (no DB, no network) and a FakePoster."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.config import settings
from src.edge.social import blind, cards, content, headshots, ledger, poster as poster_mod, runner
from src.edge.social.poster import PostResult, UrlLookup

NOV16_11ET = datetime(2026, 11, 16, 16, 0, tzinfo=timezone.utc)  # EST: 11:00 ET = 16:00 UTC
OCT21_11ET = datetime(2026, 10, 21, 15, 0, tzinfo=timezone.utc)  # EDT: 11:00 ET = 15:00 UTC
DAY = date(2026, 11, 16)
TIP = datetime(2026, 11, 17, 0, 30, tzinfo=timezone.utc)  # 7:30 PM ET 11/16


class FakePoster:
    def __init__(self, dry_run=False, ok=True, retryable=False, url=None):
        self.dry_run, self.ok, self.retryable, self.url = dry_run, ok, retryable, url
        self.calls = []

    def post(self, texts, images):
        self.calls.append((texts, images))
        return PostResult(self.ok, "sub-1" if self.ok else None, [], self.url, retryable=self.retryable,
                          error=None if self.ok else "boom")

    def lookup(self, sid):
        return UrlLookup("unknown")


class FakeLedger:
    def __init__(self):
        self.rows, self.brake, self.next_id = {}, False, 1

    def claim(self, conn, sport, kind, key, now, *, dry_run=False):
        r = self.rows.get((kind, key))
        if r is None:
            r = self.rows[(kind, key)] = {"id": self.next_id, "sport": sport, "kind": kind, "key": key,
                                          "status": "pending", "attempts": 1, "image": None, "post_url": None}
            self.next_id += 1
            return ledger.Claim(r["id"], 1)
        if r["status"] == "failed" and r["attempts"] < ledger.MAX_ATTEMPTS:
            r["status"], r["attempts"] = "pending", r["attempts"] + 1
            return ledger.Claim(r["id"], r["attempts"])
        return None

    def finish(self, conn, claim, status, now, *, text, image=None, media_url=None, submission_id=None,
               post_url=None, detail=None):
        r = next(r for r in self.rows.values() if r["id"] == claim.id)
        r.update(status=status, text=text, image=image, submission_id=submission_id, post_url=post_url,
                 detail=detail)
        return True

    def find(self, conn, kind, key):
        return self.rows.get((kind, key))

    def statuses(self, kind=None):
        return {k: r["status"] for (kd, k), r in self.rows.items() if kind is None or kd == kind}


class World:
    """What the faked DB helpers return."""

    def __init__(self):
        self.live_rows, self.gate, self.legs, self.blind_plays = [], "pass", [], []
        self.final_events, self.receipts, self.cheatsheet = [], [], []
        self.recorded = []
        self.second = None  # None: the second source agrees with the event's score


def _play(n=1, **kw):
    return {"sport": "nba", "n": n, "slate_date": DAY, "event_id": "e1", "market": "player_points",
            "market_label": "Points", "player": "A One", "side": "over", "line": 20.5, "book": "fanduel",
            "price": 116, "fair_prob": 0.49, "ev": 0.08, "n_books": 9, "form": None, "position": "F",
            "panel": None, "commence_time": TIP, **kw}


def _live(**kw):
    return {"event_id": "e1", "market": "player_points", "player": "A One", "side": "over", "line": 20.5,
            "book": "fanduel", "price": 116, "ev": 0.06, "commence_time": TIP.isoformat(), "n_books": 9,
            "verify": False, "form_flag": False, **kw}


@pytest.fixture
def world(monkeypatch):
    w, led = World(), FakeLedger()
    monkeypatch.setattr(settings, "social_sports", "nba")
    monkeypatch.setattr(settings, "social_kinds", "cheatsheet,value,blind,final,receipt")
    for name in ("claim", "finish", "find"):
        monkeypatch.setattr(ledger, name, getattr(led, name))
    monkeypatch.setattr(ledger, "brake_tripped", lambda conn, sport, now: led.brake)
    monkeypatch.setattr(ledger, "unresolved", lambda conn, now, limit=5: [])
    monkeypatch.setattr(ledger, "unknown_submissions", lambda conn, now, limit=5: [])
    monkeypatch.setattr(ledger, "note_submission", lambda conn, claim, sid, media_url=None: None)
    monkeypatch.setattr(runner, "_live_rows", lambda conn, sport, now: w.live_rows)
    monkeypatch.setattr(runner, "_gate", lambda conn, sport: w.gate)
    monkeypatch.setattr(runner, "_top_legs", lambda conn, sport, day: w.legs)
    monkeypatch.setattr(runner, "_blind_to_post", lambda conn, sport, now: w.blind_plays)
    monkeypatch.setattr(runner, "_slate_tips", lambda conn, sport, day: {"e1": TIP})
    monkeypatch.setattr(runner, "_today_teams", lambda conn, sport, day: set())
    monkeypatch.setattr(runner, "_final_events", lambda conn, sport, now: w.final_events)
    monkeypatch.setattr(runner, "_receipt_candidates", lambda conn, sport, now: w.receipts)
    monkeypatch.setattr(runner, "_assign_teams", lambda conn, sport, rows: None)
    monkeypatch.setattr(runner, "_box_line", lambda conn, sport, play: (None, None))
    monkeypatch.setattr(runner, "_cheatsheet_rows", lambda conn, sport, rows: w.cheatsheet)
    monkeypatch.setattr(runner, "_second_source", lambda conn, sport, ev: w.second if w.second is not None
                        else (ev["home_score"], ev["away_score"]))
    monkeypatch.setattr(runner, "_headshots_due", lambda conn, sport, now: False)
    monkeypatch.setattr(runner, "_blind_due", lambda conn, sport, now: True)
    monkeypatch.setattr(blind, "build_panels", lambda conn, sport, rows, now: {})
    monkeypatch.setattr(blind, "select", lambda rows, top5, panels, now: [r for r in rows if r.get("pick")])

    def record(conn, sport, picks, panels, now):
        w.recorded.extend(picks)
        return [len(w.recorded)] if picks else []
    monkeypatch.setattr(blind, "record", record)
    for name in ("render_blind", "render_cheatsheet", "render_value", "render_final", "render_receipt"):
        monkeypatch.setattr(cards, name, lambda *a, _n=name, **k: f"png:{_n}".encode())

    async def no_png(*a, **k):
        return None
    monkeypatch.setattr(headshots, "fetch_png", no_png)
    monkeypatch.setattr(headshots, "fetch_logo", no_png)
    w.ledger = led
    return w


async def step(now, p):
    return await runner.social_step(object(), now, poster=p, http=object())


# ---- the brief's tests ------------------------------------------------------------------------

async def test_dry_run_never_calls_blotato(world, monkeypatch):
    called = []
    monkeypatch.setattr(poster_mod.blotato, "upload_media", lambda *a, **k: called.append("upload"))
    monkeypatch.setattr(poster_mod.blotato, "post_thread", lambda *a, **k: called.append("post"))
    monkeypatch.setattr(settings, "twitter_posting_enabled", True)
    monkeypatch.setattr(settings, "social_dry_run", True)
    world.cheatsheet = [{"player": f"P{i}", "matchup": "A @ B"} for i in range(6)]
    await runner.social_step(object(), NOV16_11ET, http=object())  # default poster
    assert called == []
    row = world.ledger.rows[("cheatsheet", "cheatsheet:nba:2026-11-16")]
    assert row["status"] == "dry_run" and row["image"] == b"png:render_cheatsheet"


def test_default_poster_is_dry_run_unless_both_switches_say_post(monkeypatch):
    for enabled, dry, real in ((False, False, False), (True, True, False), (False, True, False), (True, False, True)):
        monkeypatch.setattr(settings, "twitter_posting_enabled", enabled)
        monkeypatch.setattr(settings, "social_dry_run", dry)
        assert isinstance(runner.default_poster(), poster_mod.BlotatoPoster) is real


async def test_brake_stops_all_posting(world):
    world.ledger.brake = True
    world.cheatsheet = [{"player": f"P{i}", "matchup": "A @ B"} for i in range(6)]
    world.blind_plays = [_play()]
    world.live_rows = [_live()]
    p = FakePoster()
    out = await step(NOV16_11ET, p)
    assert p.calls == [] and world.ledger.rows == {}
    assert out["nba"]["brake"] is True


async def test_blind_waits_for_gate_pass(world):
    world.live_rows = [_live(pick=True)]
    p = FakePoster()
    for gate in (None, "fail", "pending"):
        world.gate = gate
        await step(NOV16_11ET, p)
        assert world.recorded == [] and p.calls == []
    world.gate = "pass"
    world.blind_plays = [_play()]
    await step(NOV16_11ET, p)
    assert len(world.recorded) == 1
    assert world.ledger.statuses("blind") == {"blind:nba:1": "posted"}
    texts, images = p.calls[0]
    assert "Blind Play #1" in texts[0] and "/blind/nba/1" in texts[1]
    assert images == [[b"png:render_blind"], []]


async def test_same_key_not_posted_twice_across_ticks(world):
    world.blind_plays = [_play()]
    world.live_rows = [_live()]
    world.cheatsheet = [{"player": f"P{i}", "matchup": "A @ B"} for i in range(6)]
    p = FakePoster()
    await step(NOV16_11ET, p)
    await step(NOV16_11ET + timedelta(minutes=1), p)
    await step(NOV16_11ET + timedelta(hours=2), p)
    kinds = sorted(t[0][:12] for t, _ in p.calls)
    assert len(p.calls) == 2, kinds  # one cheat sheet, one blind play


@pytest.mark.parametrize("at_11", [OCT21_11ET, NOV16_11ET])
async def test_cheatsheet_fires_at_11_et_across_dst(world, at_11):
    world.cheatsheet = [{"player": f"P{i}", "matchup": "A @ B"} for i in range(6)]
    p = FakePoster()
    await step(at_11 - timedelta(minutes=1), p)
    assert p.calls == []
    await step(at_11, p)
    assert len(p.calls) == 1 and "Cheat Sheet" in p.calls[0][0][0]
    day = at_11.date().isoformat()
    assert world.ledger.statuses("cheatsheet") == {f"cheatsheet:nba:{day}": "posted"}


async def test_social_error_does_not_break_tick(world, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db gone")
    monkeypatch.setattr(runner, "_live_rows", boom)
    monkeypatch.setattr(ledger, "unresolved", boom)
    world.final_events = []
    out = await step(NOV16_11ET, FakePoster())
    assert isinstance(out, dict)


async def test_receipt_uses_two_images_without_post_url(world):
    world.ledger.rows[("blind", "blind:nba:1")] = {"id": 99, "status": "posted", "image": b"orig-card",
                                                    "post_url": None}
    world.receipts = [_receipt()]
    p = FakePoster()
    await step(NOV16_11ET + timedelta(days=1), p)
    texts, images = p.calls[0]
    assert images == [[b"png:render_receipt", b"orig-card"]]
    assert "Blind Play #1 was A One" in texts[0] and "/blind/nba/1" in texts[0]


async def test_receipt_quotes_known_post_url_with_one_image(world):
    world.ledger.rows[("blind", "blind:nba:1")] = {"id": 99, "status": "posted", "image": b"orig-card",
                                                    "post_url": "https://x.com/trulineapp/status/1"}
    world.receipts = [_receipt()]
    p = FakePoster()
    await step(NOV16_11ET + timedelta(days=1), p)
    texts, images = p.calls[0]
    assert images == [[b"png:render_receipt"]] and "https://x.com/trulineapp/status/1" in texts[0]


# ---- safety -----------------------------------------------------------------------------------

async def test_blind_skipped_when_ev_gone_or_inside_60_minutes(world):
    world.blind_plays = [_play(1), _play(2, player="B Two")]
    world.live_rows = [_live(player="B Two", commence_time=(NOV16_11ET + timedelta(minutes=59)).isoformat())]
    p = FakePoster()
    await step(NOV16_11ET, p)
    assert p.calls == []
    assert world.ledger.statuses("blind") == {"blind:nba:1": "skipped", "blind:nba:2": "skipped"}


async def test_blind_needs_the_same_book_still_plus_ev(world):
    world.blind_plays = [_play()]
    world.live_rows = [_live(book="betmgm")]
    p = FakePoster()
    await step(NOV16_11ET, p)
    assert p.calls == [] and world.ledger.statuses("blind") == {"blind:nba:1": "skipped"}


def _leg(rank, tip=TIP):
    return {"rank": rank, "player": f"P{rank}", "matchup": "A @ B", "commence_time": tip, "event_id": f"e{rank}",
            "market": "player_points", "market_label": "Points", "side": "over", "line": 20.5, "bet": "x",
            "book": "fanduel", "price": 110, "fair_prob": 0.5, "ev": 0.05, "n_books": 8,
            "locked_at": TIP - timedelta(hours=2), "slate_date": DAY}


async def test_value_posts_locked_legs_once_and_skips_if_a_game_started(world):
    world.legs = [_leg(0), _leg(1)]
    p = FakePoster()
    await step(TIP - timedelta(minutes=30), p)  # inside 60 min is fine for value plays
    assert world.ledger.statuses("value") == {"value:nba:2026-11-16": "posted"}
    assert "Top 5" in p.calls[0][0][0]
    world.ledger.rows.clear()
    world.legs = [_leg(0), _leg(1, tip=TIP - timedelta(hours=1))]
    p = FakePoster()
    await step(TIP - timedelta(minutes=30), p)
    assert p.calls == [] and world.ledger.statuses("value") == {"value:nba:2026-11-16": "skipped"}


def _event(**kw):
    return {"event_id": "e1", "home_team": "Boston Celtics", "away_team": "Detroit Pistons", "commence_time": TIP,
            "status": "final", "completed": True, "home_score": 110, "away_score": 104, **kw}


async def test_final_waits_for_second_source_and_grading(world, monkeypatch):
    world.ledger.rows[("blind", "blind:nba:1")] = {"id": 99, "status": "posted", "image": None, "post_url": None}
    world.final_events = [_event()]
    plays = [{"kind": "blind", "sport": "nba", "n": 1, "player": "A One", "market": "player_points",
              "side": "over", "line": 20.5, "price": 116, "result": None, "profit_units": None}]
    monkeypatch.setattr(content, "posted_plays", lambda conn, eid: plays)
    monkeypatch.setattr(content, "team_records", lambda conn, sport, start: {})
    world.second = (110, 103)
    p = FakePoster()
    now = TIP + timedelta(hours=3)
    await step(now, p)
    assert p.calls == [] and "final" not in {k for k, _ in world.ledger.rows}  # ungraded: wait
    plays[0].update(result="win", profit_units=1.16)
    await step(now, p)
    assert p.calls == []  # second source disagrees: wait, never post
    world.second = (110, 104)
    await step(now, p)
    assert world.ledger.statuses("final") == {"final:e1": "posted"}
    assert "Detroit Pistons 104, Boston Celtics 110" in p.calls[0][0][0] and "1-0" in p.calls[0][0][0]


async def test_final_ignores_plays_that_were_never_posted(world, monkeypatch):
    world.final_events = [_event()]
    monkeypatch.setattr(content, "posted_plays", lambda conn, eid: [
        {"kind": "blind", "sport": "nba", "n": 7, "player": "A", "market": "player_points", "side": "over",
         "line": 1.5, "price": 100, "result": "win", "profit_units": 1.0}])
    p = FakePoster()
    await step(TIP + timedelta(hours=3), p)
    assert world.ledger.statuses("final") == {}


def _receipt(**kw):
    return {"kind": "blind", "sport": "nba", "n": 1, "rank": None, "slate_date": DAY, "event_id": "e1",
            "player": "A One", "market": "player_points", "market_label": "Points", "side": "over", "line": 20.5,
            "price": 116, "ev": 0.08, "panel": None, "position": "F", "result": "win",
            "event": _event(), "matchup": "Detroit Pistons @ Boston Celtics", **kw}


async def test_live_mode_never_receipts_a_dry_run_play(world, monkeypatch):
    world.ledger.rows[("blind", "blind:nba:1")] = {"id": 99, "status": "dry_run", "image": b"o", "post_url": None}
    world.receipts = [_receipt()]
    p = FakePoster()
    await step(NOV16_11ET + timedelta(days=1), p)
    assert p.calls == []
    await step(NOV16_11ET + timedelta(days=1), FakePoster(dry_run=True))
    assert world.ledger.statuses("receipt") == {"receipt:blind:nba:1": "dry_run"}


async def test_receipt_skipped_when_box_score_disagrees_with_the_grade(world, monkeypatch):
    world.ledger.rows[("blind", "blind:nba:1")] = {"id": 99, "status": "posted", "image": b"o", "post_url": None}
    monkeypatch.setattr(runner, "_box_line", lambda conn, sport, play: (18.0, "18 pts · 2 reb · 1 ast"))
    world.receipts = [_receipt()]
    p = FakePoster()
    await step(NOV16_11ET + timedelta(days=1), p)
    assert p.calls == [] and world.ledger.statuses("receipt") == {"receipt:blind:nba:1": "skipped"}


async def test_render_error_marks_failed_and_other_posts_continue(world, monkeypatch):
    def bad(*a, **k):
        raise ValueError("font missing")
    monkeypatch.setattr(cards, "render_cheatsheet", bad)
    world.cheatsheet = [{"player": f"P{i}", "matchup": "A @ B"} for i in range(6)]
    world.blind_plays = [_play()]
    world.live_rows = [_live()]
    p = FakePoster()
    await step(NOV16_11ET, p)
    assert world.ledger.statuses() == {"cheatsheet:nba:2026-11-16": "failed", "blind:nba:1": "posted"}


async def test_unsure_post_is_never_retried_but_upload_failure_is(world):
    world.blind_plays = [_play()]
    world.live_rows = [_live()]
    await step(NOV16_11ET, FakePoster(ok=False, retryable=False))
    assert world.ledger.statuses("blind") == {"blind:nba:1": "skipped"}
    world.ledger.rows.clear()
    p = FakePoster(ok=False, retryable=True)
    await step(NOV16_11ET, p)
    assert world.ledger.statuses("blind") == {"blind:nba:1": "failed"}
    await step(NOV16_11ET, p)
    assert len(p.calls) == 2


@pytest.mark.parametrize("when,due", [
    (datetime(2026, 11, 19, 15, 0, tzinfo=timezone.utc), True),   # Thu 10:00 EST
    (datetime(2026, 11, 22, 15, 0, tzinfo=timezone.utc), True),   # Sun
    (datetime(2026, 11, 23, 15, 0, tzinfo=timezone.utc), True),   # Mon
    (datetime(2026, 11, 21, 15, 0, tzinfo=timezone.utc), False),  # Sat
    (datetime(2026, 11, 19, 14, 59, tzinfo=timezone.utc), False),  # Thu 9:59
    (datetime(2026, 10, 22, 14, 0, tzinfo=timezone.utc), True),   # Thu 10:00 EDT
])
def test_nfl_cheatsheet_days(when, due):
    assert (runner.cheatsheet_due("nfl", when) is not None) is due


# ---- poster -----------------------------------------------------------------------------------

def test_blotato_poster_uploads_then_posts_thread(monkeypatch):
    up, posted = [], []
    monkeypatch.setattr(poster_mod.blotato, "upload_media", lambda b, filename: up.append(b) or f"u{len(up)}")
    monkeypatch.setattr(poster_mod.blotato, "post_thread",
                        lambda texts, schedule_at, media_urls_per_post: posted.append((texts, schedule_at,
                                                                                       media_urls_per_post))
                        or {"postSubmissionId": "s1"})
    monkeypatch.setattr(poster_mod.blotato, "get_post_status",
                        lambda sid: {"status": "published", "publicUrl": "https://x.com/t/status/9"})
    r = poster_mod.BlotatoPoster().post(["a", "b"], [[b"1", b"2"], []])
    assert posted == [(["a", "b"], None, [["u1", "u2"], []])]
    assert r.ok and r.submission_id == "s1" and r.post_url == "https://x.com/t/status/9"


def test_blotato_poster_failure_modes(monkeypatch):
    monkeypatch.setattr(poster_mod.blotato, "upload_media", lambda b, filename: None)
    r = poster_mod.BlotatoPoster().post(["a"], [[b"1"]])
    assert not r.ok and r.retryable
    monkeypatch.setattr(poster_mod.blotato, "upload_media", lambda b, filename: "u")
    monkeypatch.setattr(poster_mod.blotato, "post_thread", lambda *a, **k: None)
    r = poster_mod.BlotatoPoster().post(["a"], [[b"1"]])
    assert not r.ok and not r.retryable


async def test_post_urls_are_looked_up_later(world, monkeypatch):
    noted = []
    monkeypatch.setattr(ledger, "unresolved", lambda conn, now, limit=5: [(5, "s5")])
    monkeypatch.setattr(ledger, "note_lookup", lambda conn, i, state, post_url=None, error=None:
                        noted.append((i, state, post_url)))

    class P(FakePoster):
        def lookup(self, sid):
            return UrlLookup("published", "https://x.com/t/status/5")
    await step(NOV16_11ET - timedelta(hours=5), P())
    assert noted == [(5, "published", "https://x.com/t/status/5")]


# ---- preview CLI ------------------------------------------------------------------------------

def test_preview_writes_png_and_txt(tmp_path: Path, monkeypatch):
    from src.edge.social import __main__ as cli
    rows = [{"id": 3, "kind": "blind", "key": "blind:nba:1", "text": "hello", "image": memoryview(b"\x89PNG")},
            {"id": 4, "kind": "final", "key": "final:e/1", "text": "bye", "image": None}]
    monkeypatch.setattr(cli, "_dry_runs", lambda conn, since: rows)
    n = cli.export(object(), tmp_path, date(2026, 11, 16))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert n == 2 and files == ["00003_blind_blind-nba-1.png", "00003_blind_blind-nba-1.txt",
                                "00004_final_final-e-1.txt"]
    assert (tmp_path / "00003_blind_blind-nba-1.png").read_bytes() == b"\x89PNG"


def test_blind_pass_runs_every_five_minutes_and_headshots_once_a_day(monkeypatch):
    st = {}
    monkeypatch.setattr(runner.store, "get_state", lambda conn: dict(st))
    monkeypatch.setattr(runner.store, "set_state", lambda conn, k, v, now: st.__setitem__(k, v))
    t = NOV16_11ET
    assert runner._blind_due(None, "nba", t) and not runner._blind_due(None, "nba", t + timedelta(minutes=4))
    assert runner._blind_due(None, "nba", t + timedelta(minutes=5)) and runner._blind_due(None, "nfl", t)
    assert runner._headshots_due(None, "nba", t) and not runner._headshots_due(None, "nba", t + timedelta(hours=12))
    assert runner._headshots_due(None, "nba", t + timedelta(hours=13))  # midnight ET passed


async def test_unknown_rows_are_reconciled_only_when_blotato_confirms(world, monkeypatch):
    confirmed = []
    monkeypatch.setattr(ledger, "unknown_submissions", lambda conn, now, limit=5: [(7, "s7"), (8, "s8")])
    monkeypatch.setattr(ledger, "confirm_posted", lambda conn, i, url: confirmed.append((i, url)) or True)

    class P(FakePoster):
        def lookup(self, sid):
            return UrlLookup("published", "https://x.com/t/status/7") if sid == "s7" else UrlLookup("pending")
    await step(NOV16_11ET - timedelta(hours=5), P())
    assert confirmed == [(7, "https://x.com/t/status/7")]
    await step(NOV16_11ET - timedelta(hours=5), FakePoster(dry_run=True))  # dry run never asks Blotato
    assert len(confirmed) == 1


async def test_unknown_key_is_never_posted_again(world):
    world.ledger.rows[("blind", "blind:nba:1")] = {"id": 50, "status": "unknown", "attempts": 1, "image": None,
                                                    "post_url": None}
    world.blind_plays = [_play()]
    world.live_rows = [_live()]
    p = FakePoster()
    await step(NOV16_11ET, p)
    assert p.calls == [] and world.ledger.statuses("blind") == {"blind:nba:1": "unknown"}
