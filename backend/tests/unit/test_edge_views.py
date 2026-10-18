from datetime import datetime, timedelta, timezone

from src.edge import tagging, views
from src.edge.config import NBA, NFL
from src.edge.engine import Fair, Priced
from src.edge.quotes import Quote

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def _setup(monkeypatch, tagger, home="Kansas City Chiefs", away="Jacksonville Jaguars"):
    ev = {"event_id": "e1", "home_team": home, "away_team": away, "commence_time": NOW + timedelta(hours=2)}
    mk = ["player_rush_yds", "player_rush_yds", "h2h"]
    quotes = [Quote("e1", m, p, "over" if m != "h2h" else "home", 60.5, "fanduel", -110, None, NOW)
              for m, p in zip(mk, ["A One", "B Two", ""])]
    evs = [0.035, 0.030, 0.050]
    priced = [Priced(q, Fair(0.55, 4, False), e, True, False) for q, e in zip(quotes, evs)]
    monkeypatch.setattr(views.store, "upcoming_events", lambda *a, **k: [ev])
    monkeypatch.setattr(views.store, "load_quotes", lambda *a, **k: quotes)
    monkeypatch.setattr(views.schedule, "fresh", lambda q, *a: q)
    monkeypatch.setattr(views, "_factor", lambda conn: 1)
    monkeypatch.setattr(views.nfl_players, "player_ids", lambda conn, teams: {})
    monkeypatch.setattr(views, "_player_ids", lambda conn: {})
    monkeypatch.setattr(views, "price_quotes", lambda *a, **k: priced)
    monkeypatch.setattr(views.tagging, "live_tagger", lambda conn, now: tagger)
    return priced


def _tag(verdict, lift):
    return {"key": "def_soft", "label": "Soft matchup", "direction": "tailwind", "verdict": verdict,
            "lift": lift, "detail": "d"}


def test_ev_rows_rank_by_ev_plus_proven_lift_and_skip_game_lines(monkeypatch):
    calls = []

    def tagger(player, teams, market, side):
        calls.append((player, teams, market, side))
        return [_tag("proven", 0.006)] if player == "B Two" else []

    _setup(monkeypatch, tagger)
    rows = views.ev_rows(None, NOW, min_ev=0.0, sport=NFL)
    by = {r["player"]: r for r in rows}
    assert by["B Two"]["tags"][0]["verdict"] == "proven" and by[""]["tags"] == []
    assert calls[0][1] == ("KC", "JAX")  # (home, away) BallDontLie abbreviations
    assert all(c[0] for c in calls)  # the game line was never tagged
    pl = [r["player"] for r in rows if r["player"]]
    assert pl == ["B Two", "A One"]  # 3.0% + 0.6% = 3.6% beats 3.5%
    assert by["B Two"]["ev"] == 0.03  # tags never change EV


def test_no_tag_status_means_neutral_and_pure_ev_sort(monkeypatch):
    _setup(monkeypatch, lambda *a: [_tag("neutral", 0.0)])
    rows = views.ev_rows(None, NOW, min_ev=0.0, sport=NFL)
    assert [r["ev"] for r in rows] == [0.05, 0.035, 0.03]


def test_stale_matchups_means_empty_tags_but_the_board_serves(monkeypatch):
    _setup(monkeypatch, None)
    rows = views.ev_rows(None, NOW, min_ev=0.0, sport=NFL)
    assert len(rows) == 3 and all(r["tags"] == [] for r in rows)


def test_unmappable_teams_give_empty_tags(monkeypatch):
    _setup(monkeypatch, lambda *a: [_tag("proven", 0.01)], home="Springfield Isotopes")
    assert all(r["tags"] == [] for r in views.ev_rows(None, NOW, min_ev=0.0, sport=NFL))


def test_nba_rows_have_empty_tags(monkeypatch):
    _setup(monkeypatch, lambda *a: [_tag("proven", 0.5)], home="Boston Celtics", away="Detroit Pistons")
    rows = views.ev_rows(None, NOW, min_ev=0.0, sport=NBA)
    assert all(r["tags"] == [] for r in rows)
    assert [r["ev"] for r in rows] == [0.05, 0.035, 0.03]


def test_results_by_tag_groups_by_key_and_direction():
    def r(clv, tags):
        return {"clv": clv, "result": "win", "profit_units": 1.0, "market": "m", "book": "b", "tags": tags}

    soft_t = [{"key": "def_soft", "direction": "tailwind"}]
    rows = [r(0.02, soft_t), r(0.04, soft_t + [{"key": "usage_up", "direction": "tailwind"}]),
            r(0.01, [{"key": "def_soft", "direction": "headwind"}]), r(0.0, None), r(0.0, [])]
    s = views.results_summary(rows)
    assert set(s["by_tag"]) == {"def_soft:tailwind", "def_soft:headwind", "usage_up:tailwind"}
    assert s["by_tag"]["def_soft:tailwind"]["n"] == 2
    assert s["by_tag"]["def_soft:tailwind"]["clv"]["mean_clv"] == 0.03
    assert views.results_summary([r(0.0, None)])["by_tag"] == {}
