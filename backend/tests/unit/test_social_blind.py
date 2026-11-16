import dataclasses
from datetime import datetime, timedelta, timezone

from src.edge import nightly
from src.edge.social import blind
from src.edge.social.nba_matchup import Panel
from src.edge.social.nfl_matchup import NflPanel

NOW = datetime(2026, 11, 16, 20, tzinfo=timezone.utc)
TIP = NOW + timedelta(hours=3)


def row(player="A One", event="e1", ev=0.10, tip=TIP, market="player_points", side="over", line=20.5, **kw):
    r = {"event_id": event, "matchup": "Detroit Pistons @ Boston Celtics", "commence_time": tip.isoformat(),
         "market": market, "market_label": "Points", "player": player, "side": side, "line": line,
         "bet": "x", "book": "fanduel", "price": 116, "fair_prob": 0.49, "ev": ev, "n_books": 9,
         "verify": False, "form_flag": False, "form": {"hits": 8, "games": 10, "rate": 0.8}}
    r.update(kw)
    return r


def panel(fav=True):
    return Panel(fav, 25, 20, 10, 25.0, 1.0, 34.0)


def panels_for(rows, fav=True):
    return {blind.panel_key(r): panel(fav) for r in rows}


def test_select_excludes_top5_players_and_unfavorable():
    rows = [row("A", "e1", 0.12), row("B", "e2", 0.11), row("C", "e3", 0.10)]
    panels = panels_for(rows)
    panels[blind.panel_key(rows[2])] = panel(False)
    out = blind.select(rows, {"A"}, panels, NOW)
    assert [r["player"] for r in out] == ["B"]


def test_select_missing_panel_is_not_favorable():
    rows = [row("A", "e1")]
    assert blind.select(rows, set(), {}, NOW) == []


def test_select_accepts_nfl_panel():
    r = row("A")
    p = {blind.panel_key(r): NflPanel(True, 3, 0.2, None, ["def_soft"])}
    assert blind.select([r], set(), p, NOW) == [r]


def test_select_requires_60_min_before_tip():
    ok = row("A", "e1", tip=NOW + timedelta(minutes=60))
    late = row("B", "e2", tip=NOW + timedelta(minutes=59))
    started = row("C", "e3", tip=NOW - timedelta(minutes=5))
    rows = [ok, late, started]
    assert [r["player"] for r in blind.select(rows, set(), panels_for(rows), NOW)] == ["A"]


def test_select_one_per_game_and_max_three():
    rows = [row("A", "e1", 0.20), row("B", "e1", 0.19), row("C", "e2", 0.18), row("D", "e3", 0.17),
            row("E", "e4", 0.16)]
    out = blind.select(rows, set(), panels_for(rows), NOW)
    assert [r["player"] for r in out] == ["A", "C", "D"]


def test_select_drops_ineligible_and_non_props():
    thin = row("A", "e1", n_books=4)
    verify = row("B", "e2", verify=True)
    game = row("", "e3", market="h2h")
    rows = [thin, verify, game]
    assert blind.select(rows, set(), panels_for(rows), NOW) == []


def test_zero_qualifying_returns_empty():
    assert blind.select([], set(), {}, NOW) == []


def test_panel_key_normalizes_line():
    assert blind.panel_key(row(line=20.5)) == blind.panel_key(row(line="20.50"))


def test_blind_card_has_no_identity_fields():
    names = {f.name for f in dataclasses.fields(blind.BlindCard)}
    assert not names & {"player", "team", "opponent", "matchup", "event_id", "home_team", "away_team"}


def play(**kw):
    p = {"sport": "nba", "n": 7, "position": "F", "market_label": "Points", "side": "over", "line": 20.5,
         "price": 116, "ev": 0.1, "n_books": 9, "commence_time": TIP,
         "form": {"hits": 8, "games": 10, "rate": 0.8, "avg": 24.3}, "panel": dataclasses.asdict(panel())}
    p.update(kw)
    return p


def test_tip_label_hides_a_lone_tip_time():
    other = TIP + timedelta(hours=1)
    assert blind.to_card(play(), {"e1": TIP}).tip_label == "Tonight"
    assert blind.to_card(play(), {"e1": TIP, "e2": other}).tip_label == "Tonight"  # a time would name the game
    assert blind.to_card(play(), {}).tip_label == "Tonight"


def test_tip_label_is_et_when_games_share_the_tip():
    # 23:00 UTC in November is 6:00 PM ET
    two = {"e1": TIP, "e2": TIP, "e3": TIP + timedelta(hours=1)}
    assert blind.to_card(play(), two).tip_label == "6:00 PM ET"


def test_tip_label_counts_games_not_props():
    # several props from one lone game are one tip: still "Tonight"
    assert blind.tip_label(TIP, {"e1": TIP}) == "Tonight"
    assert blind.to_card(play(), {"e1": TIP, "e2": TIP + timedelta(hours=2)}).tip_label == "Tonight"


def test_to_card_carries_form_and_no_identity():
    c = blind.to_card(play(), {"e1": TIP, "e2": TIP})
    assert (c.l10_hits, c.l10_games, c.l10_avg) == (8, 10, 24.3)
    assert c.n == 7 and c.position == "F" and c.price == 116
    assert c.panel["opp_vs_pos_rank"] == 25
    assert blind.to_card(play(form=None), {"e1": TIP}).l10_hits is None


def test_explain_template():
    p = {"n_books": 9, "fair_prob": 0.49, "book": "fanduel", "price": 116}
    assert blind.explain(p) == "9 books price this at 49%; FanDuel pays +116, which needs only 46%."
    assert "-110" in blind.explain({**p, "price": -110}) and "52%" in blind.explain({**p, "price": -110})


# ---- build_panels (DB reads faked) ----------------------------------------------------------

from src.edge import matchups, matchups_store, players  # noqa: E402
from src.edge.social import nba_matchup  # noqa: E402


class FakeCur:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.rollbacks = 0

    def cursor(self, **kw):
        return FakeCur(self.rows)

    def commit(self):
        pass

    def rollback(self):
        self.rollbacks += 1


def _good():
    return nba_matchup.TeamRanks("X", {("F", "player_points"): 25}, drtg_rank=20, pace_rank=5)


def _bad():
    return nba_matchup.TeamRanks("Y", {("F", "player_points"): 1}, drtg_rank=2, pace_rank=5)


def test_nba_panel_uses_the_opponents_ranks_for_home_and_away(monkeypatch):
    usage = nba_matchup.PlayerUsage(25.0, 25.0, 34.0, 34.0, 40)
    # DET (away) is a soft defense, BOS (home) a tough one
    monkeypatch.setattr(nba_matchup, "load_context",
                        lambda conn, season: ({"DET": _good(), "BOS": _bad()}, lambda pid: usage))
    games = [{"game_date": NOW.date() - timedelta(days=i + 1), "minutes": 30.0, "pts": 20 + i, "reb": 0,
              "ast": 0, "fg3m": 0} for i in range(10)]
    monkeypatch.setattr(players, "load_recent_games", lambda conn, ids: {1: games, 2: games})
    home = row("Home Guy", "e1", bdl_player_id=1)
    away = row("Away Guy", "e1", bdl_player_id=2)
    for r in (home, away):
        r["matchup"] = "Detroit Pistons @ Boston Celtics"
    conn = FakeConn([("Home Guy", "BOS", "F"), ("Away Guy", "DET", "F")])
    out = blind.build_panels(conn, "nba", [home, away], NOW)
    assert out[blind.panel_key(home)].favorable is True   # faces DET's ranks
    assert out[blind.panel_key(away)].favorable is False  # faces BOS's ranks
    assert out[blind.panel_key(home)].opp_vs_pos_rank == 25 and out[blind.panel_key(away)].opp_vs_pos_rank == 1
    assert (home["team"], home["position"]) == ("BOS", "F") and (away["team"], away["position"]) == ("DET", "F")
    assert home["home_team"] == "Boston Celtics" and home["away_team"] == "Detroit Pistons"
    assert home["form"]["avg"] == 24.5 and home["form"]["hits"] == 8  # mean of 20..29


def test_nba_panels_skip_players_not_in_the_game_and_fail_closed_on_errors(monkeypatch):
    usage = nba_matchup.PlayerUsage(25.0, 25.0, 34.0, 34.0, 40)
    monkeypatch.setattr(nba_matchup, "load_context", lambda conn, season: ({"DET": _good()}, lambda pid: usage))
    monkeypatch.setattr(players, "load_recent_games", lambda conn, ids: {})
    r = row("Trade Guy", bdl_player_id=1)
    assert blind.build_panels(FakeConn([("Trade Guy", "LAL", "F")]), "nba", [r], NOW) == {}

    def boom(conn, season):
        raise RuntimeError("no db")
    monkeypatch.setattr(nba_matchup, "load_context", boom)
    conn = FakeConn([("Trade Guy", "BOS", "F")])
    assert blind.build_panels(conn, "nba", [r], NOW) == {} and conn.rollbacks == 1


def test_nfl_panel_is_favorable_on_def_soft_tailwind(monkeypatch):
    ctx = matchups.build_context([], [], {}, 2026, 99)
    monkeypatch.setattr(matchups_store, "load", lambda conn, seasons: ([], []))
    monkeypatch.setattr(matchups, "build_context", lambda *a: ctx)
    monkeypatch.setattr(matchups, "resolve_player", lambda c, player, teams: ("k", "KC", "RB", "DET"))
    tag = [{"key": "def_soft", "direction": "tailwind", "detail": ""}]
    good = row("Rb One", "e1", market="player_rush_yds", tags=tag, matchup="Detroit Lions @ Kansas City Chiefs")
    flat = row("Rb Two", "e2", market="player_rush_yds", tags=[], matchup="Detroit Lions @ Kansas City Chiefs")
    out = blind.build_panels(FakeConn(), "nfl", [good, flat], NOW)
    assert out[blind.panel_key(good)].favorable is True
    assert out[blind.panel_key(flat)].favorable is False
    assert (good["team"], good["position"]) == ("KC", "RB")
