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
    assert blind.to_card(play(), [TIP]).tip_label == "Tonight"
    assert blind.to_card(play(), [TIP, other]).tip_label == "Tonight"  # nobody else tips then: a time names the game
    assert blind.to_card(play(), []).tip_label == "Tonight"


def test_tip_label_is_et_when_games_share_the_tip():
    # 23:00 UTC in November is 6:00 PM ET
    assert blind.to_card(play(), [TIP, TIP, TIP + timedelta(hours=1)]).tip_label == "6:00 PM ET"


def test_to_card_carries_form_and_no_identity():
    c = blind.to_card(play(), [TIP, TIP])
    assert (c.l10_hits, c.l10_games, c.l10_avg) == (8, 10, 24.3)
    assert c.n == 7 and c.position == "F" and c.price == 116
    assert c.panel["opp_vs_pos_rank"] == 25
    assert blind.to_card(play(form=None), [TIP]).l10_hits is None


def test_explain_template():
    p = {"n_books": 9, "fair_prob": 0.49, "book": "fanduel", "price": 116}
    assert blind.explain(p) == "9 books price this at 49%; FanDuel pays +116, which needs only 46%."
    assert "-110" in blind.explain({**p, "price": -110}) and "52%" in blind.explain({**p, "price": -110})
