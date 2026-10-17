import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.edge.backtest import replay
from src.edge.backtest.nflverse import Game
from src.edge.engine import american_to_decimal, price_quotes
from src.edge.ingest import parse_event

UTC = timezone.utc
KICK = datetime(2023, 11, 12, 18, 3, tzinfo=UTC)
GAME = Game("2023_10_CLE_BAL", 2023, 10, "BAL", "CLE", KICK, 31, 33)
FIX = Path(__file__).resolve().parents[1] / "fixtures" / "edge" / "bt_props_2023.json"


def event(prices, player="Amari Cooper", line=55.5, market="player_reception_yds"):
    """prices: {book: (over, under)}"""
    return {"id": "e1", "home_team": "Baltimore Ravens", "away_team": "Cleveland Browns",
            "commence_time": "2023-11-12T18:03:00Z",
            "bookmakers": [{"key": b, "markets": [{"key": market, "outcomes": [
                {"name": "Over", "description": player, "price": o, "point": line},
                {"name": "Under", "description": player, "price": u, "point": line}]}]}
                for b, (o, u) in prices.items()]}


EVEN = {"pinnacle": (-110, -110), "draftkings": (-110, -110), "betmgm": (-110, -110)}
SOFT = {**EVEN, "fanduel": (120, -150)}
STAT = {"name": "amari cooper", "team": "CLE", "pass_att": 0, "pass_cmp": 0, "pass_yds": 0, "pass_tds": 0,
        "pass_int": 0, "rush_att": 0, "rush_yds": 0, "receptions": 4, "rec_yds": 60}


def snaps(t6h=None, close=None, close_at=KICK - timedelta(minutes=5)):
    out = {}
    if t6h is not None:
        out["t6h"] = {"props": replay.Snap(t6h, KICK - timedelta(hours=6))}
    if close is not None:
        out["close"] = {"props": replay.Snap(close, close_at)}
    return out


def fd_over(picks):
    return next(p for p in picks if p["book"] == "fanduel" and p["side"] == "over")


def test_picks_equal_live_engine_output_on_real_2023_payload():
    payload = json.loads(FIX.read_text())["data"]
    at = KICK - timedelta(hours=6)
    picks = replay.replay_game(GAME, {"t6h": {"props": replay.Snap(payload, at)}}, [], None, -0.05, 0.15)
    want = {(p.quote.book, p.quote.player, p.quote.side, p.quote.line)
            for p in price_quotes(parse_event(payload, at), -0.05, 0.15) if p.is_ev}
    assert want and {(p["book"], p["player"], p["side"], p["line"]) for p in picks} == want
    assert all(p["label"] == "t6h" and "pinnacle" in p["sharps"].split(",") for p in picks)


def test_pick_fields_and_clv_against_the_close():
    picks = replay.replay_game(GAME, snaps(event(SOFT), event(SOFT)), [STAT], {"amari cooper"}, 0.02, 0.15)
    p = fd_over(picks)
    assert (p["event_id"], p["market"], p["player"], p["line"], p["price"]) == \
        ("e1", "player_reception_yds", "Amari Cooper", 55.5, 120)
    assert p["fair_prob"] == pytest.approx(0.5) and p["ev"] == pytest.approx(0.1) and p["verify"] is False
    assert p["n_books"] == 3 and p["sharps"] == "pinnacle"
    assert p["closing_prob"] is not None
    assert p["clv"] == pytest.approx(p["closing_prob"] - 1 / american_to_decimal(120))
    assert p["result"] == "win" and p["profit"] == pytest.approx(1.2)


def test_no_clv_when_the_line_moved_before_close():
    picks = replay.replay_game(GAME, snaps(event(SOFT), event(SOFT, line=57.5)), [STAT], {"amari cooper"},
                               0.02, 0.15)
    p = fd_over(picks)
    assert p["closing_prob"] is None and p["clv"] is None


def test_close_snapshot_after_kickoff_is_ignored():
    picks = replay.replay_game(GAME, snaps(event(SOFT), event(SOFT), close_at=KICK + timedelta(minutes=1)),
                               [STAT], {"amari cooper"}, 0.02, 0.15)
    assert fd_over(picks)["clv"] is None


def test_flag_snapshot_at_or_after_kickoff_flags_nothing():
    late = {"t1h": {"props": replay.Snap(event(SOFT), KICK)}}
    assert replay.replay_game(GAME, late, [STAT], {"amari cooper"}, 0.02, 0.15) == []


def test_empty_stub_payloads_are_skipped():
    stub = {"t6h": {"props": replay.Snap({"id": "e1", "bookmakers": [], "missing": 404}, None)}}
    assert replay.replay_game(GAME, stub, [], None, 0.02, 0.15) == []


def test_pinnacle_only_ev_leaves_novig_and_prophetx_out():
    books = {**SOFT, "novig": (110, -130), "prophetx": (110, -130)}
    p = fd_over(replay.replay_game(GAME, snaps(event(books)), [STAT], {"amari cooper"}, -1.0, 0.15))
    assert p["sharps"] == "novig,pinnacle,prophetx"
    assert p["pin_ev"] == pytest.approx(0.1)
    assert p["ev"] < p["pin_ev"]


@pytest.mark.parametrize("stats,names,side,want", [
    ([STAT], {"amari cooper"}, "over", "win"),             # stat row grades
    ([], {"amari cooper"}, "over", "loss"),                # played, no row: graded as 0
    ([], {"someone else"}, "over", "void"),                # snaps published, not in them: inactive
    ([], None, "over", None),                              # no snap counts for the game: ungradable
    ([STAT, {**STAT, "team": "BAL"}], {"amari cooper"}, "over", None),  # two players share the name
])
def test_prop_grading(stats, names, side, want):
    assert replay.grade("player_reception_yds", "Amari Cooper", side, 55.5, GAME, stats, names) == want


@pytest.mark.parametrize("odds_name,display", [
    ("Kenneth Walker III", "Kenneth Walker III"), ("D.J. Moore", "DJ Moore"), ("Ja'Marr Chase", "Ja'Marr Chase")])
def test_prop_grading_matches_names_with_suffixes_and_punctuation(odds_name, display):
    from src.edge.players import normalize_name
    row = {**STAT, "name": normalize_name(display), "rec_yds": 80}
    assert replay.grade("player_reception_yds", odds_name, "over", 55.5, GAME, [row], {row["name"]}) == "win"


def test_game_line_grading_uses_final_scores():
    assert replay.grade("h2h", "", "away", 0.0, GAME, [], None) == "win"         # CLE 33, BAL 31
    assert replay.grade("spreads", "", "home", 3.5, GAME, [], None) == "win"     # 31 + 3.5 > 33
    assert replay.grade("totals", "", "over", 63.5, GAME, [], None) == "win"
    no_score = Game(**{**GAME.__dict__, "home_score": None})
    assert replay.grade("h2h", "", "home", 0.0, no_score, [], None) is None


def test_void_and_ungradable_picks_have_no_profit():
    picks = replay.replay_game(GAME, snaps(event(SOFT)), [], {"someone else"}, 0.02, 0.15)
    p = fd_over(picks)
    assert p["result"] == "void" and p["profit"] is None
