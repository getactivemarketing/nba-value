from datetime import date, datetime, timedelta, timezone

import pytest

from src.edge import nightly

NOW = datetime(2026, 11, 11, 23, 0, tzinfo=timezone.utc)  # 6 pm ET


def row(player, game, book="betmgm", ev=0.05, price=110, fair=0.5, n_books=8, flag=False, verify=False,
        market="player_points"):
    return {"event_id": game, "matchup": f"{game} matchup", "commence_time": "2026-11-12T00:10:00+00:00",
            "market": market, "market_label": "Points", "player": player, "side": "over", "line": 20.5,
            "bet": f"{player} Over 20.5 Points", "book": book, "price": price, "fair_prob": fair, "ev": ev,
            "n_books": n_books, "verify": verify, "form": {"hits": 6, "games": 10, "rate": 0.6},
            "form_flag": flag}


def test_eligible_keeps_only_well_priced_props_that_are_not_cold():
    rows = [row("A", "g1"), row("B", "g2", n_books=4), row("C", "g3", flag=True), row("D", "g4", verify=True),
            row("", "g5", market="h2h")]
    assert [r["player"] for r in nightly.eligible(rows)] == ["A"]


def test_top_picks_take_the_best_edge_per_game_and_per_player():
    rows = [row("A", "g1", ev=0.09), row("B", "g1", ev=0.08), row("A", "g2", ev=0.07, book="bovada"),
            row("C", "g2", ev=0.03), row("D", "g3", ev=0.06), row("E", "g4", ev=0.05), row("F", "g5", ev=0.04),
            row("G", "g6", ev=0.02)]
    assert [r["player"] for r in nightly.top_picks(rows)] == ["A", "D", "E", "F", "C"]


def test_parlay_is_built_at_the_one_book_with_the_most_legs_then_the_best_value():
    rows = [row(p, f"g{i}", book="betmgm", ev=0.04) for i, p in enumerate("ABCDE")]
    rows += [row(p, f"g{i}", book="bovada", ev=0.10) for i, p in enumerate("FGHI")]  # 4 legs only
    rows += [row("J", "g0", book="betmgm", ev=0.01)]  # a weaker 2nd leg in g0: one leg per game
    p = nightly.best_parlay(rows)
    assert p["book"] == "betmgm" and [l["player"] for l in p["legs"]] == list("ABCDE")


def test_parlay_math_multiplies_prices_and_chances():
    rows = [row(p, f"g{i}", price=100, fair=0.55) for i, p in enumerate("ABC")]
    p = nightly.best_parlay(rows)
    assert p["decimal"] == pytest.approx(8.0) and p["american"] == 700
    assert p["hit_prob"] == pytest.approx(0.55 ** 3, abs=1e-4)
    assert p["ev"] == pytest.approx(8 * 0.55 ** 3 - 1, abs=1e-4)


def test_no_parlay_under_three_legs():
    assert nightly.best_parlay([row("A", "g1"), row("B", "g2")]) is None


@pytest.mark.parametrize("results, outcome, profit", [
    (["win", "win", "win"], "win", 7.0),
    (["win", "loss", None], "loss", -1.0),
    (["win", None, "win"], None, None),
    (["win", "push", "win"], "win", 3.0),  # a push drops out: pays 2 x 2
    (["push", "void", "push"], "push", 0.0),
])
def test_parlay_result(results, outcome, profit):
    legs = [{"price": 100, "result": r} for r in results]
    assert nightly.parlay_result(legs) == (outcome, profit)


def test_lock_time_is_an_hour_before_the_first_game_of_the_et_day():
    tips = [datetime(2026, 11, 12, 0, 10, tzinfo=timezone.utc), datetime(2026, 11, 12, 3, 10, tzinfo=timezone.utc)]
    assert nightly.lock_at(tips) == datetime(2026, 11, 11, 23, 10, tzinfo=timezone.utc)
    assert nightly.lock_at([]) is None
    assert nightly.slate_date(NOW) == date(2026, 11, 11)
    assert nightly.slate_date(NOW + timedelta(hours=4)) == date(2026, 11, 11)  # 11 pm ET, same slate
