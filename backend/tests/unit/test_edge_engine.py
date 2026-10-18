# backend/tests/unit/test_edge_engine.py
from datetime import datetime, timedelta, timezone

import pytest

from src.edge import engine
from src.edge.backtest import plan
from src.edge.config import TD_MARKET
from src.edge.views import describe
from src.edge.quotes import Quote
from src.services.mlb.clv import clv_from_closing

T0 = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)


def q(book, side, price, line=27.5, market="player_points", player="Jayson Tatum", at=T0):
    return Quote("ev1", market, player, side, line, book, price, None, at)


def market(*books):
    """books: (book, over_price, under_price)"""
    out = []
    for b, o, u in books:
        out += [q(b, "over", o), q(b, "under", u)]
    return out


class TestOddsMath:
    def test_american_to_decimal(self):
        assert engine.american_to_decimal(150) == pytest.approx(2.5)
        assert engine.american_to_decimal(-200) == pytest.approx(1.5)

    def test_prob_to_american_round_trips(self):
        assert engine.prob_to_american(0.5) == -100
        assert engine.prob_to_american(0.6) == -150
        assert engine.prob_to_american(0.4) == 150

    def test_power_devig_sums_to_one_and_respects_favourite(self):
        a, b = engine.devig_power(engine.implied_prob(-150), engine.implied_prob(130))
        assert a + b == pytest.approx(1.0, abs=1e-9)
        assert a > 0.55 and b < 0.45

    def test_power_devig_of_fair_market_is_identity(self):
        a, b = engine.devig_power(0.5, 0.5)
        assert a == pytest.approx(0.5) and b == pytest.approx(0.5)


class TestFairPrice:
    def test_leave_one_out_excludes_the_target_book(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110),
                        ("draftkings", -110, -110), ("betmgm", 300, -500))
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        with_mgm = engine.fair_prob(books, "player_points", "over", exclude_book=None)
        without_mgm = engine.fair_prob(books, "player_points", "over", exclude_book="betmgm")
        assert without_mgm.prob == pytest.approx(0.5, abs=1e-6)
        assert with_mgm.prob != pytest.approx(0.5, abs=1e-3)

    def test_thin_without_a_sharp_book(self):
        quotes = market(("fanduel", -110, -110), ("draftkings", -110, -110), ("betmgm", -110, -110))
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        f = engine.fair_prob(books, "player_points", "over", exclude_book=None)
        assert f.thin and f.n_books == 3 and not f.has_sharp

    def test_thin_with_fewer_than_three_books(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110))
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        assert engine.fair_prob(books, "player_points", "over", exclude_book=None).thin

    def test_one_sided_quotes_are_not_consensus_inputs(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110), ("draftkings", -110, -110))
        quotes.append(q("betmgm", "over", 500))  # over only
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        assert engine.fair_prob(books, "player_points", "over", exclude_book=None).n_books == 3

    def test_dfs_books_never_count(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110),
                        ("prizepicks", -137, -137), ("underdog", -137, -137))
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        assert engine.fair_prob(books, "player_points", "over", exclude_book=None).n_books == 2

    def test_different_lines_are_different_markets(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110), ("draftkings", -110, -110))
        quotes += [q("betmgm", "over", 200, line=25.5), q("betmgm", "under", -250, line=25.5)]
        groups = engine.group(quotes)
        assert ("ev1", "player_points", "Jayson Tatum", 25.5) in groups
        assert "betmgm" not in groups[("ev1", "player_points", "Jayson Tatum", 27.5)]

    def test_wide_hold_pairs_are_not_consensus_inputs(self):
        # Exchanges post placeholder two-way quotes (-120/-245 is ~25% hold); devigged and
        # weighted as sharps they invented +20% EV rows on the live NFL board.
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110),
                        ("draftkings", -110, -110), ("novig", -120, -245))
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        f = engine.fair_prob(books, "player_points", "over", exclude_book=None)
        assert f.n_books == 3 and f.prob == pytest.approx(0.5, abs=1e-6)

    def test_ordinary_retail_hold_still_counts(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -130, -110), ("draftkings", -110, -110))
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        assert engine.fair_prob(books, "player_points", "over", exclude_book=None).n_books == 3

    def test_a_wide_sharp_quote_does_not_satisfy_the_sharp_rule(self):
        quotes = market(("fanduel", -110, -110), ("draftkings", -110, -110),
                        ("betmgm", -110, -110), ("prophetx", -112, -325))
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        f = engine.fair_prob(books, "player_points", "over", exclude_book=None)
        assert f.thin and not f.has_sharp

    def test_weights_pull_toward_pinnacle(self):
        quotes = market(("pinnacle", -150, 130), ("fanduel", -110, -110), ("draftkings", -110, -110))
        books = engine.group(quotes)[("ev1", "player_points", "Jayson Tatum", 27.5)]
        f = engine.fair_prob(books, "player_points", "over", exclude_book=None)
        pin_over, _ = engine.devig_power(engine.implied_prob(-150), engine.implied_prob(130))
        unweighted = (pin_over + 0.5 + 0.5) / 3
        assert f.prob > unweighted


class TestPricing:
    def test_soft_book_above_consensus_is_ev(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110),
                        ("draftkings", -110, -110), ("betmgm", 120, -145))
        priced = engine.price_quotes(quotes, min_ev=0.02, verify_ev=0.15)
        hit = [p for p in priced if p.quote.book == "betmgm" and p.quote.side == "over"][0]
        assert hit.is_ev and hit.ev == pytest.approx(0.5 * 2.2 - 1, abs=1e-6)
        assert not hit.verify

    def test_huge_edges_are_flagged_verify(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110),
                        ("draftkings", -110, -110), ("betmgm", 200, -260))
        hit = [p for p in engine.price_quotes(quotes, 0.02, 0.15)
               if p.quote.book == "betmgm" and p.quote.side == "over"][0]
        assert hit.verify

    def test_thin_markets_have_no_ev(self):
        quotes = market(("fanduel", -110, -110), ("betmgm", 150, -180))
        assert all(p.ev is None and not p.is_ev for p in engine.price_quotes(quotes, 0.02, 0.15))

    def test_dfs_books_are_not_priced(self):
        quotes = market(("pinnacle", -110, -110), ("fanduel", -110, -110),
                        ("draftkings", -110, -110), ("prizepicks", -137, -137))
        assert not [p for p in engine.price_quotes(quotes, 0.02, 0.15) if p.quote.book == "prizepicks"]


class TestClosing:
    def test_latest_before_takes_last_quote_per_key_inside_window(self):
        early = q("fanduel", "over", -105, at=T0 - timedelta(hours=3))
        late = q("fanduel", "over", -120, at=T0 - timedelta(minutes=5))
        after = q("fanduel", "over", -150, at=T0 + timedelta(minutes=5))
        out = engine.latest_before([early, late, after], cutoff=T0, window=timedelta(minutes=60))
        assert [x.price for x in out] == [-120]

    def test_latest_before_drops_quotes_older_than_window(self):
        stale = q("fanduel", "over", -105, at=T0 - timedelta(hours=3))
        assert engine.latest_before([stale], cutoff=T0, window=timedelta(minutes=30)) == []

    def test_clv_matches_the_mlb_definition(self):
        for price, p in ((120, 0.48), (-150, 0.62), (100, 0.5)):
            expected = clv_from_closing(engine.american_to_decimal(price), p)
            assert engine.clv(p, price) == pytest.approx(expected, abs=1e-12)


def test_group_keeps_the_best_price_per_book_side():
    qs = [q("prophetx", "over", 130, line=0.5, market="player_anytime_td"),
          q("prophetx", "over", 132, line=0.5, market="player_anytime_td"),
          q("prophetx", "over", 129, line=0.5, market="player_anytime_td")]
    assert engine.group(qs)[("ev1", "player_anytime_td", "Jayson Tatum", 0.5)]["prophetx"]["over"].price == 132


def test_group_best_price_handles_negative_american_odds():
    qs = [q("novig", "under", -150), q("novig", "under", -120), q("novig", "under", -140)]
    assert engine.group(qs)[("ev1", "player_points", "Jayson Tatum", 27.5)]["novig"]["under"].price == -120


def _td(b, o, u):
    return [q(b, "over", o, 0.5, "player_anytime_td"), q(b, "under", u, 0.5, "player_anytime_td")]


def test_td_consensus_needs_two_two_sided_sharps_and_ignores_retail():
    books = engine.group(_td("pinnacle", 125, -152) + _td("novig", 141, -144)
                         + _td("draftkings", 125, -160) + [q("fanduel", "over", 130, 0.5, "player_anytime_td")])
    f = engine.fair_prob(books[("ev1", "player_anytime_td", "Jayson Tatum", 0.5)], "player_anytime_td", "over", None)
    assert f.n_books == 2 and not f.thin   # draftkings two-sided but not an input; fanduel yes-only


def test_td_with_one_sharp_is_thin():
    retail = [x for b in ("draftkings", "fanduel", "betmgm")
              for x in [q(b, "over", 130, 0.5, "player_anytime_td")]]
    books = engine.group(_td("pinnacle", 125, -152) + retail)
    f = engine.fair_prob(books[("ev1", "player_anytime_td", "Jayson Tatum", 0.5)], "player_anytime_td", "over", None)
    assert f.thin and f.n_books == 1


def test_td_still_applies_the_max_hold_filter():
    books = engine.group(_td("pinnacle", 125, -152) + _td("novig", -120, -245))  # novig ~25% hold
    f = engine.fair_prob(books[("ev1", "player_anytime_td", "Jayson Tatum", 0.5)], "player_anytime_td", "over", None)
    assert f.thin and f.n_books == 1


def test_other_markets_keep_three_books_and_a_sharp():
    qs = []
    for b in ("pinnacle", "novig"):
        qs += [q(b, "over", -110, 60.5, "player_rush_yds"), q(b, "under", -110, 60.5, "player_rush_yds")]
    books = engine.group(qs)[("ev1", "player_rush_yds", "Jayson Tatum", 60.5)]
    assert engine.fair_prob(books, "player_rush_yds", "over", None).thin
    for b in ("draftkings",):
        qs += [q(b, "over", -110, 60.5, "player_rush_yds"), q(b, "under", -110, 60.5, "player_rush_yds")]
    books = engine.group(qs)[("ev1", "player_rush_yds", "Jayson Tatum", 60.5)]
    assert not engine.fair_prob(books, "player_rush_yds", "over", None).thin


def test_backtest_props_cost_stays_200_without_td():
    assert plan.PROPS_COST == 200 and TD_MARKET not in plan.BT_PROP_MARKETS


def test_describe_anytime_td_reads_yes_no():
    assert describe("player_anytime_td", "Woody Marks", "over", 0.5, "H", "A") == "Woody Marks Anytime TD — Yes"
    assert describe("player_anytime_td", "Woody Marks", "under", 0.5, "H", "A") == "Woody Marks Anytime TD — No"
