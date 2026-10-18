# backend/tests/unit/test_edge_store.py
from datetime import datetime, timezone

from src.edge import opportunities
from src.edge.engine import Fair, Priced
from src.edge.quotes import Quote
from src.edge.store import changed_quotes

NOW = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)


def q(book="fanduel", price=-110, line=27.5, side="over"):
    return Quote("e1", "player_points", "Jayson Tatum", side, line, book, price, "https://x", NOW)


def test_new_keys_and_price_changes_are_history_but_repeats_are_not():
    existing = {q().key: -110, q(book="draftkings").key: -115}
    new = [q(), q(book="draftkings", price=-120), q(book="betmgm")]
    changed = changed_quotes(existing, new)
    assert [c.book for c in changed] == ["draftkings", "betmgm"]


def test_a_new_line_is_a_new_key():
    existing = {q().key: -110}
    assert changed_quotes(existing, [q(line=28.5)]) == [q(line=28.5)]


def test_opportunity_rows_only_for_ev():
    hit = Priced(q(book="betmgm", price=120), Fair(0.5, 4, True), 0.10, True, False)
    miss = Priced(q(), Fair(0.5, 4, True), -0.045, False, False)
    rows = opportunities.opportunity_rows([hit, miss], NOW)
    assert len(rows) == 1
    r = rows[0]
    assert r[:8] == ("e1", "player_points", "Jayson Tatum", "over", 27.5, "betmgm", 120, "https://x")
    assert r[8] == NOW and r[9] == 0.5 and r[10] == 0.10 and r[11] == 4
    assert r[12:] == (NOW, 0.10, "power", False, None)  # last_seen_at, max_ev, devig_method, closing_done, tags
