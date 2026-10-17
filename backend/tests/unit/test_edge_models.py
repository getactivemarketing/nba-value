# backend/tests/unit/test_edge_models.py
from src.database import Base
import src.models  # noqa: F401  registers all tables


def _uniques(table):
    return {tuple(c.name for c in u.columns) for u in table.constraints
            if u.__class__.__name__ == "UniqueConstraint"}


def test_edge_tables_registered():
    for name in ("edge_events", "edge_quotes", "edge_quote_history", "edge_players",
                 "edge_player_games", "edge_opportunities", "edge_api_calls", "edge_worker_state"):
        assert name in Base.metadata.tables, name


def test_quote_and_opportunity_keys():
    t = Base.metadata.tables
    assert ("event_id", "market", "player", "side", "line", "book") in _uniques(t["edge_quotes"])
    assert ("event_id", "market", "player", "side", "line", "book", "price") in _uniques(t["edge_opportunities"])
    assert ("bdl_player_id", "game_date") in _uniques(t["edge_player_games"])


def test_backtest_tables_registered_with_keys():
    t = Base.metadata.tables
    assert ("event_id", "kind", "label") in _uniques(t["edge_bt_snapshots"])
    assert ("run_id", "event_id", "label", "market", "player", "side", "line", "book") in _uniques(t["edge_bt_picks"])
    assert t["edge_bt_snapshots"].c.payload.type.__class__.__name__ == "JSONB"
