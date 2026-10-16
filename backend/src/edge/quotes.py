# backend/src/edge/quotes.py
"""One book's price for one side of one market, as observed by a poll.

Line convention (spec Global Constraints): spreads carry the HOME-perspective
point so both sides of a spread share a market key; totals/props carry the
point; h2h carries 0.0.
"""
from dataclasses import dataclass
from datetime import datetime

GAME_SIDES = ("home", "away")
OU_SIDES = ("over", "under")


def sides_for(market: str) -> tuple[str, str]:
    return OU_SIDES if market == "totals" or market.startswith("player_") else GAME_SIDES


@dataclass(frozen=True)
class Quote:
    event_id: str
    market: str
    player: str        # "" for game lines
    side: str          # home|away|over|under
    line: float
    book: str
    price: int         # American
    link: str | None
    observed_at: datetime

    @property
    def market_key(self) -> tuple[str, str, str, float]:
        return (self.event_id, self.market, self.player, float(self.line))

    @property
    def key(self) -> tuple[str, str, str, float, str, str]:
        return (*self.market_key, self.side, self.book)
