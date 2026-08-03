"""Append-only parallel shadow store for NFL.

One row per (game, scoring timestamp, market candidate, MODEL). Every model
scores every candidate at every timestamp — market-only baseline, the frozen
incumbent, the rolling-origin challenger, and any future v2 — so they are
compared on identical inputs rather than on separately-collected samples.

Built BEFORE the scheduler is enabled, deliberately. MLB's equivalent arrived
four months into the season and the pre-existing history is unrecoverable;
worse, `mlb_markets` only ever kept the CURRENT line, so the historical
candidate set is gone and selection experiments are impossible forever. This
keeps every candidate, not only the one a model would have picked.

DURABILITY: prediction columns are IMMUTABLE once written. The settlement
columns (closing lines, CLV, outcome) are filled once after the game, in the
same spirit as `mlb_prediction_snapshots` — grading is an outcome, not a
rewrite of the forecast.

Nothing here feeds `best_bet`. All three NFL markets are disabled; promotion
requires the pre-registered gate in docs/engineering/07-nfl-promotion-gates.md.
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class NFLShadowPrediction(Base):
    """One model's view of one market candidate at one scoring timestamp."""

    __tablename__ = "nfl_shadow_predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # --- identity -------------------------------------------------------
    game_id: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Negative after kickoff. Edge is expected to vary by this, so it is a
    # first-class reporting dimension rather than an afterthought.
    minutes_to_kickoff: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- the candidate being priced -------------------------------------
    market_type: Mapped[str] = mapped_column(String(12), nullable=False)   # spread|total|moneyline
    side: Mapped[str] = mapped_column(String(8), nullable=False)           # home|away|over|under
    book: Mapped[str] = mapped_column(String(50), nullable=False)          # or 'consensus'
    line: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    odds_decimal: Mapped[Decimal | None] = mapped_column(Numeric(7, 3), nullable=True)

    # --- which model is speaking ----------------------------------------
    model_key: Mapped[str] = mapped_column(String(32), nullable=False)     # market_only|incumbent|challenger|v2
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Stable identity for this priced side at this book: game/book/market/
    # side/line/odds. Idempotency and "preserve every market update" pull
    # against each other, so the key is candidate CONTENT, not the timestamp —
    # re-scoring unchanged prices collides and is a no-op, while a moved line
    # or shaded juice hashes differently and becomes a new row.
    candidate_hash: Mapped[str] = mapped_column(String(40), nullable=False)

    # Why this row does or does not carry a prediction. A model that cannot
    # score still writes a row: omitting it makes "model failed"
    # indistinguishable from "candidate never existed".
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="ok")
    status_detail: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # --- the forecast ----------------------------------------------------
    predicted_value: Mapped[Decimal | None] = mapped_column(Numeric(7, 3), nullable=True)  # margin or total
    probability: Mapped[Decimal | None] = mapped_column(Numeric(6, 5), nullable=True)      # P(this side wins)
    # The residual scale used to turn predicted_value into probability.
    # Recorded per row because it IS the calibration, and the whole point of
    # the rolling-origin refactor was that this number had been wrong.
    resid_std: Mapped[Decimal | None] = mapped_column(Numeric(7, 3), nullable=True)
    # Fraction of model features that were non-null. A model scoring on
    # defaults is not the same model, and MLB spent a season proving that.
    feature_coverage: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    # --- settlement (written once, after the game) -----------------------
    closing_line: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    closing_odds: Mapped[Decimal | None] = mapped_column(Numeric(7, 3), nullable=True)
    closing_novig_prob: Mapped[Decimal | None] = mapped_column(Numeric(6, 5), nullable=True)
    line_clv: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    price_clv: Mapped[Decimal | None] = mapped_column(Numeric(7, 5), nullable=True)
    crossed_key: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    actual_value: Mapped[Decimal | None] = mapped_column(Numeric(6, 1), nullable=True)  # margin or total
    side_won: Mapped[bool | None] = mapped_column(Boolean, nullable=True)               # None = push
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # --- reporting dimensions, frozen at scoring time --------------------
    # Captured now rather than re-derived later: weather and lineup state at
    # kickoff are not recoverable afterwards, and segmenting CLV by them is a
    # requirement of the gate.
    is_dome: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    wind_mph: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    temp_f: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    qb_continuity: Mapped[str | None] = mapped_column(String(16), nullable=True)  # both|home_change|away_change|both_change

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(), nullable=False
    )

    __table_args__ = (
        # Idempotency: one row per (candidate content, model). Re-running a
        # scoring pass writes nothing new; a market move writes a new row.
        UniqueConstraint("candidate_hash", "model_key", name="uq_nfl_shadow_candidate_model"),
        Index("ix_nfl_shadow_game_model", "game_id", "model_key", "scored_at"),
        Index("ix_nfl_shadow_status", "season", "model_key", "status"),
        Index("ix_nfl_shadow_settle", "game_id", "settled_at"),
        Index("ix_nfl_shadow_season_week", "season", "week", "market_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<NFLShadow {self.game_id} {self.model_key} {self.market_type}/{self.side} "
            f"{self.book} line={self.line} p={self.probability}>"
        )
