"""Evaluation database model for post-game outcome tracking."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import String, DateTime, Numeric, Boolean, Integer, Index, BigInteger, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class Evaluation(Base):
    """Post-game evaluation of a prediction/bet.

    Tracks whether predictions were correct and calculates
    performance metrics like CLV and ROI.
    """

    __tablename__ = "evaluations"

    eval_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # References
    value_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("value_scores.value_id"), nullable=False
    )
    game_id: Mapped[str] = mapped_column(String(50), nullable=False)
    market_id: Mapped[str] = mapped_column(String(50), nullable=False)

    # Evaluation timing
    eval_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Game outcome
    home_score: Mapped[int] = mapped_column(Integer, nullable=False)
    away_score: Mapped[int] = mapped_column(Integer, nullable=False)
    home_margin: Mapped[int] = mapped_column(Integer, nullable=False)
    total_points: Mapped[int] = mapped_column(Integer, nullable=False)

    # Prediction at time of score
    predicted_prob: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)  # p_true
    market_prob_at_bet: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)

    # Value Score at time of bet
    algo_a_score_at_bet: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    algo_b_score_at_bet: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)

    # Outcome
    bet_won: Mapped[bool] = mapped_column(Boolean, nullable=False)
    bet_pushed: Mapped[bool] = mapped_column(Boolean, default=False)

    # CLV (Closing Line Value) - did we beat the closing line?
    closing_prob: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    clv_edge: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)  # p_bet - p_close
    beat_closing_line: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # ROI calculation (assuming $100 bet at odds)
    odds_at_bet: Mapped[Decimal] = mapped_column(Numeric(6, 3), nullable=False)
    profit_loss: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)  # +/- dollars
    roi_pct: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)  # % return

    # Bucketing for analysis
    value_score_bucket: Mapped[str | None] = mapped_column(String(20), nullable=True)  # "80-90", "90-100"
    edge_bucket: Mapped[str | None] = mapped_column(String(20), nullable=True)
    confidence_bucket: Mapped[str | None] = mapped_column(String(20), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("idx_eval_game", "game_id"),
        Index("idx_eval_market", "market_id"),
        Index("idx_eval_value_id", "value_id"),
        Index("idx_eval_bucket", "value_score_bucket"),
        Index("idx_eval_time", "eval_time"),
    )
