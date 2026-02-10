"""MLB Game Context database model for weather and park factors."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import String, Integer, DateTime, Numeric, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class MLBGameContext(Base):
    """Game context including weather and park factors."""

    __tablename__ = "mlb_game_context"

    context_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(50), nullable=False, unique=True, index=True)

    # Venue info
    venue_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    venue_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Park factor - run scoring environment (1.0 = neutral, >1 = hitter friendly)
    park_factor: Mapped[Decimal | None] = mapped_column(Numeric(4, 2), nullable=True)

    # Weather conditions
    temperature: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Fahrenheit
    wind_speed: Mapped[int | None] = mapped_column(Integer, nullable=True)  # MPH
    wind_direction: Mapped[str | None] = mapped_column(String(20), nullable=True)  # "out to center", "in from left"
    humidity: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Percentage
    precipitation_pct: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Rain probability
    sky_condition: Mapped[str | None] = mapped_column(String(20), nullable=True)  # "clear", "cloudy", "dome"

    # Venue characteristics
    is_dome: Mapped[bool] = mapped_column(Boolean, default=False)
    is_retractable: Mapped[bool] = mapped_column(Boolean, default=False)
    roof_status: Mapped[str | None] = mapped_column(String(10), nullable=True)  # "open", "closed"

    # Derived factors
    wind_factor: Mapped[Decimal | None] = mapped_column(Numeric(4, 2), nullable=True)  # Impact on scoring
    weather_factor: Mapped[Decimal | None] = mapped_column(Numeric(4, 2), nullable=True)  # Combined weather impact

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
