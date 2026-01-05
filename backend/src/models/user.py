"""User database model."""

from datetime import datetime
from typing import Literal

from sqlalchemy import String, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class User(Base):
    """User account for authentication and subscription."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    # Subscription
    tier: Mapped[str] = mapped_column(String(20), default="free")
    subscription_expires: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    @property
    def is_paid(self) -> bool:
        """Check if user has active paid subscription."""
        if self.tier != "paid":
            return False
        if self.subscription_expires is None:
            return False
        return self.subscription_expires > datetime.utcnow()
