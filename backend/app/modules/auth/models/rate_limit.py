"""PostgreSQL persistence model for authentication rate-limit state.

The table stores one GCRA (Generic Cell Rate Algorithm) state row per
rate-limit scope and HMAC-derived subject key. Raw IP addresses, email
addresses, user identifiers, and session identifiers are never persisted.

The model is deliberately independent of User and AuthSession: rate-limit
state must also protect unauthenticated registration, login, verification,
and password-recovery requests, and should not disappear when an account or
session is deleted.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Index,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AuthenticationRateLimitEntry(Base):
    """Persisted GCRA state for one authentication rate-limit subject."""

    __tablename__ = "auth_rate_limit_entries"

    __table_args__ = (
        CheckConstraint(
            "length(scope) BETWEEN 1 AND 80",
            name="ck_auth_rate_limit_entries_scope_length",
        ),
        CheckConstraint(
            "length(key_hash) = 64",
            name="ck_auth_rate_limit_entries_key_hash_length",
        ),
        Index(
            "ix_auth_rate_limit_entries_expires_at",
            "expires_at",
        ),
    )

    scope: Mapped[str] = mapped_column(
        String(80),
        primary_key=True,
    )

    key_hash: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    # GCRA theoretical arrival time (TAT). A request is admissible when
    # TAT <= now + burst_tolerance. The repository advances TAT atomically.
    theoretical_arrival_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Rows become logically expired once TAT has passed. Physical cleanup is
    # intentionally decoupled from request handling and can run periodically.
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        # Do not include key_hash: even pseudonymous identifiers should not
        # become routine log output through model representations.
        return (
            "AuthenticationRateLimitEntry("
            f"scope={self.scope!r}, "
            f"expires_at={self.expires_at!r}"
            ")"
        )


__all__ = ["AuthenticationRateLimitEntry"]
