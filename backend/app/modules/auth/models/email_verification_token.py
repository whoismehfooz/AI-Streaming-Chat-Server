"""Persisted, single-use email verification and email-change tokens."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Uuid,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.modules.users.models.user import User  # noqa: F401


class EmailVerificationToken(Base):
    """A hashed verification token for signup or a pending email change.

    ``pending_email`` is NULL for initial account verification. For an email
    change, it contains the normalized destination address and the current
    ``User.email`` is left untouched until this token is consumed.
    """

    __tablename__ = "auth_email_verification_tokens"

    __table_args__ = (
        UniqueConstraint(
            "token_hash",
            name="uq_auth_email_verification_tokens_token_hash",
        ),
        CheckConstraint(
            "expires_at > created_at",
            name="ck_auth_email_verification_tokens_expiry_after_creation",
        ),
        CheckConstraint(
            "length(token_hash) = 64",
            name="ck_auth_email_verification_tokens_hash_length",
        ),
        CheckConstraint(
            "used_at IS NULL OR revoked_at IS NULL",
            name="ck_auth_email_verification_tokens_not_used_and_revoked",
        ),
        CheckConstraint(
            """
            pending_email IS NULL OR (
                pending_email = lower(trim(pending_email))
                AND length(pending_email) > 0
            )
            """,
            name="ck_auth_email_verification_tokens_pending_email_normalized",
        ),
        Index(
            "ix_auth_email_verification_tokens_user_id_expires_at",
            "user_id",
            "expires_at",
        ),
        Index(
            "ix_auth_email_verification_tokens_expires_at",
            "expires_at",
        ),
        Index(
            "uq_auth_email_verification_tokens_one_outstanding_per_user",
            "user_id",
            unique=True,
            postgresql_where=text("used_at IS NULL AND revoked_at IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    token_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    pending_email: Mapped[str | None] = mapped_column(
        String(320),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    user: Mapped["User"] = relationship(
        "User",
        lazy="raise",
    )

    @property
    def is_consumed(self) -> bool:
        """Whether this token has been used or revoked."""

        return self.used_at is not None or self.revoked_at is not None

    @property
    def is_email_change(self) -> bool:
        """Whether this token verifies a pending email change."""

        return self.pending_email is not None

    def __repr__(self) -> str:
        return (
            "EmailVerificationToken("
            f"id={self.id!r}, "
            f"user_id={self.user_id!r}, "
            f"email_change={self.is_email_change!r}, "
            f"consumed={self.is_consumed!r}, "
            f"expires_at={self.expires_at!r}"
            ")"
        )


__all__ = ["EmailVerificationToken"]
