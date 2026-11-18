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


class PasswordResetToken(Base):

    __tablename__ = "auth_password_reset_tokens"

    __table_args__ = (
        UniqueConstraint(
            "token_hash",
            name="uq_auth_password_reset_tokens_token_hash",
        ),
        CheckConstraint(
            "expires_at > created_at",
            name="ck_auth_password_reset_tokens_expiry_after_creation",
        ),
        CheckConstraint(
            "length(token_hash) = 64",
            name="ck_auth_password_reset_tokens_hash_length",
        ),
        CheckConstraint(
            "used_at IS NULL OR revoked_at IS NULL",
            name="ck_auth_password_reset_tokens_not_used_and_revoked",
        ),
        Index(
            "ix_auth_password_reset_tokens_user_id_expires_at",
            "user_id",
            "expires_at",
        ),
        Index(
            "ix_auth_password_reset_tokens_expires_at",
            "expires_at",
        ),
        Index(
            "uq_auth_password_reset_tokens_one_outstanding_per_user",
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
        return self.used_at is not None or self.revoked_at is not None

    def __repr__(self) -> str:
        return (
            "PasswordResetToken("
            f"id={self.id!r}, "
            f"user_id={self.user_id!r}, "
            f"consumed={self.is_consumed!r}, "
            f"expires_at={self.expires_at!r}"
            ")"
        )
