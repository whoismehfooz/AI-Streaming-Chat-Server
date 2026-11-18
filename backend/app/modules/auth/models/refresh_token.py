from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Uuid,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


if TYPE_CHECKING:
    from app.modules.auth.models.session import AuthSession


class RefreshToken(Base):

    __tablename__ = "auth_refresh_tokens"

    __table_args__ = (
        UniqueConstraint(
            "token_hash",
            name="uq_auth_refresh_tokens_token_hash",
        ),
        UniqueConstraint(
            "parent_token_id",
            name="uq_auth_refresh_tokens_parent_token_id",
        ),
        CheckConstraint(
            "expires_at > created_at",
            name="ck_auth_refresh_tokens_expiry_after_creation",
        ),
        CheckConstraint(
            "parent_token_id IS NULL OR parent_token_id <> id",
            name="ck_auth_refresh_tokens_parent_not_self",
        ),
        CheckConstraint(
            "length(token_hash) = 64",
            name="ck_auth_refresh_tokens_hash_length",
        ),
        Index(
            "ix_auth_refresh_tokens_session_id_expires_at",
            "session_id",
            "expires_at",
        ),
        Index(
            "ix_auth_refresh_tokens_family_id",
            "family_id",
        ),
        Index(
            "ix_auth_refresh_tokens_expires_at",
            "expires_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("auth_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    family_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        default=uuid.uuid4,
        nullable=False,
    )

    parent_token_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("auth_refresh_tokens.id", ondelete="SET NULL"),
        nullable=True,
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

    session: Mapped["AuthSession"] = relationship(
        "AuthSession",
        back_populates="refresh_tokens",
        lazy="raise",
    )

    def __repr__(self) -> str:
        return (
            "RefreshToken("
            f"id={self.id!r}, "
            f"session_id={self.session_id!r}, "
            f"family_id={self.family_id!r}, "
            f"used={self.used_at is not None!r}, "
            f"revoked={self.revoked_at is not None!r}, "
            f"expires_at={self.expires_at!r}"
            ")"
        )
