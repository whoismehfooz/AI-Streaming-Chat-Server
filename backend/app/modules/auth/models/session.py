from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum as SQLAlchemyEnum,
    ForeignKey,
    Index,
    String,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import SessionRevocationReason
from app.db.base import Base
from app.modules.users.models.user import User


if TYPE_CHECKING:
    from app.modules.auth.models.refresh_token import RefreshToken


_SESSION_REVOCATION_REASON_TYPE = SQLAlchemyEnum(
    SessionRevocationReason,
    name="session_revocation_reason",
    native_enum=False,
    create_constraint=True,
    validate_strings=True,
    values_callable=lambda enum_type: [member.value for member in enum_type],
    length=32,
)


class AuthSession(Base):
    """An authenticated session owned by exactly one user."""

    __tablename__ = "auth_sessions"

    __table_args__ = (
        CheckConstraint(
            "expires_at > created_at",
            name="ck_auth_sessions_expiry_after_creation",
        ),
        CheckConstraint(
            """
            (revoked_at IS NULL AND revocation_reason IS NULL)
            OR
            (revoked_at IS NOT NULL AND revocation_reason IS NOT NULL)
            """,
            name="ck_auth_sessions_revocation_state_consistent",
        ),
        Index(
            "ix_auth_sessions_user_id_created_at",
            "user_id",
            "created_at",
        ),
        Index(
            "ix_auth_sessions_user_id_expires_at",
            "user_id",
            "expires_at",
        ),
        Index(
            "ix_auth_sessions_revoked_at_expires_at",
            "revoked_at",
            "expires_at",
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

    client_ip: Mapped[str | None] = mapped_column(
        String(45),
        nullable=True,
    )

    user_agent: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
    )

    device_label: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    revocation_reason: Mapped[SessionRevocationReason | None] = mapped_column(
        _SESSION_REVOCATION_REASON_TYPE,
        nullable=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        "User",
        lazy="raise",
    )

    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        "RefreshToken",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="RefreshToken.created_at",
        lazy="raise",
    )

    @property
    def is_revoked(self) -> bool:
        """Whether this session has been explicitly revoked."""

        return self.revoked_at is not None

    def __repr__(self) -> str:
        return (
            "AuthSession("
            f"id={self.id!r}, "
            f"user_id={self.user_id!r}, "
            f"revoked={self.is_revoked!r}, "
            f"expires_at={self.expires_at!r}"
            ")"
        )
