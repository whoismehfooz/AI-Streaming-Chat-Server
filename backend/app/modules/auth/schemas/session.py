from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import SessionRevocationReason


class SessionResponse(BaseModel):

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
    )

    id: UUID
    created_at: datetime
    last_used_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    revocation_reason: SessionRevocationReason | None
    client_ip: str | None
    user_agent: str | None
    device_label: str | None
    is_revoked: bool
    is_current: bool = Field(
        default=False,
        description=(
            "Set by the service when this session matches the session used "
            "for the current request. It is not persisted on the ORM model."
        ),
    )


class SessionListResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    items: list[SessionResponse]
    total_count: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class SessionRevocationResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    revoked: bool
    revoked_at: datetime | None
    reason: SessionRevocationReason


class GlobalSessionRevocationResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    revoked_count: int = Field(ge=0)
