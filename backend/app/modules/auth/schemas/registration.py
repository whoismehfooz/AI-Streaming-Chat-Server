from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.modules.users.schemas.user import UserCreate, UserResponse


class RegistrationRequest(UserCreate):
    """Registration payload.

    Reuses the canonical user-creation validation so registration and other
    user-creation flows cannot drift on email normalization or password rules.
    """

    pass


class RegistrationResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    user: UserResponse
    verification_email_sent: bool = Field(
        description=(
            "Whether the verification email provider confirmed delivery. "
            "False means the persisted verification workflow can be retried."
        )
    )
