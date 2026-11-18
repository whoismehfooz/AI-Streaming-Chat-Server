from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    field_validator,
)

from app.modules.auth.schemas.tokens import TokenPairResponse
from app.modules.users.schemas.user import UserResponse


class LoginRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: SecretStr = Field(
        min_length=1,
        max_length=1024,
        repr=False,
        description="Account password. Never log or expose this value.",
    )

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: object) -> object:
        """Match the normalized email representation used by User."""

        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("Password cannot be empty or whitespace.")
        return value


class LoginResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    user: UserResponse
    tokens: TokenPairResponse
