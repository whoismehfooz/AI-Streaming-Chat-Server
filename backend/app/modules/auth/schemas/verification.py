from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    field_validator,
)


_TOKEN_MIN_LENGTH = 43
_TOKEN_MAX_LENGTH = 256

_GENERIC_VERIFICATION_MESSAGE = (
    "If the account is eligible, an email verification message will be sent."
)


class EmailVerificationRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    token: SecretStr = Field(
        min_length=_TOKEN_MIN_LENGTH,
        max_length=_TOKEN_MAX_LENGTH,
        repr=False,
        description="Opaque email-verification token from the email link.",
    )


class EmailVerificationResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    verified: Literal[True] = True
    message: str = "Email address verified successfully."


class ResendVerificationRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    email: EmailStr

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class ResendVerificationResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    accepted: Literal[True] = True
    message: Literal["If the account is eligible, an email verification message will be sent."] = _GENERIC_VERIFICATION_MESSAGE


class EmailChangeRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    new_email: EmailStr
    current_password: SecretStr = Field(
        min_length=1,
        max_length=1024,
        repr=False,
    )

    @field_validator("new_email", mode="before")
    @classmethod
    def normalize_new_email(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("current_password")
    @classmethod
    def validate_current_password(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("Password cannot be empty or whitespace.")
        return value


class EmailChangeVerificationRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    token: SecretStr = Field(
        min_length=_TOKEN_MIN_LENGTH,
        max_length=_TOKEN_MAX_LENGTH,
        repr=False,
    )


class EmailChangeRequestResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    accepted: Literal[True] = True
    delivered: bool
    message: str = "If the request is eligible, a verification message will be sent to the new email address."
