from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


_PASSWORD_MIN_LENGTH = 8
_PASSWORD_MAX_LENGTH = 128
_TOKEN_MIN_LENGTH = 43
_TOKEN_MAX_LENGTH = 256

_PASSWORD_RESET_ACCEPTED_MESSAGE = (
    "If an account matches the supplied email address, "
    "password-reset instructions will be sent."
)


class PasswordChangeRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    current_password: SecretStr = Field(
        min_length=1,
        max_length=1024,
        repr=False,
        description="Current account password. Never log this value.",
    )
    new_password: SecretStr = Field(
        min_length=_PASSWORD_MIN_LENGTH,
        max_length=_PASSWORD_MAX_LENGTH,
        repr=False,
        description="New password, subject to the canonical user password limits.",
    )
    confirm_new_password: SecretStr = Field(
        min_length=_PASSWORD_MIN_LENGTH,
        max_length=_PASSWORD_MAX_LENGTH,
        repr=False,
        description="Must exactly match new_password.",
    )

    @field_validator("current_password", "new_password", "confirm_new_password")
    @classmethod
    def reject_whitespace_only_passwords(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("Password cannot be empty or whitespace.")
        return value

    @model_validator(mode="after")
    def validate_password_change(self) -> "PasswordChangeRequest":
        current = self.current_password.get_secret_value()
        new = self.new_password.get_secret_value()
        confirmation = self.confirm_new_password.get_secret_value()

        if new != confirmation:
            raise ValueError("New password and confirmation do not match.")

        if new == current:
            raise ValueError("New password must differ from the current password.")

        return self


class PasswordResetRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    email: EmailStr

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class PasswordResetRequestResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    accepted: Literal[True] = True
    message: Literal["If an account matches the supplied email address, "
    "password-reset instructions will be sent."] = _PASSWORD_RESET_ACCEPTED_MESSAGE


class PasswordResetConfirmRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    token: SecretStr = Field(
        min_length=_TOKEN_MIN_LENGTH,
        max_length=_TOKEN_MAX_LENGTH,
        repr=False,
        description="Opaque, single-use password-reset token.",
    )
    new_password: SecretStr = Field(
        min_length=_PASSWORD_MIN_LENGTH,
        max_length=_PASSWORD_MAX_LENGTH,
        repr=False,
    )
    confirm_new_password: SecretStr = Field(
        min_length=_PASSWORD_MIN_LENGTH,
        max_length=_PASSWORD_MAX_LENGTH,
        repr=False,
    )

    @field_validator("new_password", "confirm_new_password")
    @classmethod
    def reject_whitespace_only_passwords(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("Password cannot be empty or whitespace.")
        return value

    @model_validator(mode="after")
    def validate_password_confirmation(self) -> "PasswordResetConfirmRequest":
        if (
            self.new_password.get_secret_value()
            != self.confirm_new_password.get_secret_value()
        ):
            raise ValueError("New password and confirmation do not match.")
        return self
