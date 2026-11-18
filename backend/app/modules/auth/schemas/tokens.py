from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


BearerTokenType = Literal["bearer"]


class RefreshTokenRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    refresh_token: SecretStr = Field(
        min_length=43,
        max_length=256,
        repr=False,
        description=(
            "Opaque URL-safe refresh token. The raw value is accepted only "
            "for verification and must never be logged or persisted."
        ),
    )


class AccessTokenResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    access_token: str = Field(
        min_length=1,
        max_length=8192,
        repr=False,
        description="Short-lived signed JWT access token.",
    )
    token_type: BearerTokenType = "bearer"
    expires_in: int = Field(
        gt=0,
        description="Access-token lifetime in seconds from issuance.",
    )


class TokenPairResponse(AccessTokenResponse):

    refresh_token: str = Field(
        min_length=43,
        max_length=256,
        repr=False,
        description=(
            "Opaque refresh token. Store it securely on the client and never "
            "write it to application logs."
        ),
    )
    refresh_expires_in: int = Field(
        gt=0,
        description="Refresh-token lifetime in seconds from issuance.",
    )
