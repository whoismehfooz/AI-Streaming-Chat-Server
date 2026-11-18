"""FastAPI dependencies for authentication and verified-user authorization.

Authentication establishes *who* is making a request. Verified-user
authorization establishes whether that authenticated account has completed
email verification. These are deliberately separate dependencies so public
account-recovery and verification operations can remain available to an
authenticated-but-unverified account where their endpoint policy permits it.

The dependency chain is:

    Authorization: Bearer <access JWT>
        -> TokenService JWT validation
        -> persistent PostgreSQL session validation
        -> active-account validation
        -> AuthenticatedPrincipal
        -> CurrentUser
        -> CurrentVerifiedUser (when an operation requires verification)

This module is an HTTP dependency layer only. It does not own database
transactions, mutate account state, implement resource ownership, or impose
email verification on every authenticated endpoint. The request-scoped DB
dependency owns session cleanup and service methods own write transactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated
import uuid

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.dependencies import get_db_session
from app.exceptions.authentication import (
    AuthenticationRequiredException,
    EmailVerificationRequiredException,
)
from app.modules.auth.models.session import AuthSession
from app.modules.auth.services.token_service import AccessTokenClaims, TokenService
from app.modules.users.models.user import User


# auto_error=False allows the application to return its standard authentication
# exception for missing credentials instead of FastAPI's default HTTP response.
bearer_access_token_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="BearerAccessToken",
    bearerFormat="JWT",
    description="Short-lived JWT access token issued by the authentication service.",
)


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    """Identity resolved from a valid access JWT and live database session."""

    user: User
    session: AuthSession
    claims: AccessTokenClaims

    @property
    def user_id(self) -> uuid.UUID:
        """Stable authenticated user identifier for service-layer ownership checks."""

        return self.user.id


async def get_current_principal(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_access_token_scheme),
    ],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> AuthenticatedPrincipal:
    """Authenticate the request and resolve its user and persistent session.

    TokenService validates the JWT signature, configured algorithm, issuer,
    audience, required claims, token type and time claims. It then validates the
    persistent PostgreSQL session and confirms that the account still exists
    and is active. No raw access token is logged, stored or returned here.
    """

    if credentials is None:
        raise AuthenticationRequiredException()

    if credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise AuthenticationRequiredException()

    token_service = TokenService(session)
    user, auth_session, claims = await token_service.authenticate_access_token(
        credentials.credentials
    )

    return AuthenticatedPrincipal(
        user=user,
        session=auth_session,
        claims=claims,
    )


async def get_current_user(
    principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)],
) -> User:
    """Return the active user authenticated for the current request."""

    return principal.user


async def get_current_session(
    principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)],
) -> AuthSession:
    """Return the active persistent session authenticated for the request."""

    return principal.session


async def get_current_user_id(
    principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)],
) -> uuid.UUID:
    """Return the authenticated user's UUID for service-layer ownership checks."""

    return principal.user_id


async def get_current_access_token_claims(
    principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)],
) -> AccessTokenClaims:
    """Return validated access-token claims without exposing the raw JWT."""

    return principal.claims


async def get_verified_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Require an authenticated, active user whose email is verified.

    Use this dependency only on operations whose policy requires verified
    email ownership, such as protected application functionality. Do not apply
    it globally: login, logout, refresh, verification, resend, password-reset
    and other account-recovery routes may intentionally need different rules.

    The verification state is read from the database-resolved User returned by
    TokenService, never from a JWT claim or client-supplied field. This keeps
    the authorization decision tied to persisted account state and means a
    verification-state change is observed on the next authenticated request.
    """

    if current_user.is_verified is not True:
        raise EmailVerificationRequiredException()

    return current_user


# Reusable dependency annotations keep endpoint signatures concise. FastAPI's
# per-request dependency cache ensures these values share the same principal
# and database dependency resolution when used together in one request.
CurrentPrincipal = Annotated[AuthenticatedPrincipal, Depends(get_current_principal)]
CurrentUser = Annotated[User, Depends(get_current_user)]
CurrentVerifiedUser = Annotated[User, Depends(get_verified_user)]
CurrentAuthSession = Annotated[AuthSession, Depends(get_current_session)]
CurrentUserId = Annotated[uuid.UUID, Depends(get_current_user_id)]
CurrentAccessTokenClaims = Annotated[
    AccessTokenClaims,
    Depends(get_current_access_token_claims),
]


__all__ = [
    "AuthenticatedPrincipal",
    "CurrentAccessTokenClaims",
    "CurrentAuthSession",
    "CurrentPrincipal",
    "CurrentUser",
    "CurrentUserId",
    "CurrentVerifiedUser",
    "bearer_access_token_scheme",
    "get_current_access_token_claims",
    "get_current_principal",
    "get_current_session",
    "get_current_user",
    "get_current_user_id",
    "get_verified_user",
]
