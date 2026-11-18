"""Request-scoped construction of application services.

This module is the composition root for HTTP endpoints: it wires the existing
SQLAlchemy request session to Auth, password, verification, session, user and
rate-limit services. Business rules remain in their domain services, while
email delivery remains behind the provider-neutral service interfaces.

The Resend adapter is resolved lazily. A missing email API key therefore does
not prevent unrelated operations such as login or logout from constructing
an Auth service; an operation that actually sends email receives a safe,
provider-neutral configuration error.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.dependencies import get_db_session
from app.integrations.email.base import EmailProviderError
from app.integrations.email.resend_client import ResendEmailClient
from app.modules.auth.services.auth_service import AuthService
from app.modules.auth.services.email_verification_service import EmailVerificationService
from app.modules.auth.services.password_service import PasswordService
from app.modules.auth.services.rate_limit_service import AuthenticationRateLimiter
from app.modules.auth.services.session_service import SessionService
from app.modules.auth.services.token_service import TokenService
from app.modules.users.services.user_service import UserService


class _LazyResendEmailSender:
    """Provider-neutral lazy proxy implementing all Auth email contracts."""

    def __init__(self) -> None:
        self._client: ResendEmailClient | None = None

    def _get_client(self, *, operation: str) -> ResendEmailClient:
        if settings.RESEND_API_KEY is None:
            raise EmailProviderError(operation=operation, retryable=False)
        if self._client is None:
            self._client = ResendEmailClient(settings)
        return self._client

    async def send_verification_email(self, *, recipient: str, token: str) -> None:
        await self._get_client(operation="send_verification_email").send_verification_email(
            recipient=recipient,
            token=token,
        )

    async def send_email_change_verification_email(
        self,
        *,
        recipient: str,
        token: str,
    ) -> None:
        await self._get_client(
            operation="send_email_change_verification_email"
        ).send_email_change_verification_email(
            recipient=recipient,
            token=token,
        )

    async def send_password_reset_email(self, *, recipient: str, token: str) -> None:
        await self._get_client(operation="send_password_reset_email").send_password_reset_email(
            recipient=recipient,
            token=token,
        )


def get_email_sender() -> _LazyResendEmailSender:
    """Return a per-request lazy email sender shared by Auth sub-services."""
    return _LazyResendEmailSender()


def get_email_verification_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    email_sender: Annotated[_LazyResendEmailSender, Depends(get_email_sender)],
) -> EmailVerificationService:
    return EmailVerificationService(session, email_sender=email_sender)


def get_auth_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    verification_service: Annotated[
        EmailVerificationService,
        Depends(get_email_verification_service),
    ],
) -> AuthService:
    return AuthService(
        session,
        email_verification_workflow=verification_service,
    )


def get_password_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    email_sender: Annotated[_LazyResendEmailSender, Depends(get_email_sender)],
) -> PasswordService:
    return PasswordService(session, email_sender=email_sender)


def get_session_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> SessionService:
    return SessionService(session)


def get_token_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> TokenService:
    return TokenService(session)


def get_user_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserService:
    return UserService(session)


def get_auth_rate_limiter() -> AuthenticationRateLimiter:
    """Create a lightweight limiter using its independent session factory."""
    return AuthenticationRateLimiter()


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
EmailVerificationServiceDep = Annotated[
    EmailVerificationService,
    Depends(get_email_verification_service),
]
PasswordServiceDep = Annotated[PasswordService, Depends(get_password_service)]
SessionServiceDep = Annotated[SessionService, Depends(get_session_service)]
TokenServiceDep = Annotated[TokenService, Depends(get_token_service)]
UserServiceDep = Annotated[UserService, Depends(get_user_service)]
AuthenticationRateLimiterDep = Annotated[
    AuthenticationRateLimiter,
    Depends(get_auth_rate_limiter),
]


__all__ = [
    "AuthServiceDep",
    "AuthenticationRateLimiterDep",
    "EmailVerificationServiceDep",
    "PasswordServiceDep",
    "SessionServiceDep",
    "TokenServiceDep",
    "UserServiceDep",
    "get_auth_rate_limiter",
    "get_auth_service",
    "get_email_sender",
    "get_email_verification_service",
    "get_password_service",
    "get_session_service",
    "get_token_service",
    "get_user_service",
]
