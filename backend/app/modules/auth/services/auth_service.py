from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import hash_password, verify_password_and_update
from app.exceptions.authentication import (
    AccountInactiveException,
    EmailVerificationRequiredException,
    InvalidCredentialsException,
)
from app.modules.auth.models.session import AuthSession
from app.modules.auth.repositories.session_repository import SessionRepository
from app.modules.auth.schemas.login import LoginRequest, LoginResponse
from app.modules.auth.schemas.registration import (
    RegistrationRequest,
    RegistrationResponse,
)
from app.modules.auth.services.token_service import TokenService
from app.modules.users.models.user import User
from app.modules.users.repositories.user_repository import UserRepository
from app.modules.users.schemas.user import UserResponse
from app.modules.users.services.user_service import UserService


logger = logging.getLogger(__name__)


class EmailVerificationWorkflow(Protocol):

    async def start_for_user(self, user: User) -> bool:
        """Persist a verification workflow and report provider delivery."""
        ...


class AuthService:

    def __init__(
        self,
        session: AsyncSession,
        *,
        email_verification_workflow: EmailVerificationWorkflow,
    ) -> None:
        if email_verification_workflow is None:
            raise ValueError("email_verification_workflow is required.")

        self.session = session
        self.user_service = UserService(session)
        self.user_repository = UserRepository(session)
        self.session_repository = SessionRepository(session)
        self.token_service = TokenService(session)
        self.email_verification_workflow = email_verification_workflow

    @staticmethod
    def _utc_now(value: datetime | None = None) -> datetime:
        now = datetime.now(timezone.utc) if value is None else value
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime instance.")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware.")
        return now.astimezone(timezone.utc)

    async def register(
        self,
        request: RegistrationRequest,
    ) -> RegistrationResponse:

        if not isinstance(request, RegistrationRequest):
            raise TypeError("request must be a RegistrationRequest instance.")

        password = request.password.get_secret_value()
        password_hash = hash_password(password)

        user = await self.user_service.create_user(
            email=str(request.email),
            password_hash=password_hash,
            commit=True,
        )

        email_sent = await self.email_verification_workflow.start_for_user(user)

        return RegistrationResponse(
            user=UserResponse.model_validate(user),
            verification_email_sent=email_sent,
        )

    async def _get_user_for_login(self, email: str) -> User | None:

        statement = (
            select(User)
            .where(User.email == email)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def login(
        self,
        request: LoginRequest,
        *,
        client_ip: str | None = None,
        user_agent: str | None = None,
        device_label: str | None = None,
        now: datetime | None = None,
    ) -> LoginResponse:

        if not isinstance(request, LoginRequest):
            raise TypeError("request must be a LoginRequest instance.")

        current_time = self._utc_now(now)
        email = str(request.email).strip().lower()
        password = request.password.get_secret_value()

        try:
            user = await self._get_user_for_login(email)
            if user is None:

                try:
                    hash_password(password)
                except (TypeError, ValueError):
                    pass
                raise InvalidCredentialsException()

            try:
                password_valid, replacement_hash = verify_password_and_update(
                    password,
                    user.password_hash,
                )
            except (TypeError, ValueError):
                password_valid, replacement_hash = False, None

            if not password_valid:
                raise InvalidCredentialsException()

            if not user.is_active:
                raise AccountInactiveException()

            if not user.is_verified:
                raise EmailVerificationRequiredException()

            if replacement_hash is not None:
                await self.user_repository.update(
                    user,
                    password_hash=replacement_hash,
                )

            session_expires_at = current_time + timedelta(
                days=settings.REFRESH_TOKEN_EXPIRE_DAYS
            )
            auth_session: AuthSession = await self.session_repository.create(
                user.id,
                expires_at=session_expires_at,
                client_ip=client_ip,
                user_agent=user_agent,
                device_label=device_label,
                last_used_at=current_time,
            )

            token_pair = await self.token_service.issue_token_pair(
                user_id=user.id,
                session_id=auth_session.id,
                now=current_time,
            )

            await self.session.commit()

            return LoginResponse(
                user=UserResponse.model_validate(user),
                tokens=token_pair,
            )

        except BaseException:
            try:
                await self.session.rollback()
            except SQLAlchemyError:
                logger.exception("Database rollback failed during login.")
            raise
