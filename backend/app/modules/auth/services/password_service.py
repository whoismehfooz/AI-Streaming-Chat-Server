"""Password change and password-reset workflows.

Password changes and successful resets invalidate every existing session and
refresh credential. Password-reset tokens are opaque, single-use values whose
SHA-256 digests alone are persisted. Email delivery is injected through a
small protocol so this service does not depend on Resend or its SDK.

The email sender used in Round 2 must implement ``send_password_reset_email``.
The service commits the reset-token record before calling it and updates
``sent_at`` in a separate transaction after successful provider delivery.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.enums import SessionRevocationReason
from app.core.security import generate_secure_token, hash_password, hash_token, verify_password
from app.exceptions.authentication import (
    AccountInactiveException,
    ExpiredPasswordResetTokenException,
    InvalidPasswordException,
    InvalidPasswordResetTokenException,
    PasswordPolicyViolationException,
    UsedPasswordResetTokenException,
)
from app.exceptions.resources import UserNotFoundException
from app.modules.auth.models.password_reset_token import PasswordResetToken
from app.modules.auth.repositories.session_repository import SessionRepository
from app.modules.auth.repositories.token_repository import TokenRepository
from app.modules.auth.schemas.password import (
    PasswordChangeRequest,
    PasswordResetConfirmRequest,
    PasswordResetRequest,
    PasswordResetRequestResponse,
)
from app.modules.users.models.user import User
from app.modules.users.repositories.user_repository import UserRepository


logger = logging.getLogger(__name__)


class PasswordResetEmailSender(Protocol):
    """Provider-neutral contract implemented by the Round-2 email adapter."""

    async def send_password_reset_email(
        self,
        *,
        recipient: str,
        token: str,
    ) -> None:
        """Send a reset link containing the one-time raw token."""
        ...


class PasswordService:
    """Change account passwords and manage password-recovery tokens."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        email_sender: PasswordResetEmailSender,
    ) -> None:
        if email_sender is None:
            raise ValueError("email_sender is required.")

        self.session = session
        self.user_repository = UserRepository(session)
        self.token_repository = TokenRepository(session)
        self.session_repository = SessionRepository(session)
        self.email_sender = email_sender

    @staticmethod
    def _utc_now(value: datetime | None = None) -> datetime:
        now = datetime.now(timezone.utc) if value is None else value
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime instance.")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware.")
        return now.astimezone(timezone.utc)

    @staticmethod
    def _normalize_email(email: str) -> str:
        if not isinstance(email, str):
            raise TypeError("email must be a string.")
        normalized = email.strip().lower()
        if not normalized:
            raise ValueError("email cannot be empty.")
        if len(normalized) > 320:
            raise ValueError("email cannot exceed 320 characters.")
        return normalized

    async def _get_user_for_update(self, user_id: uuid.UUID) -> User:
        statement = (
            select(User)
            .where(User.id == user_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        user = result.scalar_one_or_none()
        if user is None:
            raise UserNotFoundException()
        return user

    async def _revoke_user_credentials(
        self,
        user_id: uuid.UUID,
        *,
        reason: SessionRevocationReason,
        now: datetime,
    ) -> None:
        # The user row is locked by the caller. Repository lock order remains
        # user -> session rows -> token rows for password-security operations.
        await self.session_repository.revoke_all_for_user(
            user_id,
            reason=reason,
            revoked_at=now,
        )
        await self.token_repository.revoke_refresh_tokens_for_user(
            user_id,
            revoked_at=now,
        )
        await self.token_repository.revoke_password_reset_tokens_for_user(
            user_id,
            revoked_at=now,
        )

    async def change_password(
        self,
        user_id: uuid.UUID,
        request: PasswordChangeRequest,
        *,
        now: datetime | None = None,
    ) -> None:
        """Verify the current password, replace it and revoke all sessions."""

        if not isinstance(user_id, uuid.UUID):
            raise TypeError("user_id must be a UUID instance.")
        if not isinstance(request, PasswordChangeRequest):
            raise TypeError("request must be a PasswordChangeRequest instance.")

        current_time = self._utc_now(now)
        try:
            user = await self._get_user_for_update(user_id)
            if not user.is_active:
                raise AccountInactiveException()

            current_password = request.current_password.get_secret_value()
            new_password = request.new_password.get_secret_value()

            if not verify_password(current_password, user.password_hash):
                raise InvalidPasswordException()
            if verify_password(new_password, user.password_hash):
                raise PasswordPolicyViolationException(
                    "The new password must differ from the current account password."
                )

            try:
                new_password_hash = hash_password(new_password)
            except (TypeError, ValueError) as exc:
                raise PasswordPolicyViolationException() from exc

            await self.user_repository.update(
                user,
                password_hash=new_password_hash,
            )
            await self._revoke_user_credentials(
                user.id,
                reason=SessionRevocationReason.PASSWORD_CHANGE,
                now=current_time,
            )
            await self.session.commit()

        except BaseException:
            await self._rollback_safely("password change")
            raise

    async def request_password_reset(
        self,
        request: PasswordResetRequest,
        *,
        now: datetime | None = None,
    ) -> PasswordResetRequestResponse:
        """Start a non-enumerating password-reset workflow.

        The generic response is returned for unknown and inactive accounts.
        For a valid account, the token hash is committed before the provider
        call. Provider failures are logged without deleting the persisted
        workflow; a subsequent request can issue a replacement token.
        """

        if not isinstance(request, PasswordResetRequest):
            raise TypeError("request must be a PasswordResetRequest instance.")

        normalized_email = self._normalize_email(str(request.email))
        current_time = self._utc_now(now)

        try:
            user_statement = (
                select(User)
                .where(User.email == normalized_email)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            user_result = await self.session.execute(user_statement)
            user = user_result.scalar_one_or_none()
            if user is None or not user.is_active:
                await self.session.rollback()
                return PasswordResetRequestResponse()

            raw_token = generate_secure_token()
            expires_at = current_time + timedelta(
                minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES
            )
            reset_token: PasswordResetToken | None = (
                await self.token_repository.create_password_reset_token(
                    user.id,
                    token_hash=hash_token(raw_token),
                    expires_at=expires_at,
                    created_at=current_time,
                )
            )
            if reset_token is None:
                await self.session.rollback()
                return PasswordResetRequestResponse()

            # Commit before the network call. Never hold a DB transaction open
            # while waiting for the email provider.
            await self.session.commit()

        except BaseException:
            await self._rollback_safely("password-reset token creation")
            raise

        try:
            await self.email_sender.send_password_reset_email(
                recipient=user.email,
                token=raw_token,
            )
        except Exception:
            # Do not log the raw token or provider payload. The committed token
            # remains valid and can be replaced by a later reset request.
            logger.exception("Password-reset email delivery failed.")
            return PasswordResetRequestResponse()

        try:
            await self.token_repository.mark_password_reset_email_sent(
                reset_token,
                sent_at=self._utc_now(),
            )
            await self.session.commit()
        except Exception:
            await self._rollback_safely("password-reset delivery timestamp update")
            # Delivery already succeeded and the token remains usable. Do not
            # turn a timestamp bookkeeping failure into a false delivery claim.
            logger.exception(
                "Password-reset email was sent, but sent_at could not be persisted."
            )

        return PasswordResetRequestResponse()

    async def confirm_password_reset(
        self,
        request: PasswordResetConfirmRequest,
        *,
        now: datetime | None = None,
    ) -> None:
        """Consume a reset token once, replace the password and revoke sessions."""

        if not isinstance(request, PasswordResetConfirmRequest):
            raise TypeError("request must be a PasswordResetConfirmRequest instance.")

        current_time = self._utc_now(now)
        raw_token = request.token.get_secret_value()
        token_hash_value = hash_token(raw_token)

        try:
            reset_token = await self.token_repository.get_password_reset_token_by_hash_for_update(
                token_hash_value
            )
            if reset_token is None:
                raise InvalidPasswordResetTokenException()
            if reset_token.used_at is not None:
                raise UsedPasswordResetTokenException()
            if reset_token.revoked_at is not None:
                raise InvalidPasswordResetTokenException()
            if reset_token.expires_at <= current_time:
                raise ExpiredPasswordResetTokenException()

            user = await self._get_user_for_update(reset_token.user_id)
            if not user.is_active:
                raise AccountInactiveException()

            new_password = request.new_password.get_secret_value()
            if verify_password(new_password, user.password_hash):
                raise PasswordPolicyViolationException(
                    "The new password must differ from the current account password."
                )
            try:
                new_password_hash = hash_password(new_password)
            except (TypeError, ValueError) as exc:
                raise PasswordPolicyViolationException() from exc

            consumed = await self.token_repository.mark_password_reset_token_used(
                reset_token,
                used_at=current_time,
            )
            if not consumed:
                # The token changed state or expired between lookup and update.
                raise InvalidPasswordResetTokenException()

            await self.user_repository.update(
                user,
                password_hash=new_password_hash,
            )
            await self._revoke_user_credentials(
                user.id,
                reason=SessionRevocationReason.PASSWORD_RESET,
                now=current_time,
            )
            await self.session.commit()

        except BaseException:
            await self._rollback_safely("password reset confirmation")
            raise

    async def _rollback_safely(self, operation: str) -> None:
        try:
            await self.session.rollback()
        except SQLAlchemyError:
            logger.exception("Database rollback failed during %s.", operation)
