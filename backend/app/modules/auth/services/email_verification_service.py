"""Account email verification and verified email-address change workflows.

Raw tokens are generated for delivery only. SHA-256 digests are persisted,
and database transactions are committed before awaiting the email provider.
Registration tokens and email-change tokens are deliberately separated by
``pending_email`` so one workflow can never complete the other.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import generate_secure_token, hash_token, verify_password
from app.exceptions.authentication import (
    AccountInactiveException,
    EmailAlreadyVerifiedException,
    ExpiredEmailVerificationTokenException,
    InvalidEmailVerificationTokenException,
    InvalidPasswordException,
    UsedEmailVerificationTokenException,
)
from app.exceptions.base import EmailAlreadyRegisteredException
from app.modules.auth.models.email_verification_token import EmailVerificationToken
from app.modules.auth.repositories.token_repository import TokenRepository
from app.modules.auth.schemas.verification import (
    EmailChangeRequest,
    EmailChangeRequestResponse,
    EmailChangeVerificationRequest,
    EmailVerificationRequest,
    EmailVerificationResponse,
    ResendVerificationRequest,
    ResendVerificationResponse,
)
from app.modules.users.models.user import User
from app.modules.users.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)


class VerificationEmailSender(Protocol):
    async def send_verification_email(self, *, recipient: str, token: str) -> None: ...
    async def send_email_change_verification_email(self, *, recipient: str, token: str) -> None: ...


class EmailVerificationService:
    """Issue, deliver and consume registration and email-change tokens."""

    def __init__(self, session: AsyncSession, *, email_sender: VerificationEmailSender) -> None:
        if email_sender is None:
            raise ValueError("email_sender is required.")
        self.session = session
        self.token_repository = TokenRepository(session)
        self.user_repository = UserRepository(session)
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

    async def _rollback_safely(self, operation: str) -> None:
        try:
            await self.session.rollback()
        except SQLAlchemyError:
            logger.exception("Database rollback failed during %s.", operation)

    async def _lock_user_by_id(self, user_id: uuid.UUID) -> User | None:
        result = await self.session.execute(
            select(User).where(User.id == user_id).with_for_update()
            .execution_options(populate_existing=True)
        )
        return result.scalar_one_or_none()

    async def _deliver_and_mark_sent(
        self, token_record: EmailVerificationToken, *, recipient: str,
        raw_token: str, email_change: bool,
    ) -> bool:
        try:
            if email_change:
                await self.email_sender.send_email_change_verification_email(
                    recipient=recipient, token=raw_token
                )
            else:
                await self.email_sender.send_verification_email(
                    recipient=recipient, token=raw_token
                )
        except Exception:
            logger.exception("Verification email delivery failed.")
            return False

        try:
            marked = await self.token_repository.mark_email_verification_email_sent(
                token_record, sent_at=self._utc_now()
            )
            if marked:
                await self.session.commit()
            else:
                await self.session.rollback()
                logger.warning("Email was delivered but sent_at was not updated because the token is no longer active.")
        except Exception:
            await self._rollback_safely("verification email sent timestamp update")
            logger.exception("Email was delivered but sent_at could not be persisted.")
        return True

    async def _issue_registration_token(self, user: User, *, now: datetime) -> bool:
        if not user.is_active or user.is_verified:
            await self.session.rollback()
            return False
        raw_token = generate_secure_token()
        try:
            record = await self.token_repository.create_email_verification_token(
                user.id, token_hash=hash_token(raw_token),
                expires_at=now + timedelta(minutes=settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_MINUTES),
                created_at=now,
            )
            if record is None:
                await self.session.rollback()
                return False
            await self.session.commit()
        except BaseException:
            await self._rollback_safely("registration verification token creation")
            raise
        return await self._deliver_and_mark_sent(
            record, recipient=user.email, raw_token=raw_token, email_change=False
        )

    async def start_for_user(self, user: User, *, now: datetime | None = None) -> bool:
        """AuthService-compatible registration-verification entry point."""
        if not isinstance(user, User):
            raise TypeError("user must be a User instance.")
        current_time = self._utc_now(now)
        try:
            locked_user = await self._lock_user_by_id(user.id)
            if locked_user is None:
                await self.session.rollback()
                return False
            return await self._issue_registration_token(locked_user, now=current_time)
        except BaseException:
            await self._rollback_safely("starting registration verification")
            raise

    async def resend_verification_email(
        self, request: ResendVerificationRequest, *, now: datetime | None = None,
    ) -> ResendVerificationResponse:
        if not isinstance(request, ResendVerificationRequest):
            raise TypeError("request must be a ResendVerificationRequest instance.")
        normalized_email = self._normalize_email(str(request.email))
        current_time = self._utc_now(now)
        try:
            result = await self.session.execute(
                select(User).where(User.email == normalized_email).with_for_update()
                .execution_options(populate_existing=True)
            )
            user = result.scalar_one_or_none()
            if user is None or not user.is_active or user.is_verified:
                await self.session.rollback()
                return ResendVerificationResponse()
            await self._issue_registration_token(user, now=current_time)
            # Keep response indistinguishable for unknown, inactive and verified accounts.
            return ResendVerificationResponse()
        except BaseException:
            await self._rollback_safely("resending registration verification")
            raise

    async def verify_email(
        self, request: EmailVerificationRequest, *, now: datetime | None = None,
    ) -> EmailVerificationResponse:
        """Consume registration-only token and mark the account verified."""
        if not isinstance(request, EmailVerificationRequest):
            raise TypeError("request must be an EmailVerificationRequest instance.")
        current_time = self._utc_now(now)
        token_digest = hash_token(request.token.get_secret_value())
        try:
            record = await self.token_repository.get_registration_verification_token_by_hash_for_update(token_digest)

            if record is None:
                raise InvalidEmailVerificationTokenException()

            self._validate_token_state(record, current_time)
            user = await self.user_repository.get_by_id(record.user_id)

            if user is None:
                raise InvalidEmailVerificationTokenException()
            if not user.is_active:
                raise AccountInactiveException()
            if user.is_verified:
                if not await self.token_repository.mark_email_verification_token_used(record, used_at=current_time):
                    raise InvalidEmailVerificationTokenException()
                await self.session.commit()
                raise EmailAlreadyVerifiedException()
            user.is_verified = True
            if not await self.token_repository.mark_email_verification_token_used(record, used_at=current_time):
                raise InvalidEmailVerificationTokenException()
            await self.session.flush()
            await self.session.commit()
            return EmailVerificationResponse()
        except EmailAlreadyVerifiedException:
            raise
        except BaseException:
            await self._rollback_safely("registration email verification")
            raise

    @staticmethod
    def _validate_token_state(record: EmailVerificationToken, now: datetime) -> None:
        if record.used_at is not None:
            raise UsedEmailVerificationTokenException()
        if record.revoked_at is not None:
            raise InvalidEmailVerificationTokenException()
        if record.expires_at <= now:
            raise ExpiredEmailVerificationTokenException()

    async def request_email_change(
        self, user_id: uuid.UUID, request: EmailChangeRequest, *, now: datetime | None = None,
    ) -> EmailChangeRequestResponse:
        """Authenticate the request, bind the target address to a token, then deliver."""
        if not isinstance(user_id, uuid.UUID):
            raise TypeError("user_id must be a UUID instance.")
        if not isinstance(request, EmailChangeRequest):
            raise TypeError("request must be an EmailChangeRequest instance.")
        current_time = self._utc_now(now)
        pending_email = self._normalize_email(str(request.new_email))
        current_password = request.current_password.get_secret_value()
        try:
            user = await self._lock_user_by_id(user_id)
            if user is None:
                from app.exceptions.resources import UserNotFoundException
                raise UserNotFoundException()
            if not user.is_active:
                raise AccountInactiveException()
            if not user.is_verified:
                from app.exceptions.authentication import EmailVerificationRequiredException
                raise EmailVerificationRequiredException()
            if not verify_password(current_password, user.password_hash):
                raise InvalidPasswordException()
            if pending_email == user.email:
                raise EmailAlreadyRegisteredException()
            existing = await self.user_repository.get_by_email(pending_email)
            if existing is not None and existing.id != user.id:
                raise EmailAlreadyRegisteredException()

            raw_token = generate_secure_token()
            record = await self.token_repository.create_email_verification_token(
                user.id, token_hash=hash_token(raw_token), pending_email=pending_email,
                expires_at=current_time + timedelta(minutes=settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_MINUTES),
                created_at=current_time,
            )
            if record is None:
                await self.session.rollback()
                from app.exceptions.resources import UserNotFoundException
                raise UserNotFoundException()
            await self.session.commit()
        except BaseException:
            await self._rollback_safely("email-change request")
            raise

        delivered = await self._deliver_and_mark_sent(
            record, recipient=pending_email, raw_token=raw_token, email_change=True
        )
        return EmailChangeRequestResponse(accepted=True, delivered=delivered)

    async def verify_email_change(
        self, request: EmailChangeVerificationRequest, *, now: datetime | None = None,
    ) -> EmailVerificationResponse:
        """Atomically replace User.email only after target-address ownership proof."""
        if not isinstance(request, EmailChangeVerificationRequest):
            raise TypeError("request must be an EmailChangeVerificationRequest instance.")
        current_time = self._utc_now(now)
        token_digest = hash_token(request.token.get_secret_value())
        try:
            record = await self.token_repository.get_email_change_token_by_hash_for_update(token_digest)
            if record is None or record.pending_email is None:
                raise InvalidEmailVerificationTokenException()
            self._validate_token_state(record, current_time)
            user = await self.user_repository.get_by_id(record.user_id)
            if user is None:
                raise InvalidEmailVerificationTokenException()
            if not user.is_active:
                raise AccountInactiveException()
            pending_email = self._normalize_email(record.pending_email)
            existing = await self.user_repository.get_by_email(pending_email)
            if existing is not None and existing.id != user.id:
                raise EmailAlreadyRegisteredException()
            user.email = pending_email
            if not await self.token_repository.mark_email_verification_token_used(record, used_at=current_time):
                raise InvalidEmailVerificationTokenException()
            await self.session.flush()
            await self.session.commit()
            return EmailVerificationResponse(message="Email address changed and verified successfully.")
        except IntegrityError as exc:
            await self._rollback_safely("email-change verification uniqueness conflict")
            raise EmailAlreadyRegisteredException() from exc
        except BaseException:
            await self._rollback_safely("email-change verification")
            raise
