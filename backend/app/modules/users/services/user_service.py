"""Application-level user account operations.

This service protects account-security invariants that must remain true even
when operations are called outside HTTP endpoints:

* generic profile updates cannot bypass verified email change;
* password-hash changes invalidate existing authenticated sessions;
* account deactivation revokes sessions and outstanding authentication tokens;
* administrative un-verification invalidates existing sessions;
* all security transitions lock the user row before locking dependent rows.

Repositories flush but never commit. ``commit=True`` makes this service own
its transaction; ``commit=False`` leaves commit/rollback to the caller so the
operation can participate in a larger unit of work.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import SessionRevocationReason
from app.exceptions.authentication import AccountInactiveException
from app.exceptions.base import EmailAlreadyRegisteredException
from app.exceptions.resources import UserNotFoundException
from app.modules.auth.repositories.session_repository import SessionRepository
from app.modules.auth.repositories.token_repository import TokenRepository
from app.modules.users.models.user import User
from app.modules.users.repositories.user_repository import UserRepository

if TYPE_CHECKING:
    from app.modules.users.schemas.user import UserUpdate


logger = logging.getLogger(__name__)

_EMAIL_UNIQUE_CONSTRAINT = "uq_users_email"
_MAX_EMAIL_LENGTH = 320
_MAX_PASSWORD_HASH_LENGTH = 255


class UserService:
    """Coordinate user persistence with authentication security state."""

    def __init__(self, session: AsyncSession) -> None:
        if session is None:
            raise ValueError("session is required.")
        self.session = session
        self.repository = UserRepository(session)
        self.session_repository = SessionRepository(session)
        self.token_repository = TokenRepository(session)

    async def _rollback_if_owned(self, *, commit: bool, operation: str) -> None:
        if not commit:
            return
        try:
            await self.session.rollback()
        except SQLAlchemyError:
            logger.exception("Database rollback failed during %s.", operation)

    @staticmethod
    def _validate_uuid(user_id: uuid.UUID, *, field_name: str = "user_id") -> None:
        if not isinstance(user_id, uuid.UUID):
            raise TypeError(f"{field_name} must be a UUID instance.")

    @staticmethod
    def _normalize_email(email: str) -> str:
        if not isinstance(email, str):
            raise TypeError("email must be a string.")
        normalized_email = email.strip().lower()
        if not normalized_email:
            raise ValueError("email cannot be empty.")
        if len(normalized_email) > _MAX_EMAIL_LENGTH:
            raise ValueError("email cannot exceed 320 characters.")
        return normalized_email

    @staticmethod
    def _validate_password_hash(password_hash: str) -> None:
        if not isinstance(password_hash, str) or not password_hash.strip():
            raise ValueError("password_hash must be a non-empty string.")
        if len(password_hash) > _MAX_PASSWORD_HASH_LENGTH:
            raise ValueError("password_hash cannot exceed 255 characters.")

    @staticmethod
    def _validate_pagination(*, limit: int, offset: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer.")

    @staticmethod
    def _is_email_unique_violation(exc: IntegrityError) -> bool:
        original_error = exc.orig
        candidates = (
            original_error,
            getattr(original_error, "__cause__", None),
            getattr(original_error, "__context__", None),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            diagnostic = getattr(candidate, "diag", None)
            constraint_name = (
                getattr(diagnostic, "constraint_name", None)
                or getattr(candidate, "constraint_name", None)
            )
            if constraint_name == _EMAIL_UNIQUE_CONSTRAINT:
                return True
        error_text = str(original_error).lower()
        return (
            _EMAIL_UNIQUE_CONSTRAINT in error_text
            or "unique constraint failed: users.email" in error_text
        )

    async def _finish_write(self, entity: User, *, commit: bool) -> User:
        if commit:
            await self.session.commit()
            await self.session.refresh(entity)
        return entity

    async def _lock_user(self, user_id: uuid.UUID) -> User:
        user = await self.repository.get_by_id_for_update(user_id)
        if user is None:
            raise UserNotFoundException()
        return user

    async def _revoke_user_credentials(
        self,
        user_id: uuid.UUID,
        *,
        reason: SessionRevocationReason,
        now: datetime,
        revoke_email_tokens: bool = True,
        revoke_password_reset_tokens: bool = True,
    ) -> None:
        """Revoke all persisted credentials after the user row is locked."""
        # ``now`` is validated by each repository operation. The user lock is
        # acquired before dependent session/token locks to keep a stable order.
        await self.session_repository.revoke_all_for_user(
            user_id,
            reason=reason,
            revoked_at=now,
        )
        await self.token_repository.revoke_refresh_tokens_for_user(
            user_id,
            revoked_at=now,
        )
        if revoke_password_reset_tokens:
            await self.token_repository.revoke_password_reset_tokens_for_user(
                user_id,
                revoked_at=now,
            )
        if revoke_email_tokens:
            await self.token_repository.revoke_email_verification_tokens_for_user(
                user_id,
                revoked_at=now,
            )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    async def create_user(
        self,
        email: str,
        password_hash: str,
        *,
        commit: bool = True,
    ) -> User:
        normalized_email = self._normalize_email(email)
        self._validate_password_hash(password_hash)
        try:
            existing_user = await self.repository.get_by_email(normalized_email)
            if existing_user is not None:
                raise EmailAlreadyRegisteredException()
            user = await self.repository.create(normalized_email, password_hash)
            return await self._finish_write(user, commit=commit)
        except IntegrityError as exc:
            await self._rollback_if_owned(commit=commit, operation="user creation")
            if self._is_email_unique_violation(exc):
                raise EmailAlreadyRegisteredException() from exc
            raise
        except BaseException:
            await self._rollback_if_owned(commit=commit, operation="user creation")
            raise

    async def get_user(self, user_id: uuid.UUID) -> User:
        self._validate_uuid(user_id)
        user = await self.repository.get_by_id(user_id)
        if user is None:
            raise UserNotFoundException()
        return user

    async def get_user_by_email(self, email: str) -> User:
        normalized_email = self._normalize_email(email)
        user = await self.repository.get_by_email(normalized_email)
        if user is None:
            raise UserNotFoundException()
        return user

    async def list_users(self, limit: int, offset: int) -> list[User]:
        self._validate_pagination(limit=limit, offset=offset)
        return await self.repository.lists(limit=limit, offset=offset)

    async def update_user(
        self,
        user_id: uuid.UUID,
        user_data: UserUpdate,
        *,
        commit: bool = True,
    ) -> User:
        """Reject generic email updates; verified email change has its own flow."""
        self._validate_uuid(user_id)
        if user_data is None:
            raise TypeError("user_data is required.")
        updates = user_data.model_dump(exclude_unset=True)
        if not updates:
            raise ValueError("At least one user field must be provided.")
        if "email" in updates:
            raise ValueError(
                "Email addresses must be changed through "
                "EmailVerificationService.request_email_change()."
            )
        raise ValueError("UserUpdate contains unsupported fields.")

    async def set_user_password_hash(
        self,
        user_id: uuid.UUID,
        password_hash: str,
        *,
        commit: bool = True,
    ) -> User:
        """Set a trusted pre-hashed password and invalidate all old sessions.

        User-facing password changes and resets must use PasswordService,
        which verifies the current/reset credential and applies password policy.
        This method is retained for trusted administrative/internal workflows.
        """
        self._validate_uuid(user_id)
        self._validate_password_hash(password_hash)
        try:
            user = await self._lock_user(user_id)
            if not user.is_active:
                raise AccountInactiveException()
            await self.repository.update(user, password_hash=password_hash)
            await self._revoke_user_credentials(
                user.id,
                reason=SessionRevocationReason.SECURITY_EVENT,
                now=self._utc_now(),
            )
            return await self._finish_write(user, commit=commit)
        except BaseException:
            await self._rollback_if_owned(commit=commit, operation="password hash update")
            raise

    async def set_user_active_status(
        self,
        user_id: uuid.UUID,
        is_active: bool,
        *,
        commit: bool = True,
    ) -> User:
        self._validate_uuid(user_id)
        if not isinstance(is_active, bool):
            raise TypeError("is_active must be a boolean.")
        try:
            user = await self._lock_user(user_id)
            was_active = user.is_active
            if was_active == is_active:
                return await self._finish_write(user, commit=commit)

            await self.repository.update(user, is_active=is_active)
            if not is_active:
                await self._revoke_user_credentials(
                    user.id,
                    reason=SessionRevocationReason.ACCOUNT_DEACTIVATED,
                    now=self._utc_now(),
                )
            return await self._finish_write(user, commit=commit)
        except BaseException:
            await self._rollback_if_owned(commit=commit, operation="account status update")
            raise

    async def set_user_verified_status(
        self,
        user_id: uuid.UUID,
        is_verified: bool,
        *,
        commit: bool = True,
    ) -> User:
        """Allow trusted un-verification, never bypass proof to verify an email."""
        self._validate_uuid(user_id)
        if not isinstance(is_verified, bool):
            raise TypeError("is_verified must be a boolean.")
        if is_verified:
            raise ValueError(
                "Email verification can only be completed through "
                "EmailVerificationService.verify_email()."
            )
        try:
            user = await self._lock_user(user_id)
            if not user.is_verified:
                return await self._finish_write(user, commit=commit)
            await self.repository.update(user, is_verified=False)
            await self._revoke_user_credentials(
                user.id,
                reason=SessionRevocationReason.SECURITY_EVENT,
                now=self._utc_now(),
            )
            return await self._finish_write(user, commit=commit)
        except BaseException:
            await self._rollback_if_owned(commit=commit, operation="email verification status update")
            raise

    async def delete_user(
        self,
        user_id: uuid.UUID,
        *,
        commit: bool = True,
    ) -> None:
        self._validate_uuid(user_id)
        try:
            user = await self._lock_user(user_id)
            # Database ON DELETE CASCADE relationships remove dependent
            # sessions and token records in the same transaction.
            await self.repository.delete(user)
            if commit:
                await self.session.commit()
        except BaseException:
            await self._rollback_if_owned(commit=commit, operation="user deletion")
            raise


__all__ = ["UserService"]
