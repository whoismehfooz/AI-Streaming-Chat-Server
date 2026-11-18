"""Persistence operations for user accounts.

The repository owns SQL construction and flushing only. It never commits or
rolls back; transaction boundaries belong to application services. Methods
that participate in account-security transitions expose explicit row-locking
variants so callers can serialize account state changes with session/token
revocation.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.users.models.user import User


class UserRepository:
    """Database access for the User aggregate."""

    def __init__(self, session: AsyncSession) -> None:
        if session is None:
            raise ValueError("session is required.")
        self.session = session

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
        if len(normalized_email) > 320:
            raise ValueError("email cannot exceed 320 characters.")
        return normalized_email

    @staticmethod
    def _validate_pagination(*, limit: int, offset: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer.")

    async def create(self, email: str, password_hash: str) -> User:
        normalized_email = self._normalize_email(email)
        if not isinstance(password_hash, str) or not password_hash.strip():
            raise ValueError("password_hash must be a non-empty string.")
        if len(password_hash) > 255:
            raise ValueError("password_hash cannot exceed 255 characters.")

        user = User(email=normalized_email, password_hash=password_hash)
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        self._validate_uuid(user_id)
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_id_for_update(self, user_id: uuid.UUID) -> User | None:
        """Return and lock one user row for a security-sensitive transition."""
        self._validate_uuid(user_id)
        statement = (
            select(User)
            .where(User.id == user_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        normalized_email = self._normalize_email(email)
        result = await self.session.execute(
            select(User).where(User.email == normalized_email)
        )
        return result.scalar_one_or_none()

    async def get_by_email_for_update(self, email: str) -> User | None:
        """Return and lock an account selected by its normalized email."""
        normalized_email = self._normalize_email(email)
        statement = (
            select(User)
            .where(User.email == normalized_email)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def lists(self, limit: int, offset: int) -> list[User]:
        self._validate_pagination(limit=limit, offset=offset)
        statement = (
            select(User)
            .order_by(User.created_at.desc(), User.id.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def update(
        self,
        user: User,
        *,
        email: str | None = None,
        password_hash: str | None = None,
        is_active: bool | None = None,
        is_verified: bool | None = None,
    ) -> User:
        """Apply explicitly supplied persistence fields and flush changes.

        This is a trusted internal persistence method, not an authorization
        boundary. Public email changes must go through EmailVerificationService.
        """
        if not isinstance(user, User):
            raise TypeError("user must be a User instance.")
        if email is not None:
            user.email = self._normalize_email(email)
        if password_hash is not None:
            if not isinstance(password_hash, str) or not password_hash.strip():
                raise ValueError("password_hash must be a non-empty string.")
            if len(password_hash) > 255:
                raise ValueError("password_hash cannot exceed 255 characters.")
            user.password_hash = password_hash
        if is_active is not None:
            if not isinstance(is_active, bool):
                raise TypeError("is_active must be a boolean.")
            user.is_active = is_active
        if is_verified is not None:
            if not isinstance(is_verified, bool):
                raise TypeError("is_verified must be a boolean.")
            user.is_verified = is_verified

        await self.session.flush()
        return user

    async def delete(self, user: User) -> None:
        if not isinstance(user, User):
            raise TypeError("user must be a User instance.")
        await self.session.delete(user)
        await self.session.flush()


__all__ = ["UserRepository"]
