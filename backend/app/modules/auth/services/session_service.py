"""Application workflows for authenticated-session management.

Session ownership is enforced in repository queries. Every operation that
changes a session and its refresh credentials is committed as one database
transaction. Repositories only flush; this service owns the transaction.

This service deliberately does not implement rate limiting. The supplied
Auth models contain no rate-limit persistence model, so a PostgreSQL-backed
limiter must be added as a separate persistence concern before the Auth
rate-limiting requirement can be considered complete.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import SessionRevocationReason
from app.exceptions.resources import SessionNotFoundException, UserNotFoundException
from app.modules.auth.repositories.session_repository import SessionRepository
from app.modules.auth.repositories.token_repository import TokenRepository
from app.modules.auth.schemas.session import (
    GlobalSessionRevocationResponse,
    SessionListResponse,
    SessionResponse,
    SessionRevocationResponse,
)
from app.modules.users.repositories.user_repository import UserRepository


logger = logging.getLogger(__name__)


class SessionService:
    """Manage the authenticated user's own sessions and refresh credentials."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.session_repository = SessionRepository(session)
        self.token_repository = TokenRepository(session)
        self.user_repository = UserRepository(session)

    @staticmethod
    def _utc_now(value: datetime | None = None) -> datetime:
        now = datetime.now(timezone.utc) if value is None else value
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime instance.")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware.")
        return now.astimezone(timezone.utc)

    @staticmethod
    def _validate_uuid(value: uuid.UUID, *, field_name: str) -> None:
        if not isinstance(value, uuid.UUID):
            raise TypeError(f"{field_name} must be a UUID instance.")

    async def _rollback_safely(self, operation: str) -> None:
        try:
            await self.session.rollback()
        except SQLAlchemyError:
            logger.exception("Database rollback failed during %s.", operation)

    async def list_sessions(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 20,
        offset: int = 0,
        current_session_id: uuid.UUID | None = None,
    ) -> SessionListResponse:
        """Return a paginated session history restricted to ``user_id``."""

        self._validate_uuid(user_id, field_name="user_id")
        if current_session_id is not None:
            self._validate_uuid(
                current_session_id,
                field_name="current_session_id",
            )

        # Repository validates limit and offset, keeping the same pagination
        # contract across all persistence operations.
        sessions = await self.session_repository.list_by_user(
            user_id,
            limit=limit,
            offset=offset,
        )
        total_count = await self.session_repository.count_by_user(user_id)

        items = [
            SessionResponse.model_validate(auth_session).model_copy(
                update={
                    "is_current": auth_session.id == current_session_id,
                }
            )
            for auth_session in sessions
        ]
        return SessionListResponse(
            items=items,
            total_count=total_count,
            limit=limit,
            offset=offset,
        )

    async def revoke_session(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        *,
        now: datetime | None = None,
    ) -> SessionRevocationResponse:
        """Revoke one session belonging to the authenticated user.

        A session belonging to another user is indistinguishable from a
        missing session. Its outstanding refresh credentials are revoked in
        the same transaction as the session.
        """

        self._validate_uuid(user_id, field_name="user_id")
        self._validate_uuid(session_id, field_name="session_id")
        current_time = self._utc_now(now)

        try:
            auth_session = await self.session_repository.get_by_id_for_user_for_update(
                session_id=session_id,
                user_id=user_id,
            )
            if auth_session is None:
                raise SessionNotFoundException()

            was_revoked = auth_session.revoked_at is not None
            if not was_revoked:
                await self.session_repository.revoke(
                    auth_session,
                    reason=SessionRevocationReason.LOGOUT,
                    revoked_at=current_time,
                )

            await self.token_repository.revoke_refresh_tokens_for_session(
                auth_session.id,
                revoked_at=current_time,
            )

            # Validate and materialize the response before commit. This avoids
            # depending on ORM attribute access after commit, even if a caller
            # supplies a sessionmaker with expire_on_commit=True.
            if auth_session.revocation_reason is None or auth_session.revoked_at is None:
                raise RuntimeError(
                    "Session has an inconsistent revocation state."
                )
            response = SessionRevocationResponse(
                session_id=auth_session.id,
                revoked=not was_revoked,
                revoked_at=auth_session.revoked_at,
                reason=auth_session.revocation_reason,
            )
            await self.session.commit()
            return response

        except BaseException:
            await self._rollback_safely("single-session revocation")
            raise

    async def logout(
        self,
        user_id: uuid.UUID,
        current_session_id: uuid.UUID,
        *,
        now: datetime | None = None,
    ) -> SessionRevocationResponse:
        """Log out the current device by revoking its session and refresh tokens."""

        return await self.revoke_session(
            user_id=user_id,
            session_id=current_session_id,
            now=now,
        )

    async def revoke_all_sessions(
        self,
        user_id: uuid.UUID,
        *,
        now: datetime | None = None,
    ) -> GlobalSessionRevocationResponse:
        """Revoke every session and refresh credential for one user."""

        self._validate_uuid(user_id, field_name="user_id")
        current_time = self._utc_now(now)

        try:
            user = await self.user_repository.get_by_id(user_id)
            if user is None:
                raise UserNotFoundException()

            revoked_sessions = await self.session_repository.revoke_all_for_user(
                user_id,
                reason=SessionRevocationReason.LOGOUT,
                revoked_at=current_time,
            )
            # Includes refresh tokens attached to already-revoked or expired
            # sessions, ensuring no outstanding refresh credential survives.
            await self.token_repository.revoke_refresh_tokens_for_user(
                user_id,
                revoked_at=current_time,
            )
            await self.session.commit()

            return GlobalSessionRevocationResponse(
                revoked_count=revoked_sessions,
            )

        except BaseException:
            await self._rollback_safely("global session revocation")
            raise
