from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import SessionRevocationReason
from app.modules.auth.models.session import AuthSession
from app.modules.users.models.user import User


class SessionRepository:

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _validate_pagination(*, limit: int, offset: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Limit must be a positive integer.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("Offset must be a non-negative integer.")

    @staticmethod
    def _validate_aware_datetime(value: datetime, *, field_name: str) -> None:
        if not isinstance(value, datetime):
            raise TypeError(f"{field_name} must be a datetime instance.")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field_name} must be timezone-aware.")

    async def _lock_user(self, user_id: uuid.UUID) -> bool:
        """Serialize session creation and account-wide session revocation."""

        statement = select(User.id).where(User.id == user_id).with_for_update()
        result = await self.session.execute(statement)
        return result.scalar_one_or_none() is not None

    @staticmethod
    def _validate_revocation_reason(reason: SessionRevocationReason) -> None:
        if not isinstance(reason, SessionRevocationReason):
            raise TypeError("reason must be a SessionRevocationReason member.")

    async def create(
        self,
        user_id: uuid.UUID,
        *,
        expires_at: datetime,
        client_ip: str | None = None,
        user_agent: str | None = None,
        device_label: str | None = None,
        last_used_at: datetime | None = None,
    ) -> AuthSession:

        self._validate_aware_datetime(expires_at, field_name="expires_at")
        if last_used_at is not None:
            self._validate_aware_datetime(last_used_at, field_name="last_used_at")

        if client_ip is not None and len(client_ip) > 45:
            raise ValueError("client_ip cannot exceed 45 characters.")
        if user_agent is not None and len(user_agent) > 1024:
            raise ValueError("user_agent cannot exceed 1024 characters.")
        if device_label is not None and len(device_label) > 120:
            raise ValueError("device_label cannot exceed 120 characters.")


        if not await self._lock_user(user_id):
            raise ValueError("user_id does not identify an existing user.")

        auth_session = AuthSession(
            user_id=user_id,
            expires_at=expires_at,
            client_ip=client_ip,
            user_agent=user_agent,
            device_label=device_label,
        )
        if last_used_at is not None:
            auth_session.last_used_at = last_used_at

        self.session.add(auth_session)
        await self.session.flush()
        return auth_session

    async def get_by_id(self, session_id: uuid.UUID) -> AuthSession | None:

        statement = select(AuthSession).where(AuthSession.id == session_id)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_by_id_for_user(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> AuthSession | None:

        statement = select(AuthSession).where(
            AuthSession.id == session_id,
            AuthSession.user_id == user_id,
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_by_id_for_user_for_update(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> AuthSession | None:

        statement = (
            select(AuthSession)
            .where(
                AuthSession.id == session_id,
                AuthSession.user_id == user_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_active_by_id_for_user(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        now: datetime,
        for_update: bool = False,
    ) -> AuthSession | None:

        self._validate_aware_datetime(now, field_name="now")
        statement = select(AuthSession).where(
            AuthSession.id == session_id,
            AuthSession.user_id == user_id,
            AuthSession.revoked_at.is_(None),
            AuthSession.expires_at > now,
        )
        if for_update:
            statement = statement.with_for_update().execution_options(
                populate_existing=True
            )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: uuid.UUID,
        *,
        limit: int,
        offset: int,
    ) -> list[AuthSession]:

        self._validate_pagination(limit=limit, offset=offset)
        statement = (
            select(AuthSession)
            .where(AuthSession.user_id == user_id)
            .order_by(AuthSession.created_at.desc(), AuthSession.id.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def count_by_user(self, user_id: uuid.UUID) -> int:

        statement = (
            select(func.count())
            .select_from(AuthSession)
            .where(AuthSession.user_id == user_id)
        )
        result = await self.session.execute(statement)
        return int(result.scalar_one())

    async def list_active_by_user(
        self,
        user_id: uuid.UUID,
        *,
        now: datetime,
        limit: int,
        offset: int,
    ) -> list[AuthSession]:

        self._validate_aware_datetime(now, field_name="now")
        self._validate_pagination(limit=limit, offset=offset)
        statement = (
            select(AuthSession)
            .where(
                AuthSession.user_id == user_id,
                AuthSession.revoked_at.is_(None),
                AuthSession.expires_at > now,
            )
            .order_by(AuthSession.created_at.desc(), AuthSession.id.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def count_active_by_user(
        self,
        user_id: uuid.UUID,
        *,
        now: datetime,
    ) -> int:

        self._validate_aware_datetime(now, field_name="now")
        statement = (
            select(func.count())
            .select_from(AuthSession)
            .where(
                AuthSession.user_id == user_id,
                AuthSession.revoked_at.is_(None),
                AuthSession.expires_at > now,
            )
        )
        result = await self.session.execute(statement)
        return int(result.scalar_one())

    async def touch(
        self,
        auth_session: AuthSession,
        *,
        last_used_at: datetime,
    ) -> bool:
        """Advance ``last_used_at`` without moving it backwards.

        The caller should obtain the row with ``FOR UPDATE`` when this update
        participates in refresh/session state transitions.
        """

        self._validate_aware_datetime(last_used_at, field_name="last_used_at")
        if (
            auth_session.revoked_at is not None
            or auth_session.expires_at <= last_used_at
        ):
            return False

        current_value = auth_session.last_used_at
        if current_value is not None:
            self._validate_aware_datetime(
                current_value,
                field_name="session.last_used_at",
            )
            if current_value >= last_used_at:
                return False

        auth_session.last_used_at = last_used_at
        await self.session.flush()
        return True

    async def revoke(
        self,
        auth_session: AuthSession,
        *,
        reason: SessionRevocationReason,
        revoked_at: datetime,
    ) -> bool:

        self._validate_revocation_reason(reason)
        self._validate_aware_datetime(revoked_at, field_name="revoked_at")

        if auth_session.revoked_at is not None:
            return False

        auth_session.revoked_at = revoked_at
        auth_session.revocation_reason = reason
        await self.session.flush()
        return True

    async def revoke_by_id_for_user(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        reason: SessionRevocationReason,
        revoked_at: datetime,
    ) -> AuthSession | None:

        auth_session = await self.get_by_id_for_user_for_update(
            session_id=session_id,
            user_id=user_id,
        )
        if auth_session is None:
            return None

        await self.revoke(
            auth_session,
            reason=reason,
            revoked_at=revoked_at,
        )
        return auth_session

    async def revoke_all_for_user(
        self,
        user_id: uuid.UUID,
        *,
        reason: SessionRevocationReason,
        revoked_at: datetime,
        exclude_session_id: uuid.UUID | None = None,
    ) -> int:
        """Lock and revoke every currently unrevoked session for a user.

        Expired sessions are included so security events leave a complete
        revocation record. The optional exclusion is intended only for an
        explicitly designed "sign out other sessions" workflow.
        """

        self._validate_revocation_reason(reason)
        self._validate_aware_datetime(revoked_at, field_name="revoked_at")
        if not await self._lock_user(user_id):
            return 0

        conditions = [
            AuthSession.user_id == user_id,
            AuthSession.revoked_at.is_(None),
        ]
        if exclude_session_id is not None:
            conditions.append(AuthSession.id != exclude_session_id)

        statement = (
            select(AuthSession)
            .where(*conditions)
            .order_by(AuthSession.id.asc())
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        sessions = list(result.scalars().all())

        for auth_session in sessions:
            auth_session.revoked_at = revoked_at
            auth_session.revocation_reason = reason

        if sessions:
            await self.session.flush()

        return len(sessions)
