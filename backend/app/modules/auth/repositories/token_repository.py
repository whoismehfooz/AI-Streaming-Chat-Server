from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.models.email_verification_token import EmailVerificationToken
from app.modules.auth.models.password_reset_token import PasswordResetToken
from app.modules.auth.models.refresh_token import RefreshToken
from app.modules.auth.models.session import AuthSession
from app.modules.users.models.user import User


_SHA256_HEX_LENGTH = 64


class TokenRepository:
    """Database access for refresh, email-verification and password-reset tokens."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _validate_aware_datetime(value: datetime, *, field_name: str) -> None:
        if not isinstance(value, datetime):
            raise TypeError(f"{field_name} must be a datetime instance.")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field_name} must be timezone-aware.")

    @staticmethod
    def _normalize_token_hash(token_hash: str) -> str:
        if not isinstance(token_hash, str):
            raise TypeError("token_hash must be a string.")
        if token_hash != token_hash.strip():
            raise ValueError("token_hash must not contain surrounding whitespace.")
        normalized = token_hash.lower()
        if len(normalized) != _SHA256_HEX_LENGTH:
            raise ValueError("token_hash must be a 64-character SHA-256 hex digest.")
        try:
            int(normalized, 16)
        except ValueError as exc:
            raise ValueError(
                "token_hash must contain only hexadecimal characters."
            ) from exc
        return normalized

    @staticmethod
    def _normalize_pending_email(pending_email: str | None) -> str | None:
        """Normalize an optional email-change target before persistence."""

        if pending_email is None:
            return None
        if not isinstance(pending_email, str):
            raise TypeError("pending_email must be a string or None.")

        normalized = pending_email.strip().lower()
        if not normalized:
            raise ValueError("pending_email cannot be empty.")
        if len(normalized) > 320:
            raise ValueError("pending_email cannot exceed 320 characters.")
        return normalized

    async def _lock_session(self, session_id: uuid.UUID) -> bool:
        statement = (
            select(AuthSession.id)
            .where(AuthSession.id == session_id)
            .with_for_update()
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none() is not None

    async def _lock_user(self, user_id: uuid.UUID) -> bool:
        statement = (
            select(User.id)
            .where(User.id == user_id)
            .with_for_update()
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none() is not None

    # ------------------------------------------------------------------
    # Refresh tokens
    # ------------------------------------------------------------------

    async def create_refresh_token(
        self,
        session_id: uuid.UUID,
        *,
        family_id: uuid.UUID,
        token_hash: str,
        expires_at: datetime,
        parent_token_id: uuid.UUID | None = None,
    ) -> RefreshToken:

        normalized_hash = self._normalize_token_hash(token_hash)
        self._validate_aware_datetime(expires_at, field_name="expires_at")

        # Serialize all token creation/rotation operations for this session.
        # The session is the aggregate lock shared by refresh and revocation
        # workflows, so acquire it before touching any refresh-token row.
        if not await self._lock_session(session_id):
            raise ValueError("session_id does not identify an existing session.")

        family_locator_statement = (
            select(RefreshToken.session_id)
            .where(RefreshToken.family_id == family_id)
            .distinct()
        )
        family_locator_result = await self.session.execute(family_locator_statement)
        family_session_ids = set(family_locator_result.scalars().all())
        if family_session_ids and family_session_ids != {session_id}:
            raise ValueError(
                "A refresh-token family must belong to exactly one session."
            )
        if parent_token_id is None and family_session_ids:
            raise ValueError("A refresh-token family can have only one root token.")
        if parent_token_id is not None and not family_session_ids:
            raise ValueError(
                "A rotated refresh token must reference an existing family."
            )

        if parent_token_id is not None:
            parent_statement = (
                select(RefreshToken)
                .where(RefreshToken.id == parent_token_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            parent_result = await self.session.execute(parent_statement)
            parent = parent_result.scalar_one_or_none()
            if parent is None:
                raise ValueError(
                    "parent_token_id does not identify an existing refresh token."
                )
            if parent.session_id != session_id or parent.family_id != family_id:
                raise ValueError(
                    "A refresh-token parent must belong to the same session and family."
                )
            if parent.used_at is None or parent.revoked_at is not None:
                raise ValueError(
                    "A rotated refresh-token parent must be consumed and not revoked."
                )

        refresh_token = RefreshToken(
            session_id=session_id,
            family_id=family_id,
            parent_token_id=parent_token_id,
            token_hash=normalized_hash,
            expires_at=expires_at,
        )
        self.session.add(refresh_token)
        await self.session.flush()
        return refresh_token

    async def get_refresh_token_by_hash(self, token_hash: str) -> RefreshToken | None:
        """Find a refresh token by its stored digest, without locking it."""

        normalized_hash = self._normalize_token_hash(token_hash)
        statement = select(RefreshToken).where(
            RefreshToken.token_hash == normalized_hash
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_refresh_token_by_hash_for_update(
        self,
        token_hash: str,
    ) -> RefreshToken | None:
        """Lock the owning session, then lock and return the matching token.

        Locking the session first serializes all refresh-token transitions in
        that session. A replay waiting behind a successful rotation therefore
        observes the committed ``used_at`` state before deciding on reuse
        detection and family revocation.
        """

        normalized_hash = self._normalize_token_hash(token_hash)
        locator_statement = select(
            RefreshToken.id,
            RefreshToken.session_id,
        ).where(RefreshToken.token_hash == normalized_hash)
        locator_result = await self.session.execute(locator_statement)
        locator = locator_result.one_or_none()
        if locator is None:
            return None

        if not await self._lock_session(locator.session_id):
            return None

        statement = (
            select(RefreshToken)
            .where(
                RefreshToken.id == locator.id,
                RefreshToken.token_hash == normalized_hash,
                RefreshToken.session_id == locator.session_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_refresh_token_by_id(
        self,
        refresh_token_id: uuid.UUID,
    ) -> RefreshToken | None:
        statement = select(RefreshToken).where(RefreshToken.id == refresh_token_id)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_refresh_token_by_id_for_update(
        self,
        refresh_token_id: uuid.UUID,
    ) -> RefreshToken | None:
        locator_statement = select(RefreshToken.session_id).where(
            RefreshToken.id == refresh_token_id
        )
        locator_result = await self.session.execute(locator_statement)
        session_id = locator_result.scalar_one_or_none()
        if session_id is None or not await self._lock_session(session_id):
            return None

        statement = (
            select(RefreshToken)
            .where(
                RefreshToken.id == refresh_token_id,
                RefreshToken.session_id == session_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def list_refresh_tokens_by_session(
        self,
        session_id: uuid.UUID,
    ) -> list[RefreshToken]:
        statement = (
            select(RefreshToken)
            .where(RefreshToken.session_id == session_id)
            .order_by(RefreshToken.created_at.asc(), RefreshToken.id.asc())
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def list_refresh_tokens_by_family(
        self,
        family_id: uuid.UUID,
        *,
        for_update: bool = False,
    ) -> list[RefreshToken]:
        """List one token family, optionally locking its session and members."""

        if for_update:
            locator_statement = (
                select(RefreshToken.session_id)
                .where(RefreshToken.family_id == family_id)
                .distinct()
                .order_by(RefreshToken.session_id.asc())
            )
            locator_result = await self.session.execute(locator_statement)
            session_ids = list(locator_result.scalars().all())
            if not session_ids:
                return []
            # A family is an invariantly session-scoped chain. Refuse to
            # mutate inconsistent legacy data rather than acquiring several
            # session locks in an order that could conflict with an in-flight
            # refresh transaction.
            if len(session_ids) != 1:
                raise ValueError(
                "A refresh-token family must belong to exactly one session."
            )
            if not await self._lock_session(session_ids[0]):
                return []

        statement = (
            select(RefreshToken)
            .where(RefreshToken.family_id == family_id)
            .order_by(RefreshToken.created_at.asc(), RefreshToken.id.asc())
        )
        if for_update:
            statement = statement.with_for_update().execution_options(
                populate_existing=True
            )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def mark_refresh_token_used(
        self,
        refresh_token: RefreshToken,
        *,
        used_at: datetime,
    ) -> bool:
        """Atomically consume an unexpired, unrevoked refresh token once."""

        self._validate_aware_datetime(used_at, field_name="used_at")
        statement = (
            update(RefreshToken)
            .where(
                RefreshToken.id == refresh_token.id,
                RefreshToken.used_at.is_(None),
                RefreshToken.revoked_at.is_(None),
                RefreshToken.expires_at > used_at,
            )
            .values(used_at=used_at)
            .returning(RefreshToken.id)
            .execution_options(synchronize_session="fetch")
        )
        result = await self.session.execute(statement)
        consumed_id = result.scalar_one_or_none()
        if consumed_id is None:
            return False
        await self.session.flush()
        return True

    async def revoke_refresh_token(
        self,
        refresh_token: RefreshToken,
        *,
        revoked_at: datetime,
    ) -> bool:
        """Revoke a refresh token once, including a previously used token."""

        self._validate_aware_datetime(revoked_at, field_name="revoked_at")
        statement = (
            update(RefreshToken)
            .where(
                RefreshToken.id == refresh_token.id,
                RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=revoked_at)
            .returning(RefreshToken.id)
            .execution_options(synchronize_session="fetch")
        )
        result = await self.session.execute(statement)
        revoked_id = result.scalar_one_or_none()
        if revoked_id is None:
            return False
        await self.session.flush()
        return True

    async def revoke_refresh_token_family(
        self,
        family_id: uuid.UUID,
        *,
        revoked_at: datetime,
    ) -> int:
        """Revoke every still-unrevoked member of a refresh-token family."""

        self._validate_aware_datetime(revoked_at, field_name="revoked_at")
        family_tokens = await self.list_refresh_tokens_by_family(
            family_id,
            for_update=True,
        )
        changed = 0
        for refresh_token in family_tokens:
            if refresh_token.revoked_at is None:
                refresh_token.revoked_at = revoked_at
                changed += 1
        if changed:
            await self.session.flush()
        return changed

    async def revoke_refresh_tokens_for_user(
        self,
        user_id: uuid.UUID,
        *,
        revoked_at: datetime,
        exclude_session_id: uuid.UUID | None = None,
    ) -> int:
        """Revoke every unrevoked refresh token across a user's sessions.

        All owning session rows are locked in deterministic UUID order before
        token rows are locked. This is the bulk counterpart to per-session
        revocation and is intended for password changes, password resets,
        account deactivation and global sign-out workflows.
        """

        self._validate_aware_datetime(revoked_at, field_name="revoked_at")
        if not await self._lock_user(user_id):
            return 0

        session_conditions = [AuthSession.user_id == user_id]
        if exclude_session_id is not None:
            session_conditions.append(AuthSession.id != exclude_session_id)
        session_id_statement = (
            select(AuthSession.id)
            .where(*session_conditions)
            .order_by(AuthSession.id.asc())
        )
        session_id_result = await self.session.execute(session_id_statement)
        session_ids = list(session_id_result.scalars().all())
        if not session_ids:
            return 0

        for session_id in session_ids:
            if not await self._lock_session(session_id):
                # A concurrently deleted session has no remaining refresh
                # tokens because the model's foreign key uses ON DELETE CASCADE.
                continue

        statement = (
            select(RefreshToken)
            .join(AuthSession, AuthSession.id == RefreshToken.session_id)
            .where(
                *session_conditions,
                RefreshToken.revoked_at.is_(None),
            )
            .order_by(RefreshToken.session_id.asc(), RefreshToken.id.asc())
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        refresh_tokens = list(result.scalars().all())
        for refresh_token in refresh_tokens:
            refresh_token.revoked_at = revoked_at

        if refresh_tokens:
            await self.session.flush()
        return len(refresh_tokens)

    async def revoke_refresh_tokens_for_session(
        self,
        session_id: uuid.UUID,
        *,
        revoked_at: datetime,
    ) -> int:
        """Revoke all remaining refresh credentials attached to a session."""

        self._validate_aware_datetime(revoked_at, field_name="revoked_at")
        if not await self._lock_session(session_id):
            return 0

        statement = (
            select(RefreshToken)
            .where(
                RefreshToken.session_id == session_id,
                RefreshToken.revoked_at.is_(None),
            )
            .order_by(RefreshToken.id.asc())
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        refresh_tokens = list(result.scalars().all())
        for refresh_token in refresh_tokens:
            refresh_token.revoked_at = revoked_at

        if refresh_tokens:
            await self.session.flush()
        return len(refresh_tokens)


    async def create_email_verification_token(
        self,
        user_id: uuid.UUID,
        *,
        token_hash: str,
        expires_at: datetime,
        created_at: datetime,
        pending_email: str | None = None,
    ) -> EmailVerificationToken | None:

        normalized_hash = self._normalize_token_hash(token_hash)
        normalized_pending_email = self._normalize_pending_email(pending_email)
        self._validate_aware_datetime(expires_at, field_name="expires_at")
        self._validate_aware_datetime(created_at, field_name="created_at")
        if expires_at <= created_at:
            raise ValueError("expires_at must be later than created_at.")
        if not await self._lock_user(user_id):
            return None

        existing_statement = (
            select(EmailVerificationToken)
            .where(
                EmailVerificationToken.user_id == user_id,
                EmailVerificationToken.used_at.is_(None),
                EmailVerificationToken.revoked_at.is_(None),
            )
            .order_by(EmailVerificationToken.id.asc())
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        existing_result = await self.session.execute(existing_statement)
        existing_tokens = list(existing_result.scalars().all())
        for existing_token in existing_tokens:
            existing_token.revoked_at = created_at
        if existing_tokens:
            # Free the partial unique-index slot before inserting its successor.
            await self.session.flush()

        verification_token = EmailVerificationToken(
            user_id=user_id,
            token_hash=normalized_hash,
            pending_email=normalized_pending_email,
            created_at=created_at,
            expires_at=expires_at,
        )
        self.session.add(verification_token)
        await self.session.flush()
        return verification_token

    async def get_email_verification_token_by_hash(
        self,
        token_hash: str,
    ) -> EmailVerificationToken | None:
        normalized_hash = self._normalize_token_hash(token_hash)
        statement = select(EmailVerificationToken).where(
            EmailVerificationToken.token_hash == normalized_hash
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_email_verification_token_by_hash_for_update(
        self,
        token_hash: str,
    ) -> EmailVerificationToken | None:
        normalized_hash = self._normalize_token_hash(token_hash)
        locator_statement = select(
            EmailVerificationToken.id,
            EmailVerificationToken.user_id,
        ).where(EmailVerificationToken.token_hash == normalized_hash)
        locator_result = await self.session.execute(locator_statement)
        locator = locator_result.one_or_none()
        if locator is None or not await self._lock_user(locator.user_id):
            return None

        statement = (
            select(EmailVerificationToken)
            .where(
                EmailVerificationToken.id == locator.id,
                EmailVerificationToken.token_hash == normalized_hash,
                EmailVerificationToken.user_id == locator.user_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_registration_verification_token_by_hash_for_update(
        self,
        token_hash: str,
    ) -> EmailVerificationToken | None:
        """Lock and return a token explicitly issued for account registration."""

        normalized_hash = self._normalize_token_hash(token_hash)
        locator_statement = select(
            EmailVerificationToken.id,
            EmailVerificationToken.user_id,
        ).where(
            EmailVerificationToken.token_hash == normalized_hash,
            EmailVerificationToken.pending_email.is_(None),
        )
        locator_result = await self.session.execute(locator_statement)
        locator = locator_result.one_or_none()
        if locator is None or not await self._lock_user(locator.user_id):
            return None

        statement = (
            select(EmailVerificationToken)
            .where(
                EmailVerificationToken.id == locator.id,
                EmailVerificationToken.token_hash == normalized_hash,
                EmailVerificationToken.user_id == locator.user_id,
                EmailVerificationToken.pending_email.is_(None),
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_email_change_token_by_hash_for_update(
        self,
        token_hash: str,
    ) -> EmailVerificationToken | None:
        """Lock and return a token explicitly bound to a pending email address."""

        normalized_hash = self._normalize_token_hash(token_hash)
        locator_statement = select(
            EmailVerificationToken.id,
            EmailVerificationToken.user_id,
        ).where(
            EmailVerificationToken.token_hash == normalized_hash,
            EmailVerificationToken.pending_email.is_not(None),
        )
        locator_result = await self.session.execute(locator_statement)
        locator = locator_result.one_or_none()
        if locator is None or not await self._lock_user(locator.user_id):
            return None

        statement = (
            select(EmailVerificationToken)
            .where(
                EmailVerificationToken.id == locator.id,
                EmailVerificationToken.token_hash == normalized_hash,
                EmailVerificationToken.user_id == locator.user_id,
                EmailVerificationToken.pending_email.is_not(None),
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_active_email_change_token_for_user(
        self,
        user_id: uuid.UUID,
        *,
        now: datetime,
        for_update: bool = False,
    ) -> EmailVerificationToken | None:
        """Return the user's unconsumed, unexpired email-change token."""

        self._validate_aware_datetime(now, field_name="now")
        if for_update and not await self._lock_user(user_id):
            return None

        statement = select(EmailVerificationToken).where(
            EmailVerificationToken.user_id == user_id,
            EmailVerificationToken.pending_email.is_not(None),
            EmailVerificationToken.used_at.is_(None),
            EmailVerificationToken.revoked_at.is_(None),
            EmailVerificationToken.expires_at > now,
        )
        if for_update:
            statement = statement.with_for_update().execution_options(
                populate_existing=True
            )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_active_email_verification_token_for_user(
        self,
        user_id: uuid.UUID,
        *,
        now: datetime,
        for_update: bool = False,
    ) -> EmailVerificationToken | None:
        self._validate_aware_datetime(now, field_name="now")
        if for_update and not await self._lock_user(user_id):
            return None

        statement = select(EmailVerificationToken).where(
            EmailVerificationToken.user_id == user_id,
            EmailVerificationToken.used_at.is_(None),
            EmailVerificationToken.revoked_at.is_(None),
            EmailVerificationToken.expires_at > now,
        )
        if for_update:
            statement = statement.with_for_update().execution_options(
                populate_existing=True
            )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def mark_email_verification_token_used(
        self,
        verification_token: EmailVerificationToken,
        *,
        used_at: datetime,
    ) -> bool:
        """Atomically consume a verification token only once and before expiry."""

        self._validate_aware_datetime(used_at, field_name="used_at")
        statement = (
            update(EmailVerificationToken)
            .where(
                EmailVerificationToken.id == verification_token.id,
                EmailVerificationToken.used_at.is_(None),
                EmailVerificationToken.revoked_at.is_(None),
                EmailVerificationToken.expires_at > used_at,
            )
            .values(used_at=used_at)
            .returning(EmailVerificationToken.id)
            .execution_options(synchronize_session="fetch")
        )
        result = await self.session.execute(statement)
        consumed_id = result.scalar_one_or_none()
        if consumed_id is None:
            return False
        await self.session.flush()
        return True

    async def revoke_email_verification_token(
        self,
        verification_token: EmailVerificationToken,
        *,
        revoked_at: datetime,
    ) -> bool:
        self._validate_aware_datetime(revoked_at, field_name="revoked_at")
        statement = (
            update(EmailVerificationToken)
            .where(
                EmailVerificationToken.id == verification_token.id,
                EmailVerificationToken.used_at.is_(None),
                EmailVerificationToken.revoked_at.is_(None),
            )
            .values(revoked_at=revoked_at)
            .returning(EmailVerificationToken.id)
            .execution_options(synchronize_session="fetch")
        )
        result = await self.session.execute(statement)
        revoked_id = result.scalar_one_or_none()
        if revoked_id is None:
            return False
        await self.session.flush()
        return True

    async def revoke_email_verification_tokens_for_user(
        self,
        user_id: uuid.UUID,
        *,
        revoked_at: datetime,
    ) -> int:
        self._validate_aware_datetime(revoked_at, field_name="revoked_at")
        if not await self._lock_user(user_id):
            return 0

        statement = (
            update(EmailVerificationToken)
            .where(
                EmailVerificationToken.user_id == user_id,
                EmailVerificationToken.used_at.is_(None),
                EmailVerificationToken.revoked_at.is_(None),
            )
            .values(revoked_at=revoked_at)
            .returning(EmailVerificationToken.id)
            .execution_options(synchronize_session="fetch")
        )
        result = await self.session.execute(statement)
        revoked_ids = result.scalars().all()
        if revoked_ids:
            await self.session.flush()
        return len(revoked_ids)

    async def revoke_email_change_tokens_for_user(
        self,
        user_id: uuid.UUID,
        *,
        revoked_at: datetime,
    ) -> int:
        """Revoke outstanding email-change tokens without touching signup tokens."""

        self._validate_aware_datetime(revoked_at, field_name="revoked_at")
        if not await self._lock_user(user_id):
            return 0

        statement = (
            update(EmailVerificationToken)
            .where(
                EmailVerificationToken.user_id == user_id,
                EmailVerificationToken.pending_email.is_not(None),
                EmailVerificationToken.used_at.is_(None),
                EmailVerificationToken.revoked_at.is_(None),
            )
            .values(revoked_at=revoked_at)
            .returning(EmailVerificationToken.id)
            .execution_options(synchronize_session="fetch")
        )
        result = await self.session.execute(statement)
        revoked_ids = result.scalars().all()
        if revoked_ids:
            await self.session.flush()
        return len(revoked_ids)

    async def mark_email_verification_email_sent(
        self,
        verification_token: EmailVerificationToken,
        *,
        sent_at: datetime,
    ) -> bool:
        """Record provider success without reviving a consumed/revoked token."""

        self._validate_aware_datetime(sent_at, field_name="sent_at")
        statement = (
            update(EmailVerificationToken)
            .where(
                EmailVerificationToken.id == verification_token.id,
                EmailVerificationToken.used_at.is_(None),
                EmailVerificationToken.revoked_at.is_(None),
            )
            .values(sent_at=sent_at)
            .returning(EmailVerificationToken.id)
            .execution_options(synchronize_session="fetch")
        )
        result = await self.session.execute(statement)
        updated_id = result.scalar_one_or_none()
        if updated_id is None:
            return False
        await self.session.flush()
        return True

    # ------------------------------------------------------------------
    # Password-reset tokens
    # ------------------------------------------------------------------

    async def create_password_reset_token(
        self,
        user_id: uuid.UUID,
        *,
        token_hash: str,
        expires_at: datetime,
        created_at: datetime,
    ) -> PasswordResetToken | None:
        """Issue a reset token, replacing any previous outstanding token.

        User-row locking serializes concurrent reset requests. All unconsumed
        rows are revoked, including expired rows that still occupy the model's
        partial unique index. Returns ``None`` if the user no longer exists.
        """

        normalized_hash = self._normalize_token_hash(token_hash)
        self._validate_aware_datetime(expires_at, field_name="expires_at")
        self._validate_aware_datetime(created_at, field_name="created_at")
        if expires_at <= created_at:
            raise ValueError("expires_at must be later than created_at.")
        if not await self._lock_user(user_id):
            return None

        existing_statement = (
            select(PasswordResetToken)
            .where(
                PasswordResetToken.user_id == user_id,
                PasswordResetToken.used_at.is_(None),
                PasswordResetToken.revoked_at.is_(None),
            )
            .order_by(PasswordResetToken.id.asc())
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        existing_result = await self.session.execute(existing_statement)
        existing_tokens = list(existing_result.scalars().all())
        for existing_token in existing_tokens:
            existing_token.revoked_at = created_at
        if existing_tokens:
            # Free the partial unique-index slot before inserting its successor.
            await self.session.flush()

        reset_token = PasswordResetToken(
            user_id=user_id,
            token_hash=normalized_hash,
            created_at=created_at,
            expires_at=expires_at,
        )
        self.session.add(reset_token)
        await self.session.flush()
        return reset_token

    async def get_password_reset_token_by_hash(
        self,
        token_hash: str,
    ) -> PasswordResetToken | None:
        normalized_hash = self._normalize_token_hash(token_hash)
        statement = select(PasswordResetToken).where(
            PasswordResetToken.token_hash == normalized_hash
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_password_reset_token_by_hash_for_update(
        self,
        token_hash: str,
    ) -> PasswordResetToken | None:
        normalized_hash = self._normalize_token_hash(token_hash)
        locator_statement = select(
            PasswordResetToken.id,
            PasswordResetToken.user_id,
        ).where(PasswordResetToken.token_hash == normalized_hash)
        locator_result = await self.session.execute(locator_statement)
        locator = locator_result.one_or_none()
        if locator is None or not await self._lock_user(locator.user_id):
            return None

        statement = (
            select(PasswordResetToken)
            .where(
                PasswordResetToken.id == locator.id,
                PasswordResetToken.token_hash == normalized_hash,
                PasswordResetToken.user_id == locator.user_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_active_password_reset_token_for_user(
        self,
        user_id: uuid.UUID,
        *,
        now: datetime,
        for_update: bool = False,
    ) -> PasswordResetToken | None:
        self._validate_aware_datetime(now, field_name="now")
        if for_update and not await self._lock_user(user_id):
            return None

        statement = select(PasswordResetToken).where(
            PasswordResetToken.user_id == user_id,
            PasswordResetToken.used_at.is_(None),
            PasswordResetToken.revoked_at.is_(None),
            PasswordResetToken.expires_at > now,
        )
        if for_update:
            statement = statement.with_for_update().execution_options(
                populate_existing=True
            )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def mark_password_reset_token_used(
        self,
        reset_token: PasswordResetToken,
        *,
        used_at: datetime,
    ) -> bool:
        """Atomically consume a reset token only once and before expiry."""

        self._validate_aware_datetime(used_at, field_name="used_at")
        statement = (
            update(PasswordResetToken)
            .where(
                PasswordResetToken.id == reset_token.id,
                PasswordResetToken.used_at.is_(None),
                PasswordResetToken.revoked_at.is_(None),
                PasswordResetToken.expires_at > used_at,
            )
            .values(used_at=used_at)
            .returning(PasswordResetToken.id)
            .execution_options(synchronize_session="fetch")
        )
        result = await self.session.execute(statement)
        consumed_id = result.scalar_one_or_none()
        if consumed_id is None:
            return False
        await self.session.flush()
        return True

    async def revoke_password_reset_token(
        self,
        reset_token: PasswordResetToken,
        *,
        revoked_at: datetime,
    ) -> bool:
        self._validate_aware_datetime(revoked_at, field_name="revoked_at")
        statement = (
            update(PasswordResetToken)
            .where(
                PasswordResetToken.id == reset_token.id,
                PasswordResetToken.used_at.is_(None),
                PasswordResetToken.revoked_at.is_(None),
            )
            .values(revoked_at=revoked_at)
            .returning(PasswordResetToken.id)
            .execution_options(synchronize_session="fetch")
        )
        result = await self.session.execute(statement)
        revoked_id = result.scalar_one_or_none()
        if revoked_id is None:
            return False
        await self.session.flush()
        return True

    async def revoke_password_reset_tokens_for_user(
        self,
        user_id: uuid.UUID,
        *,
        revoked_at: datetime,
    ) -> int:
        self._validate_aware_datetime(revoked_at, field_name="revoked_at")
        if not await self._lock_user(user_id):
            return 0

        statement = (
            update(PasswordResetToken)
            .where(
                PasswordResetToken.user_id == user_id,
                PasswordResetToken.used_at.is_(None),
                PasswordResetToken.revoked_at.is_(None),
            )
            .values(revoked_at=revoked_at)
            .returning(PasswordResetToken.id)
            .execution_options(synchronize_session="fetch")
        )
        result = await self.session.execute(statement)
        revoked_ids = result.scalars().all()
        if revoked_ids:
            await self.session.flush()
        return len(revoked_ids)

    async def mark_password_reset_email_sent(
        self,
        reset_token: PasswordResetToken,
        *,
        sent_at: datetime,
    ) -> bool:
        """Record provider success without reviving a consumed/revoked token."""

        self._validate_aware_datetime(sent_at, field_name="sent_at")
        statement = (
            update(PasswordResetToken)
            .where(
                PasswordResetToken.id == reset_token.id,
                PasswordResetToken.used_at.is_(None),
                PasswordResetToken.revoked_at.is_(None),
            )
            .values(sent_at=sent_at)
            .returning(PasswordResetToken.id)
            .execution_options(synchronize_session="fetch")
        )
        result = await self.session.execute(statement)
        updated_id = result.scalar_one_or_none()
        if updated_id is None:
            return False
        await self.session.flush()
        return True
