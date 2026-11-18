"""JWT access-token and persistent refresh-token lifecycle management.

JWT handling belongs to the Auth domain, while reusable cryptographic
primitives remain in ``app.core.security``. Refresh credentials are opaque,
random values; only their SHA-256 digests are persisted.

All write workflows own their transaction. ``issue_token_pair`` is the one
exception: it only flushes so AuthService can atomically commit it with the
new session and any password-hash upgrade.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from jwt import InvalidTokenError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.enums import AuthTokenType, SessionRevocationReason
from app.core.security import generate_secure_token, hash_token
from app.exceptions.authentication import (
    AccountInactiveException,
    ExpiredAccessTokenException,
    ExpiredRefreshTokenException,
    InvalidAccessTokenException,
    InvalidRefreshTokenException,
    InvalidSessionException,
    InvalidTokenAlgorithmException,
    InvalidTokenAudienceException,
    InvalidTokenClaimsException,
    InvalidTokenIssuerException,
    InvalidTokenSignatureException,
    MalformedAccessTokenException,
    MissingTokenClaimsException,
    RefreshTokenReuseDetectedException,
    RefreshTokenRotationFailedException,
    RevokedRefreshTokenException,
    SessionExpiredException,
    SessionRevokedException,
    UnsupportedTokenTypeException,
)
from app.modules.auth.models.session import AuthSession
from app.modules.auth.repositories.session_repository import SessionRepository
from app.modules.auth.repositories.token_repository import TokenRepository
from app.modules.auth.schemas.tokens import RefreshTokenRequest, TokenPairResponse
from app.modules.users.models.user import User
from app.modules.users.repositories.user_repository import UserRepository


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AccessTokenClaims:
    """Validated identity and session claims extracted from an access JWT."""

    user_id: uuid.UUID
    session_id: uuid.UUID
    token_id: uuid.UUID
    issued_at: datetime
    not_before: datetime
    expires_at: datetime


class TokenService:
    """Issue, decode, validate, rotate and revoke authentication credentials."""

    _JWT_REQUIRED_CLAIMS = ("sub", "sid", "jti", "iss", "aud", "iat", "nbf", "exp", "token_type")
    _MAX_ACCESS_TOKEN_LENGTH = 8192

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
    def _timestamp(value: datetime) -> int:
        return int(value.timestamp())

    def create_access_token(
        self,
        *,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        now: datetime | None = None,
    ) -> str:
        """Create a short-lived signed access JWT for one user session."""

        if not isinstance(user_id, uuid.UUID) or not isinstance(session_id, uuid.UUID):
            raise TypeError("user_id and session_id must be UUID instances.")

        issued_at = self._utc_now(now)
        expires_at = issued_at + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
        token_id = uuid.uuid4()

        payload = {
            "sub": str(user_id),
            "sid": str(session_id),
            "jti": str(token_id),
            "iss": settings.JWT_ISSUER,
            "aud": settings.JWT_AUDIENCE,
            "iat": self._timestamp(issued_at),
            "nbf": self._timestamp(issued_at),
            "exp": self._timestamp(expires_at),
            "token_type": AuthTokenType.ACCESS.value,
        }

        return jwt.encode(
            payload,
            settings.JWT_SECRET_KEY.get_secret_value(),
            algorithm=settings.JWT_ALGORITHM,
            headers={"typ": "JWT"},
        )

    async def issue_token_pair(
        self,
        *,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        now: datetime | None = None,
        family_id: uuid.UUID | None = None,
        parent_token_id: uuid.UUID | None = None,
    ) -> TokenPairResponse:
        """Issue access and refresh credentials without committing the session.

        This method is used by AuthService during login and by this service
        during refresh rotation. The raw refresh token is returned once to the
        caller; only its digest is sent to the repository.
        """

        current_time = self._utc_now(now)
        auth_session = await self.session_repository.get_by_id_for_user(
            session_id=session_id,
            user_id=user_id,
        )
        if auth_session is None:
            raise InvalidSessionException()
        if auth_session.revoked_at is not None:
            raise SessionRevokedException()
        if auth_session.expires_at <= current_time:
            raise SessionExpiredException()

        refresh_expires_at = min(
            current_time + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            auth_session.expires_at,
        )
        raw_refresh_token = generate_secure_token()
        resolved_family_id = family_id or uuid.uuid4()

        await self.token_repository.create_refresh_token(
            session_id,
            family_id=resolved_family_id,
            token_hash=hash_token(raw_refresh_token),
            expires_at=refresh_expires_at,
            parent_token_id=parent_token_id,
        )

        access_token = self.create_access_token(
            user_id=user_id,
            session_id=session_id,
            now=current_time,
        )

        return TokenPairResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=raw_refresh_token,
            refresh_expires_in=max(
                1,
                int((refresh_expires_at - current_time).total_seconds()),
            ),
        )

    def decode_access_token(
        self,
        encoded_token: str,
        *,
        now: datetime | None = None,
    ) -> AccessTokenClaims:
        """Verify JWT signature, algorithm, issuer, audience, time and claims."""

        if not isinstance(encoded_token, str) or not encoded_token:
            raise MalformedAccessTokenException()
        if len(encoded_token) > self._MAX_ACCESS_TOKEN_LENGTH:
            raise MalformedAccessTokenException()

        try:
            payload: dict[str, Any] = jwt.decode(
                encoded_token,
                settings.JWT_SECRET_KEY.get_secret_value(),
                algorithms=[settings.JWT_ALGORITHM],
                issuer=settings.JWT_ISSUER,
                audience=settings.JWT_AUDIENCE,
                options={
                    "require": list(self._JWT_REQUIRED_CLAIMS),
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
                leeway=0,
            )
        except jwt.ExpiredSignatureError as exc:
            raise ExpiredAccessTokenException() from exc
        except jwt.InvalidAlgorithmError as exc:
            raise InvalidTokenAlgorithmException() from exc
        except jwt.InvalidSignatureError as exc:
            raise InvalidTokenSignatureException() from exc
        except jwt.InvalidIssuerError as exc:
            raise InvalidTokenIssuerException() from exc
        except jwt.InvalidAudienceError as exc:
            raise InvalidTokenAudienceException() from exc
        except jwt.MissingRequiredClaimError as exc:
            raise MissingTokenClaimsException() from exc
        except jwt.DecodeError as exc:
            raise MalformedAccessTokenException() from exc
        except InvalidTokenError as exc:
            raise InvalidAccessTokenException() from exc

        if payload.get("token_type") != AuthTokenType.ACCESS.value:
            raise UnsupportedTokenTypeException()

        try:
            user_id = uuid.UUID(payload["sub"])
            session_id = uuid.UUID(payload["sid"])
            token_id = uuid.UUID(payload["jti"])
            issued_at_raw = payload["iat"]
            not_before_raw = payload["nbf"]
            expires_at_raw = payload["exp"]
            for claim_name, claim_value in (
                ("iat", issued_at_raw),
                ("nbf", not_before_raw),
                ("exp", expires_at_raw),
            ):
                if isinstance(claim_value, bool) or not isinstance(claim_value, (int, float)):
                    raise ValueError(f"{claim_name} must be numeric.")
            issued_at = datetime.fromtimestamp(issued_at_raw, tz=timezone.utc)
            not_before = datetime.fromtimestamp(not_before_raw, tz=timezone.utc)
            expires_at = datetime.fromtimestamp(expires_at_raw, tz=timezone.utc)
        except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
            raise InvalidTokenClaimsException() from exc

        current_time = self._utc_now(now)
        if expires_at <= issued_at or not_before > expires_at:
            raise InvalidTokenClaimsException()
        if issued_at > current_time or not_before > current_time:
            raise InvalidTokenClaimsException()

        return AccessTokenClaims(
            user_id=user_id,
            session_id=session_id,
            token_id=token_id,
            issued_at=issued_at,
            not_before=not_before,
            expires_at=expires_at,
        )

    async def authenticate_access_token(
        self,
        encoded_token: str,
        *,
        now: datetime | None = None,
    ) -> tuple[User, AuthSession, AccessTokenClaims]:
        """Resolve a valid JWT to an active user and persistent session.

        This performs a database-backed session check. It does not make
        stateless access JWTs instantly revocable: session state is checked on
        every call, so consumers must consistently use this method for that
        guarantee to apply.
        """

        current_time = self._utc_now(now)
        claims = self.decode_access_token(encoded_token, now=current_time)
        auth_session = await self.session_repository.get_active_by_id_for_user(
            session_id=claims.session_id,
            user_id=claims.user_id,
            now=current_time,
        )
        if auth_session is None:
            existing_session = await self.session_repository.get_by_id_for_user(
                session_id=claims.session_id,
                user_id=claims.user_id,
            )
            if existing_session is None:
                raise InvalidSessionException()
            if existing_session.revoked_at is not None:
                raise SessionRevokedException()
            if existing_session.expires_at <= current_time:
                raise SessionExpiredException()
            raise InvalidSessionException()

        user = await self.user_repository.get_by_id(claims.user_id)
        if user is None:
            raise InvalidAccessTokenException()
        if not user.is_active:
            raise AccountInactiveException()

        return user, auth_session, claims

    async def refresh(
        self,
        request: RefreshTokenRequest,
        *,
        now: datetime | None = None,
    ) -> TokenPairResponse:
        """Rotate a refresh token, detecting replay and revoking its session."""

        if not isinstance(request, RefreshTokenRequest):
            raise TypeError("request must be a RefreshTokenRequest instance.")

        current_time = self._utc_now(now)
        raw_refresh_token = request.refresh_token.get_secret_value()
        token_hash_value = hash_token(raw_refresh_token)

        try:
            refresh_token = await self.token_repository.get_refresh_token_by_hash_for_update(
                token_hash_value
            )
            if refresh_token is None:
                raise InvalidRefreshTokenException()

            if refresh_token.used_at is not None:
                await self.token_repository.revoke_refresh_token_family(
                    refresh_token.family_id,
                    revoked_at=current_time,
                )
                await self.token_repository.revoke_refresh_tokens_for_session(
                    refresh_token.session_id,
                    revoked_at=current_time,
                )
                auth_session = await self.session_repository.get_by_id(
                    refresh_token.session_id
                )
                if auth_session is not None:
                    await self.session_repository.revoke(
                        auth_session,
                        reason=SessionRevocationReason.REFRESH_TOKEN_REUSE,
                        revoked_at=current_time,
                    )
                # Security revocation must commit before the exception is
                # propagated. A surrounding exception handler must not undo it.
                await self.session.commit()
                raise RefreshTokenReuseDetectedException()

            if refresh_token.revoked_at is not None:
                raise RevokedRefreshTokenException()
            if refresh_token.expires_at <= current_time:
                raise ExpiredRefreshTokenException()

            auth_session = await self.session_repository.get_by_id(
                refresh_token.session_id
            )
            if auth_session is None:
                raise InvalidSessionException()
            if auth_session.revoked_at is not None:
                raise SessionRevokedException()
            if auth_session.expires_at <= current_time:
                raise SessionExpiredException()

            user = await self.user_repository.get_by_id(auth_session.user_id)
            if user is None:
                raise InvalidRefreshTokenException()
            if not user.is_active:
                await self.token_repository.revoke_refresh_tokens_for_session(
                    auth_session.id,
                    revoked_at=current_time,
                )
                await self.session_repository.revoke(
                    auth_session,
                    reason=SessionRevocationReason.ACCOUNT_DEACTIVATED,
                    revoked_at=current_time,
                )
                await self.session.commit()
                raise AccountInactiveException()

            consumed = await self.token_repository.mark_refresh_token_used(
                refresh_token,
                used_at=current_time,
            )
            if not consumed:
                raise RefreshTokenRotationFailedException()

            await self.session_repository.touch(
                auth_session,
                last_used_at=current_time,
            )
            token_pair = await self.issue_token_pair(
                user_id=user.id,
                session_id=auth_session.id,
                now=current_time,
                family_id=refresh_token.family_id,
                parent_token_id=refresh_token.id,
            )
            await self.session.commit()
            return token_pair

        except BaseException:
            # A committed replay/deactivation response has no active transaction;
            # rollback is therefore harmless and cannot undo the security event.
            try:
                await self.session.rollback()
            except SQLAlchemyError:
                logger.exception("Database rollback failed during refresh rotation.")
            raise
