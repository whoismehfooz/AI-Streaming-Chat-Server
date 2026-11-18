from __future__ import annotations

import hashlib
import hmac
import ipaddress
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Final

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.exceptions.authentication import AuthenticationRateLimitExceededException
from app.modules.auth.repositories.rate_limit_repository import (
    AuthenticationRateLimitRepository,
)


class AuthenticationRateLimitOperation(StrEnum):
    """Sensitive authentication operations protected by the service."""

    REGISTRATION = "registration"
    LOGIN = "login"
    PASSWORD_RESET = "password_reset"
    EMAIL_VERIFICATION = "email_verification"
    TOKEN_REFRESH = "token_refresh"


@dataclass(frozen=True, slots=True)
class RateLimitRule:
    """A GCRA rate with an explicit maximum burst capacity."""

    limit: int
    window_seconds: int
    burst_capacity: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("limit", self.limit),
            ("window_seconds", self.window_seconds),
            ("burst_capacity", self.burst_capacity),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer.")
        if self.burst_capacity > self.limit:
            raise ValueError("burst_capacity cannot exceed limit.")

    @property
    def emission_interval(self) -> timedelta:
        return timedelta(seconds=self.window_seconds / self.limit)

    @property
    def burst_tolerance(self) -> timedelta:
        return self.emission_interval * (self.burst_capacity - 1)


# Defaults are intentionally proportionate and independently overridable by
# injecting a custom policy mapping in tests or deployment configuration.
# Each operation has a source dimension and, where applicable, an identity
# dimension. The identity dimension is HMACed before persistence.
DEFAULT_AUTH_RATE_LIMIT_POLICIES: Final[
    Mapping[AuthenticationRateLimitOperation, Mapping[str, RateLimitRule]]
] = {
    AuthenticationRateLimitOperation.REGISTRATION: {
        "ip": RateLimitRule(limit=5, window_seconds=3_600, burst_capacity=5),
    },
    AuthenticationRateLimitOperation.LOGIN: {
        "ip": RateLimitRule(limit=20, window_seconds=900, burst_capacity=10),
        "identity": RateLimitRule(limit=10, window_seconds=900, burst_capacity=5),
    },
    AuthenticationRateLimitOperation.PASSWORD_RESET: {
        "ip": RateLimitRule(limit=5, window_seconds=3_600, burst_capacity=3),
        "identity": RateLimitRule(limit=3, window_seconds=3_600, burst_capacity=2),
    },
    AuthenticationRateLimitOperation.EMAIL_VERIFICATION: {
        "ip": RateLimitRule(limit=10, window_seconds=3_600, burst_capacity=5),
        "identity": RateLimitRule(limit=5, window_seconds=3_600, burst_capacity=3),
    },
    AuthenticationRateLimitOperation.TOKEN_REFRESH: {
        "ip": RateLimitRule(limit=60, window_seconds=3_600, burst_capacity=15),
        "identity": RateLimitRule(limit=30, window_seconds=3_600, burst_capacity=10),
    },
}

_SCOPE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_IDENTITY_DIMENSIONS: Final[frozenset[str]] = frozenset(
    {"email", "user_id", "session_id", "identity"}
)


def get_client_ip(request: Request) -> str:
    """Return the socket peer address without trusting spoofable headers.

    ``X-Forwarded-For`` and ``Forwarded`` are intentionally not parsed here.
    If the API is behind a reverse proxy, configure Uvicorn/FastAPI trusted
    proxy handling with an explicit allow-list so ``request.client.host`` is
    populated from a trusted proxy chain. Never trust arbitrary client-sent
    forwarding headers for an abuse-control boundary.
    """

    if request.client is None or not request.client.host:
        raise ValueError("A client IP address is required for rate limiting.")

    try:
        return str(ipaddress.ip_address(request.client.host))
    except ValueError as exc:
        raise ValueError("The request client address is not a valid IP address.") from exc


class AuthenticationRateLimiter:
    """Enforce shared authentication limits using PostgreSQL GCRA state."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
        policies: Mapping[
            AuthenticationRateLimitOperation,
            Mapping[str, RateLimitRule],
        ] = DEFAULT_AUTH_RATE_LIMIT_POLICIES,
        hash_secret: str | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._policies = policies
        secret = (
            hash_secret
            if hash_secret is not None
            else settings.JWT_SECRET_KEY.get_secret_value()
        )
        if not isinstance(secret, str) or len(secret.encode("utf-8")) < 32:
            raise ValueError("Rate-limit HMAC secret must contain at least 32 bytes.")
        self._hash_secret = secret.encode("utf-8")

        self._validate_policies()

    async def enforce(
        self,
        operation: AuthenticationRateLimitOperation | str,
        *,
        identifiers: Mapping[str, str],
    ) -> None:
        """Consume all configured dimensions or raise HTTP 429.

        ``identifiers`` must provide every dimension configured for the
        operation. Typical usage:

            await limiter.enforce(
                AuthenticationRateLimitOperation.LOGIN,
                identifiers={"ip": get_client_ip(request), "identity": email},
            )

        The method deliberately does not reveal which dimension was exceeded.
        """

        try:
            resolved_operation = AuthenticationRateLimitOperation(operation)
        except ValueError as exc:
            raise ValueError("Unsupported authentication rate-limit operation.") from exc

        rules = self._policies.get(resolved_operation)
        if not rules:
            raise ValueError("No rate-limit policy is configured for this operation.")

        missing_dimensions = set(rules).difference(identifiers)
        if missing_dimensions:
            raise ValueError(
                "Missing required rate-limit identifier dimension(s): "
                + ", ".join(sorted(missing_dimensions))
            )

        now = datetime.now(timezone.utc)
        blocked_retry_after = 0

        # Use a dedicated session and transaction. Rate-limit accounting must
        # commit even if later endpoint work fails or the request DB session
        # rolls back. It must not commit unrelated application mutations.
        async with self._session_factory() as session:
            async with session.begin():
                repository = AuthenticationRateLimitRepository(session)

                for dimension, rule in rules.items():
                    identifier = identifiers[dimension]
                    normalized_identifier = self._normalize_identifier(
                        dimension,
                        identifier,
                    )
                    scope = f"auth.{resolved_operation.value}.{dimension}"
                    key_hash = self._hash_identifier(scope, normalized_identifier)

                    allowed, theoretical_arrival_time = await repository.consume(
                        scope=scope,
                        key_hash=key_hash,
                        now=now,
                        emission_interval=rule.emission_interval,
                        burst_tolerance=rule.burst_tolerance,
                    )

                    if not allowed:
                        retry_after = math.ceil(
                            (
                                theoretical_arrival_time
                                - rule.burst_tolerance
                                - now
                            ).total_seconds()
                        )
                        blocked_retry_after = max(
                            blocked_retry_after,
                            max(1, retry_after),
                        )

        if blocked_retry_after:
            raise AuthenticationRateLimitExceededException(
                retry_after_seconds=blocked_retry_after,
            )

    @staticmethod
    def _normalize_identifier(dimension: str, identifier: str) -> str:
        if not isinstance(identifier, str):
            raise TypeError("Rate-limit identifiers must be strings.")

        normalized = identifier.strip()
        if not normalized:
            raise ValueError("Rate-limit identifiers cannot be empty.")

        if dimension == "ip":
            try:
                return str(ipaddress.ip_address(normalized))
            except ValueError as exc:
                raise ValueError("The IP rate-limit identifier is invalid.") from exc

        if dimension == "email":
            return normalized.casefold()

        # UUID/session identifiers and generic identity values are treated as
        # opaque, case-insensitive identifiers. Their raw value is never stored.
        if dimension in _IDENTITY_DIMENSIONS:
            return normalized.casefold()

        return normalized

    def _hash_identifier(self, scope: str, identifier: str) -> str:
        message = f"{scope}\0{identifier}".encode("utf-8")
        return hmac.new(
            self._hash_secret,
            message,
            hashlib.sha256,
        ).hexdigest()

    def _validate_policies(self) -> None:
        if not self._policies:
            raise ValueError("At least one authentication rate-limit policy is required.")

        for operation, dimensions in self._policies.items():
            try:
                resolved_operation = AuthenticationRateLimitOperation(operation)
            except ValueError as exc:
                raise ValueError("A rate-limit policy uses an unknown operation.") from exc

            if not dimensions:
                raise ValueError(
                    f"Rate-limit operation {resolved_operation.value!r} has no dimensions."
                )

            for dimension, rule in dimensions.items():
                if not _SCOPE_PATTERN.fullmatch(dimension):
                    raise ValueError(
                        f"Invalid rate-limit dimension name: {dimension!r}."
                    )
                if not isinstance(rule, RateLimitRule):
                    raise TypeError("Every rate-limit policy value must be RateLimitRule.")


__all__ = [
    "AuthenticationRateLimitOperation",
    "AuthenticationRateLimiter",
    "DEFAULT_AUTH_RATE_LIMIT_POLICIES",
    "RateLimitRule",
    "get_client_ip",
]
