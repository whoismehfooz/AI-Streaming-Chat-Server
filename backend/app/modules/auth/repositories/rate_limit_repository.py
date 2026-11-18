"""Atomic PostgreSQL repository operations for authentication rate limits."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import case, func, or_, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.models.rate_limit import AuthenticationRateLimitEntry


class AuthenticationRateLimitRepository:
    """Persist and atomically advance GCRA rate-limit state.

    This repository never commits or rolls back. The rate-limit service owns
    the dedicated transaction used for each enforcement operation.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def consume(
        self,
        *,
        scope: str,
        key_hash: str,
        now: datetime,
        emission_interval: timedelta,
        burst_tolerance: timedelta,
    ) -> tuple[bool, datetime]:
        """Consume one request slot using a concurrency-safe PostgreSQL UPSERT.

        Returns ``(allowed, theoretical_arrival_time)``. The conditional
        ``ON CONFLICT DO UPDATE`` is evaluated while PostgreSQL holds the
        conflicting row lock, so concurrent workers cannot over-admit based
        on stale in-process state.
        """

        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware.")
        if emission_interval <= timedelta(0):
            raise ValueError("emission_interval must be positive.")
        if burst_tolerance < timedelta(0):
            raise ValueError("burst_tolerance cannot be negative.")

        table = AuthenticationRateLimitEntry.__table__
        proposed_tat = now + emission_interval
        existing_state_expired = table.c.expires_at <= now

        next_tat = case(
            (
                existing_state_expired,
                proposed_tat,
            ),
            else_=func.greatest(
                table.c.theoretical_arrival_time,
                now,
            ) + emission_interval,
        )

        statement = (
            postgresql_insert(table)
            .values(
                scope=scope,
                key_hash=key_hash,
                theoretical_arrival_time=proposed_tat,
                expires_at=proposed_tat,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=[table.c.scope, table.c.key_hash],
                set_={
                    "theoretical_arrival_time": next_tat,
                    "expires_at": next_tat,
                    "updated_at": now,
                },
                where=or_(
                    existing_state_expired,
                    table.c.theoretical_arrival_time
                    <= now + burst_tolerance,
                ),
            )
            .returning(table.c.theoretical_arrival_time)
        )

        accepted_tat = await self._session.scalar(statement)
        if accepted_tat is not None:
            return True, accepted_tat

        # The conditional UPSERT did not update the row, which means the
        # request is over the limit. Read the current TAT to calculate a
        # standards-compatible Retry-After value in the service layer.
        current_tat = await self._session.scalar(
            select(table.c.theoretical_arrival_time).where(
                table.c.scope == scope,
                table.c.key_hash == key_hash,
            )
        )

        # A conflicting row cannot disappear during this transaction unless
        # an external cleanup process deletes it. Treat that rare race as a
        # retryable repository condition rather than silently allowing access.
        if current_tat is None:
            raise RuntimeError(
                "Rate-limit state disappeared during a rejected UPSERT."
            )

        return False, current_tat

    async def delete_expired(self, *, now: datetime, batch_size: int = 1_000) -> int:
        """Delete a bounded batch of expired state rows.

        This is intended for a scheduled maintenance job, not request paths.
        SKIP LOCKED lets multiple cleanup workers run without blocking active
        request transactions. The caller owns the surrounding transaction.
        """

        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        table = AuthenticationRateLimitEntry.__table__
        candidates = (
            select(table.c.scope, table.c.key_hash)
            .where(table.c.expires_at <= now)
            .order_by(table.c.expires_at)
            .limit(batch_size)
            .with_for_update(skip_locked=True)
            .cte("expired_rate_limit_entries")
        )

        from sqlalchemy import delete

        result = await self._session.execute(
            delete(table)
            .where(
                table.c.scope == candidates.c.scope,
                table.c.key_hash == candidates.c.key_hash,
            )
            .returning(table.c.scope)
        )
        return len(result.scalars().all())


__all__ = ["AuthenticationRateLimitRepository"]
