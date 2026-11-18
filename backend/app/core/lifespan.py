from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import resend
from fastapi import FastAPI
from sqlalchemy import text

from app.core.config import settings
from app.db.session import dispose_engine, engine


logger = logging.getLogger(__name__)


async def _verify_database_connectivity() -> None:
    """Fail application startup when PostgreSQL cannot be reached."""

    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


def _configure_resend() -> tuple[str | None, resend.AsyncHTTPClient | None]:
    """Configure Resend and return the SDK globals that must be restored.

    Resend's SDK configuration is process-global. Save the previous values so
    lifespan shutdown (including startup failure) does not leave mutated SDK
    state behind, which is especially useful for test processes and embedded
    ASGI applications.
    """

    previous_api_key = resend.api_key
    previous_async_http_client = resend.default_async_http_client

    if settings.RESEND_API_KEY is None:
        resend.api_key = ""
        logger.warning(
            "RESEND_API_KEY is not configured; outbound email delivery is unavailable."
        )
        return previous_api_key, previous_async_http_client

    configured_http_client = resend.HTTPXClient(
        timeout=settings.RESEND_TIMEOUT_SECONDS,
    )
    resend.api_key = settings.RESEND_API_KEY.get_secret_value()
    resend.default_async_http_client = configured_http_client
    logger.info(
        "Resend async integration configured (timeout=%s seconds).",
        settings.RESEND_TIMEOUT_SECONDS,
    )
    return previous_api_key, previous_async_http_client


def _restore_resend(
    previous_api_key: str | None,
    previous_async_http_client: resend.AsyncHTTPClient | None,
) -> None:
    """Restore process-global SDK state without closing SDK-owned resources.

    The configured Resend HTTPXClient is stateless with respect to network
    resources in the pinned SDK version: each request creates and closes its
    own HTTPX AsyncClient. Restoring the prior SDK reference is sufficient.
    """

    resend.api_key = previous_api_key
    resend.default_async_http_client = previous_async_http_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own shared infrastructure for one FastAPI application lifespan."""

    logger.info(
        "Starting application: %s (environment=%s)",
        settings.APP_NAME,
        settings.APP_ENV,
    )

    previous_resend_api_key: str | None = None
    previous_resend_http_client: resend.AsyncHTTPClient | None = None
    resend_state_captured = False
    primary_error: BaseException | None = None

    try:
        try:
            await _verify_database_connectivity()
        except Exception:
            logger.exception(
                "Database connectivity check failed; application startup cannot continue."
            )
            raise

        logger.info("Database connectivity verified successfully.")

        (
            previous_resend_api_key,
            previous_resend_http_client,
        ) = _configure_resend()
        resend_state_captured = True

        # Expose only non-secret operational state. Never place the API key or
        # SDK client itself on app.state where unrelated code could mutate it.
        app.state.email_delivery_configured = settings.RESEND_API_KEY is not None

        yield

    except BaseException as exc:
        # Preserve startup, request-serving, and cancellation exceptions if a
        # later cleanup operation also fails.
        primary_error = exc
        raise

    finally:
        logger.info("Shutting down application: %s", settings.APP_NAME)

        if resend_state_captured:
            _restore_resend(
                previous_resend_api_key,
                previous_resend_http_client,
            )

        try:
            await dispose_engine()
        except Exception:
            if primary_error is None:
                logger.exception("Database engine disposal failed during shutdown.")
                raise
            logger.exception(
                "Database engine disposal also failed while handling an earlier error."
            )
        else:
            logger.info("Database engine disposed successfully.")


__all__ = ["lifespan"]
