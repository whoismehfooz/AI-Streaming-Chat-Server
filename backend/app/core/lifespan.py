
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.core.config import settings
from app.db.session import dispose_engine, engine


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "Starting application: %s",
        settings.APP_NAME,
    )

    try:
        # Verify that the database is reachable before accepting requests.
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

        logger.info("Database connectivity verified successfully.")

        yield

    finally:
        logger.info(
            "Shutting down application: %s",
            settings.APP_NAME,
        )

        await dispose_engine()

        logger.info("Database engine disposed successfully.")
