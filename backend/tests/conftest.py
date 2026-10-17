import sys
from pathlib import Path

BACKEND_DIR=Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0,str(BACKEND_DIR))

import pytest_asyncio

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.db.dependencies import get_db_session
from app.main import app



@pytest_asyncio.fixture(autouse=True)
async def test_database():
    engine = create_async_engine(
        settings.TEST_DATABASE_URL,
        poolclass=NullPool
    )

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async def override_get_db_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session

    try:
        yield session_factory
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()
