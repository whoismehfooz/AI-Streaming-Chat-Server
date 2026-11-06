import uuid

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import (create_async_engine, async_sessionmaker)

from app.core.config import settings
from app.models.conversation import Conversation
from app.models.message import Message
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository


@pytest.mark.asyncio
async def test_conversation_and_message_presistence():
    engine = create_async_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True
    )

    SessionLocal = async_sessionmaker(
        bind=engine,
        expire_on_commit=False
    )

    async with SessionLocal() as session:
        conversation_repository = ConversationRepository(session)
        message_repository = MessageRepository(session)

        conversation = await conversation_repository.create(
            title="Repository Test"
        )

        assert conversation.id is not None

        message = await message_repository.create(
            conversation_id=conversation.id,
            role="user",
            content="Hello from the repository test."
        )

        assert message.id is not None
        assert message.conversation_id == conversation.id

        await session.commit()

        loaded_conversation = (
            await conversation_repository.get_by_id(
                conversation.id
            )
        )

        loaded_messages= (
            await message_repository.list_by_conversation(
                conversation.id,
                limit=10,
                offset=0
            )
        )

        assert loaded_conversation is not None
        assert loaded_conversation.title == "Repository Test"

        assert len(loaded_messages)== 1
        assert loaded_messages[0].content == "Hello from the repository test."

        await session.execute(
            delete(Message).where(
                Message.conversation_id == conversation.id
            )
        )

        await session.execute(
            delete(Conversation).where(
                Conversation.id == conversation.id
            )
        )

        await session.commit()

    await engine.dispose()
