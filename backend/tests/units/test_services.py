import uuid

import pytest
from sqlalchemy.ext.asyncio import (create_async_engine, async_sessionmaker)

from app.core.config import settings
from app.services import ConversationService, MessageService


@pytest.mark.asyncio
async def test_conversation_service():
    engine = create_async_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True
    )

    SessionLocal = async_sessionmaker(
        bind=engine,
        expire_on_commit=False
    )

    conversation_id = None

    async with SessionLocal() as session:
        service = ConversationService(session)

        conversation = await service.create_conversation(
            title="Service Test"
        )

        conversation_id = conversation.id

        assert conversation_id is not None
        assert conversation.title == "Service Test"

        loaded_conversation = await service.get_conversation(
            conversation.id
        )

        assert loaded_conversation is not None
        assert loaded_conversation.id == conversation.id

        conversations = await service.list_conversations(
            limit=10,
            offset=0
        )

        assert any(
            item.id == conversation.id
            for item in conversations
        )

        deleted_conversation = await service.delete_conversation(
            conversation.id
        )

        assert deleted_conversation is True

        missing = await service.get_conversation(
            conversation.id
        )

        assert missing is None

    await engine.dispose()


@pytest.mark.asyncio
async def test_message_service():
    engine = create_async_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True
    )

    SessionLocal = async_sessionmaker(
        bind=engine,
        expire_on_commit=False
    )

    async with SessionLocal() as session:
        conversation_service = ConversationService(session)
        message_service = MessageService(session)

        conversation = await conversation_service.create_conversation(
            title="Message Service Test"
        )

        message = await message_service.create_message(
            conversation_id=conversation.id,
            role="user",
            content="Hello from the service."
        )

        assert message.id is not None
        assert message.conversation_id == conversation.id

        loaded_message = await message_service.get_message(
            message.id
        )

        assert loaded_message is not None
        assert loaded_message.content == "Hello from the service."

        messages = await message_service.list_messages(
            conversation_id=conversation.id,
            limit=10,
            offset=0
        )

        assert len(messages) == 1
        assert messages[0].id == message.id

        deleted_message = await conversation_service.delete_conversation(
            conversation.id
        )

        assert deleted_message is True

        missing_message = await message_service.get_message(
            message.id
        )

        assert missing_message is None

    await engine.dispose()
