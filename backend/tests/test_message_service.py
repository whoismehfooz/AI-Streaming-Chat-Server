from datetime import datetime, timedelta, timezone

import pytest
import uuid

from app.models import Conversation, Message
from app.services import MessageService
from app.exceptions.custom_exceptions import ConversationNotFoundException
from app.exceptions.custom_exceptions import ConversationNotFoundException


@pytest.mark.asyncio
async def test_create_message_updates_conversation_updated_at(
    test_database
):
    old_timestamp = datetime.now(timezone.utc) - timedelta(
        minutes=10
    )

    async with test_database() as session:
        conversation = Conversation(
            title="Updated At Test",
            updated_at=old_timestamp
        )

        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)

        assert conversation.updated_at == old_timestamp

        service = MessageService(session)

        message = await service.create_message(
            conversation_id=conversation.id,
            role="user",
            content="First message."
        )

        await session.refresh(conversation)

        assert message.conversation_id == conversation.id

        assert conversation.updated_at > old_timestamp


@pytest.mark.asyncio
async def test_create_message_persists_conversation_timestamp_and_message(
    test_database
):
    async with test_database() as session:
        conversation = Conversation(
            title="Transaction Test"
        )

        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)

        service = MessageService(session)

        message = await service.create_message(
            conversation_id=conversation.id,
            role="user",
            content="Transactional message."
        )

        message_id = message.id
        conversation_id = conversation.id

    async with test_database() as session:
        persisted_conversation = await session.get(
            Conversation,
            conversation_id
        )

        assert persisted_conversation is not None

        assert persisted_conversation.updated_at is not None

        persisted_message = await session.get(
            Message,
            message_id
        )

        assert persisted_message is not None

        assert persisted_message.content == (
            "Transactional message."
        )


@pytest.mark.asyncio
async def test_create_message_rejects_missing_conversation(
    test_database,
):
    async with test_database() as session:
        service = MessageService(session)

        with pytest.raises(ConversationNotFoundException) as exc_info:
            await service.create_message(
                conversation_id=uuid.uuid4(),
                role="user",
                content="This should fail."
            )

        assert exc_info.value.status_code == 404
