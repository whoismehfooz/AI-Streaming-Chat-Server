import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.message import Message


class MessageRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )

        self.session.add(message)
        await self.session.flush()

        return message

    async def get_by_id(
        self,
        message_id: uuid.UUID,
    ) -> Message | None:
        result = await self.session.execute(
            select(Message).where(
                Message.id == message_id
            )
        )

        return result.scalar_one_or_none()

    async def get_by_id_for_user(
        self,
        message_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Message | None:
        result = await self.session.execute(
            select(Message)
            .join(
                Conversation,
                Message.conversation_id == Conversation.id,
            )
            .where(
                Message.id == message_id,
                Conversation.user_id == user_id,
            )
        )

        return result.scalar_one_or_none()

    async def list_by_conversation(
        self,
        conversation_id: uuid.UUID,
        limit: int,
        offset: int,
    ) -> list[Message]:
        result = await self.session.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id
            )
            .order_by(
                Message.created_at.asc(),
                Message.id.asc(),
            )
            .limit(limit)
            .offset(offset)
        )

        return list(result.scalars().all())

    async def list_by_conversation_for_user(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int,
        offset: int,
    ) -> list[Message]:
        result = await self.session.execute(
            select(Message)
            .join(
                Conversation,
                Message.conversation_id == Conversation.id,
            )
            .where(
                Message.conversation_id == conversation_id,
                Conversation.user_id == user_id,
            )
            .order_by(
                Message.created_at.asc(),
                Message.id.asc(),
            )
            .limit(limit)
            .offset(offset)
        )

        return list(result.scalars().all())

    async def list_recent_by_conversation(
        self,
        conversation_id: uuid.UUID,
        limit: int,
    ) -> list[Message]:
        result = await self.session.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id
            )
            .order_by(
                Message.created_at.desc(),
                Message.id.desc(),
            )
            .limit(limit)
        )

        messages = list(result.scalars().all())
        messages.reverse()

        return messages
