from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.conversations.models.conversation import Conversation
from app.modules.conversations.models.message import Message


class MessageRepository:


    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _validate_pagination(*, limit: int, offset: int) -> None:
        """Validate pagination arguments before issuing a database query."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Limit must be a positive integer.")

        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("Offset must be a non-negative integer.")

    @staticmethod
    def _validate_limit(limit: int) -> None:
        """Validate a limit used by recent-message queries."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Limit must be a positive integer.")

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

        statement = select(Message).where(Message.id == message_id)
        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def get_by_id_for_user(
        self,
        message_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Message | None:

        statement = (
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
        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def get_by_id_for_user_for_update(
        self,
        message_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Message | None:
        """Fetch a message only through its owner's conversation and lock it.

        The joined parent conversation is included in the ownership predicate.
        ``FOR UPDATE OF conversations`` locks the ownership boundary without
        unnecessarily locking unrelated joined rows. This is PostgreSQL's
        concurrency guard for user-scoped message mutations.
        """
        statement = (
            select(Message)
            .join(
                Conversation,
                Message.conversation_id == Conversation.id,
            )
            .where(
                Message.id == message_id,
                Conversation.user_id == user_id,
            )
            .with_for_update(of=Conversation)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def list_by_conversation(
        self,
        conversation_id: uuid.UUID,
        limit: int,
        offset: int,
    ) -> list[Message]:

        self._validate_pagination(limit=limit, offset=offset)

        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(
                Message.created_at.asc(),
                Message.id.asc(),
            )
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(statement)

        return list(result.scalars().all())

    async def list_by_conversation_for_user(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int,
        offset: int,
    ) -> list[Message]:
        self._validate_pagination(limit=limit, offset=offset)

        statement = (
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
        result = await self.session.execute(statement)

        return list(result.scalars().all())

    async def list_recent_by_conversation(
        self,
        conversation_id: uuid.UUID,
        limit: int,
    ) -> list[Message]:

        self._validate_limit(limit)

        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(
                Message.created_at.desc(),
                Message.id.desc(),
            )
            .limit(limit)
        )
        result = await self.session.execute(statement)

        messages = list(result.scalars().all())
        messages.reverse()

        return messages

    async def list_recent_by_conversation_for_user(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int,
    ) -> list[Message]:

        self._validate_limit(limit)

        statement = (
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
                Message.created_at.desc(),
                Message.id.desc(),
            )
            .limit(limit)
        )
        result = await self.session.execute(statement)

        messages = list(result.scalars().all())
        messages.reverse()

        return messages

    async def delete(self, message: Message) -> None:

        await self.session.delete(message)
        await self.session.flush()
