from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.conversations.models.conversation import Conversation


class ConversationRepository:

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _validate_pagination(*, limit: int, offset: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Limit must be a positive integer.")

        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("Offset must be a non-negative integer.")

    async def create(
        self,
        user_id: uuid.UUID,
        title: str | None = None,
    ) -> Conversation:
        conversation = Conversation(
            user_id=user_id,
            title=title,
        )

        self.session.add(conversation)
        await self.session.flush()

        return conversation

    async def get_by_id(
        self,
        conversation_id: uuid.UUID,
    ) -> Conversation | None:

        statement = select(Conversation).where(
            Conversation.id == conversation_id,
        )
        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def get_by_id_for_user(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Conversation | None:

        statement = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def get_by_id_for_user_for_update(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Conversation | None:
        """Fetch and lock a conversation inside the caller's transaction.

        Ownership is part of the SQL predicate, not a post-fetch Python check.
        PostgreSQL's row lock serializes mutations against concurrent updates or
        deletion of this conversation. The caller owns commit/rollback.
        """
        statement = (
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
            .with_for_update(of=Conversation)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_by_id_for_update(
        self,
        conversation_id: uuid.UUID,
    ) -> Conversation | None:
        """Fetch and lock a conversation for trusted internal workflows only.

        Do not use this method for user-facing requests. Those must use an
        ownership-scoped lookup such as ``get_by_id_for_user_for_update``.
        The caller owns commit/rollback.
        """
        statement = (
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .with_for_update(of=Conversation)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: uuid.UUID,
        limit: int,
        offset: int,
    ) -> list[Conversation]:
        self._validate_pagination(limit=limit, offset=offset)

        statement = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(
                Conversation.updated_at.desc(),
                Conversation.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(statement)

        return list(result.scalars().all())

    async def update(
        self,
        conversation: Conversation,
        title: str | None,
    ) -> Conversation:
        conversation.title = title

        await self.session.flush()

        return conversation

    async def delete(self, conversation: Conversation) -> None:

        await self.session.delete(conversation)
        await self.session.flush()
