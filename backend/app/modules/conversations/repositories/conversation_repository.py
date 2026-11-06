import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

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
        result = await self.session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id
            )
        )

        return result.scalar_one_or_none()

    async def get_by_id_for_user(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Conversation | None:
        result = await self.session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )

        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: uuid.UUID,
        limit: int,
        offset: int,
    ) -> list[Conversation]:
        result = await self.session.execute(
            select(Conversation)
            .where(
                Conversation.user_id == user_id
            )
            .order_by(
                Conversation.updated_at.desc(),
                Conversation.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )

        return list(result.scalars().all())

    async def update(
        self,
        conversation: Conversation,
        title: str | None,
    ) -> Conversation:
        conversation.title = title

        await self.session.flush()

        return conversation

    async def delete(
        self,
        conversation: Conversation,
    ) -> None:
        await self.session.delete(conversation)
        await self.session.flush()
