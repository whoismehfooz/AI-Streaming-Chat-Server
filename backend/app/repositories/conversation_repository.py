import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation



class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session  = session

    async def create(
            self,
            title: str | None = None
    ) -> Conversation:

        conversation = Conversation(title=title)

        self.session.add(conversation)
        await self.session.flush()

        return conversation

    async def get_by_id(
            self,
            conversation_id: uuid.UUID
    ) -> Conversation | None:

        result = await self.session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id
            )
        )

        return result.scalar_one_or_none()

    async def list(
            self,
            limit: int,
            offset: int
    ) -> list[Conversation]:

        result = await self.session.execute(
            select(Conversation)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )

        return list(result.scalars().all())

    async def delete(
            self,
            conversation: Conversation
    ) -> None:

        await self.session.delete(conversation)
        await self.session.flush()
