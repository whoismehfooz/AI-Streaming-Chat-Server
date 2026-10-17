import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.repositories.conversation_repository import ConversationRepository


class ConversationService:
    def __init__(self,session: AsyncSession):
        self.session = session
        self.repository = ConversationRepository(session)


    async def create_conversation(
            self,
            title: str | None = None
    ) -> Conversation:

        conversation = await self.repository.create(
            title=title
        )

        await self.session.commit()
        await self.session.refresh(conversation)

        return conversation

    async def get_conversation(
            self,
            conversation_id: uuid.UUID
    ) -> Conversation | None:

        return await self.repository.get_by_id(
            conversation_id
        )

    async def list_conversations(
            self,
            limit: int,
            offset: int
    ) -> list[Conversation]:

        return await self.repository.list(
            limit=limit,
            offset=offset
        )

    async def delete_conversation(
            self,
            conversation_id: uuid.UUID
    ) -> bool:

        conversation = await self.repository.get_by_id(
            conversation_id
        )

        if conversation is None:
            return False

        await self.repository.delete(conversation)
        await self.session.commit()

        return True
