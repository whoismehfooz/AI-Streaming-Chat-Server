from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation
from app.repositories.message_repository import MessageRepository

from app.exceptions.custom_exceptions import ConversationNotFoundException

class MessageService:
    def __init__(self,session: AsyncSession):
        self.session = session
        self.repository = MessageRepository(session)

    async def create_message(
            self,
            conversation_id,
            role,
            content
    ):
        conversation = await self.session.get(
            Conversation,
            conversation_id
        )

        if conversation is None:
            raise ConversationNotFoundException(
                "Conversation Not Found.",
                status_code=404
            )

        message = await self.repository.create(
            conversation_id=conversation_id,
            role=role,
            content=content
        )

        conversation.updated_at = datetime.now(timezone.utc)

        await self.session.commit()
        await self.session.refresh(message)

        return message

    async def get_message(self, message_id):
        return await self.repository.get_by_id(message_id)

    async def list_messages(
            self,
            conversation_id,
            limit,
            offset,
    ):
        return await self.repository.list_by_conversation(
            conversation_id=conversation_id,
            limit=limit,
            offset=offset
        )

    async def list_recent_messages(
            self,
            conversation_id,
            limit
    ):
        return await self.repository.list_recent_by_conversation(
            conversation_id=conversation_id,
            limit=limit
        )
