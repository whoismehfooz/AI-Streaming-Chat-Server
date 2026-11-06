import uuid

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions.custom_exceptions import ConversationNotFoundException
from app.models.message import Message
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository


class MessageService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = MessageRepository(session)
        self.conversation_repository = ConversationRepository(session)

    @staticmethod
    def _validate_content(content: str) -> None:
        if not content or not content.strip():
            raise ValueError("Message content cannot be empty or whitespace.")

    @staticmethod
    def _validate_pagination(
        limit: int,
        offset: int,
    ) -> None:
        if limit < 1:
            raise ValueError("Limit must be greater than zero.")

        if offset < 0:
            raise ValueError("Offset cannot be negative.")

    async def create_message(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        content: str,
    ) -> Message:
        self._validate_content(content)

        try:
            conversation = await self.conversation_repository.get_by_id_for_user(
                conversation_id=conversation_id,
                user_id=user_id,
            )

            if conversation is None:
                raise ConversationNotFoundException(
                    "Conversation not found.",
                    status_code=404,
                )

            conversation.updated_at = func.now()

            message = await self.repository.create(
                conversation_id=conversation_id,
                role="user",
                content=content,
            )

            await self.session.commit()
            await self.session.refresh(message)

            return message

        except Exception:
            await self.session.rollback()
            raise

    async def create_assistant_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
    ) -> Message:
        self._validate_content(content)

        try:
            conversation = await self.conversation_repository.get_by_id(
                conversation_id=conversation_id,
            )

            if conversation is None:
                raise ConversationNotFoundException(
                    "Conversation not found.",
                    status_code=404,
                )

            conversation.updated_at = func.now()

            message = await self.repository.create(
                conversation_id=conversation_id,
                role="assistant",
                content=content,
            )

            await self.session.commit()
            await self.session.refresh(message)

            return message

        except Exception:
            await self.session.rollback()
            raise

    async def get_message(
        self,
        message_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Message:
        message = await self.repository.get_by_id_for_user(
            message_id=message_id,
            user_id=user_id,
        )

        if message is None:
            raise ConversationNotFoundException(
                "Message not found.",
                status_code=404,
            )

        return message

    async def list_messages(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int,
        offset: int,
    ) -> list[Message]:
        self._validate_pagination(
            limit=limit,
            offset=offset,
        )

        conversation = await self.conversation_repository.get_by_id_for_user(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if conversation is None:
            raise ConversationNotFoundException(
                "Conversation not found.",
                status_code=404,
            )

        return await self.repository.list_by_conversation_for_user(
            conversation_id=conversation_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

    async def list_recent_messages(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int,
    ) -> list[Message]:
        if limit < 1:
            raise ValueError("Limit must be greater than zero.")

        conversation = await self.conversation_repository.get_by_id_for_user(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if conversation is None:
            raise ConversationNotFoundException(
                "Conversation not found.",
                status_code=404,
            )

        return await self.repository.list_recent_by_conversation(
            conversation_id=conversation_id,
            limit=limit,
        )
