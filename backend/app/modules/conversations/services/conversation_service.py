import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions.custom_exceptions import ConversationNotFoundException
from app.models.conversation import Conversation
from app.repositories.conversation_repository import ConversationRepository


class ConversationService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = ConversationRepository(session)

    @staticmethod
    def _normalize_title(title: str | None) -> str | None:
        if title is None:
            return None

        normalized_title = title.strip()

        if not normalized_title:
            return None

        if len(normalized_title) > 255:
            raise ValueError("Conversation title cannot exceed 255 characters.")

        return normalized_title

    async def create_conversation(
        self,
        user_id: uuid.UUID,
        title: str | None = None,
    ) -> Conversation:
        normalized_title = self._normalize_title(title)

        try:
            conversation = await self.repository.create(
                user_id=user_id,
                title=normalized_title,
            )

            await self.session.commit()
            await self.session.refresh(conversation)

            return conversation

        except Exception:
            await self.session.rollback()
            raise

    async def get_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Conversation:
        conversation = await self.repository.get_by_id_for_user(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if conversation is None:
            raise ConversationNotFoundException(
                "Conversation not found.",
                status_code=404,
            )

        return conversation

    async def list_conversations(
        self,
        user_id: uuid.UUID,
        limit: int,
        offset: int,
    ) -> list[Conversation]:
        if limit < 1:
            raise ValueError("Limit must be greater than zero.")

        if offset < 0:
            raise ValueError("Offset cannot be negative.")

        return await self.repository.list_by_user(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

    async def update_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        title: str | None,
    ) -> Conversation:
        normalized_title = self._normalize_title(title)

        try:
            conversation = await self.repository.get_by_id_for_user(
                conversation_id=conversation_id,
                user_id=user_id,
            )

            if conversation is None:
                raise ConversationNotFoundException(
                    "Conversation not found.",
                    status_code=404,
                )

            updated_conversation = await self.repository.update(
                conversation=conversation,
                title=normalized_title,
            )

            await self.session.commit()
            await self.session.refresh(updated_conversation)

            return updated_conversation

        except Exception:
            await self.session.rollback()
            raise

    async def delete_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        try:
            conversation = await self.repository.get_by_id_for_user(
                conversation_id=conversation_id,
                user_id=user_id,
            )

            if conversation is None:
                raise ConversationNotFoundException(
                    "Conversation not found.",
                    status_code=404,
                )

            await self.repository.delete(conversation)
            await self.session.commit()

        except Exception:
            await self.session.rollback()
            raise
