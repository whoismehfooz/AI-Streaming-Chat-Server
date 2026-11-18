from __future__ import annotations

import logging
import uuid
from typing import Final

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions.resources import (
    ConversationNotFoundException,
    MessageNotFoundException,
)
from app.modules.conversations.models.conversation import Conversation
from app.modules.conversations.models.message import Message
from app.modules.conversations.repositories.conversation_repository import (
    ConversationRepository,
)
from app.modules.conversations.repositories.message_repository import (
    MessageRepository,
)


logger = logging.getLogger(__name__)

USER_MESSAGE_ROLE: Final[str] = "user"
ASSISTANT_MESSAGE_ROLE: Final[str] = "assistant"


class MessageService:

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = MessageRepository(session)
        self.conversation_repository = ConversationRepository(session)

    async def _rollback_if_owned(self, *, commit: bool) -> None:
        if not commit:
            return

        try:
            await self.session.rollback()
        except Exception:
            logger.exception("Rollback failed while handling a message operation.")

    @staticmethod
    def _validate_content(content: str) -> None:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content cannot be empty or whitespace.")

    @staticmethod
    def _validate_pagination(*, limit: int, offset: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Limit must be a positive integer.")

        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("Offset must be a non-negative integer.")

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Limit must be a positive integer.")

    async def _get_owned_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Conversation:
        conversation = await self.conversation_repository.get_by_id_for_user(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if conversation is None:
            raise ConversationNotFoundException()

        return conversation

    async def _finish_write(
        self,
        message: Message,
        *,
        commit: bool,
    ) -> Message:
        if commit:
            await self.session.commit()
            await self.session.refresh(message)

        return message

    async def create_message(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        content: str,
        *,
        commit: bool = True,
    ) -> Message:
        self._validate_content(content)

        try:
            conversation = await self._get_owned_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
            )

            conversation.updated_at = func.now()

            message = await self.repository.create(
                conversation_id=conversation_id,
                role=USER_MESSAGE_ROLE,
                content=content,
            )

            return await self._finish_write(message, commit=commit)

        except BaseException:
            await self._rollback_if_owned(commit=commit)
            raise

    async def create_assistant_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
        *,
        user_id: uuid.UUID | None = None,
        commit: bool = True,
    ) -> Message:
        self._validate_content(content)

        try:
            if user_id is None:
                conversation = await self.conversation_repository.get_by_id(
                    conversation_id=conversation_id,
                )
            else:
                conversation = await self.conversation_repository.get_by_id_for_user(
                    conversation_id=conversation_id,
                    user_id=user_id,
                )

            if conversation is None:
                raise ConversationNotFoundException()

            conversation.updated_at = func.now()

            message = await self.repository.create(
                conversation_id=conversation_id,
                role=ASSISTANT_MESSAGE_ROLE,
                content=content,
            )

            return await self._finish_write(message, commit=commit)

        except BaseException:
            await self._rollback_if_owned(commit=commit)
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
            raise MessageNotFoundException()

        return message

    async def list_messages(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int,
        offset: int,
    ) -> list[Message]:
        self._validate_pagination(limit=limit, offset=offset)

        await self._get_owned_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
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
        self._validate_limit(limit)

        await self._get_owned_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        return await self.repository.list_recent_by_conversation_for_user(
            conversation_id=conversation_id,
            user_id=user_id,
            limit=limit,
        )

    async def delete_message(
        self,
        message_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        commit: bool = True,
    ) -> None:
        try:
            message = await self.repository.get_by_id_for_user(
                message_id=message_id,
                user_id=user_id,
            )

            if message is None:
                raise MessageNotFoundException()

            conversation = await self._get_owned_conversation(
                conversation_id=message.conversation_id,
                user_id=user_id,
            )
            conversation.updated_at = func.now()

            await self.repository.delete(message)

            if commit:
                await self.session.commit()

        except BaseException:
            await self._rollback_if_owned(commit=commit)
            raise
