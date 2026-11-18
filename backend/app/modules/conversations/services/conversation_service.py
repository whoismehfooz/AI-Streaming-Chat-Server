from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions.resources import ConversationNotFoundException
from app.modules.conversations.models.conversation import Conversation
from app.modules.conversations.repositories.conversation_repository import (
    ConversationRepository,
)


logger = logging.getLogger(__name__)
_MAX_TITLE_LENGTH = 255


class ConversationService:

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = ConversationRepository(session)

    async def _rollback_if_owned(self, *, commit: bool) -> None:
        if not commit:
            return

        try:
            await self.session.rollback()
        except Exception:
            logger.exception(
                "Rollback failed while handling a conversation operation."
            )

    @staticmethod
    def _normalize_title(title: str | None) -> str | None:
        if title is None:
            return None

        if not isinstance(title, str):
            raise ValueError("Conversation title must be a string or None.")

        normalized_title = title.strip()

        if not normalized_title:
            return None

        if len(normalized_title) > _MAX_TITLE_LENGTH:
            raise ValueError("Conversation title cannot exceed 255 characters.")

        return normalized_title

    @staticmethod
    def _validate_pagination(*, limit: int, offset: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Limit must be a positive integer.")

        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("Offset must be a non-negative integer.")

    async def _finish_write(
        self,
        conversation: Conversation,
        *,
        commit: bool,
    ) -> Conversation:
        if commit:
            await self.session.commit()
            await self.session.refresh(conversation)

        return conversation

    async def create_conversation(
        self,
        user_id: uuid.UUID,
        title: str | None = None,
        *,
        commit: bool = True,
    ) -> Conversation:

        normalized_title = self._normalize_title(title)

        try:
            conversation = await self.repository.create(
                user_id=user_id,
                title=normalized_title,
            )

            return await self._finish_write(conversation, commit=commit)

        except BaseException:
            await self._rollback_if_owned(commit=commit)
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
            raise ConversationNotFoundException()

        return conversation

    async def list_conversations(
        self,
        user_id: uuid.UUID,
        limit: int,
        offset: int,
    ) -> list[Conversation]:

        self._validate_pagination(limit=limit, offset=offset)

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
        *,
        commit: bool = True,
    ) -> Conversation:

        normalized_title = self._normalize_title(title)

        try:
            conversation = await self.repository.get_by_id_for_user_for_update(
                conversation_id=conversation_id,
                user_id=user_id,
            )

            if conversation is None:
                raise ConversationNotFoundException()

            updated_conversation = await self.repository.update(
                conversation=conversation,
                title=normalized_title,
            )

            return await self._finish_write(
                updated_conversation,
                commit=commit,
            )

        except BaseException:
            await self._rollback_if_owned(commit=commit)
            raise

    async def delete_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        commit: bool = True,
    ) -> None:

        try:
            conversation = await self.repository.get_by_id_for_user_for_update(
                conversation_id=conversation_id,
                user_id=user_id,
            )

            if conversation is None:
                raise ConversationNotFoundException()

            await self.repository.delete(conversation)

            if commit:
                await self.session.commit()

        except BaseException:
            await self._rollback_if_owned(commit=commit)
            raise
