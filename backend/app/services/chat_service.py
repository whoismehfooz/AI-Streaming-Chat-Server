
import asyncio
import logging
from collections.abc import AsyncGenerator
from uuid import UUID

from app.exceptions.custom_exceptions import (
    AIProviderException,
    ConversationNotFoundException,
)
from app.services.ai_service import AIService
from app.services.context_builder import ContextBuilder
from app.services.conversation_service import ConversationService
from app.services.message_service import MessageService


logger = logging.getLogger(__name__)


class ChatService:

    def __init__(
        self,
        conversation_service: ConversationService,
        message_service: MessageService,
        context_builder: ContextBuilder,
        ai_service: AIService,
    ) -> None:

        self.conversation_service = conversation_service
        self.message_service = message_service
        self.context_builder = context_builder
        self.ai_service = ai_service

    async def stream_message(
        self,
        conversation_id: UUID,
        content: str,
    ) -> AsyncGenerator[str, None]:

        conversation = await self.conversation_service.get_conversation(
            conversation_id
        )

        if conversation is None:
            raise ConversationNotFoundException(
                "Conversation Not Found.",
                status_code=404,
            )

        await self.message_service.create_message(
            conversation_id=conversation_id,
            role="user",
            content=content,
        )

        try:
            recent_messages = (
                await self.message_service.list_recent_messages(
                    conversation_id=conversation_id,
                    limit=100,
                )
            )

            context = self.context_builder.build(recent_messages)

        except Exception:
            logger.exception(
                "Failed to build LLM context for conversation %s",
                conversation_id,
            )
            raise

        response_chunks: list[str] = []
        stream_completed = False

        try:
            async for chunk in self.ai_service.stream_response(context):

                if not isinstance(chunk, str) or not chunk:
                    continue

                response_chunks.append(chunk)

                yield chunk

            stream_completed = True

        except asyncio.CancelledError:

            logger.info(
                (
                    "Chat stream cancelled: conversation=%s "
                    "partial_characters=%d"
                ),
                conversation_id,
                sum(len(part) for part in response_chunks),
            )

            await self._persist_partial_response(
                conversation_id=conversation_id,
                response_chunks=response_chunks,
                reason="cancelled",
            )
            raise

        except Exception:
            logger.exception(
                (
                    "Chat stream failed: conversation=%s "
                    "partial_characters=%d"
                ),
                conversation_id,
                sum(len(part) for part in response_chunks),
            )

            await self._persist_partial_response(
                conversation_id=conversation_id,
                response_chunks=response_chunks,
                reason="interrupted",
            )
            raise

        finally:
            if not stream_completed:
                logger.debug(
                    "Chat stream ended without normal completion: conversation=%s",
                    conversation_id,
                )

        full_response = "".join(response_chunks)

        if not full_response:
            logger.warning(
                "LLM stream completed without text: conversation=%s",
                conversation_id,
            )
            return

        try:
            await self.message_service.create_message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
            )

        except Exception:

            logger.exception(
                (
                    "Failed to persist completed assistant response: "
                    "conversation=%s characters=%d"
                ),
                conversation_id,
                len(full_response),
            )
            raise

        logger.info(
            (
                "Chat stream completed and persisted: conversation=%s "
                "characters=%d chunks=%d"
            ),
            conversation_id,
            len(full_response),
            len(response_chunks),
        )

    async def _persist_partial_response(
        self,
        conversation_id: UUID,
        response_chunks: list[str],
        reason: str,
    ) -> None:

        if not response_chunks:
            return

        partial_response = "".join(response_chunks)

        marked_response = (
            f"{partial_response}\n\n[Stream {reason.capitalize()}]"
        )

        try:
            await self.message_service.create_message(
                conversation_id=conversation_id,
                role="assistant",
                content=marked_response,
            )

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                (
                    "Failed to persist partial assistant response: "
                    "conversation=%s reason=%s characters=%d"
                ),
                conversation_id,
                reason,
                len(partial_response),
            )
