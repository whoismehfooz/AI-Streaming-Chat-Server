from uuid import UUID
from typing import AsyncGenerator
import logging

from app.services.ai_service import AIService
from app.services.conversation_service import ConversationService
from app.services.message_service import MessageService
from app.services.context_builder import ContextBuilder

from app.schemas.chat import ChatResponse
from app.exceptions.custom_exceptions import ConversationNotFoundException

logger = logging.getLogger(__name__)


class ChatService:

    def __init__(
            self,
            conversation_service: ConversationService,
            message_service: MessageService,
            context_builder: ContextBuilder,
            ai_service: AIService
    ):
        self.conversation_service = conversation_service
        self.message_service = message_service
        self.context_builder = context_builder
        self.ai_service = ai_service


    async def stream_message(
            self,
            conversation_id: UUID,
            content: str
    ) -> AsyncGenerator[str,None]:

        conversation = await self.conversation_service.get_conversation(
            conversation_id
        )

        if conversation is None:
            raise ConversationNotFoundException(
                "Conversation Not Found.",
                status_code=404
            )

        user_message = await self.message_service.create_message(
            conversation_id=conversation_id,
            role="user",
            content=content
        )
        try:
            messages = await self.message_service.list_recent_messages(
                conversation_id=conversation_id,
                limit=100
            )

            context = self.context_builder.build(messages)

        except Exception as exc:
            logger.error("Failed to build context for streaming conversation %s: %s",
                        conversation_id,
                        str(exc))
            raise

        full_assistant_content = ""

        try: 
            async for token in self.ai_service.stream_response(context):
                full_assistant_content += token
                yield token

        except Exception as exc:
            logger.error("Streaming intrrupted for conversation %s after %d chars: %s",
                        conversation_id,
                        len(full_assistant_content),
                        str(exc))

            if full_assistant_content:
                try:
                    await self.message_service.create_message(
                        conversation_id,
                        role="assistant",
                        content=full_assistant_content + "\n\n[Stream Interrupted]"
                    )
                except Exception as save_exc:
                    logger.error("Failed to save partial message: %s", 
                                str(save_exc))
            raise 

        if full_assistant_content:
            await self.message_service.create_message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_assistant_content
            )
