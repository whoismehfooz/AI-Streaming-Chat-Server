import logging
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services import ChatService
from app.api.dependencies import get_chat_service

from app.exceptions.custom_exceptions import ConversationNotFoundException, AIProviderException

logger = logging.getLogger(__name__)

chat_router = APIRouter(
    prefix="/conversations",
    tags=["Chat"]
)


@chat_router.post("/{conversation_id}/chat/stream",status_code=status.HTTP_200_OK)
async def stream_chat_message(
    conversation_id: UUID,
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    async def event_generator():
        try:
            async for token in chat_service.stream_message(
                conversation_id=conversation_id,
                content=request.content
            ):
                yield f"data: {token}\n\n"

            yield "data: [DONE]\n\n"

        except ConversationNotFoundException as exc:
            logger.warning("Stream failed - Conversation not found: %s",
                        conversation_id)
            yield f"data: [ERROR: {str(exc)}]\n\n"

        except AIProviderException as exc:
            logger.error("Stream failed due to AI Provider error on conversation %s: %s",
                        conversation_id,
                        str(exc))
            yield f"data: [ERROR: {str(exc)}]\n\n"

        except Exception as exc:
            logger.error("Unexpected error during SSE stream for conversation %s: %s",
                        conversation_id)
            yield f"data: [ERROR: An unexpected streaming error occured.]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
