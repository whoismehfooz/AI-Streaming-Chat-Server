import json
import logging
from collections.abc import AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service
from app.exceptions.custom_exceptions import (
    AIProviderException,
    ConversationNotFoundException,
)
from app.schemas.chat import ChatRequest
from app.services import ChatService


logger = logging.getLogger(__name__)

chat_router = APIRouter(
    prefix="/conversations",
    tags=["Chat"],
)


def _encode_sse_event(event: str, data: str) -> str:
    """
    Encode one Server-Sent Event.

    JSON encoding keeps arbitrary generated text, including embedded
    newlines and Unicode characters, inside a valid single-line data field.
    """
    encoded_data = json.dumps({"text": data}, ensure_ascii=False)
    return f"event: {event}\ndata: {encoded_data}\n\n"


def _encode_sse_error(message: str) -> str:
    """Encode a safe, structured error event."""
    encoded_data = json.dumps(
        {"message": message},
        ensure_ascii=False,
    )
    return f"event: error\ndata: {encoded_data}\n\n"


@chat_router.post(
    "/{conversation_id}/chat/stream",
    status_code=status.HTTP_200_OK,
)
async def stream_chat_message(
    conversation_id: UUID,
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Stream an assistant response using Server-Sent Events (SSE).

    Event contract:
      - event: token  -> data: {"text": "..."}
      - event: done   -> data: {"text": "[DONE]"}
      - event: error  -> data: {"message": "..."}

    Once the HTTP response has started, its status code cannot be changed.
    Failures occurring during stream consumption are therefore delivered
    through an SSE error event.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in chat_service.stream_message(
                conversation_id=conversation_id,
                content=request.content,
            ):
                yield _encode_sse_event("token", chunk)

            yield _encode_sse_event("done", "[DONE]")

        except ConversationNotFoundException:
            logger.warning(
                "Streaming requested for a missing conversation: %s",
                conversation_id,
            )

            yield _encode_sse_error("Conversation Not Found.")

        except AIProviderException as exc:
            logger.warning(
                (
                    "AI provider failed during streaming: "
                    "conversation=%s status_code=%s"
                ),
                conversation_id,
                exc.status_code,
            )
            yield _encode_sse_error(str(exc))

        except GeneratorExit:
            logger.debug(
                "SSE consumer closed the stream: conversation=%s",
                conversation_id,
            )
            raise

        except Exception:
            logger.exception(
                "Unexpected error during SSE streaming: conversation=%s",
                conversation_id,
            )

            yield _encode_sse_error(
                "An unexpected streaming error occurred."
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
