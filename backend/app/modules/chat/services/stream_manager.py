"""
Semantic lifecycle manager for one Chat streaming response.

StreamManager consumes normalized LLMStreamChunk objects from
ResponseGenerator and emits typed ChatStreamEvent objects. It owns:
    - semantic event ordering and per-request sequence numbers;
    - generated-content accumulation and output-size enforcement;
    - terminal completion/error behavior;
    - persistence callback coordination;
    - safe application/provider error mapping;
    - interruption notification on cancellation and consumer closure.

It does not serialize SSE, access FastAPI Request, perform SQL queries,
authorize conversation ownership, or create/close provider clients.

Persistence contract with the current MessageService:
    - The users message has already been committed before streaming starts.
    - MessageService creates assistant messages only after complete content
      is available and the database generates their UUID.
    - Therefore no assistant_message_id is accepted or emitted at stream
      start or for content deltas.
    - ``on_completed`` persists the complete assistant content and returns
      the real database-generated assistant message UUID.
    - The completion event is emitted only after that callback succeeds.
    - ``on_interrupted`` receives partial content and may apply the
      application's explicit partial-response persistence policy.

The manager never fabricates a database message identifier.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias
from uuid import UUID

from app.exceptions.base import ApplicationException
from app.exceptions.error_codes import ErrorCode
from app.integrations.llm.exceptions import LLMProviderException, LLMProviderResponseException
from app.integrations.llm.schemas import LLMFinishReason, LLMMessage, LLMUsage
from app.modules.chat.schemas.chat_request import ChatGenerationOptions
from app.modules.chat.schemas.stream_event import (
    ChatContentDeltaEvent,
    ChatStreamCompletedEvent,
    ChatStreamErrorEvent,
    ChatStreamEvent,
    ChatStreamStartedEvent,
)
from app.modules.chat.services.response_generator import ResponseGenerator


logger = logging.getLogger(__name__)


class StreamInterruptionReason(StrEnum):
    """Internal reason supplied to the interruption persistence callback."""

    CANCELLED = "cancelled"
    CONSUMER_CLOSED = "consumer_closed"
    PROVIDER_ERROR = "provider_error"
    PERSISTENCE_ERROR = "persistence_error"


@dataclass(frozen=True, slots=True)
class StreamResult:
    """
    Immutable snapshot supplied to ChatService callbacks.

    ``content`` may be partial when ``completed`` is false. It contains only
    model-generated text, never provider diagnostics or exception text.
    There is intentionally no assistant_message_id: persistence has not
    necessarily created an assistant record for an interrupted stream.
    """

    conversation_id: UUID
    user_message_id: UUID
    request_id: str
    content: str
    finish_reason: LLMFinishReason | None
    usage: LLMUsage | None
    provider_name: str | None
    model: str | None
    provider_request_id: str | None
    chunks_received: int
    completed: bool
    interruption_reason: StreamInterruptionReason | None = None


CompletionHandler: TypeAlias = Callable[[StreamResult], Awaitable[UUID]]
InterruptionHandler: TypeAlias = Callable[
    [StreamResult, BaseException | None],
    Awaitable[None],
]


class StreamManager:
    """
    Manage one generation stream from accepted users turn to terminal outcome.

    ``on_completed`` must persist the complete assistant response and return
    its real database UUID. ``on_interrupted`` owns the application's policy
    for partial generated content. Both callbacks are required so persistence
    behavior cannot be silently omitted by a caller.
    """

    def __init__(
        self,
        response_generator: ResponseGenerator,
        *,
        max_response_characters: int = 1_000_000,
    ) -> None:
        if not isinstance(response_generator, ResponseGenerator):
            raise TypeError(
                "response_generator must be a ResponseGenerator instance."
            )

        if (
            isinstance(max_response_characters, bool)
            or not isinstance(max_response_characters, int)
            or max_response_characters <= 0
        ):
            raise ValueError(
                "max_response_characters must be a positive integer."
            )

        self._response_generator = response_generator
        self._max_response_characters = max_response_characters

    async def stream(
        self,
        *,
        conversation_id: UUID,
        user_message_id: UUID,
        request_id: str,
        messages: Sequence[LLMMessage],
        generation_options: ChatGenerationOptions | None,
        on_completed: CompletionHandler,
        on_interrupted: InterruptionHandler,
    ) -> AsyncIterator[ChatStreamEvent]:
        """
        Yield ordered semantic events for one assistant response.

        Event sequence numbers are independent of provider chunk sequence
        numbers. They begin at zero and increment once per emitted event.
        """

        if not isinstance(conversation_id, UUID):
            raise TypeError("conversation_id must be a UUID.")
        if not isinstance(user_message_id, UUID):
            raise TypeError("user_message_id must be a UUID.")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string.")
        if not callable(on_completed) or not callable(on_interrupted):
            raise TypeError("Both stream persistence callbacks are required.")

        event_sequence = 0
        chunks_received = 0
        accumulated_characters = 0
        interruption_notified = False
        terminal_event_emitted = False

        content_parts: list[str] = []
        finish_reason: LLMFinishReason | None = None
        usage: LLMUsage | None = None
        provider_name: str | None = None
        model: str | None = None
        provider_request_id: str | None = None

        def snapshot(
            *,
            completed: bool,
            interruption_reason: StreamInterruptionReason | None = None,
        ) -> StreamResult:
            return StreamResult(
                conversation_id=conversation_id,
                user_message_id=user_message_id,
                request_id=request_id,
                content="".join(content_parts),
                finish_reason=finish_reason,
                usage=usage,
                provider_name=provider_name,
                model=model,
                provider_request_id=provider_request_id,
                chunks_received=chunks_received,
                completed=completed,
                interruption_reason=interruption_reason,
            )

        async def notify_interrupted(
            reason: StreamInterruptionReason,
            error: BaseException | None,
        ) -> None:
            nonlocal interruption_notified

            if interruption_notified:
                return

            # Set before awaiting to prevent duplicate persistence attempts if
            # cancellation occurs while the callback is executing.
            interruption_notified = True

            try:
                await on_interrupted(
                    snapshot(
                        completed=False,
                        interruption_reason=reason,
                    ),
                    error,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Chat stream interruption callback failed: conversation_id=%s request_id=%s reason=%s",
                    conversation_id,
                    request_id,
                    reason.value,
                )

        def make_error_event(
            error: BaseException,
            *,
            persistence_failure: bool = False,
        ) -> ChatStreamErrorEvent:
            if persistence_failure:
                error_code = ErrorCode.PERSISTENCE_ERROR
                public_message = "The generated response could not be saved."
                retryable = False
                retry_after_seconds = None

            elif isinstance(error, ApplicationException):
                error_code = error.error_code
                public_message = error.message
                retryable = (
                    error.retryable
                    if isinstance(error, LLMProviderException)
                    else False
                )
                retry_after_seconds = (
                    error.retry_after_seconds
                    if isinstance(error, LLMProviderException)
                    else None
                )

            else:
                error_code = ErrorCode.INTERNAL_SERVER_ERROR
                public_message = "The response could not be completed."
                retryable = False
                retry_after_seconds = None

            return ChatStreamErrorEvent(
                conversation_id=conversation_id,
                request_id=request_id,
                sequence_number=event_sequence,
                error_code=error_code,
                message=public_message,
                retryable=retryable,
                retry_after_seconds=retry_after_seconds,
            )

        try:
            yield ChatStreamStartedEvent(
                conversation_id=conversation_id,
                request_id=request_id,
                sequence_number=event_sequence,
                user_message_id=user_message_id,
            )
            event_sequence += 1

            async for chunk in self._response_generator.stream(
                messages,
                generation_options=generation_options,
                request_id=request_id,
            ):
                chunks_received += 1
                provider_name = chunk.provider_name
                model = chunk.model
                provider_request_id = (
                    chunk.provider_request_id or provider_request_id
                )
                finish_reason = chunk.finish_reason or finish_reason
                usage = chunk.usage or usage

                if not chunk.delta:
                    continue

                next_character_count = accumulated_characters + len(chunk.delta)
                if next_character_count > self._max_response_characters:
                    raise LLMProviderResponseException(
                        message=(
                            "The generated response exceeded the configured "
                            "output size limit."
                        ),
                        provider_name=chunk.provider_name,
                        provider_request_id=chunk.provider_request_id,
                    )

                content_parts.append(chunk.delta)
                accumulated_characters = next_character_count

                yield ChatContentDeltaEvent(
                    conversation_id=conversation_id,
                    request_id=request_id,
                    sequence_number=event_sequence,
                    delta=chunk.delta,
                )
                event_sequence += 1

            final_content = "".join(content_parts)
            if not final_content.strip():
                raise LLMProviderResponseException(
                    message="The LLM provider completed without producing text.",
                    provider_name=provider_name or self._response_generator.provider_name,
                    provider_request_id=provider_request_id,
                )

            completed_result = snapshot(completed=True)

            try:
                assistant_message_id = await on_completed(completed_result)
                if not isinstance(assistant_message_id, UUID):
                    raise TypeError(
                        "on_completed must return the persisted assistant message UUID."
                    )
            except asyncio.CancelledError:
                raise
            except Exception as persistence_error:
                logger.exception(
                    "Failed to persist completed assistant response: conversation_id=%s request_id=%s",
                    conversation_id,
                    request_id,
                )
                await notify_interrupted(
                    StreamInterruptionReason.PERSISTENCE_ERROR,
                    persistence_error,
                )
                yield make_error_event(
                    persistence_error,
                    persistence_failure=True,
                )
                terminal_event_emitted = True
                return

            # Mark terminal before yielding. If the consumer closes the
            # iterator immediately after receiving this event, the GeneratorExit
            # handler must not misclassify a successfully persisted turn as
            # interrupted.
            terminal_event_emitted = True
            yield ChatStreamCompletedEvent(
                conversation_id=conversation_id,
                request_id=request_id,
                sequence_number=event_sequence,
                assistant_message_id=assistant_message_id,
                finish_reason=finish_reason,
                usage=usage,
            )

        except asyncio.CancelledError:
            logger.info(
                "Chat stream cancelled: conversation_id=%s request_id=%s chunks=%d",
                conversation_id,
                request_id,
                chunks_received,
            )
            await notify_interrupted(
                StreamInterruptionReason.CANCELLED,
                None,
            )
            raise

        except GeneratorExit:
            logger.info(
                "Chat stream consumer closed: conversation_id=%s request_id=%s chunks=%d",
                conversation_id,
                request_id,
                chunks_received,
            )
            await notify_interrupted(
                StreamInterruptionReason.CONSUMER_CLOSED,
                None,
            )
            raise

        except Exception as error:
            logger.exception(
                "Chat stream failed: conversation_id=%s request_id=%s chunks=%d",
                conversation_id,
                request_id,
                chunks_received,
            )
            await notify_interrupted(
                StreamInterruptionReason.PROVIDER_ERROR,
                error,
            )

            if not terminal_event_emitted:
                yield make_error_event(error)
                terminal_event_emitted = True


__all__ = [
    "StreamInterruptionReason",
    "StreamResult",
    "CompletionHandler",
    "InterruptionHandler",
    "StreamManager",
]
