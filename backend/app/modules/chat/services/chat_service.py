"""
Authenticated application service for Chat turns and SSE streaming.

ChatService coordinates the existing User/Conversation/Message domain
services with ChatOrchestrator, StreamManager, and SSEStreamAdapter.

The actual Conversations-domain contracts are used directly:
    ConversationService.get_conversation(conversation_id, user_id)
    MessageService.list_recent_messages(conversation_id, user_id, limit)
    MessageService.create_message(conversation_id, user_id, content)
    MessageService.create_assistant_message(conversation_id, content)

Ownership is enforced by the existing scoped ConversationService and
MessageService methods. The authenticated users ID must come from the
authentication dependency; it is never accepted from ChatRequest.

The MessageService commits each write independently. Therefore a users
message remains persisted if generation later fails. A successful assistant
message is persisted before the final Chat response or stream completion
event is returned.

Generation entitlement/quota checks and cross-worker concurrency control
are delegated to a mandatory ChatExecutionPolicy implementation. This file
defines the required port but does not pretend that an in-process lock is a
distributed concurrency control mechanism.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from app.exceptions.base import ApplicationException
from app.exceptions.error_codes import ErrorCode
from app.exceptions.resources import ConversationNotFoundException
from app.modules.chat.schemas.chat_request import (
    ChatGenerationOptions,
    ChatRequest,
)
from app.modules.chat.schemas.chat_response import ChatResponse
from app.modules.chat.services.chat_orchestrator import (
    ChatOrchestrator,
    PreparedChatContext,
)
from app.modules.chat.services.stream_adapter import SSEStreamAdapter
from app.modules.chat.services.stream_manager import (
    StreamInterruptionReason,
    StreamResult,
)

logger = logging.getLogger(__name__)


class ChatPersistenceException(ApplicationException):
    """Raised when a Chat message cannot be safely persisted."""

    def __init__(
        self,
        message: str = "The chat response could not be saved.",
    ) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code=ErrorCode.PERSISTENCE_ERROR,
        )


class ChatGenerationLimitException(ApplicationException):
    """Raised when a client requests an output limit above application policy."""

    def __init__(
        self,
        message: str = "The requested output token limit exceeds the allowed maximum.",
    ) -> None:
        super().__init__(
            message=message,
            status_code=422,
            error_code=ErrorCode.REQUEST_VALIDATION_ERROR,
        )


@runtime_checkable
class ConversationRecord(Protocol):
    """Minimum conversation model surface required by ChatService."""

    id: UUID


@runtime_checkable
class MessageRecord(Protocol):
    """Minimum persisted message model surface required by ChatService."""

    id: UUID
    conversation_id: UUID
    role: str
    content: str
    created_at: datetime


class ConversationServicePort(Protocol):
    """Ownership-scoped interface implemented by the existing ConversationService."""

    async def get_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID,
    ) -> ConversationRecord:
        ...


class MessageServicePort(Protocol):
    """Exact message operations exposed by the existing MessageService."""

    async def create_message(
        self,
        conversation_id: UUID,
        user_id: UUID,
        content: str,
    ) -> MessageRecord:
        ...

    async def create_assistant_message(
        self,
        conversation_id: UUID,
        content: str,
    ) -> MessageRecord:
        ...

    async def list_recent_messages(
        self,
        conversation_id: UUID,
        user_id: UUID,
        limit: int,
    ) -> list[MessageRecord]:
        ...


class ChatExecutionPolicy(Protocol):
    """
    Mandatory generation authorization and concurrency boundary.

    A concrete implementation must:
        - enforce model/feature entitlement and usage/token/cost quotas;
        - enforce per-users and per-conversation generation limits;
        - provide a generation slot coordinated across all application
          workers, using a shared mechanism appropriate to deployment;
        - release the slot on normal completion, exception, and cancellation.

    It should raise the existing authorization exceptions, for example
    AIUsageQuotaExceededException, AITokenLimitExceededException,
    AIModelAccessDeniedException, AIConcurrentGenerationLimitExceededException,
    or AIRequestLimitExceededException.

    Conversation ownership is not duplicated here: the scoped
    ConversationService and MessageService methods already enforce it.
    """

    async def authorize_generation(
        self,
        *,
        user_id: UUID,
        conversation: ConversationRecord,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        ...

    def generation_slot(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> AbstractAsyncContextManager[None]:
        ...


DisconnectCheck = Callable[[], Awaitable[bool]]


class ChatService:
    """
    Execute authenticated non-streaming and SSE Chat turns.

    The mandatory execution policy is intentionally injected. This service
    will not silently substitute a process-local asyncio.Lock for a
    deployment-wide concurrency guarantee.
    """

    def __init__(
        self,
        conversation_service: ConversationServicePort,
        message_service: MessageServicePort,
        chat_orchestrator: ChatOrchestrator,
        stream_adapter: SSEStreamAdapter,
        execution_policy: ChatExecutionPolicy,
        *,
        history_limit: int = 100,
        default_max_output_tokens: int = 512,
        maximum_output_tokens: int = 4096,
        interrupted_response_marker: bool = True,
    ) -> None:
        for name, value in (
            ("history_limit", history_limit),
            ("default_max_output_tokens", default_max_output_tokens),
            ("maximum_output_tokens", maximum_output_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if default_max_output_tokens > maximum_output_tokens:
            raise ValueError(
                "default_max_output_tokens cannot exceed maximum_output_tokens."
            )
        if not isinstance(interrupted_response_marker, bool):
            raise TypeError("interrupted_response_marker must be a bool.")

        self._conversation_service = conversation_service
        self._message_service = message_service
        self._chat_orchestrator = chat_orchestrator
        self._stream_adapter = stream_adapter
        self._execution_policy = execution_policy
        self._history_limit = history_limit
        self._default_max_output_tokens = default_max_output_tokens
        self._maximum_output_tokens = maximum_output_tokens
        self._interrupted_response_marker = interrupted_response_marker

    @staticmethod
    def _validate_identity(
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> None:
        if not isinstance(user_id, UUID):
            raise TypeError("user_id must be the authenticated users's UUID.")
        if not isinstance(conversation_id, UUID):
            raise TypeError("conversation_id must be a UUID.")

    @staticmethod
    def _validate_request(request: ChatRequest) -> None:
        if not isinstance(request, ChatRequest):
            raise TypeError("request must be a ChatRequest instance.")

    def _output_token_limit(self, request: ChatRequest) -> int:
        generation = request.generation
        requested = (
            generation.max_output_tokens
            if generation is not None
            and generation.max_output_tokens is not None
            else self._default_max_output_tokens
        )

        if requested > self._maximum_output_tokens:
            raise ChatGenerationLimitException()

        return requested

    async def _get_owned_conversation(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> ConversationRecord:
        """
        Resolve a conversation through the ownership-scoped domain service.

        The service intentionally receives both IDs. A missing or
        non-owned conversation is represented by the existing
        ConversationNotFoundException, preventing resource enumeration.
        """

        conversation = await self._conversation_service.get_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if conversation is None:
            # Defensive compatibility for a service implementation that
            # returns None rather than raising the documented not-found error.
            raise ConversationNotFoundException()

        actual_conversation_id = getattr(conversation, "id", None)
        if not isinstance(actual_conversation_id, UUID):
            raise ChatPersistenceException(
                "The conversation service returned an invalid conversation record."
            )
        if actual_conversation_id != conversation_id:
            raise ChatPersistenceException(
                "The conversation service returned a mismatched conversation record."
            )

        return conversation

    @staticmethod
    def _effective_generation_options(
        request: ChatRequest,
        output_tokens: int,
    ) -> ChatGenerationOptions:
        """
        Resolve the output limit once for budgeting, policy, and provider I/O.

        This prevents ChatService's context reservation from disagreeing with
        ResponseGenerator's configured default when the client omits
        max_output_tokens.
        """

        options = request.generation or ChatGenerationOptions()
        return options.model_copy(update={"max_output_tokens": output_tokens})

    async def _prepare_and_persist_user_message(
        self,
        *,
        user_id: UUID,
        conversation: ConversationRecord,
        request: ChatRequest,
        output_tokens: int,
    ) -> tuple[PreparedChatContext, MessageRecord]:
        """
        Build and authorize the final context before persisting the users turn.

        The pending users message is included in context before it is written.
        This lets token-budget and generation-entitlement checks fail without
        creating a durable users message. MessageService rechecks conversation
        ownership when the users message is finally persisted.
        """

        recent_messages = await self._message_service.list_recent_messages(
            conversation_id=conversation.id,
            user_id=user_id,
            limit=self._history_limit,
        )

        pending_user_message = {
            "role": "users",
            "content": request.content,
        }

        prepared_context = self._chat_orchestrator.prepare(
            (*recent_messages, pending_user_message),
            reserved_output_tokens=output_tokens,
        )

        await self._execution_policy.authorize_generation(
            user_id=user_id,
            conversation=conversation,
            input_tokens=prepared_context.input_tokens,
            output_tokens=output_tokens,
        )

        user_message = await self._message_service.create_message(
            conversation_id=conversation.id,
            user_id=user_id,
            content=request.content,
        )

        if not isinstance(getattr(user_message, "id", None), UUID):
            raise ChatPersistenceException(
                "The message service returned an invalid users message identifier."
            )
        if getattr(user_message, "conversation_id", None) != conversation.id:
            raise ChatPersistenceException(
                "The message service returned a users message for another conversation."
            )
        if getattr(user_message, "role", None) != "users":
            raise ChatPersistenceException(
                "The message service did not persist the users turn with the users role."
            )

        return prepared_context, user_message

    async def send_message(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        request: ChatRequest,
        request_id: str | None = None,
    ) -> ChatResponse:
        """
        Execute one non-streaming turn and persist the assistant response.

        The users message is committed before calling the LLM. If generation
        fails, that users message remains in history and no assistant record is
        fabricated. The assistant message is committed before ChatResponse
        is constructed.
        """

        self._validate_identity(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        self._validate_request(request)

        resolved_request_id = self._resolve_request_id(request_id)
        output_tokens = self._output_token_limit(request)

        conversation = await self._get_owned_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )

        async with self._execution_policy.generation_slot(
            user_id=user_id,
            conversation_id=conversation_id,
        ):
            prepared_context, user_message = (
                await self._prepare_and_persist_user_message(
                    user_id=user_id,
                    conversation=conversation,
                    request=request,
                    output_tokens=output_tokens,
                )
            )

            response = await self._chat_orchestrator.generate(
                prepared_context,
                generation_options=self._effective_generation_options(
                    request,
                    output_tokens,
                ),
                request_id=resolved_request_id,
            )

            try:
                assistant_message = (
                    await self._message_service.create_assistant_message(
                        conversation_id=conversation_id,
                        content=response.content,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "Assistant response persistence failed: conversation_id=%s request_id=%s",
                    conversation_id,
                    resolved_request_id,
                )
                raise ChatPersistenceException() from exc

            assistant_message_id = getattr(assistant_message, "id", None)
            created_at = getattr(assistant_message, "created_at", None)

            if not isinstance(assistant_message_id, UUID):
                raise ChatPersistenceException(
                    "The message service returned an invalid assistant message identifier."
                )
            if getattr(assistant_message, "conversation_id", None) != conversation_id:
                raise ChatPersistenceException(
                    "The message service returned an assistant message for another conversation."
                )
            if getattr(assistant_message, "role", None) != "assistant":
                raise ChatPersistenceException(
                    "The message service did not persist the assistant turn with the assistant role."
                )
            if not isinstance(created_at, datetime):
                raise ChatPersistenceException(
                    "The message service returned an invalid assistant timestamp."
                )
            if created_at.tzinfo is None or created_at.utcoffset() is None:
                raise ChatPersistenceException(
                    "The message service returned a timezone-naive assistant timestamp."
                )

            logger.info(
                (
                    "Chat turn completed: conversation_id=%s request_id=%s "
                    "provider=%s model=%s input_tokens=%s output_tokens=%s"
                ),
                conversation_id,
                resolved_request_id,
                response.provider_name,
                response.model,
                response.usage.input_tokens if response.usage else None,
                response.usage.output_tokens if response.usage else None,
            )

            return ChatResponse(
                conversation_id=conversation_id,
                user_message_id=user_message.id,
                assistant_message_id=assistant_message_id,
                content=response.content,
                finish_reason=response.finish_reason,
                usage=response.usage,
                created_at=created_at,
            )

    @staticmethod
    def _resolve_request_id(request_id: str | None) -> str:
        if request_id is None:
            return str(uuid4())

        if (
            not isinstance(request_id, str)
            or not request_id.strip()
            or len(request_id) > 128
            or any(character in request_id for character in ("\\r", "\\n", "\\x00"))
        ):
            raise ValueError(
                "request_id must be a safe, non-empty string of at most 128 characters."
            )

        return request_id

    async def _persist_interrupted_response(
        self,
        result: StreamResult,
    ) -> None:
        """
        Persist generated partial text when the stream ends before completion.

        The marker distinguishes incomplete assistant output from a normal
        completed answer in subsequent conversation history. A persistence
        failure is logged and does not replace the original stream outcome.

        If completion persistence itself failed, partial persistence is
        skipped because the original commit outcome may be ambiguous and the
        current MessageService does not provide idempotent message IDs.
        """

        if not result.content.strip():
            return

        if (
            result.interruption_reason
            is StreamInterruptionReason.PERSISTENCE_ERROR
        ):
            logger.error(
                "Skipping partial persistence after ambiguous completion persistence failure: conversation_id=%s request_id=%s",
                result.conversation_id,
                result.request_id,
            )
            return

        if self._interrupted_response_marker:
            if (
                result.interruption_reason
                is StreamInterruptionReason.CANCELLED
                or result.interruption_reason
                is StreamInterruptionReason.CONSUMER_CLOSED
            ):
                marker = "[Response interrupted]"
            else:
                marker = "[Response incomplete]"

            content = f"{result.content.rstrip()}\n\n{marker}"
        else:
            content = result.content

        try:
            await self._message_service.create_assistant_message(
                conversation_id=result.conversation_id,
                content=content,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Partial assistant response persistence failed: conversation_id=%s request_id=%s",
                result.conversation_id,
                result.request_id,
            )

    async def stream_message(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        request: ChatRequest,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream one authenticated Chat turn as SSE frames.

        Authentication and ownership checks, context preparation, quota
        authorization, and users-message persistence occur before the first
        semantic event. The generation slot remains held until the stream
        completes, fails, is cancelled, or the client disconnects.
        """

        self._validate_identity(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        self._validate_request(request)

        if is_disconnected is not None and not callable(is_disconnected):
            raise TypeError("is_disconnected must be an async callable or None.")

        resolved_request_id = self._resolve_request_id(request_id)
        output_tokens = self._output_token_limit(request)

        conversation = await self._get_owned_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )

        async with self._execution_policy.generation_slot(
            user_id=user_id,
            conversation_id=conversation_id,
        ):
            prepared_context, user_message = (
                await self._prepare_and_persist_user_message(
                    user_id=user_id,
                    conversation=conversation,
                    request=request,
                    output_tokens=output_tokens,
                )
            )

            assistant_persisted = False

            async def on_completed(result: StreamResult) -> UUID:
                nonlocal assistant_persisted

                try:
                    assistant_message = (
                        await self._message_service.create_assistant_message(
                            conversation_id=result.conversation_id,
                            content=result.content,
                        )
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception(
                        "Completed assistant response persistence failed: conversation_id=%s request_id=%s",
                        result.conversation_id,
                        result.request_id,
                    )
                    raise ChatPersistenceException() from exc

                # Set immediately after the service returns: the database
                # commit may have succeeded even if subsequent validation fails.
                assistant_persisted = True

                assistant_message_id = getattr(assistant_message, "id", None)
                if not isinstance(assistant_message_id, UUID):
                    raise ChatPersistenceException(
                        "The message service returned an invalid assistant message identifier."
                    )
                if getattr(assistant_message, "conversation_id", None) != conversation_id:
                    raise ChatPersistenceException(
                        "The message service returned an assistant message for another conversation."
                    )
                if getattr(assistant_message, "role", None) != "assistant":
                    raise ChatPersistenceException(
                        "The message service did not persist the assistant turn with the assistant role."
                    )

                return assistant_message_id

            async def on_interrupted(
                result: StreamResult,
                error: BaseException | None,
            ) -> None:
                # Current StreamManager sends GeneratorExit through this
                # callback even when the consumer closes immediately after
                # receiving the terminal completion event. Never duplicate
                # the already committed assistant record.
                if assistant_persisted:
                    return

                if error is not None:
                    logger.warning(
                        (
                            "Chat stream interrupted: conversation_id=%s "
                            "request_id=%s reason=%s error_type=%s"
                        ),
                        result.conversation_id,
                        result.request_id,
                        (
                            result.interruption_reason.value
                            if result.interruption_reason is not None
                            else "unknown"
                        ),
                        type(error).__name__,
                    )

                await self._persist_interrupted_response(result)

            semantic_events = self._chat_orchestrator.stream(
                prepared_context,
                conversation_id=conversation_id,
                user_message_id=user_message.id,
                request_id=resolved_request_id,
                generation_options=self._effective_generation_options(
                    request,
                    output_tokens,
                ),
                on_completed=on_completed,
                on_interrupted=on_interrupted,
            )

            async for frame in self._stream_adapter.adapt(
                semantic_events,
                is_disconnected=is_disconnected,
            ):
                yield frame


__all__ = [
    "ChatPersistenceException",
    "ChatGenerationLimitException",
    "ConversationRecord",
    "MessageRecord",
    "ConversationServicePort",
    "MessageServicePort",
    "ChatExecutionPolicy",
    "ChatService",
]
