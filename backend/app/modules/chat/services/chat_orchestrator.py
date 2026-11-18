"""
Coordinate prompt resolution, context budgeting, and LLM generation.

ChatOrchestrator is the application workflow coordinator between:
    - PromptBuilder, which resolves trusted versioned instructions;
    - ContextBuilder, which selects history within the token budget;
    - ResponseGenerator, which invokes the provider-neutral LLM contract;
    - StreamManager, which manages semantic stream events and persistence
      callbacks.

It does not access the database, authorize users, serialize SSE, or create
and close provider clients. Those responsibilities remain in ChatService,
the Conversations domain, the transport adapter, and application lifespan.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from app.integrations.llm.schemas import LLMMessage, LLMResponse
from app.modules.chat.schemas.chat_request import ChatGenerationOptions
from app.modules.chat.schemas.stream_event import ChatStreamEvent

from .context_builder import ContextBuilder
from .prompt_builder import PromptBuilder
from .response_generator import ResponseGenerator
from .stream_manager import (
    CompletionHandler,
    InterruptionHandler,
    StreamManager,
)


@dataclass(frozen=True, slots=True)
class PreparedChatContext:
    """Immutable canonical prompt and its estimated input-token count."""

    messages: tuple[LLMMessage, ...]
    input_tokens: int
    reserved_output_tokens: int

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("Prepared chat context must contain messages.")
        if any(not isinstance(item, LLMMessage) for item in self.messages):
            raise TypeError("Prepared context must contain only LLMMessage objects.")
        if self.input_tokens < 0:
            raise ValueError("input_tokens cannot be negative.")
        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens cannot be negative.")


class ChatOrchestrator:
    """
    Prepare a bounded prompt and delegate complete or streaming generation.

    All collaborators are injected. The orchestrator performs no global
    configuration lookup and owns no external resources.
    """

    def __init__(
        self,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        response_generator: ResponseGenerator,
        stream_manager: StreamManager,
    ) -> None:
        if not isinstance(context_builder, ContextBuilder):
            raise TypeError("context_builder must be a ContextBuilder instance.")
        if not isinstance(prompt_builder, PromptBuilder):
            raise TypeError("prompt_builder must be a PromptBuilder instance.")
        if not isinstance(response_generator, ResponseGenerator):
            raise TypeError(
                "response_generator must be a ResponseGenerator instance."
            )
        if not isinstance(stream_manager, StreamManager):
            raise TypeError("stream_manager must be a StreamManager instance.")

        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._response_generator = response_generator
        self._stream_manager = stream_manager

    def prepare(
        self,
        conversation_history: Sequence[object],
        *,
        prompt_variables: Mapping[str, str] | None = None,
        prompt_version: str | None = None,
        reserved_output_tokens: int = 0,
    ) -> PreparedChatContext:
        """
        Resolve trusted instructions, select history, and validate the budget.

        PromptBuilder runs before ContextBuilder so the trusted system or
        developer prompt is included in the same budget as conversation
        history. This avoids under-counting the final provider request.
        """

        trusted_and_history = self._prompt_builder.build(
            conversation_history,
            prompt_variables=prompt_variables,
            prompt_version=prompt_version,
        )

        bounded_context = self._context_builder.build(
            trusted_and_history,
            reserved_output_tokens=reserved_output_tokens,
        )

        canonical_messages = tuple(
            LLMMessage.model_validate(item)
            for item in bounded_context
        )

        input_tokens = self._context_builder.validate_llm_messages(
            canonical_messages,
            reserved_output_tokens=reserved_output_tokens,
        )

        return PreparedChatContext(
            messages=canonical_messages,
            input_tokens=input_tokens,
            reserved_output_tokens=reserved_output_tokens,
        )

    async def generate(
        self,
        prepared_context: PreparedChatContext,
        *,
        generation_options: ChatGenerationOptions | None,
        request_id: str,
    ) -> LLMResponse:
        """Generate one normalized non-streaming completion."""

        if not isinstance(prepared_context, PreparedChatContext):
            raise TypeError(
                "prepared_context must be a PreparedChatContext instance."
            )

        return await self._response_generator.generate(
            prepared_context.messages,
            generation_options=generation_options,
            request_id=request_id,
        )

    def stream(
        self,
        prepared_context: PreparedChatContext,
        *,
        conversation_id: UUID,
        user_message_id: UUID,
        request_id: str,
        generation_options: ChatGenerationOptions | None,
        on_completed: CompletionHandler,
        on_interrupted: InterruptionHandler,
    ) -> AsyncIterator[ChatStreamEvent]:
        """
        Return the lazy StreamManager semantic event iterator.

        The assistant message ID is intentionally not passed in advance.
        The current MessageService creates the assistant record after the
        generated content is complete and the database assigns its UUID.
        StreamManager emits that real UUID only in message_complete.
        """

        if not isinstance(prepared_context, PreparedChatContext):
            raise TypeError(
                "prepared_context must be a PreparedChatContext instance."
            )

        return self._stream_manager.stream(
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            request_id=request_id,
            messages=prepared_context.messages,
            generation_options=generation_options,
            on_completed=on_completed,
            on_interrupted=on_interrupted,
        )


__all__ = ["PreparedChatContext", "ChatOrchestrator"]
