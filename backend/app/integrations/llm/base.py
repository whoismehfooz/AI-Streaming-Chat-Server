"""
Provider abstraction for asynchronous LLM integrations.

Application/domain services depend on ``BaseLLMProvider`` rather than a
vendor SDK. Concrete adapters (for example OpenAI-compatible providers)
translate the shared schemas to/from their SDK-specific request and response
types.

Lifecycle and reliability contract:
    - ``complete`` performs one non-streaming completion.
    - ``stream`` yields normalized chunks through a pull-based async iterator.
    - Consumers provide natural backpressure by requesting the next chunk.
      Adapters must not create unbounded internal queues.
    - Cancellation must propagate. Implementations must not swallow
      ``asyncio.CancelledError`` and must close upstream streams in ``finally``.
    - A stream may retry transient failures only before its first yielded
      chunk. It must never replay already-delivered output.
    - ``close`` releases owned network clients and is safe to call during
      application shutdown. Implementations should make it idempotent.
    - Provider SDK types must not escape this integration boundary.

Token counting, conversation-history selection, context truncation, usage
charging, and users authorization belong to application/domain services.
The adapter may enforce provider/model hard limits as a final safety check.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.integrations.llm.exceptions import (
    LLMModelNotSupportedException,
    LLMProviderConfigurationException,
    LLMProviderRequestRejectedException,
    LLMStreamingNotSupportedException,
)
from app.integrations.llm.schemas import (
    LLMModelCapabilities,
    LLMProviderCapabilities,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
)


class BaseLLMProvider(ABC):
    """
    Abstract asynchronous interface implemented by every LLM provider.

    Concrete implementations own their SDK client and transport resources.
    They must expose a normalized capability description so factories and
    application services can inspect them without depending on provider-
    specific classes.
    """

    @property
    @abstractmethod
    def capabilities(self) -> LLMProviderCapabilities:
        """Immutable, provider-neutral model and feature capabilities."""

    @property
    def provider_name(self) -> str:
        """Stable provider identifier supplied by the capability descriptor."""
        return self.capabilities.provider_name

    @property
    def default_model(self) -> str | None:
        """
        Configured default model, if the adapter owns that configuration.

        An application may instead provide a model on each ``LLMRequest``.
        """
        return None

    def resolve_model(self, requested_model: str | None) -> str:
        """
        Resolve the request model against the adapter's configured default.

        A missing model is a local configuration error, not an upstream
        authentication or users-validation error.
        """
        model = requested_model or self.default_model
        if model is None or not model.strip():
            raise LLMProviderConfigurationException(
                provider_name=self.provider_name,
            )
        return model

    def get_model_capabilities(
        self,
        model: str,
    ) -> LLMModelCapabilities | None:
        """Return known capabilities for a model, if the adapter publishes them."""
        for model_capabilities in self.capabilities.models:
            if model_capabilities.model == model:
                return model_capabilities
        return None

    def validate_request(
        self,
        request: LLMRequest,
        *,
        streaming: bool = False,
    ) -> str:
        """
        Perform common provider/model capability checks before network I/O.

        Returns the resolved model name. Concrete adapters should call this
        at the start of both ``complete`` and ``stream`` so unsupported
        models/parameters fail consistently and before opening a connection.

        This does not count input tokens or truncate conversation history;
        those responsibilities belong to the AI application/service layer.
        """
        model = self.resolve_model(request.model)
        capabilities = self.capabilities
        model_capabilities = self.get_model_capabilities(model)

        if capabilities.models and model_capabilities is None:
            raise LLMModelNotSupportedException(
                provider_name=self.provider_name,
            )

        if streaming and not capabilities.supports_streaming:
            raise LLMStreamingNotSupportedException(
                provider_name=self.provider_name,
            )

        if (
            streaming
            and model_capabilities is not None
            and not model_capabilities.supports_streaming
        ):
            raise LLMStreamingNotSupportedException(
                provider_name=self.provider_name,
            )

        if model_capabilities is None:
            return model

        generation = request.generation

        if (
            model_capabilities.max_output_tokens is not None
            and generation.max_output_tokens > model_capabilities.max_output_tokens
        ):
            raise LLMProviderRequestRejectedException(
                message=(
                    "The requested output token limit exceeds the selected "
                    "model's maximum."
                ),
                status_code=400,
                provider_name=self.provider_name,
            )

        unsupported_parameters = (
            generation.temperature is not None
            and not model_capabilities.supports_temperature,
            generation.top_p is not None
            and not model_capabilities.supports_top_p,
            generation.frequency_penalty is not None
            and not model_capabilities.supports_frequency_penalty,
            generation.presence_penalty is not None
            and not model_capabilities.supports_presence_penalty,
            generation.stop_sequences is not None
            and not model_capabilities.supports_stop_sequences,
        )

        if any(unsupported_parameters):
            raise LLMProviderRequestRejectedException(
                message=(
                    "The selected model does not support one or more "
                    "requested generation parameters."
                ),
                status_code=400,
                provider_name=self.provider_name,
            )

        return model

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Generate one complete assistant response.

        Implementations must apply request deadlines and retry policy, map
        provider failures to ``app.integrations.llm.exceptions``, and return
        only normalized ``LLMResponse`` data.
        """

    @abstractmethod
    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """
        Stream normalized assistant output incrementally.

        The returned iterator must be lazy: network work should begin when
        iteration starts, not merely when ``stream`` is called. It must
        propagate cancellation and close the upstream stream on exhaustion,
        failure, or consumer disconnect.
        """

    @abstractmethod
    async def close(self) -> None:
        """
        Release resources owned by this provider adapter.

        This normally closes its asynchronous SDK/HTTP client. It should be
        idempotent so lifespan shutdown and explicit cleanup can both call it.
        """

    async def __aenter__(self) -> BaseLLMProvider:
        """Allow provider adapters to be used with ``async with``."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Close provider-owned resources when leaving an async context."""
        await self.close()


__all__ = ["BaseLLMProvider"]
