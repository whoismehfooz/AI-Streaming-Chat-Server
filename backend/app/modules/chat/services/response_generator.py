from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from uuid import uuid4

from app.integrations.llm.base import BaseLLMProvider
from app.integrations.llm.exceptions import LLMProviderResponseException
from app.integrations.llm.schemas import (
    LLMGenerationConfig,
    LLMMessage,
    LLMRequest,
    LLMRequestOptions,
    LLMResponse,
    LLMStreamChunk,
)
from app.modules.chat.schemas.chat_request import ChatGenerationOptions


logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generate complete or incremental assistant output through BaseLLMProvider.

    The provider instance is injected and owned by the application lifespan.
    This service never closes it. ``default_model`` may be omitted when the
    provider adapter owns a configured default model.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        default_max_output_tokens: int = 512,
        request_options: LLMRequestOptions | None = None,
        default_model: str | None = None,
    ) -> None:
        if not isinstance(provider, BaseLLMProvider):
            raise TypeError("provider must implement BaseLLMProvider.")

        if (
            isinstance(default_max_output_tokens, bool)
            or not isinstance(default_max_output_tokens, int)
            or default_max_output_tokens <= 0
        ):
            raise ValueError(
                "default_max_output_tokens must be a positive integer."
            )

        if default_model is not None and (
            not isinstance(default_model, str) or not default_model.strip()
        ):
            raise ValueError(
                "default_model must be a non-empty string or None."
            )

        if request_options is not None and not isinstance(
            request_options,
            LLMRequestOptions,
        ):
            raise TypeError("request_options must be an LLMRequestOptions instance.")

        self._provider = provider
        self._default_max_output_tokens = default_max_output_tokens
        self._request_options = request_options or LLMRequestOptions()
        self._default_model = default_model

    @property
    def provider_name(self) -> str:
        """Return the provider's normalized identifier for observability."""

        return self._provider.provider_name

    def _normalize_messages(
        self,
        messages: Sequence[LLMMessage],
    ) -> tuple[LLMMessage, ...]:
        if isinstance(messages, (str, bytes)):
            raise TypeError("messages must be a sequence of LLMMessage objects.")

        try:
            normalized = tuple(messages)
        except TypeError as exc:
            raise TypeError(
                "messages must be an iterable of LLMMessage objects."
            ) from exc

        if not normalized:
            raise ValueError("At least one prepared LLM message is required.")

        if any(not isinstance(message, LLMMessage) for message in normalized):
            raise TypeError("messages must contain only LLMMessage instances.")

        return normalized

    def _build_request(
        self,
        messages: Sequence[LLMMessage],
        *,
        generation_options: ChatGenerationOptions | None,
        request_id: str | None,
    ) -> LLMRequest:
        normalized_messages = self._normalize_messages(messages)

        if generation_options is not None and not isinstance(
            generation_options,
            ChatGenerationOptions,
        ):
            raise TypeError(
                "generation_options must be ChatGenerationOptions or None."
            )

        options = generation_options or ChatGenerationOptions()

        generation = LLMGenerationConfig(
            max_output_tokens=(
                options.max_output_tokens
                if options.max_output_tokens is not None
                else self._default_max_output_tokens
            ),
            temperature=options.temperature,
            top_p=options.top_p,
            frequency_penalty=options.frequency_penalty,
            presence_penalty=options.presence_penalty,
            stop_sequences=options.stop_sequences,
        )

        resolved_request_id = request_id or str(uuid4())
        if not isinstance(resolved_request_id, str) or not resolved_request_id.strip():
            raise ValueError("request_id must be a non-empty string.")

        return LLMRequest(
            messages=normalized_messages,
            model=self._default_model,
            generation=generation,
            options=self._request_options,
            request_id=resolved_request_id,
        )

    async def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        generation_options: ChatGenerationOptions | None = None,
        request_id: str | None = None,
    ) -> LLMResponse:
        """Perform one non-streaming completion and validate its result."""

        request = self._build_request(
            messages,
            generation_options=generation_options,
            request_id=request_id,
        )

        try:
            response = await self._provider.complete(request)
        except asyncio.CancelledError:
            raise

        if not isinstance(response, LLMResponse):
            raise LLMProviderResponseException(
                message="The LLM provider returned an invalid completion result.",
                provider_name=self.provider_name,
            )

        if not response.content.strip():
            raise LLMProviderResponseException(
                message="The LLM provider returned an empty completion.",
                provider_name=response.provider_name,
                provider_request_id=response.provider_request_id,
            )

        return response

    async def stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        generation_options: ChatGenerationOptions | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Yield normalized provider chunks in strict sequence order.

        The upstream iterator is explicitly closed even when the consumer
        stops early, a response-size guard fails downstream, or cancellation
        occurs. The provider remains responsible for translating vendor SDK
        exceptions and applying its configured retry/deadline policy.
        """

        request = self._build_request(
            messages,
            generation_options=generation_options,
            request_id=request_id,
        )

        upstream = self._provider.stream(request).__aiter__()
        last_sequence_number = -1
        stream_provider: str | None = None
        stream_model: str | None = None
        final_chunk_seen = False

        try:
            async for chunk in upstream:
                if not isinstance(chunk, LLMStreamChunk):
                    raise LLMProviderResponseException(
                        message="The LLM provider returned an invalid stream chunk.",
                        provider_name=self.provider_name,
                    )

                if final_chunk_seen:
                    raise LLMProviderResponseException(
                        message=(
                            "The LLM provider emitted data after the final "
                            "stream chunk."
                        ),
                        provider_name=chunk.provider_name,
                        provider_request_id=chunk.provider_request_id,
                    )

                if chunk.sequence_number != last_sequence_number + 1:
                    raise LLMProviderResponseException(
                        message=(
                            "The LLM provider returned missing or out-of-order "
                            "stream chunks."
                        ),
                        provider_name=chunk.provider_name,
                        provider_request_id=chunk.provider_request_id,
                    )

                if stream_provider is None:
                    stream_provider = chunk.provider_name
                    stream_model = chunk.model
                elif (
                    chunk.provider_name != stream_provider
                    or chunk.model != stream_model
                ):
                    raise LLMProviderResponseException(
                        message=(
                            "The LLM provider changed stream identity "
                            "mid-response."
                        ),
                        provider_name=chunk.provider_name,
                        provider_request_id=chunk.provider_request_id,
                    )

                last_sequence_number = chunk.sequence_number
                final_chunk_seen = chunk.is_final
                yield chunk

        except asyncio.CancelledError:
            raise

        except LLMProviderResponseException:
            raise

        except Exception:
            # Adapters should normalize upstream failures. Unexpected errors
            # are logged with traceback but are not exposed to API consumers.
            logger.exception(
                "Unexpected failure consuming normalized LLM stream: provider=%s request_id=%s",
                self.provider_name,
                request.request_id,
            )
            raise

        finally:
            close = getattr(upstream, "aclose", None)
            if callable(close):
                try:
                    await close()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Failed to close normalized LLM stream: provider=%s request_id=%s",
                        self.provider_name,
                        request.request_id,
                    )


__all__ = ["ResponseGenerator"]
