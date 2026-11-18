"""
Asynchronous OpenAI-compatible LLM provider adapter.

This adapter implements the provider-neutral ``BaseLLMProvider`` contract
using the official OpenAI asynchronous Python SDK. It supports OpenAI and
OpenAI-compatible chat-completions endpoints, including the Groq endpoint
used by the existing AI service.

Engineering guarantees:
    - Provider SDK request/response types do not escape this module.
    - Completion and stream requests use bounded request deadlines.
    - Streaming also enforces an idle timeout between chunks.
    - SDK-level automatic retries are disabled to avoid multiplying the
      application's explicit retry policy.
    - Completion retries are bounded by LLMRetryPolicy.
    - A stream may retry only before any chunk has been delivered.
    - Cancellation propagates and upstream streams are closed in ``finally``.
    - No unbounded producer queue is created; async iteration provides
      pull-based backpressure.
    - Public exceptions never expose credentials, raw provider bodies,
      prompts, or SDK exception text.
    - Logs contain operational metadata only, never message content.

Responsibilities intentionally outside this adapter:
    - User authentication and authorization.
    - Conversation/message persistence and ownership.
    - Prompt construction and system-prompt policy.
    - Conversation-history selection and truncation.
    - Token counting and context-budget allocation.
    - User quotas, billing, and usage persistence.
    - SSE framing and HTTP response lifecycle.

The adapter accepts text chat messages only, matching the current project
scope and normalized schemas.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from app.integrations.llm.base import BaseLLMProvider
from app.integrations.llm.exceptions import (
    LLMModelNotSupportedException,
    LLMProviderAuthenticationException,
    LLMProviderConnectionException,
    LLMProviderException,
    LLMProviderRateLimitException,
    LLMProviderRequestRejectedException,
    LLMProviderResponseException,
    LLMProviderTimeoutException,
    LLMProviderUnavailableException,
    LLMStreamInterruptedException,
)
from app.integrations.llm.schemas import (
    LLMFinishReason,
    LLMProviderCapabilities,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    LLMUsage,
)


logger = logging.getLogger(__name__)


class OpenAICompatibleLLMProvider(BaseLLMProvider):
    """
    OpenAI SDK adapter for OpenAI and compatible chat-completions APIs.

    Args:
        api_key:
            Provider API credential. It is passed only to the SDK and is
            never logged or included in exceptions.
        model:
            Default model used when ``LLMRequest.model`` is omitted.
        base_url:
            Optional API base URL. Use the provider's official endpoint for
            direct OpenAI access or an OpenAI-compatible endpoint such as
            Groq's API.
        provider_name:
            Stable, safe provider identifier used in normalized results and
            logs (for example ``"openai"`` or ``"groq"``).
        capabilities:
            Optional provider capability descriptor. If omitted, a
            conservative descriptor is created without claiming an
            unverified model catalogue.
        client:
            Optional preconfigured ``AsyncOpenAI`` client, primarily useful
            for controlled dependency injection in tests. Injected clients
            remain owned by the caller and are not closed by this adapter.
        default_timeout_seconds:
            Default transport timeout for SDK operations. Per-request
            deadlines from ``LLMRequest.options`` override this value.

    The adapter owns the SDK client it creates. The application must close
    it during lifespan shutdown.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        base_url: str | None = None,
        provider_name: str = "openai",
        capabilities: LLMProviderCapabilities | None = None,
        client: AsyncOpenAI | None = None,
        default_timeout_seconds: float = 90.0,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")

        if (
            not isinstance(provider_name, str)
            or not provider_name.strip()
            or len(provider_name.strip()) > 64
        ):
            raise ValueError(
                "provider_name must be a non-empty string of at most 64 characters."
            )

        if (
            isinstance(default_timeout_seconds, bool)
            or not isinstance(default_timeout_seconds, (int, float))
            or default_timeout_seconds <= 0
        ):
            raise ValueError("default_timeout_seconds must be greater than zero.")

        if base_url is not None and (
            not isinstance(base_url, str) or not base_url.strip()
        ):
            raise ValueError("base_url must be None or a non-empty string.")

        normalized_provider_name = provider_name.strip().lower()
        normalized_capabilities = capabilities or LLMProviderCapabilities(
            provider_name=normalized_provider_name,
            models=(),
            supports_streaming=True,
            supports_streaming_usage=False,
            supports_idempotency_keys=False,
            reports_usage=True,
        )

        # Validate all caller-supplied metadata before allocating a network
        # client. A bad capability descriptor must not leak a live client.
        if normalized_capabilities.provider_name != normalized_provider_name:
            raise ValueError(
                "capabilities.provider_name must match provider_name."
            )

        if client is None:
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("api_key is required when no SDK client is injected.")

            sdk_client = AsyncOpenAI(
                api_key=api_key.strip(),
                base_url=base_url.strip() if base_url else None,
                timeout=float(default_timeout_seconds),
                # Retries are implemented here so one policy controls the
                # total number of attempts and streams are never replayed
                # after output has been delivered.
                max_retries=0,
            )
            owns_client = True
        else:
            sdk_client = client
            owns_client = False

        self._client = sdk_client
        self._owns_client = owns_client
        self._model = model.strip()
        self._provider_name = normalized_provider_name
        self._default_timeout_seconds = float(default_timeout_seconds)
        self._closed = False
        self._capabilities = normalized_capabilities

    @property
    def capabilities(self) -> LLMProviderCapabilities:
        """Return the immutable provider capability descriptor."""
        return self._capabilities

    @property
    def default_model(self) -> str:
        """Return the configured default model."""
        return self._model

    def _ensure_open(self) -> None:
        if self._closed:
            raise LLMProviderException(
                message="The LLM provider client has already been closed.",
                status_code=500,
                provider_name=self._provider_name,
            )

    def _request_client(self, timeout_seconds: float) -> AsyncOpenAI:
        """
        Return an SDK client view with this operation's timeout.

        ``with_options`` creates a lightweight client configuration view
        sharing the underlying transport. It does not create a new network
        connection pool for every request.
        """
        return self._client.with_options(
            timeout=timeout_seconds,
            max_retries=0,
        )

    def _build_request_payload(
        self,
        request: LLMRequest,
        model: str,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """Translate normalized request data into SDK-compatible parameters."""
        messages: list[dict[str, str]] = []

        for message in request.messages:
            item = {
                "role": message.role,
                "content": message.content,
            }
            if message.name is not None:
                item["name"] = message.name
            messages.append(item)

        generation = request.generation

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        # The existing Groq implementation used max_tokens. Direct OpenAI
        # requests use max_completion_tokens, which is the current normalized
        # parameter for chat-completion output limits.
        if self._provider_name == "groq":
            payload["max_tokens"] = generation.max_output_tokens
        else:
            payload["max_completion_tokens"] = generation.max_output_tokens

        if generation.temperature is not None:
            payload["temperature"] = generation.temperature

        if generation.top_p is not None:
            payload["top_p"] = generation.top_p

        if generation.frequency_penalty is not None:
            payload["frequency_penalty"] = generation.frequency_penalty

        if generation.presence_penalty is not None:
            payload["presence_penalty"] = generation.presence_penalty

        if generation.stop_sequences is not None:
            payload["stop"] = list(generation.stop_sequences)

        if (
            stream
            and self.capabilities.supports_streaming_usage
        ):
            payload["stream_options"] = {"include_usage": True}

        return payload

    @staticmethod
    def _normalize_finish_reason(value: Any) -> LLMFinishReason | None:
        if value is None:
            return None

        if value in {"stop", "length", "content_filter"}:
            return value

        # Tool/function-call finish reasons are outside the current text-only
        # scope. Preserve the fact that generation ended without leaking an
        # unsupported provider-specific enum into the shared contract.
        return "unknown"

    @staticmethod
    def _non_negative_int(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    @classmethod
    def _normalize_usage(cls, raw_usage: Any) -> LLMUsage | None:
        """Normalize provider token-usage fields, if usage was reported."""
        if raw_usage is None:
            return None

        input_tokens = cls._non_negative_int(
            getattr(raw_usage, "prompt_tokens", None)
        )
        output_tokens = cls._non_negative_int(
            getattr(raw_usage, "completion_tokens", None)
        )
        total_tokens = cls._non_negative_int(
            getattr(raw_usage, "total_tokens", None)
        )

        if input_tokens is None or output_tokens is None:
            return None

        if total_tokens is None:
            total_tokens = input_tokens + output_tokens

        prompt_details = getattr(raw_usage, "prompt_tokens_details", None)
        completion_details = getattr(
            raw_usage,
            "completion_tokens_details",
            None,
        )

        cached_tokens = cls._non_negative_int(
            getattr(prompt_details, "cached_tokens", None)
        )
        reasoning_tokens = cls._non_negative_int(
            getattr(completion_details, "reasoning_tokens", None)
        )

        return LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    @classmethod
    def _normalize_response(
        cls,
        raw_response: Any,
        *,
        provider_name: str,
        requested_model: str,
    ) -> LLMResponse:
        """Convert an SDK completion object into the provider-neutral schema."""
        choices = getattr(raw_response, "choices", None)

        if not choices:
            raise LLMProviderResponseException(
                message="The LLM provider returned an empty completion response.",
                provider_name=provider_name,
                provider_request_id=getattr(raw_response, "id", None),
            )

        first_choice = choices[0]
        raw_message = getattr(first_choice, "message", None)
        content = getattr(raw_message, "content", None)

        if content is None:
            content = ""
        elif not isinstance(content, str):
            raise LLMProviderResponseException(
                message="The LLM provider returned an unsupported response format.",
                provider_name=provider_name,
                provider_request_id=getattr(raw_response, "id", None),
            )

        raw_created = getattr(raw_response, "created", None)
        created_at: datetime | None = None

        if isinstance(raw_created, (int, float)) and not isinstance(raw_created, bool):
            try:
                created_at = datetime.fromtimestamp(
                    raw_created,
                    tz=timezone.utc,
                )
            except (OverflowError, OSError, ValueError):
                created_at = None

        response_model = getattr(raw_response, "model", None)
        if not isinstance(response_model, str) or not response_model:
            response_model = requested_model

        return LLMResponse(
            provider_name=provider_name,
            model=response_model,
            content=content,
            finish_reason=cls._normalize_finish_reason(
                getattr(first_choice, "finish_reason", None)
            ),
            usage=cls._normalize_usage(
                getattr(raw_response, "usage", None)
            ),
            provider_request_id=getattr(raw_response, "id", None),
            created_at=created_at,
        )

    def _normalize_stream_chunk(
        self,
        raw_chunk: Any,
        *,
        model: str,
        sequence_number: int,
    ) -> LLMStreamChunk | None:
        """Convert one provider stream event into a normalized chunk."""
        choices = getattr(raw_chunk, "choices", None) or []
        raw_usage = getattr(raw_chunk, "usage", None)
        usage = self._normalize_usage(raw_usage)

        delta = ""
        finish_reason: LLMFinishReason | None = None

        if choices:
            first_choice = choices[0]
            raw_delta = getattr(first_choice, "delta", None)
            raw_content = getattr(raw_delta, "content", None)

            if raw_content is not None:
                if not isinstance(raw_content, str):
                    raise LLMProviderResponseException(
                        message=(
                            "The LLM provider returned an unsupported "
                            "streaming content format."
                        ),
                        provider_name=self._provider_name,
                        provider_request_id=getattr(raw_chunk, "id", None),
                    )
                delta = raw_content

            finish_reason = self._normalize_finish_reason(
                getattr(first_choice, "finish_reason", None)
            )

        # Ignore protocol events that contain neither content, terminal
        # metadata, nor usage information.
        if not delta and finish_reason is None and usage is None:
            return None

        is_final = finish_reason is not None or (
            not choices and usage is not None
        )

        chunk_model = getattr(raw_chunk, "model", None)
        if not isinstance(chunk_model, str) or not chunk_model:
            chunk_model = model

        return LLMStreamChunk(
            provider_name=self._provider_name,
            model=chunk_model,
            sequence_number=sequence_number,
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
            provider_request_id=getattr(raw_chunk, "id", None),
            is_final=is_final,
        )

    @staticmethod
    def _retry_after_seconds(exc: APIStatusError) -> int | None:
        """Read a safe Retry-After delay from an upstream response."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)

        if headers is None:
            return None

        value = headers.get("retry-after")
        if not isinstance(value, str):
            return None

        value = value.strip()
        if not value:
            return None

        try:
            seconds = float(value)
            if seconds >= 0:
                return int(seconds)
        except ValueError:
            pass

        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)

            delay = (
                retry_at - datetime.now(timezone.utc)
            ).total_seconds()

            if delay >= 0:
                return int(delay)
        except (TypeError, ValueError, OverflowError):
            return None

        return None

    def _translate_provider_exception(
        self,
        exc: Exception,
    ) -> LLMProviderException:
        """
        Translate SDK/transport errors into the integration exception family.

        The original exception is retained as ``__cause__`` by the caller,
        allowing protected logs to retain diagnostics without exposing
        vendor-specific messages in API responses.
        """
        if isinstance(exc, LLMProviderException):
            return exc

        if isinstance(exc, (APITimeoutError, TimeoutError)):
            return LLMProviderTimeoutException(
                provider_name=self._provider_name,
            )

        if isinstance(exc, AuthenticationError):
            return LLMProviderAuthenticationException(
                provider_name=self._provider_name,
                provider_request_id=getattr(exc, "request_id", None),
            )

        if isinstance(exc, RateLimitError):
            return LLMProviderRateLimitException(
                provider_name=self._provider_name,
                provider_request_id=getattr(exc, "request_id", None),
                retry_after_seconds=self._retry_after_seconds(exc),
            )

        if isinstance(exc, APIConnectionError):
            return LLMProviderConnectionException(
                provider_name=self._provider_name,
            )

        if isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            request_id = getattr(exc, "request_id", None)

            if status_code in {401, 403}:
                return LLMProviderAuthenticationException(
                    provider_name=self._provider_name,
                    provider_request_id=request_id,
                )

            if status_code == 404:
                return LLMModelNotSupportedException(
                    provider_name=self._provider_name,
                    provider_request_id=request_id,
                )

            if status_code == 408:
                return LLMProviderTimeoutException(
                    provider_name=self._provider_name,
                    provider_request_id=request_id,
                )

            if status_code == 429:
                return LLMProviderRateLimitException(
                    provider_name=self._provider_name,
                    provider_request_id=request_id,
                    retry_after_seconds=self._retry_after_seconds(exc),
                )

            if isinstance(status_code, int) and status_code >= 500:
                return LLMProviderUnavailableException(
                    provider_name=self._provider_name,
                    provider_request_id=request_id,
                )

            return LLMProviderRequestRejectedException(
                provider_name=self._provider_name,
                provider_request_id=request_id,
            )

        return LLMProviderException(
            message="The LLM provider request failed unexpectedly.",
            provider_name=self._provider_name,
        )

    @staticmethod
    def _retry_is_enabled(
        exc: LLMProviderException,
        request: LLMRequest,
    ) -> bool:
        """Apply the request's explicit retry switches to transient failures."""
        policy = request.options.retry_policy

        if not exc.retryable:
            return False

        if isinstance(exc, LLMProviderRateLimitException):
            return policy.retry_on_rate_limit

        if isinstance(exc, LLMProviderTimeoutException):
            return policy.retry_on_timeout

        if isinstance(exc, LLMProviderConnectionException):
            return policy.retry_on_connection_error

        return True

    @staticmethod
    def _retry_delay(
        exc: LLMProviderException,
        request: LLMRequest,
        attempt_number: int,
    ) -> float:
        """Calculate bounded exponential backoff with optional jitter."""
        policy = request.options.retry_policy

        if exc.retry_after_seconds is not None:
            return float(exc.retry_after_seconds)

        exponential_delay = min(
            policy.initial_backoff_seconds
            * (policy.backoff_multiplier ** max(0, attempt_number - 1)),
            policy.max_backoff_seconds,
        )

        if policy.jitter_ratio == 0:
            return exponential_delay

        jitter_factor = random.uniform(
            1.0 - policy.jitter_ratio,
            1.0 + policy.jitter_ratio,
        )

        return min(
            exponential_delay * jitter_factor,
            policy.max_backoff_seconds,
        )

    async def _wait_before_retry(
        self,
        exc: LLMProviderException,
        request: LLMRequest,
        *,
        attempt_number: int,
        deadline: float,
    ) -> bool:
        """
        Sleep only when another attempt can fit inside the overall deadline.

        Returns False when the remaining deadline is too short for the
        calculated delay. Cancellation during sleep is intentionally allowed
        to propagate.
        """
        delay = self._retry_delay(
            exc,
            request,
            attempt_number,
        )
        remaining = deadline - asyncio.get_running_loop().time()

        if remaining <= 0 or delay >= remaining:
            return False

        logger.warning(
            "Retrying LLM provider request: provider=%s attempt=%d delay=%.2fs",
            self._provider_name,
            attempt_number + 1,
            delay,
        )

        await asyncio.sleep(delay)
        return True

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Execute a non-streaming completion with bounded retries and deadline.

        The timeout covers all attempts and retry delays, not just one SDK
        request. Retries occur only for failures marked transient and enabled
        by the request's retry policy.
        """
        self._ensure_open()
        model = self.validate_request(request, streaming=False)
        payload = self._build_request_payload(
            request,
            model,
            stream=False,
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + request.options.timeout_seconds
        policy = request.options.retry_policy
        started_at = time.perf_counter()

        for attempt in range(1, policy.max_attempts + 1):
            remaining = deadline - loop.time()

            if remaining <= 0:
                raise LLMProviderTimeoutException(
                    provider_name=self._provider_name,
                )

            try:
                request_client = self._request_client(remaining)

                async with asyncio.timeout(remaining):
                    raw_response = await (
                        request_client.chat.completions.create(**payload)
                    )

                normalized = self._normalize_response(
                    raw_response,
                    provider_name=self._provider_name,
                    requested_model=model,
                )

                logger.info(
                    (
                        "LLM completion succeeded: provider=%s model=%s "
                        "duration=%.2fs usage_reported=%s"
                    ),
                    self._provider_name,
                    model,
                    time.perf_counter() - started_at,
                    normalized.usage is not None,
                )

                return normalized

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                translated = self._translate_provider_exception(exc)

                if (
                    attempt < policy.max_attempts
                    and self._retry_is_enabled(translated, request)
                ):
                    waited = await self._wait_before_retry(
                        translated,
                        request,
                        attempt_number=attempt,
                        deadline=deadline,
                    )
                    if waited:
                        continue

                logger.warning(
                    (
                        "LLM completion failed: provider=%s model=%s "
                        "error_type=%s duration=%.2fs"
                    ),
                    self._provider_name,
                    model,
                    type(translated).__name__,
                    time.perf_counter() - started_at,
                )

                raise translated from exc

        # The loop always returns or raises. This guard protects the contract
        # if retry-policy implementation changes in the future.
        raise LLMProviderException(
            message="The LLM provider request could not be completed.",
            provider_name=self._provider_name,
        )

    async def _close_upstream_stream(self, stream: Any | None) -> None:
        """Close an upstream SDK stream without hiding the primary failure."""
        if stream is None:
            return

        close_method = getattr(stream, "close", None)
        if close_method is None:
            close_method = getattr(stream, "aclose", None)

        if close_method is None:
            return

        try:
            result = close_method()
            if hasattr(result, "__await__"):
                await result
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Failed to close upstream LLM stream: provider=%s",
                self._provider_name,
                exc_info=True,
            )

    async def stream(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Yield normalized provider chunks with timeout and safe retry behavior.

        The overall timeout bounds the complete stream lifetime. The idle
        timeout separately bounds stream establishment and each wait for the
        next chunk. Once any chunk has been yielded, the stream is never
        retried because replay could duplicate output already received by
        the chat service/client.
        """
        self._ensure_open()
        model = self.validate_request(request, streaming=True)
        payload = self._build_request_payload(
            request,
            model,
            stream=True,
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + request.options.timeout_seconds
        policy = request.options.retry_policy
        emitted_chunks = 0
        emitted_text = False
        started_at = time.perf_counter()

        logger.info(
            "Starting LLM stream: provider=%s model=%s",
            self._provider_name,
            model,
        )

        for attempt in range(1, policy.max_attempts + 1):
            upstream_stream: Any | None = None

            try:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise LLMProviderTimeoutException(
                        provider_name=self._provider_name,
                    )

                request_client = self._request_client(remaining)
                initial_wait = min(
                    request.options.stream_idle_timeout_seconds,
                    remaining,
                )

                async with asyncio.timeout(initial_wait):
                    upstream_stream = await (
                        request_client.chat.completions.create(**payload)
                    )

                iterator = upstream_stream.__aiter__()
                sequence_number = 0

                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise LLMProviderTimeoutException(
                            provider_name=self._provider_name,
                        )

                    next_chunk_timeout = min(
                        request.options.stream_idle_timeout_seconds,
                        remaining,
                    )

                    try:
                        async with asyncio.timeout(next_chunk_timeout):
                            raw_chunk = await anext(iterator)
                    except StopAsyncIteration:
                        duration = time.perf_counter() - started_at
                        logger.info(
                            (
                                "LLM stream completed: provider=%s model=%s "
                                "duration=%.2fs chunks=%d"
                            ),
                            self._provider_name,
                            model,
                            duration,
                            emitted_chunks,
                        )
                        return

                    normalized_chunk = self._normalize_stream_chunk(
                        raw_chunk,
                        model=model,
                        sequence_number=sequence_number,
                    )

                    if normalized_chunk is None:
                        continue

                    sequence_number += 1
                    emitted_chunks += 1
                    emitted_text = emitted_text or bool(normalized_chunk.delta)

                    yield normalized_chunk

            except asyncio.CancelledError:
                # Client disconnects and task cancellation must reach the
                # caller. The finally block still closes the upstream stream.
                raise

            except Exception as exc:
                translated = self._translate_provider_exception(exc)

                if emitted_chunks > 0:
                    logger.warning(
                        (
                            "LLM stream interrupted after output: provider=%s "
                            "model=%s chunks=%d"
                        ),
                        self._provider_name,
                        model,
                        emitted_chunks,
                    )

                    raise LLMStreamInterruptedException(
                        provider_name=self._provider_name,
                        provider_request_id=translated.provider_request_id,
                        chunks_emitted=emitted_chunks,
                        partial_output_emitted=emitted_text,
                    ) from exc

                if (
                    attempt < policy.max_attempts
                    and self._retry_is_enabled(translated, request)
                ):
                    waited = await self._wait_before_retry(
                        translated,
                        request,
                        attempt_number=attempt,
                        deadline=deadline,
                    )
                    if waited:
                        continue

                logger.warning(
                    (
                        "LLM stream failed before output: provider=%s "
                        "model=%s error_type=%s"
                    ),
                    self._provider_name,
                    model,
                    type(translated).__name__,
                )

                raise translated from exc

            finally:
                await self._close_upstream_stream(upstream_stream)

        raise LLMProviderException(
            message="The LLM response stream could not be completed.",
            provider_name=self._provider_name,
        )

    async def close(self) -> None:
        """
        Close the SDK client when this adapter owns it.

        Repeated calls are safe. An injected client remains the responsibility
        of the component that created and supplied it.
        """
        if self._closed:
            return

        self._closed = True

        if self._owns_client:
            await self._client.close()

        logger.debug(
            "LLM provider client closed: provider=%s owned_client=%s",
            self._provider_name,
            self._owns_client,
        )


# The architecture names this module openai_client.py. This alias offers a
# concise import name while keeping the implementation explicit about its
# OpenAI-compatible protocol support.
OpenAIClient = OpenAICompatibleLLMProvider


__all__ = [
    "OpenAICompatibleLLMProvider",
    "OpenAIClient",
]
