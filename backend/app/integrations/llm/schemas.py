"""
Provider-neutral schemas for LLM requests, responses, and streaming.

These models form the contract shared by the application and every LLM
provider adapter. Provider SDK objects must be converted into these models
at the integration boundary and must never leak into domain services.

Scope:
    - Chat messages and generation parameters.
    - Request deadlines and retry policy.
    - Normalized completion and streaming results.
    - Token usage and model/provider capabilities.

This module intentionally does not perform token counting or conversation
truncation. Those operations require model-aware tokenization and belong in
the AI application/service layer. The provider adapter must enforce the
selected model's actual limits as a final boundary check.

Only text messages are represented here. Multimodal payloads, tool calling,
structured-output protocols, and agent execution are not part of the
currently defined project scope.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


LLMMessageRole = Literal["system", "developer", "user", "assistant"]

LLMFinishReason = Literal[
    "stop",
    "length",
    "content_filter",
    "unknown",
]


class LLMMessage(BaseModel):
    """
    A provider-independent chat message.

    ``role`` identifies the speaker and ``content`` contains plain text.
    Message ordering is significant and is preserved exactly as supplied.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    role: LLMMessageRole
    content: Annotated[str, Field(min_length=1)]
    name: Annotated[str | None, Field(default=None, min_length=1, max_length=128)]

    @field_validator("content")
    @classmethod
    def content_must_not_be_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must contain non-whitespace characters.")
        return value


class LLMGenerationConfig(BaseModel):
    """
    Normalized generation parameters supported by the project.

    Provider adapters are responsible for translating these normalized
    names to provider-specific request fields and rejecting unsupported
    combinations before making a network request.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_output_tokens: Annotated[int, Field(default=1024, gt=0)]
    temperature: Annotated[float | None, Field(default=None, ge=0.0, le=2.0)]
    top_p: Annotated[float | None, Field(default=None, gt=0.0, le=1.0)]
    frequency_penalty: Annotated[
        float | None,
        Field(default=None, ge=-2.0, le=2.0),
    ]
    presence_penalty: Annotated[
        float | None,
        Field(default=None, ge=-2.0, le=2.0),
    ]
    stop_sequences: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=256)], ...] | None,
        Field(default=None, max_length=16),
    ]


class LLMRetryPolicy(BaseModel):
    """
    Bounded retry policy for transient provider failures.

    ``max_attempts`` includes the initial attempt. A streaming adapter may
    retry only before it has yielded the first output chunk. Once output has
    reached the consumer, retrying would duplicate or corrupt the response.

    Provider SDK automatic retries should be disabled or configured so they
    do not multiply this policy unexpectedly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: Annotated[int, Field(default=3, ge=1, le=5)]
    initial_backoff_seconds: Annotated[
        float,
        Field(default=0.25, gt=0.0, le=30.0),
    ]
    max_backoff_seconds: Annotated[
        float,
        Field(default=8.0, gt=0.0, le=120.0),
    ]
    backoff_multiplier: Annotated[
        float,
        Field(default=2.0, ge=1.0, le=5.0),
    ]
    jitter_ratio: Annotated[float, Field(default=0.2, ge=0.0, le=1.0)]
    retry_on_rate_limit: bool = True
    retry_on_timeout: bool = True
    retry_on_connection_error: bool = True

    @field_validator("max_backoff_seconds")
    @classmethod
    def max_backoff_must_not_be_below_initial(
        cls,
        value: float,
        info,
    ) -> float:
        initial = info.data.get("initial_backoff_seconds")
        if initial is not None and value < initial:
            raise ValueError(
                "max_backoff_seconds must be greater than or equal to "
                "initial_backoff_seconds."
            )
        return value


class LLMRequestOptions(BaseModel):
    """
    Operational options for one provider call.

    The request timeout bounds the complete provider operation. For streams,
    the idle timeout bounds the time allowed between successive chunks.
    Implementations must also configure connection/pool timeouts at the
    underlying HTTP transport layer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout_seconds: Annotated[
        float,
        Field(default=90.0, gt=0.0, le=900.0),
    ]
    stream_idle_timeout_seconds: Annotated[
        float,
        Field(default=60.0, gt=0.0, le=300.0),
    ]
    retry_policy: LLMRetryPolicy = Field(default_factory=LLMRetryPolicy)


class LLMRequest(BaseModel):
    """
    Canonical input to a provider adapter.

    ``model`` may be omitted when the configured provider adapter has a
    default model. A request must contain at least one ordered message.
    ``request_id`` is correlation metadata, not an authentication secret.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: Annotated[tuple[LLMMessage, ...], Field(min_length=1)]
    model: Annotated[str | None, Field(default=None, min_length=1, max_length=200)]
    generation: LLMGenerationConfig = Field(default_factory=LLMGenerationConfig)
    options: LLMRequestOptions = Field(default_factory=LLMRequestOptions)
    request_id: Annotated[str | None, Field(default=None, min_length=1, max_length=128)]


class LLMUsage(BaseModel):
    """
    Normalized token usage reported by an LLM provider.

    Usage may be absent for providers or streaming modes that do not report
    it. Token counts are non-negative integers; adapters map provider-specific
    prompt/completion terminology into input/output terminology.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    total_tokens: Annotated[int, Field(ge=0)]
    cached_input_tokens: Annotated[int | None, Field(default=None, ge=0)]
    reasoning_tokens: Annotated[int | None, Field(default=None, ge=0)]


class LLMResponse(BaseModel):
    """
    Normalized non-streaming completion result.

    ``content`` is the assistant's generated text. Provider-specific response
    objects and headers must remain inside the adapter.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_name: Annotated[str, Field(min_length=1, max_length=64)]
    model: Annotated[str, Field(min_length=1, max_length=200)]
    content: str
    finish_reason: LLMFinishReason | None = None
    usage: LLMUsage | None = None
    provider_request_id: Annotated[
        str | None,
        Field(default=None, min_length=1, max_length=256),
    ]
    created_at: datetime | None = None


class LLMStreamChunk(BaseModel):
    """
    One normalized unit emitted by a streaming provider adapter.

    ``delta`` may be empty for a terminal metadata/usage chunk. Sequence
    numbers are monotonically increasing within a single stream and start at
    zero. ``is_final`` marks the terminal chunk when the provider protocol
    exposes a terminal event.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_name: Annotated[str, Field(min_length=1, max_length=64)]
    model: Annotated[str, Field(min_length=1, max_length=200)]
    sequence_number: Annotated[int, Field(ge=0)]
    delta: str = ""
    finish_reason: LLMFinishReason | None = None
    usage: LLMUsage | None = None
    provider_request_id: Annotated[
        str | None,
        Field(default=None, min_length=1, max_length=256),
    ]
    is_final: bool = False


class LLMModelCapabilities(BaseModel):
    """
    Capability and limit information for one concrete model.

    Context-window limits include both input and generated output unless
    the provider's published model contract explicitly defines them
    differently. The application layer remains responsible for counting
    and budgeting input tokens before dispatch.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Annotated[str, Field(min_length=1, max_length=200)]
    context_window_tokens: Annotated[int | None, Field(default=None, gt=0)]
    max_output_tokens: Annotated[int | None, Field(default=None, gt=0)]
    supports_streaming: bool = True
    supports_temperature: bool = True
    supports_top_p: bool = True
    supports_frequency_penalty: bool = False
    supports_presence_penalty: bool = False
    supports_stop_sequences: bool = True

    @model_validator(mode="after")
    def output_limit_must_fit_context_window(self) -> LLMModelCapabilities:
        if (
            self.context_window_tokens is not None
            and self.max_output_tokens is not None
            and self.max_output_tokens > self.context_window_tokens
        ):
            raise ValueError(
                "max_output_tokens cannot exceed context_window_tokens."
            )
        return self


class LLMProviderCapabilities(BaseModel):
    """
    Public capability description for a provider adapter.

    ``models`` is an explicit allow-list when populated. An empty tuple means
    the adapter cannot provide a reliable static model catalogue, not that
    every model name is guaranteed to work.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_name: Annotated[str, Field(min_length=1, max_length=64)]
    models: tuple[LLMModelCapabilities, ...] = ()
    supports_streaming: bool = True
    supports_streaming_usage: bool = False
    supports_idempotency_keys: bool = False
    reports_usage: bool = True

    @model_validator(mode="after")
    def model_names_must_be_unique(self) -> LLMProviderCapabilities:
        names = [model.model for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError("provider model names must be unique.")
        return self


__all__ = [
    "LLMMessageRole",
    "LLMFinishReason",
    "LLMMessage",
    "LLMGenerationConfig",
    "LLMRetryPolicy",
    "LLMRequestOptions",
    "LLMRequest",
    "LLMUsage",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMModelCapabilities",
    "LLMProviderCapabilities",
]
