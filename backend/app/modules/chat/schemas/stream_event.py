from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.exceptions.error_codes import ErrorCode
from app.integrations.llm.schemas import LLMFinishReason, LLMUsage


class ChatStreamEventBase(BaseModel):
    """Shared correlation, ownership-context, and ordering metadata."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    conversation_id: UUID
    request_id: Annotated[str, Field(min_length=1, max_length=128)]
    sequence_number: Annotated[int, Field(strict=True, ge=0)]
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("request_id")
    @classmethod
    def request_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("request_id cannot be blank.")
        return value

    @field_validator("occurred_at")
    @classmethod
    def occurred_at_must_be_timezone_aware(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware.")
        return value


class ChatStreamStartedEvent(ChatStreamEventBase):
    """
    Announces that the users turn has been accepted and generation is starting.

    The assistant message ID is deliberately not included: with the current
    MessageService contract, that database-generated ID does not exist yet.
    """

    event: Literal["message_start"] = "message_start"
    user_message_id: UUID


class ChatContentDeltaEvent(ChatStreamEventBase):
    """Carries one non-empty text delta from the normalized LLM stream."""

    event: Literal["content_delta"] = "content_delta"
    delta: Annotated[str, Field(min_length=1)]

class ChatStreamCompletedEvent(ChatStreamEventBase):
    """
    Marks successful generation and persistence of the assistant message.

    This terminal event must not be emitted before the final assistant
    Message record has been committed and its database-generated ID obtained.
    """

    event: Literal["message_complete"] = "message_complete"
    assistant_message_id: UUID
    finish_reason: LLMFinishReason | None = None
    usage: LLMUsage | None = None


class ChatStreamErrorEvent(ChatStreamEventBase):
    """
    Safe, machine-readable failure notification for an established stream.

    ``message`` must be a client-safe message. Never populate it with raw
    provider SDK exception text, stack traces, SQL errors, secrets, or
    internal infrastructure details.

    Retry metadata is descriptive only. Clients must not blindly retry a
    partially delivered turn unless the API later provides an idempotency
    contract for that operation.
    """

    event: Literal["error"] = "error"
    error_code: ErrorCode
    message: Annotated[str, Field(min_length=1, max_length=512)]
    retryable: bool = False
    retry_after_seconds: Annotated[int | None, Field(default=None, strict=True, ge=0)]

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Stream error message cannot be blank.")
        return value


class ChatStreamHeartbeatEvent(ChatStreamEventBase):
    """
    Optional keep-alive event for deployments that need application traffic.

    Heartbeats are transport-liveness signals. They are not model output and
    must never be persisted as chat messages.
    """

    event: Literal["heartbeat"] = "heartbeat"


ChatStreamEvent: TypeAlias = Annotated[
    (
        ChatStreamStartedEvent
        | ChatContentDeltaEvent
        | ChatStreamCompletedEvent
        | ChatStreamErrorEvent
        | ChatStreamHeartbeatEvent
    ),
    Field(discriminator="event"),
]


__all__ = [
    "ChatStreamEventBase",
    "ChatStreamStartedEvent",
    "ChatContentDeltaEvent",
    "ChatStreamCompletedEvent",
    "ChatStreamErrorEvent",
    "ChatStreamHeartbeatEvent",
    "ChatStreamEvent",
]
