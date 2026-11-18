from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.integrations.llm.schemas import LLMFinishReason, LLMUsage


class ChatResponse(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: Literal["completed"] = "completed"

    conversation_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID

    content: Annotated[str, Field(min_length=1)]
    finish_reason: LLMFinishReason | None = None
    usage: LLMUsage | None = None

    created_at: datetime

    @field_validator("content")
    @classmethod
    def content_must_contain_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Assistant response content cannot be blank.")
        return value

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware.")
        return value


__all__ = ["ChatResponse"]
