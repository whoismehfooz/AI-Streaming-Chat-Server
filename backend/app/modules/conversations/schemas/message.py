from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


MessageRole = Literal[
    "system",
    "user",
    "assistant",
]


class MessageCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message content cannot be empty or whitespace.")

        return value


class MessageResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
    )

    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    created_at: datetime


class MessageListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[MessageResponse]
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
