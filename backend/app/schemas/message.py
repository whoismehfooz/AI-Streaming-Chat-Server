from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


MessageRole = Literal["system","user","assistant"]

class MessageCreate(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1)


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    created_at: datetime


class MessageListResponse(BaseModel):
    items: list[MessageResponse]
    limit: int
    offset: int
