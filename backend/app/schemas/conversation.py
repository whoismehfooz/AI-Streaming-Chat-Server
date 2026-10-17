from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field



class ConversationCreate(BaseModel):
    title: str | None = Field(
        default=None,
        max_length=255
    )


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]
    limit: int
    offset: int
