from datetime import datetime
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ConversationBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(
        default=None,
        max_length=255,
    )

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None


class ConversationCreate(ConversationBase):
    pass


class ConversationUpdate(ConversationBase):
    @model_validator(mode="after")
    def validate_update(self) -> "ConversationUpdate":
        if "title" not in self.model_fields_set:
            raise ValueError("At least one field must be provided.")

        return self


class ConversationResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
    )

    id: UUID
    user_id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ConversationResponse]
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
