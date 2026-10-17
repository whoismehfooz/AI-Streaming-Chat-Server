from uuid import UUID
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    content: str=Field(min_length=1)


class ChatResponse(BaseModel):
    conversation_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    content: str
