from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.dependencies import get_db_session
from app.schemas.message import (
    MessageCreate,
    MessageResponse,
    MessageListResponse
)

from app.services import ConversationService, MessageService
from app.exceptions.custom_exceptions import ConversationNotFoundException


message_router = APIRouter(
    prefix="/conversations/{conversation_id}/messages",
    tags=["Messages"]
)


@message_router.post("",response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def create_message(
    conversation_id: UUID,
    payload: MessageCreate,
    session: AsyncSession = Depends(get_db_session)
):
    conversation_service = ConversationService(session)

    conversation = await conversation_service.get_conversation(
        conversation_id
    )

    if not conversation:
        raise ConversationNotFoundException(
            "Conversation Not Found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    message_service = MessageService(session)

    return await message_service.create_message(
        conversation_id=conversation_id,
        role=payload.role,
        content=payload.content
    )


@message_router.get("",response_model=MessageListResponse, status_code=status.HTTP_200_OK)
async def list_messages(
    conversation_id: UUID,
    limit: int = Query(
        default=50,
        ge=1,
        le=100
    ),
    offset: int = Query(
        default=0,
        ge=0
    ),
    session: AsyncSession = Depends(get_db_session)
):
    conversation_service = ConversationService(session)

    conversation = await conversation_service.get_conversation(
        conversation_id
    )

    if not conversation:
        raise ConversationNotFoundException(
            "Conversation Not Found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    message_service = MessageService(session)

    messages = await message_service.list_messages(
        conversation_id=conversation_id,
        limit=limit,
        offset=offset
    )

    return MessageListResponse(
        items=messages,
        limit=limit,
        offset=offset
    )
