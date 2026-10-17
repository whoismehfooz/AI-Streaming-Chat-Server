from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession


from app.db.dependencies import get_db_session
from app.schemas.conversation import (
    ConversationCreate,
    ConversationResponse,
    ConversationListResponse
)
from app.services.conversation_service import ConversationService
from app.exceptions.custom_exceptions import ConversationNotFoundException

conversation_router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)


@conversation_router.post("",response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate,
    session: AsyncSession = Depends(get_db_session)
):
    service = ConversationService(session)

    return await service.create_conversation(
        title=payload.title
    )


@conversation_router.get("",response_model=ConversationListResponse, status_code=status.HTTP_200_OK)
async def list_conversations(
    limit: int = Query(
        default=20,
        ge=1,
        le=100
    ),
    offset: int = Query(
        default=0,
        ge=0
    ),
    session: AsyncSession = Depends(get_db_session)
):
    service = ConversationService(session)

    conversations = await service.list_conversations(
        limit=limit,
        offset=offset
    )

    return ConversationListResponse(
        items=conversations,
        limit=limit,
        offset=offset
    )


@conversation_router.get("/{conversation_id}", response_model=ConversationResponse, status_code=status.HTTP_200_OK)
async def get_conversation(
    conversation_id: UUID,
    session: AsyncSession = Depends(get_db_session)
):
    serivce = ConversationService(session)

    conversation = await serivce.get_conversation(
        conversation_id
    )

    if conversation is None:
        raise ConversationNotFoundException(
            "Conversation Not Found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    return conversation


@conversation_router.delete("/{conversation_id}",status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    session: AsyncSession = Depends(get_db_session)
):
    service = ConversationService(session)

    deleted = await service.delete_conversation(
        conversation_id
    )

    if not deleted:
        raise ConversationNotFoundException(
            "Conversation Not Found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )
