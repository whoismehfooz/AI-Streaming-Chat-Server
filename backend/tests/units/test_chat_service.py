
import asyncio
from uuid import uuid4

import pytest

from app.exceptions.custom_exceptions import (
    AIProviderException,
    ConversationNotFoundException,
)
from app.models import Conversation, Message
from app.services import ChatService


class FakeConversationService:
    def __init__(self, conversation):
        self.conversation = conversation
        self.requested_conversation_ids = []

    async def get_conversation(self, conversation_id):
        self.requested_conversation_ids.append(conversation_id)
        return self.conversation


class MissingConversationService:
    async def get_conversation(self, conversation_id):
        return None


class FakeMessageService:
    def __init__(
        self,
        messages=None,
        user_message=None,
        assistant_message=None,
        *,
        fail_on_role=None,
        create_error=None,
    ):
        self.messages = messages or []
        self.user_message = user_message
        self.assistant_message = assistant_message
        self.created_messages = []
        self.recent_message_requests = []
        self.fail_on_role = fail_on_role
        self.create_error = create_error

    async def create_message(
        self,
        conversation_id,
        role,
        content,
    ):
        if role == self.fail_on_role:
            if self.create_error is not None:
                raise self.create_error
            raise RuntimeError(f"Failed to persist {role} message.")

        self.created_messages.append(
            {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
            }
        )

        if role == "user":
            return self.user_message

        return self.assistant_message

    async def list_recent_messages(
        self,
        conversation_id,
        limit,
    ):
        self.recent_message_requests.append(
            {
                "conversation_id": conversation_id,
                "limit": limit,
            }
        )
        return self.messages


class FakeContextBuilder:
    def __init__(self, result=None, error=None):
        self.received_messages = None
        self.error = error
        self.result = result

    def build(self, messages):
        self.received_messages = messages

        if self.error is not None:
            raise self.error

        if self.result is not None:
            return self.result

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]


class FakeAIService:
    def __init__(
        self,
        chunks=None,
        error=None,
    ):
        self.chunks = (
            ["Fake ", "assistant ", "response"]
            if chunks is None
            else chunks
        )
        self.error = error
        self.received_context = None
        self.call_count = 0

    async def stream_response(self, messages):
        self.call_count += 1
        self.received_context = messages

        for chunk in self.chunks:
            yield chunk

        if self.error is not None:
            raise self.error


def build_chat_service(
    *,
    conversation=None,
    messages=None,
    chunks=None,
    ai_error=None,
    context_error=None,
    context_result=None,
    fail_on_role=None,
    persistence_error=None,
):
    conversation_id = (
        conversation.id if conversation is not None else uuid4()
    )

    user_message = Message(
        id=uuid4(),
        conversation_id=conversation_id,
        role="user",
        content="Hello",
    )

    assistant_message = Message(
        id=uuid4(),
        conversation_id=conversation_id,
        role="assistant",
        content="Fake assistant response",
    )

    conversation_service = (
        FakeConversationService(conversation)
        if conversation is not None
        else MissingConversationService()
    )

    message_service = FakeMessageService(
        messages=messages if messages is not None else [user_message],
        user_message=user_message,
        assistant_message=assistant_message,
        fail_on_role=fail_on_role,
        create_error=persistence_error,
    )

    context_builder = FakeContextBuilder(
        result=context_result,
        error=context_error,
    )

    ai_service = FakeAIService(
        chunks=chunks,
        error=ai_error,
    )

    chat_service = ChatService(
        conversation_service=conversation_service,
        message_service=message_service,
        context_builder=context_builder,
        ai_service=ai_service,
    )

    return (
        chat_service,
        conversation_id,
        message_service,
        context_builder,
        ai_service,
        user_message,
    )


@pytest.mark.asyncio
async def test_stream_message_happy_path():
    conversation_id = uuid4()

    conversation = Conversation(
        id=conversation_id,
        title="Test Conversation",
    )

    user_message = Message(
        id=uuid4(),
        conversation_id=conversation_id,
        role="user",
        content="Hello",
    )

    message_service = FakeMessageService(
        messages=[user_message],
        user_message=user_message,
        assistant_message=Message(
            id=uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content="Fake assistant response",
        ),
    )

    context_builder = FakeContextBuilder()
    ai_service = FakeAIService()

    chat_service = ChatService(
        conversation_service=FakeConversationService(conversation),
        message_service=message_service,
        context_builder=context_builder,
        ai_service=ai_service,
    )

    chunks = []

    async for chunk in chat_service.stream_message(
        conversation_id=conversation_id,
        content="Hello",
    ):
        chunks.append(chunk)

    assert "".join(chunks) == "Fake assistant response"

    # The user message is persisted before the response is generated.
    assert len(message_service.created_messages) == 2

    assert message_service.created_messages[0] == {
        "conversation_id": conversation_id,
        "role": "user",
        "content": "Hello",
    }

    assert message_service.created_messages[1] == {
        "conversation_id": conversation_id,
        "role": "assistant",
        "content": "Fake assistant response",
    }

    assert context_builder.received_messages == [user_message]

    assert ai_service.received_context == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]

    assert message_service.recent_message_requests == [
        {
            "conversation_id": conversation_id,
            "limit": 100,
        }
    ]


@pytest.mark.asyncio
async def test_stream_message_conversation_not_found():
    (
        chat_service,
        conversation_id,
        message_service,
        _,
        ai_service,
        _,
    ) = build_chat_service()

    with pytest.raises(
        ConversationNotFoundException,
        match="Conversation Not Found",
    ):
        async for _ in chat_service.stream_message(
            conversation_id=conversation_id,
            content="Hello",
        ):
            pass

    assert message_service.created_messages == []
    assert ai_service.call_count == 0


@pytest.mark.asyncio
async def test_stream_message_uses_context_builder_output():
    conversation_id = uuid4()

    conversation = Conversation(
        id=conversation_id,
        title="Context Test",
    )

    custom_context = [
        {
            "role": "system",
            "content": "Follow the test instructions.",
        },
        {
            "role": "user",
            "content": "A custom context message.",
        },
    ]

    (
        chat_service,
        _,
        _,
        _,
        ai_service,
        _,
    ) = build_chat_service(
        conversation=conversation,
        context_result=custom_context,
    )

    async for _ in chat_service.stream_message(
        conversation_id=conversation_id,
        content="Hello",
    ):
        pass

    assert ai_service.received_context == custom_context


@pytest.mark.asyncio
async def test_stream_message_ignores_empty_or_non_string_chunks():
    conversation_id = uuid4()

    conversation = Conversation(
        id=conversation_id,
        title="Chunk Validation Test",
    )

    (
        chat_service,
        _,
        message_service,
        _,
        _,
        _,
    ) = build_chat_service(
        conversation=conversation,
        chunks=["Hello", "", None, 123, " world"],
    )

    received_chunks = []

    async for chunk in chat_service.stream_message(
        conversation_id=conversation_id,
        content="Hello",
    ):
        received_chunks.append(chunk)

    assert received_chunks == ["Hello", " world"]

    assistant_messages = [
        message
        for message in message_service.created_messages
        if message["role"] == "assistant"
    ]

    assert len(assistant_messages) == 1
    assert assistant_messages[0]["content"] == "Hello world"


@pytest.mark.asyncio
async def test_stream_message_does_not_persist_empty_assistant_response():
    conversation_id = uuid4()

    conversation = Conversation(
        id=conversation_id,
        title="Empty Response Test",
    )

    (
        chat_service,
        _,
        message_service,
        _,
        _,
        _,
    ) = build_chat_service(
        conversation=conversation,
        chunks=["", None],
    )

    received_chunks = []

    async for chunk in chat_service.stream_message(
        conversation_id=conversation_id,
        content="Hello",
    ):
        received_chunks.append(chunk)

    assert received_chunks == []

    assert [
        message["role"]
        for message in message_service.created_messages
    ] == ["user"]


@pytest.mark.asyncio
async def test_stream_message_persists_partial_response_on_provider_failure():
    conversation_id = uuid4()

    conversation = Conversation(
        id=conversation_id,
        title="Provider Failure Test",
    )

    (
        chat_service,
        _,
        message_service,
        _,
        _,
        _,
    ) = build_chat_service(
        conversation=conversation,
        chunks=["Partial ", "response"],
        ai_error=AIProviderException(
            "AI provider request failed.",
            status_code=502,
        ),
    )

    received_chunks = []

    with pytest.raises(
        AIProviderException,
        match="AI provider request failed",
    ):
        async for chunk in chat_service.stream_message(
            conversation_id=conversation_id,
            content="Hello",
        ):
            received_chunks.append(chunk)

    assert received_chunks == ["Partial ", "response"]

    assert [
        message["role"]
        for message in message_service.created_messages
    ] == ["user", "assistant"]

    assert message_service.created_messages[1]["content"] == (
        "Partial response\n\n[Stream Interrupted]"
    )


@pytest.mark.asyncio
async def test_stream_message_persists_partial_response_on_cancellation():
    conversation_id = uuid4()

    conversation = Conversation(
        id=conversation_id,
        title="Cancellation Test",
    )

    (
        chat_service,
        _,
        message_service,
        _,
        _,
        _,
    ) = build_chat_service(
        conversation=conversation,
        chunks=["Partial ", "response"],
        ai_error=asyncio.CancelledError(),
    )

    received_chunks = []

    with pytest.raises(asyncio.CancelledError):
        async for chunk in chat_service.stream_message(
            conversation_id=conversation_id,
            content="Hello",
        ):
            received_chunks.append(chunk)

    assert received_chunks == ["Partial ", "response"]

    assert [
        message["role"]
        for message in message_service.created_messages
    ] == ["user", "assistant"]

    assert message_service.created_messages[1]["content"] == (
        "Partial response\n\n[Stream Cancelled]"
    )


@pytest.mark.asyncio
async def test_stream_message_persists_user_message_if_context_building_fails():
    conversation_id = uuid4()

    conversation = Conversation(
        id=conversation_id,
        title="Context Failure Test",
    )

    (
        chat_service,
        _,
        message_service,
        _,
        ai_service,
        _,
    ) = build_chat_service(
        conversation=conversation,
        context_error=RuntimeError("Context construction failed."),
    )

    with pytest.raises(
        RuntimeError,
        match="Context construction failed",
    ):
        async for _ in chat_service.stream_message(
            conversation_id=conversation_id,
            content="Hello",
        ):
            pass

    assert [
        message["role"]
        for message in message_service.created_messages
    ] == ["user"]

    assert ai_service.call_count == 0


@pytest.mark.asyncio
async def test_stream_message_propagates_assistant_persistence_failure():
    conversation_id = uuid4()

    conversation = Conversation(
        id=conversation_id,
        title="Persistence Failure Test",
    )

    (
        chat_service,
        _,
        message_service,
        _,
        _,
        _,
    ) = build_chat_service(
        conversation=conversation,
        fail_on_role="assistant",
        persistence_error=RuntimeError(
            "Assistant persistence failed."
        ),
    )

    received_chunks = []

    with pytest.raises(
        RuntimeError,
        match="Assistant persistence failed",
    ):
        async for chunk in chat_service.stream_message(
            conversation_id=conversation_id,
            content="Hello",
        ):
            received_chunks.append(chunk)

    assert "".join(received_chunks) == "Fake assistant response"

    # The user message succeeded, but the assistant message did not.
    assert [
        message["role"]
        for message in message_service.created_messages
    ] == ["user"]
