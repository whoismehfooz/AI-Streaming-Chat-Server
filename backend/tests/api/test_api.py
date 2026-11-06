import pytest
from httpx import ASGITransport, AsyncClient
from uuid import uuid4

from sqlalchemy import select

from app.main import app
from app.models import Message
from app.api.dependencies import get_ai_service
from tests.fakes.failing_ai_service import FailingAIService


class FakeAIService:
    def __init__(self):
        self.calls = []

    async def generate_response(self,messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)

        return f"Fake response {len(self.calls)}"


class FakeStreamingAIService:
        async def stream_response(self, messages):
            yield "Stream "
            yield "token!"


@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        response = await client.get("/health")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "healthy"
        assert data["service"] == "AI Streaming Chat Server"
        assert data["environment"] == "development"
        assert data["version"] == "v1.0.0"


@pytest.mark.asyncio
async def test_create_conversation():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        response = await client.post(
            "/conversations",
            json={
                "title":"API integration test"
            }
        )

        assert response.status_code == 201

        data = response.json()

        assert data["id"]
        assert data["title"] == "API integration test"
        assert data["created_at"] 
        assert data["updated_at"]


@pytest.mark.asyncio
async def test_get_conversation():
    tranpsort = ASGITransport(app=app)

    async with AsyncClient(
        transport=tranpsort,
        base_url="http://test"
    ) as client:
        create_response = await client.post(
            "/conversations",
            json={
                "title":"Retrievel test"
            }
        )

        assert create_response.status_code == 201

        conversation_id = create_response.json()["id"]

        response = await client.get(
            f"/conversations/{conversation_id}"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == conversation_id
    assert data["title"] == "Retrievel test"


@pytest.mark.asyncio
async def test_list_converstions():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        create_response = await client.post(
            "/conversations",
            json={
                "title": "List test"
            }
        )

        assert create_response.status_code == 201

        response = await client.get(
            "/conversations",
            params={
                "limit":10,
                "offset": 0
            }
        )

        assert response.status_code == 200

        data = response.json()

        assert "items" in data
        assert data["limit"] == 10
        assert data["offset"] == 0
        assert any(
            item["title"] == "List test"
            for item in data["items"]
        )


@pytest.mark.asyncio
async def test_get_nonexistent_conversation():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        response = await client.get(
            f"/conversations/{uuid4()}"
        )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Conversation Not Found."
    }


@pytest.mark.asyncio
async def test_create_message():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        conversation_response = await client.post(
            "/conversations",
            json={
                "title":"Message API test"
            }
        )

        assert conversation_response.status_code == 201

        conversation_id = conversation_response.json()["id"]

        response = await client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "role":"user",
                "content":"Hello from API integration test."
            }
        )

    assert response.status_code == 201

    data = response.json()

    assert data["id"] 
    assert data["conversation_id"] == conversation_id
    assert data["role"] == "user"
    assert data["content"] == (
        "Hello from API integration test."
    )
    assert data["created_at"]


@pytest.mark.asyncio
async def test_list_messages():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        conversation_response = await client.post(
            "/conversations",
            json={
                "title":"History API test."
            }
        )

        assert conversation_response.status_code == 201

        conversation_id = conversation_response.json()["id"]

        first_message = await client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "role":"user",
                "content":"First message"
            }
        )

        second_message = await client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "role": "assistant",
                "content":"Second message",
            }
        )

        assert first_message.status_code == 201
        assert second_message.status_code == 201

        response = await client.get(
            f"/conversations/{conversation_id}/messages",
            params={
                "limit": 10,
                "offset":0
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert len(data["items"]) == 2

    assert data["items"][0]["content"] == "First message"
    assert data["items"][1]["content"] == "Second message"


@pytest.mark.asyncio
async def test_create_message_rejects_empty_content():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        conversation_response = await client.post(
            "/conversations",
            json={
                "title":"Validation test"
            }
        )

        conversation_id = conversation_response.json()["id"]

        response = await client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "role":"user",
                "content":""
            }
        )

        assert response.status_code == 422



@pytest.mark.asyncio
async def test_create_message_for_nonexistent_conversation():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/conversations/{uuid4()}/messages",
            json={
                "role": "user",
                "content": "This should fail.",
            },
        )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Conversation Not Found."
    }


@pytest.mark.asyncio
async def test_chat_streaming_endpoint():
    class FakeStreamingAIService:
        async def stream_response(self, messages):
            yield "Stream "
            yield "token!"

    app.dependency_overrides[get_ai_service] = FakeStreamingAIService

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            conversation_response = await client.post(
                "/conversations",
                json={"title": "API Stream Test"},
            )

            assert conversation_response.status_code == 201
            conversation_id = conversation_response.json()["id"]

            response = await client.post(
                f"/conversations/{conversation_id}/chat/stream",
                json={"content": "Hello stream"},
            )

            assert response.status_code == 200
            assert response.headers["content-type"].startswith(
                "text/event-stream"
            )

            body = response.text

            assert 'event: token\ndata: {"text": "Stream "}' in body
            assert 'event: token\ndata: {"text": "token!"}' in body
            assert 'event: done\ndata: {"text": "[DONE]"}' in body
            assert "event: error" not in body

    finally:
        app.dependency_overrides.pop(get_ai_service, None)



@pytest.mark.asyncio
async def test_chat_streaming_endpoint_nonexistent_conversation():
    class FakeStreamingAIService:
        async def stream_response(self, messages):
            yield "This should not be reached"

    app.dependency_overrides[get_ai_service] = FakeStreamingAIService

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"/conversations/{uuid4()}/chat/stream",
                json={"content": "This should fail"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "text/event-stream"
        )

        body = response.text

        assert "event: error" in body
        assert "Conversation Not Found." in body
        assert "event: done" not in body

    finally:
        app.dependency_overrides.pop(get_ai_service, None)


@pytest.mark.asyncio
async def test_chat_stream_endpoint_with_persisted_message():
    class StreamingFakeAIService:
        def __init__(self):
            self.calls = []

        async def stream_response(self, messages):
            self.calls.append(messages)
            yield "Fake "
            yield "response 1"

    fake_ai_service = StreamingFakeAIService()
    app.dependency_overrides[get_ai_service] = lambda: fake_ai_service

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            conversation_response = await client.post(
                "/conversations",
                json={"title": "Persistence Streaming Test"},
            )

            assert conversation_response.status_code == 201
            conversation_id = conversation_response.json()["id"]

            response = await client.post(
                f"/conversations/{conversation_id}/chat/stream",
                json={"content": "First user message"},
            )

            assert response.status_code == 200

            body = response.text

            assert 'event: token\ndata: {"text": "Fake "}' in body
            assert 'event: token\ndata: {"text": "response 1"}' in body
            assert 'event: done\ndata: {"text": "[DONE]"}' in body
            assert "event: error" not in body

            history_response = await client.get(
                f"/conversations/{conversation_id}/messages"
            )

            assert history_response.status_code == 200

            messages = history_response.json()["items"]

            assert len(messages) == 2

            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "First user message"

            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "Fake response 1"

    finally:
        app.dependency_overrides.pop(get_ai_service, None)


@pytest.mark.asyncio
async def test_chat_stream_endpoint_preserves_multi_turn_context():
    class StreamingFakeAIService:
        def __init__(self):
            self.calls = []

        async def stream_response(self, messages):
            self.calls.append(messages)
            yield f"Fake response {len(self.calls)}"

    fake_ai_service = StreamingFakeAIService()
    app.dependency_overrides[get_ai_service] = lambda: fake_ai_service

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            conversation_response = await client.post(
                "/conversations",
                json={"title": "Multi-turn Streaming test"},
            )

            assert conversation_response.status_code == 201
            conversation_id = conversation_response.json()["id"]

            first_message = await client.post(
                f"/conversations/{conversation_id}/chat/stream",
                json={"content": "My name is Ray."},
            )

            assert first_message.status_code == 200
            assert 'event: done\ndata: {"text": "[DONE]"}' in (
                first_message.text
            )

            second_message = await client.post(
                f"/conversations/{conversation_id}/chat/stream",
                json={"content": "What is my name?"},
            )

            assert second_message.status_code == 200
            assert 'event: done\ndata: {"text": "[DONE]"}' in (
                second_message.text
            )

            assert len(fake_ai_service.calls) == 2

            assert fake_ai_service.calls[0] == [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "My name is Ray.",
                },
            ]

            assert fake_ai_service.calls[1] == [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "My name is Ray.",
                },
                {
                    "role": "assistant",
                    "content": "Fake response 1",
                },
                {
                    "role": "user",
                    "content": "What is my name?",
                },
            ]

    finally:
        app.dependency_overrides.pop(get_ai_service, None)


@pytest.mark.asyncio
async def test_chat_stream_persists_user_message_when_ai_provider_fails(
    test_database,
):
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        conversation_response = await client.post(
            "/conversations",
            json={"title": "AI Failure Streaming Test"},
        )

        assert conversation_response.status_code == 201
        conversation_id = conversation_response.json()["id"]

        app.dependency_overrides[get_ai_service] = FailingAIService

        try:
            response = await client.post(
                f"/conversations/{conversation_id}/chat/stream",
                json={
                    "content": "This message should survive an AI failure."
                },
            )

            assert response.status_code == 200
            assert response.headers["content-type"].startswith(
                "text/event-stream"
            )

            body = response.text

            assert "event: error" in body
            assert "AI provider request failed." in body
            assert "event: done" not in body

            async with test_database() as session:
                result = await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at.asc())
                )

                messages = list(result.scalars().all())

            assert len(messages) == 1
            assert messages[0].role == "user"
            assert messages[0].content == (
                "This message should survive an AI failure."
            )

            assert not any(
                message.role == "assistant"
                for message in messages
            )

        finally:
            app.dependency_overrides.pop(get_ai_service, None)
