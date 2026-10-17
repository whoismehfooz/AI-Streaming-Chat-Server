import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.dependencies import get_ai_service


class FakeStreamingAIService:
    async def stream_response(self, messages):
        yield "Hello"
        yield "from"
        yield "stream!"


@pytest.mark.asyncio
async def test_streaming_endpoint():
    app.dependency_overrides[get_ai_service] = FakeStreamingAIService

    try:
        tranport = ASGITransport(app=app)
        async with AsyncClient(
            transport=tranport,
            base_url="http://test"
        ) as client:
            conversation_response = await client.post(
                "/conversations",
                json={
                    "title":"Streaming Test"
                }
            )

            assert conversation_response.status_code == 201

            conversation_id = conversation_response.json()["id"]

            response = await client.post(
                f"/conversations/{conversation_id}/chat/stream",
                json={
                    "content":"Test stream"
                }
            )

            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

            body = response.text
            assert "data: Hello" in body
            assert "data: from" in body
            assert "data: stream!" in body
            assert "data: [DONE]" in body

    finally:
        app.dependency_overrides.clear()
