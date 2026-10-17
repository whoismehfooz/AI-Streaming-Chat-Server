from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_ai_service
from app.main import app
from tests.fakes.integration_ai_service import IntegrationAIService



@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_chat_streaming_flow_with_real_database():
    app.dependency_overrides[get_ai_service] = IntegrationAIService

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:

            create_response = await client.post(
                "/conversations",
                json={"title": "Streaming Integration Test Conversation"},
            )

            assert create_response.status_code == 201

            conversation = create_response.json()
            conversation_id = conversation["id"]

            assert UUID(conversation_id)
            assert conversation["title"] == "Streaming Integration Test Conversation"

            # Streaming endpoint hit karo
            chat_response = await client.post(
                f"/conversations/{conversation_id}/chat/stream",
                json={"content": "Hello streaming integration database"},
            )

            assert chat_response.status_code == 200
            assert chat_response.headers["content-type"].startswith("text/event-stream")

            body = chat_response.text
            assert "data: integration " in body
            assert "data: assistant " in body
            assert "data: response" in body
            assert "data: [DONE]" in body

            # Verify karo ki messages DB mein persist ho chuke hain
            messages_response = await client.get(
                f"/conversations/{conversation_id}/messages",
            )

            assert messages_response.status_code == 200

            messages = messages_response.json()["items"]

            assert len(messages) == 2

            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Hello streaming integration database"

            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "integration assistant response"

    finally:
        app.dependency_overrides.clear()
