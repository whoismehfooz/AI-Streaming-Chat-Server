import json
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
                json={
                    "title": "Streaming Integration Test Conversation",
                },
            )

            assert create_response.status_code == 201

            conversation = create_response.json()
            conversation_id = conversation["id"]

            assert UUID(conversation_id)
            assert (
                conversation["title"]
                == "Streaming Integration Test Conversation"
            )

            user_content = "Hello streaming integration database"

            chat_response = await client.post(
                f"/conversations/{conversation_id}/chat/stream",
                json={"content": user_content},
            )

            assert chat_response.status_code == 200
            assert chat_response.headers["content-type"].startswith(
                "text/event-stream"
            )

            events = []

            for block in chat_response.text.strip().split("\n\n"):
                if not block.strip():
                    continue

                event_name = None
                data_lines = []

                for line in block.splitlines():
                    if line.startswith("event: "):
                        event_name = line.removeprefix("event: ")
                    elif line.startswith("data: "):
                        data_lines.append(line.removeprefix("data: "))

                if event_name is not None and data_lines:
                    events.append(
                        {
                            "event": event_name,
                            "data": json.loads("\n".join(data_lines)),
                        }
                    )

            token_events = [
                event for event in events
                if event["event"] == "token"
            ]

            assert [
                event["data"]["text"] for event in token_events
            ] == [
                "integration ",
                "assistant ",
                "response",
            ]

            assert events[-1] == {
                "event": "done",
                "data": {"text": "[DONE]"},
            }

            assert not any(
                event["event"] == "error"
                for event in events
            )

            messages_response = await client.get(
                f"/conversations/{conversation_id}/messages",
            )

            assert messages_response.status_code == 200

            messages = messages_response.json()["items"]

            assert len(messages) == 2

            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == user_content

            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "integration assistant response"

    finally:
        app.dependency_overrides.clear()
