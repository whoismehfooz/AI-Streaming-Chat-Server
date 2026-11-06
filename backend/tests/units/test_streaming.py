
import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_ai_service
from app.main import app


class FakeStreamingAIService:
    async def stream_response(self, messages):
        yield "Hello"
        yield "from"
        yield "stream!"


@pytest.mark.asyncio
async def test_streaming_endpoint():
    app.dependency_overrides[get_ai_service] = FakeStreamingAIService

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            conversation_response = await client.post(
                "/conversations",
                json={"title": "Streaming Test"},
            )

            assert conversation_response.status_code == 201
            conversation_id = conversation_response.json()["id"]

            response = await client.post(
                f"/conversations/{conversation_id}/chat/stream",
                json={"content": "Test stream"},
            )

            assert response.status_code == 200
            assert response.headers["content-type"].startswith(
                "text/event-stream"
            )

            events = []

            for block in response.text.strip().split("\n\n"):
                if not block:
                    continue

                event_name = None
                data_lines = []

                for line in block.splitlines():
                    if line.startswith("event: "):
                        event_name = line.removeprefix("event: ")
                    elif line.startswith("data: "):
                        data_lines.append(line.removeprefix("data: "))

                if event_name is not None:
                    events.append(
                        {
                            "event": event_name,
                            "data": json.loads("\n".join(data_lines)),
                        }
                    )

            token_events = [
                item for item in events if item["event"] == "token"
            ]

            assert [item["data"]["text"] for item in token_events] == [
                "Hello",
                "from",
                "stream!",
            ]

            assert events[-1] == {
                "event": "done",
                "data": {"text": "[DONE]"},
            }

            assert not any(
                item["event"] == "error" for item in events
            )

    finally:
        app.dependency_overrides.pop(get_ai_service, None)
