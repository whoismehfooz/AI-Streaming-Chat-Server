import pytest

from app.services.ai_service import AIService


@pytest.mark.integration
@pytest.mark.asyncio
async def test_groq_real_streaming_response():
    service = AIService()

    tokens = []
    async for token in service.stream_response(
        [
            {
                "role": "user",
                "content": (
                    "Reply with exactly: "
                    "streaming integration test passed"
                ),
            }
        ]
    ):
        tokens.append(token)

    full_response = "".join(tokens)

    assert full_response
    assert isinstance(full_response, str)
    assert "streaming integration test passed" in full_response.lower()
