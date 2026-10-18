import pytest

from app.services.ai_service import AIService


@pytest.mark.integration
@pytest.mark.asyncio
async def test_groq_real_streaming_response():
    service = AIService()

    try:
        chunks = []

        async for chunk in service.stream_response(
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
            chunks.append(chunk)

        full_response = "".join(chunks)

        assert full_response
        assert isinstance(full_response, str)
        assert "streaming integration test passed" in full_response.lower()

    finally:
        await service.close()
