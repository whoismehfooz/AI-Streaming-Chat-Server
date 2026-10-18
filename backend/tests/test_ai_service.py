from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from app.exceptions.custom_exceptions import AIProviderException
from app.services import ai_service
from app.services import AIService


@pytest.fixture(autouse=True)
def mock_async_openai_client(monkeypatch):
    """Prevent unit tests from requiring real provider credentials."""
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=None),
        ),
        close=AsyncMock(),
    )

    monkeypatch.setattr(
        ai_service,
        "AsyncOpenAI",
        lambda **kwargs: fake_client,
    )


@pytest.mark.asyncio
async def test_stream_response_success():
    service = AIService()

    class FakeChunkDelta:
        content = "test-token"

    class FakeChunkChoice:
        delta = FakeChunkDelta()

    class FakeChunk:
        choices = [FakeChunkChoice()]

    async def generate_chunks():
        yield FakeChunk()
        yield FakeChunk()

    async def mock_stream(*args, **kwargs):
        return generate_chunks()

    service.client.chat.completions.create = mock_stream

    try:
        tokens = []

        async for token in service.stream_response([]):
            tokens.append(token)

        assert tokens == ["test-token", "test-token"]

    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stream_response_rate_limit_error():
    service = AIService()

    request = httpx.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
    )
    response = httpx.Response(429, request=request)

    async def raise_rate_limit(*args, **kwargs):
        raise RateLimitError(
            "rate limited",
            response=response,
            body=None,
        )

    service.client.chat.completions.create = raise_rate_limit

    try:
        with pytest.raises(AIProviderException) as exc_info:
            async for _ in service.stream_response([]):
                pass

        assert exc_info.value.status_code == 429
        assert str(exc_info.value) == (
            "AI provider rate limit or quota exceeded."
        )

    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stream_response_authentication_error():
    service = AIService()

    request = httpx.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
    )
    response = httpx.Response(401, request=request)

    async def raise_auth_error(*args, **kwargs):
        raise AuthenticationError(
            "invalid api key",
            response=response,
            body=None,
        )

    service.client.chat.completions.create = raise_auth_error

    try:
        with pytest.raises(AIProviderException) as exc_info:
            async for _ in service.stream_response([]):
                pass

        assert exc_info.value.status_code == 502
        assert str(exc_info.value) == (
            "AI provider authentication failed."
        )

    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stream_response_timeout_error():
    service = AIService()

    request = httpx.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
    )

    async def raise_timeout(*args, **kwargs):
        raise APITimeoutError(request=request)

    service.client.chat.completions.create = raise_timeout

    try:
        with pytest.raises(AIProviderException) as exc_info:
            async for _ in service.stream_response([]):
                pass

        assert exc_info.value.status_code == 504
        assert str(exc_info.value) == "AI provider timed out."

    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stream_response_connection_error():
    service = AIService()

    request = httpx.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
    )

    async def raise_connection_error(*args, **kwargs):
        raise APIConnectionError(request=request)

    service.client.chat.completions.create = raise_connection_error

    try:
        with pytest.raises(AIProviderException) as exc_info:
            async for _ in service.stream_response([]):
                pass

        assert exc_info.value.status_code == 503
        assert str(exc_info.value) == (
            "unable to connect to AI provider."
        )

    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stream_response_status_error():
    service = AIService()

    request = httpx.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
    )
    response = httpx.Response(500, request=request)

    async def raise_status_error(*args, **kwargs):
        raise APIStatusError(
            "server error",
            response=response,
            body=None,
        )

    service.client.chat.completions.create = raise_status_error

    try:
        with pytest.raises(AIProviderException) as exc_info:
            async for _ in service.stream_response([]):
                pass

        assert exc_info.value.status_code == 502
        assert str(exc_info.value) == (
            "AI provider request failed."
        )

    finally:
        await service.close()
