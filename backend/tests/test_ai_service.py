import pytest 
import httpx2

from openai import (
    RateLimitError,
    AuthenticationError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError
)

from app.exceptions.custom_exceptions import AIProviderException
from app.services import AIService



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

    async def mock_stream(*args,**kwargs):
        return generate_chunks()

    service.client.chat.completions.create = mock_stream

    tokens = []
    async for token in service.stream_response([]):
        tokens.append(token)

    assert tokens == ["test-token","test-token"]



@pytest.mark.asyncio
async def test_stream_response_rate_limit_error():
    service = AIService()

    http_request = httpx2.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions"
    )
    http_response = httpx2.Response(
        429,
        request=http_request,
    )

    async def raise_rate_limit(*args, **kwargs):
        raise RateLimitError(
            "rate limited",
            response=http_response,
            body=None,
        )

    service.client.chat.completions.create = raise_rate_limit

    with pytest.raises(AIProviderException) as exc_info:
        async for _ in service.stream_response([]):
            pass

    assert exc_info.value.status_code == 429
    assert str(exc_info.value) == "AI provider rate limit or quota exceeded."


@pytest.mark.asyncio
async def test_stream_response_authentication_error():
    service = AIService()

    http_request = httpx2.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions"
    )
    http_response = httpx2.Response(
        401,
        request=http_request
    )

    async def raise_auth_error(*args, **kwargs):
        raise AuthenticationError(
            "invalid api key",
            response=http_response,
            body=None
        )

    service.client.chat.completions.create = raise_auth_error

    with pytest.raises(AIProviderException) as exc_info:
        async for _ in service.stream_response([]):
            pass

    assert exc_info.value.status_code == 502
    assert str(exc_info.value) == "AI provider authentication failed."


@pytest.mark.asyncio
async def test_stream_response_timeout_error():
    service = AIService()

    http_request = httpx2.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions"
    )

    async def raise_timeout(*args, **kwargs):
        raise APITimeoutError(request=http_request)

    service.client.chat.completions.create = raise_timeout

    with pytest.raises(AIProviderException) as exc_info:
        async for _ in service.stream_response([]):
            pass

    assert exc_info.value.status_code == 504
    assert str(exc_info.value) == "AI provider timed out."


@pytest.mark.asyncio
async def test_stream_response_connection_error():
    service = AIService()

    http_request = httpx2.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions"
    )

    async def raise_conn_error(*args, **kwargs):
        raise APIConnectionError(request=http_request)

    service.client.chat.completions.create = raise_conn_error

    with pytest.raises(AIProviderException) as exc_info:
        async for _ in service.stream_response([]):
            pass

    assert exc_info.value.status_code == 503
    assert str(exc_info.value) == "unable to connect to AI provider."


@pytest.mark.asyncio
async def test_stream_response_status_error():
    service = AIService()

    http_request = httpx2.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions"
    )
    http_response = httpx2.Response(
        500,
        request=http_request
    )

    async def raise_status_error(*args, **kwargs):
        raise APIStatusError(
            "server error",
            response=http_response,
            body=None
        )

    service.client.chat.completions.create = raise_status_error

    with pytest.raises(AIProviderException) as exc_info:
        async for _ in service.stream_response([]):
            pass

    assert exc_info.value.status_code == 502
    assert str(exc_info.value) == "AI provider request failed."
