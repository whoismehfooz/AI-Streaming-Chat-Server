from app.exceptions.custom_exceptions import AIProviderException


class FailingAIService:
    
    async def stream_response(
        self,
        messages: list[dict[str, str]],
    ):
        raise AIProviderException(
            "AI provider request failed.",
            status_code=502,
        )
        yield  