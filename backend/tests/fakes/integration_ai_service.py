class IntegrationAIService:

    async def stream_response(
        self,
        messages: list[dict[str, str]],
    ):
        yield "integration "
        yield "assistant "
        yield "response"
