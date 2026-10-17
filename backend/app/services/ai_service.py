import logging
import time
from typing import AsyncGenerator

from openai import (
    AsyncOpenAI,
    RateLimitError,
    AuthenticationError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError
)

from app.core.config import settings
from app.exceptions.custom_exceptions import AIProviderException


logger = logging.getLogger(__name__)


class AIService:

    MODEL = settings.GROQ_MODEL

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

    async def stream_response(self, messages: list[dict[str,str]]) -> AsyncGenerator[str,None]:
        try:
            start_time=time.perf_counter()
            logger.info(
                "Starting AI provider streaming execution: model=%s",
                self.MODEL
            )

            response = await self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                max_tokens=settings.CHAT_MAX_OUTPUT_TOKENS,
                stream=True
            )

            token_count = 0

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    token_count+=1
                    yield token

            duration = time.perf_counter() - start_time
            logger.info(
                "AI provider streaming execution completed sucessfully: model=%s, duration=%.2fs, total_tokens=%d",
                self.MODEL,
                duration,
                token_count
            )

        except RateLimitError as exc:
            logger.error(
                "AI provider limit rate limit error during stream: model=%s",
                self.MODEL
            )
            raise AIProviderException(
                "AI provider rate limit or quota exceeded.",
                status_code=429
            ) from exc

        except AuthenticationError as exc:
            logger.error(
                "AI provider authentication failed during stream: model=%s",
                self.MODEL
            )
            raise AIProviderException(
                "AI provider authentication failed.",
                status_code=502
            ) from exc

        except APITimeoutError as exc:
            logger.error(
                "AI provider timed out during stream: model=%s",
                self.MODEL
            )
            raise AIProviderException(
                "AI provider timed out.",
                status_code=504
            )

        except APIConnectionError as exc:
            logger.error(
                "AI provider connection error during stream: model=%s",
                self.MODEL
            )
            raise AIProviderException(
                "unable to connect to AI provider.",
                status_code=503
            ) from exc

        except APIStatusError as exc:
            logger.error(
                "AI provider status error during stream: model=%s",
                self.MODEL
            )
            raise AIProviderException(
                "AI provider request failed.",
                status_code=502
            )
