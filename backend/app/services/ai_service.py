import logging
import time
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from app.core.config import settings
from app.exceptions.custom_exceptions import AIProviderException


logger = logging.getLogger(__name__)


class AIService:


    MODEL = settings.GROQ_MODEL
    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url=self.BASE_URL,
        )

    async def stream_response(
        self,
        messages: Sequence[dict[str, str]],
    ) -> AsyncGenerator[str, None]:

        started_at = time.perf_counter()
        chunk_count = 0
        character_count = 0
        stream: Any | None = None

        logger.info(
            "Starting LLM stream: provider=groq model=%s",
            self.MODEL,
        )

        try:
            stream = await self.client.chat.completions.create(
                model=self.MODEL,
                messages=list(messages),
                max_tokens=settings.CHAT_MAX_OUTPUT_TOKENS,
                stream=True,
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if not isinstance(content, str) or not content:
                    continue

                chunk_count += 1
                character_count += len(content)

                yield content

            duration = time.perf_counter() - started_at

            logger.info(
                (
                    "LLM stream completed: provider=groq model=%s "
                    "duration=%.2fs chunks=%d characters=%d"
                ),
                self.MODEL,
                duration,
                chunk_count,
                character_count,
            )

        except RateLimitError as exc:
            logger.warning(
                "LLM provider rate limit or quota error: model=%s",
                self.MODEL,
            )
            raise AIProviderException(
                "AI provider rate limit or quota exceeded.",
                status_code=429,
            ) from exc

        except AuthenticationError as exc:
            logger.error(
                "LLM provider authentication failed: model=%s",
                self.MODEL,
            )
            raise AIProviderException(
                "AI provider authentication failed.",
                status_code=502,
            ) from exc

        except APITimeoutError as exc:
            logger.warning(
                "LLM provider timed out: model=%s",
                self.MODEL,
            )
            raise AIProviderException(
                "AI provider timed out.",
                status_code=504,
            ) from exc

        except APIConnectionError as exc:
            logger.warning(
                "Unable to connect to LLM provider: model=%s",
                self.MODEL,
            )
            raise AIProviderException(
                "unable to connect to AI provider.",
                status_code=503,
            ) from exc

        except APIStatusError as exc:
            logger.error(
                "LLM provider returned an API error: model=%s status=%s",
                self.MODEL,
                exc.status_code,
            )
            raise AIProviderException(
                "AI provider request failed.",
                status_code=502,
            ) from exc

        except GeneratorExit:
            logger.debug(
                "LLM stream consumer closed the generator: model=%s",
                self.MODEL,
            )
            raise

        except Exception:
            logger.exception(
                "Unexpected failure while streaming from LLM provider: model=%s",
                self.MODEL,
            )
            raise

        finally:
            if stream is not None:
                try:
                    await stream.close()
                except Exception:
                    logger.warning(
                        "Failed to close upstream LLM stream: model=%s",
                        self.MODEL,
                        exc_info=True,
                    )

            duration = time.perf_counter() - started_at

            logger.debug(
                (
                    "LLM stream lifecycle ended: provider=groq model=%s "
                    "duration=%.2fs chunks=%d characters=%d"
                ),
                self.MODEL,
                duration,
                chunk_count,
                character_count,
            )

    async def close(self) -> None:
        await self.client.close()
