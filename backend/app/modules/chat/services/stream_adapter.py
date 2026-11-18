"""
Transport adapter for semantic Chat events using Server-Sent Events (SSE).

The Chat domain emits validated Pydantic event models. This module converts
those models into SSE frames and deliberately knows nothing about FastAPI,
SQLAlchemy, conversation ownership, message persistence, or LLM providers.

The endpoint is responsible for wrapping ``adapt()`` in its streaming
response and supplying ``request.is_disconnected`` (or an equivalent
asynchronous disconnect probe).

Responsibilities:
    - Validate and serialize supported Chat stream event models.
    - Produce deterministic SSE event IDs from request ID and sequence.
    - Emit transport-level comment heartbeats while the source is idle.
    - Stop consuming when the client disconnects.
    - Close the upstream async iterator on normal completion, disconnect,
      cancellation, and serialization/transport failure.

The adapter does not invent semantic events or modify generated content.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeAlias

from app.modules.chat.schemas.stream_event import (
    ChatContentDeltaEvent,
    ChatStreamCompletedEvent,
    ChatStreamErrorEvent,
    ChatStreamEvent,
    ChatStreamHeartbeatEvent,
    ChatStreamStartedEvent,
)


logger = logging.getLogger(__name__)

DisconnectCheck: TypeAlias = Callable[[], Awaitable[bool]]

_SUPPORTED_EVENT_TYPES = (
    ChatStreamStartedEvent,
    ChatContentDeltaEvent,
    ChatStreamCompletedEvent,
    ChatStreamErrorEvent,
    ChatStreamHeartbeatEvent,
)
_TERMINAL_EVENT_NAMES = frozenset({"message_complete", "error"})


class StreamAdapterError(ValueError):
    """Raised when a semantic event cannot be represented as a valid SSE frame."""


class SSEStreamAdapter:
    """
    Convert Chat semantic events into SSE text frames.

    ``heartbeat_interval_seconds`` controls comment heartbeats only. It is
    independent of the provider's request deadline and stream idle timeout.
    Set it to ``None`` to disable adapter-generated heartbeat comments.
    """

    def __init__(
        self,
        *,
        heartbeat_interval_seconds: float | None = 15.0,
    ) -> None:
        if heartbeat_interval_seconds is not None:
            if (
                isinstance(heartbeat_interval_seconds, bool)
                or not isinstance(heartbeat_interval_seconds, (int, float))
                or heartbeat_interval_seconds <= 0
            ):
                raise ValueError(
                    "heartbeat_interval_seconds must be positive or None."
                )

        self._heartbeat_interval_seconds = (
            float(heartbeat_interval_seconds)
            if heartbeat_interval_seconds is not None
            else None
        )

    @staticmethod
    def _validate_sse_field(value: str, *, field_name: str) -> str:
        """Reject characters that could inject additional SSE fields."""

        if not isinstance(value, str) or not value:
            raise StreamAdapterError(
                f"SSE {field_name} must be a non-empty string."
            )

        if any(character in value for character in ("\r", "\n", "\x00")):
            raise StreamAdapterError(
                f"SSE {field_name} contains a forbidden character."
            )

        return value

    def encode(self, event: ChatStreamEvent) -> str:
        """
        Serialize one validated Chat event.

        Pydantic JSON serialization escapes newlines inside message content.
        The SSE frame ends with the required blank line delimiter.
        """

        if not isinstance(event, _SUPPORTED_EVENT_TYPES):
            raise StreamAdapterError("Unsupported Chat stream event type.")

        event_name = self._validate_sse_field(
            event.event,
            field_name="event name",
        )
        request_id = self._validate_sse_field(
            event.request_id,
            field_name="request id",
        )

        event_id = self._validate_sse_field(
            f"{request_id}:{event.sequence_number}",
            field_name="event id",
        )

        data = event.model_dump_json(
            exclude_none=True,
            by_alias=True,
        )

        return (
            f"id: {event_id}\n"
            f"event: {event_name}\n"
            f"data: {data}\n\n"
        )

    @staticmethod
    def encode_comment(comment: str = "keep-alive") -> str:
        """
        Encode an SSE comment heartbeat.

        Newlines are replaced rather than rejected because comments are
        transport-generated diagnostics and must never create extra frames.
        """

        if not isinstance(comment, str):
            raise TypeError("SSE comment must be a string.")

        safe_comment = comment.replace("\r", " ").replace("\n", " ")
        return f": {safe_comment}\n\n"

    async def adapt(
        self,
        events: AsyncIterator[ChatStreamEvent],
        *,
        is_disconnected: DisconnectCheck | None = None,
    ) -> AsyncIterator[str]:
        """
        Adapt a semantic event iterator to an SSE text iterator.

        A single pending ``anext`` task is kept alive between heartbeat
        intervals. Repeatedly wrapping ``anext`` in ``asyncio.wait_for`` would
        cancel the pending read on each timeout and can corrupt a streaming
        provider iterator.

        The source iterator is explicitly closed in ``finally``. This allows
        StreamManager and the provider adapter to run their cleanup paths.
        """

        if is_disconnected is not None and not callable(is_disconnected):
            raise TypeError("is_disconnected must be an async callable or None.")

        iterator = events.__aiter__()
        pending_next: asyncio.Task[ChatStreamEvent] | None = None

        try:
            pending_next = asyncio.create_task(anext(iterator))

            while True:
                if self._heartbeat_interval_seconds is None:
                    done, _ = await asyncio.wait(
                        {pending_next},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                else:
                    done, _ = await asyncio.wait(
                        {pending_next},
                        timeout=self._heartbeat_interval_seconds,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                if not done:
                    if is_disconnected is not None and await is_disconnected():
                        logger.debug(
                            "SSE client disconnected while awaiting a Chat event."
                        )
                        break

                    yield self.encode_comment()
                    continue

                try:
                    event = pending_next.result()
                except StopAsyncIteration:
                    break

                yield self.encode(event)

                if event.event in _TERMINAL_EVENT_NAMES:
                    break

                if is_disconnected is not None and await is_disconnected():
                    logger.debug(
                        "SSE client disconnected after receiving a Chat event."
                    )
                    break

                pending_next = asyncio.create_task(anext(iterator))

        except asyncio.CancelledError:
            logger.debug("SSE response task was cancelled.")
            raise

        finally:
            if pending_next is not None and not pending_next.done():
                pending_next.cancel()
                await asyncio.gather(
                    pending_next,
                    return_exceptions=True,
                )

            close = getattr(iterator, "aclose", None)
            if callable(close):
                try:
                    await close()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Failed to close the Chat semantic event iterator."
                    )


# Short alias retained for dependency wiring and concise construction.
StreamAdapter = SSEStreamAdapter


__all__ = [
    "DisconnectCheck",
    "StreamAdapterError",
    "SSEStreamAdapter",
    "StreamAdapter",
]
