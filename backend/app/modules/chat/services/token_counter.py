"""
Model-aware token counting and conservative fallback estimation for Chat.

TokenCounter is the single token-estimation component used by ContextBuilder.
It supports an injected tokenizer for the configured model. If no tokenizer
is available, it uses an explicitly approximate UTF-8 byte heuristic.

Important boundaries:
    - The fallback estimate is not billing-grade usage.
    - Provider-reported LLMUsage is authoritative for actual usage.
    - The provider adapter remains responsible for enforcing hard model
      limits as the final request boundary.
    - A configured tokenizer failure is surfaced. It never silently falls
      back to a weaker heuristic because doing so could under-budget input.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class TextTokenizer(Protocol):
    """Minimal tokenizer adapter contract."""

    def encode(self, text: str) -> Sequence[object]:
        """Encode text into a sized token sequence."""


class TokenCounter:
    """
    Count text and provider-neutral chat-message framing.

    The fallback estimates one token per ``fallback_bytes_per_token`` UTF-8
    bytes, rounded up. The default of three bytes per token is deliberately
    conservative for many ordinary inputs, but is not guaranteed to be an
    upper bound for every language, tokenizer, or model.
    """

    def __init__(
        self,
        tokenizer: TextTokenizer | None = None,
        *,
        fallback_bytes_per_token: float = 3.0,
        per_message_overhead_tokens: int = 4,
        assistant_reply_overhead_tokens: int = 2,
    ) -> None:
        if tokenizer is not None and not isinstance(tokenizer, TextTokenizer):
            raise TypeError("tokenizer must provide an encode(text) method.")

        if (
            isinstance(fallback_bytes_per_token, bool)
            or not isinstance(fallback_bytes_per_token, (int, float))
            or not math.isfinite(float(fallback_bytes_per_token))
            or fallback_bytes_per_token <= 0
        ):
            raise ValueError("fallback_bytes_per_token must be a finite positive number.")

        for name, value in (
            ("per_message_overhead_tokens", per_message_overhead_tokens),
            ("assistant_reply_overhead_tokens", assistant_reply_overhead_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

        self._tokenizer = tokenizer
        self._fallback_bytes_per_token = float(fallback_bytes_per_token)
        self._per_message_overhead_tokens = per_message_overhead_tokens
        self._assistant_reply_overhead_tokens = assistant_reply_overhead_tokens

    @property
    def uses_model_tokenizer(self) -> bool:
        """Whether a model-specific tokenizer has been injected."""

        return self._tokenizer is not None

    @property
    def assistant_reply_overhead_tokens(self) -> int:
        """Estimated framing tokens reserved for the generated assistant turn."""

        return self._assistant_reply_overhead_tokens

    def count_text(self, text: str) -> int:
        """Return exact tokenizer length or the documented fallback estimate."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not text:
            return 0

        if self._tokenizer is not None:
            try:
                # tiktoken supports disallowed_special. Other tokenizer
                # adapters may expose the narrower encode(text) contract.
                try:
                    encoded = self._tokenizer.encode(  # type: ignore[call-arg]
                        text,
                        disallowed_special=(),
                    )
                except TypeError:
                    encoded = self._tokenizer.encode(text)
            except Exception as exc:
                raise RuntimeError(
                    "The configured model tokenizer failed; token budgeting cannot continue safely."
                ) from exc

            try:
                token_count = len(encoded)
            except TypeError as exc:
                raise TypeError(
                    "Tokenizer.encode() must return a sized token sequence."
                ) from exc

            if isinstance(token_count, bool) or not isinstance(token_count, int):
                raise TypeError("Tokenizer returned an invalid token count.")
            if token_count < 0:
                raise RuntimeError("Tokenizer returned a negative token count.")

            return token_count

        byte_length = len(text.encode("utf-8"))
        return max(
            1,
            math.ceil(byte_length / self._fallback_bytes_per_token),
        )

    def estimate(self, text: str) -> int:
        """Compatibility alias retained for earlier Chat code."""

        return self.count_text(text)

    def count_message(self, role: str, content: str) -> int:
        """Count message content, role text, and fixed message framing."""

        if not isinstance(role, str) or not role.strip():
            raise ValueError("role must be a non-empty string.")
        if not isinstance(content, str):
            raise TypeError("content must be a string.")

        return (
            self.count_text(content)
            + self._per_message_overhead_tokens
            + self.count_text(role)
        )

    def count_messages(
        self,
        messages: Sequence[tuple[str, str]],
    ) -> int:
        """
        Count an ordered chat payload plus assistant reply priming.

        Provider-specific message framing differs. This is an application
        planning estimate; the selected provider adapter performs the final
        hard-limit validation.
        """

        total = sum(
            self.count_message(role, content)
            for role, content in messages
        )

        if messages:
            total += self._assistant_reply_overhead_tokens

        return total


__all__ = ["TextTokenizer", "TokenCounter"]
