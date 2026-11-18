"""
Build a bounded, chronologically ordered canonical Chat context.

ContextBuilder receives the complete ordered prompt produced by PromptBuilder:
trusted system/developer instruction messages first, followed by persisted
users/assistant history. It preserves repository ordering and truncates only
whole older conversation messages.

It owns:
    - validation of prompt/history roles and message content;
    - input-token budgeting;
    - reserving the requested assistant output-token allowance;
    - selecting the newest complete history messages that fit;
    - final canonical prompt token counting.

It does not own database access, authorization, prompt-version lookup,
provider selection, or generation.

The repository is responsible for deterministic chronological ordering.
This builder intentionally does not sort by timestamps because equal
timestamps require the repository's secondary ordering key to remain intact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.exceptions.base import ContextBudgetExceededException
from app.integrations.llm.schemas import LLMMessage

from .token_counter import TokenCounter


_ALLOWED_INSTRUCTION_ROLES = frozenset({"system", "developer"})
_ALLOWED_HISTORY_ROLES = frozenset({"users", "assistant"})


class ContextBuilder:
    """
    Select the newest complete history messages that fit the input budget.

    ``context_token_budget`` is the configured maximum input-prompt budget.
    If ``model_context_window_tokens`` is supplied, the effective input
    budget is additionally constrained to:

        model_context_window_tokens - reserved_output_tokens

    The configured input budget and model context-window budget are separate
    controls. The former protects application resources; the latter protects
    the selected model's total input-plus-output capacity.
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        context_token_budget: int,
        *,
        model_context_window_tokens: int | None = None,
        default_reserved_output_tokens: int = 0,
    ) -> None:
        if not isinstance(token_counter, TokenCounter):
            raise TypeError("token_counter must be a TokenCounter instance.")

        if (
            isinstance(context_token_budget, bool)
            or not isinstance(context_token_budget, int)
            or context_token_budget <= 0
        ):
            raise ValueError("context_token_budget must be a positive integer.")

        if model_context_window_tokens is not None:
            if (
                isinstance(model_context_window_tokens, bool)
                or not isinstance(model_context_window_tokens, int)
                or model_context_window_tokens <= 0
            ):
                raise ValueError(
                    "model_context_window_tokens must be a positive integer or None."
                )

        if (
            isinstance(default_reserved_output_tokens, bool)
            or not isinstance(default_reserved_output_tokens, int)
            or default_reserved_output_tokens < 0
        ):
            raise ValueError(
                "default_reserved_output_tokens must be a non-negative integer."
            )

        if (
            model_context_window_tokens is not None
            and default_reserved_output_tokens >= model_context_window_tokens
        ):
            raise ValueError(
                "default_reserved_output_tokens must be smaller than the model context window."
            )

        self.token_counter = token_counter
        self.context_token_budget = context_token_budget
        self.model_context_window_tokens = model_context_window_tokens
        self.default_reserved_output_tokens = default_reserved_output_tokens

    @staticmethod
    def _message_fields(message: Any) -> tuple[str, str]:
        """Normalize an LLMMessage, mapping, or ORM-like message object."""

        if isinstance(message, LLMMessage):
            role = message.role
            content = message.content
        elif isinstance(message, Mapping):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)

        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(
                "Context messages must expose string role and content fields."
            )

        if not content.strip():
            raise ValueError("Context messages cannot contain empty content.")

        return role, content

    @staticmethod
    def _validate_message_order(
        normalized: Sequence[tuple[str, str]],
    ) -> tuple[tuple[str, str], ...]:
        """
        Require trusted instruction roles to form a leading prefix.

        Persisted history is restricted to users/assistant roles. A system or
        developer message appearing after conversation history is rejected
        rather than silently promoted or moved.
        """

        instruction_prefix: list[tuple[str, str]] = []
        history: list[tuple[str, str]] = []
        history_started = False

        for role, content in normalized:
            if role in _ALLOWED_INSTRUCTION_ROLES:
                if history_started:
                    raise ValueError(
                        "System/developer instructions must precede conversation history."
                    )
                instruction_prefix.append((role, content))
                continue

            if role not in _ALLOWED_HISTORY_ROLES:
                raise ValueError(
                    "Conversation history may contain only users and assistant roles."
                )

            history_started = True
            history.append((role, content))

        if not history:
            raise ValueError("At least one users/assistant conversation message is required.")

        if history[-1][0] != "users":
            raise ValueError("The latest conversation message must be a users message.")

        return (*instruction_prefix, *history)

    def _effective_input_budget(self, reserved_output_tokens: int) -> int:
        if (
            isinstance(reserved_output_tokens, bool)
            or not isinstance(reserved_output_tokens, int)
            or reserved_output_tokens < 0
        ):
            raise ValueError("reserved_output_tokens must be a non-negative integer.")

        budget = self.context_token_budget

        if self.model_context_window_tokens is not None:
            remaining_model_capacity = (
                self.model_context_window_tokens - reserved_output_tokens
            )
            if remaining_model_capacity <= 0:
                raise ContextBudgetExceededException(
                    "The requested output-token reservation leaves no model input capacity."
                )
            budget = min(budget, remaining_model_capacity)

        return budget

    def build(
        self,
        messages: Sequence[Any],
        *,
        reserved_output_tokens: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Return trusted instructions plus the newest fitting history.

        The latest users message is never silently dropped. If the trusted
        instruction prefix and latest users message cannot fit, a
        ContextBudgetExceededException is raised.
        """

        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise TypeError("messages must be a sequence of chat messages.")

        if not messages:
            raise ValueError("At least one chat message is required.")

        resolved_output_reservation = (
            self.default_reserved_output_tokens
            if reserved_output_tokens is None
            else reserved_output_tokens
        )
        effective_budget = self._effective_input_budget(
            resolved_output_reservation
        )

        normalized = tuple(self._message_fields(item) for item in messages)
        ordered = self._validate_message_order(normalized)

        instruction_prefix: list[tuple[str, str]] = []
        history: list[tuple[str, str]] = []

        history_started = False
        for role, content in ordered:
            if role in _ALLOWED_INSTRUCTION_ROLES and not history_started:
                instruction_prefix.append((role, content))
            else:
                history_started = True
                history.append((role, content))

        fixed_cost = sum(
            self.token_counter.count_message(role, content)
            for role, content in instruction_prefix
        )
        fixed_cost += self.token_counter.assistant_reply_overhead_tokens

        if fixed_cost > effective_budget:
            raise ContextBudgetExceededException(
                "The trusted prompt instructions exceed the available context token budget."
            )

        remaining_budget = effective_budget - fixed_cost
        selected_history: list[tuple[str, str]] = []

        for role, content in reversed(history):
            message_cost = self.token_counter.count_message(role, content)

            if message_cost > remaining_budget:
                if not selected_history:
                    raise ContextBudgetExceededException(
                        "The latest users message exceeds the available context token budget."
                    )
                break

            selected_history.append((role, content))
            remaining_budget -= message_cost

        selected_history.reverse()

        return [
            {"role": role, "content": content}
            for role, content in (*instruction_prefix, *selected_history)
        ]

    def count_context_tokens(self, context: Sequence[Any]) -> int:
        """Count a complete context, including role/message/reply framing."""

        normalized = tuple(self._message_fields(item) for item in context)
        return self.token_counter.count_messages(normalized)

    def count_llm_messages(self, messages: Sequence[LLMMessage]) -> int:
        """Count the final canonical LLM message sequence."""

        if any(not isinstance(message, LLMMessage) for message in messages):
            raise TypeError("messages must contain only LLMMessage instances.")

        return self.token_counter.count_messages(
            tuple((message.role, message.content) for message in messages)
        )

    def validate_llm_messages(
        self,
        messages: Sequence[LLMMessage],
        *,
        reserved_output_tokens: int | None = None,
    ) -> int:
        """
        Validate the final prompt against the same effective input budget.

        Returns the estimated input-token count for quota checks, metrics,
        and operational logging.
        """

        resolved_output_reservation = (
            self.default_reserved_output_tokens
            if reserved_output_tokens is None
            else reserved_output_tokens
        )
        effective_budget = self._effective_input_budget(
            resolved_output_reservation
        )

        token_count = self.count_llm_messages(messages)
        if token_count > effective_budget:
            raise ContextBudgetExceededException(
                "The final prompt exceeds the available context token budget."
            )

        return token_count


__all__ = ["ContextBuilder"]
