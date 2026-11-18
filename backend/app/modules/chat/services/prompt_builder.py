from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from app.exceptions.base import ApplicationException
from app.exceptions.error_codes import ErrorCode
from app.integrations.llm.schemas import LLMMessage
from app.modules.chat.prompts.prompt_registry import PromptRegistry
from app.modules.chat.prompts.templates import PromptTemplateValidationError
from app.modules.chat.prompts.versioning import PromptVersion


logger = logging.getLogger(__name__)

DEFAULT_CHAT_PROMPT_NAME = "chat.system"


class PromptBuildException(ApplicationException):
    """Raised when trusted prompt and conversation history cannot be composed."""

    def __init__(
        self,
        message: str = "The chat prompt could not be prepared.",
        *,
        status_code: int = 500,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_code=ErrorCode.APPLICATION_ERROR,
        )


class PromptBuilder:
    """
    Build the ordered LLM message tuple for one generation request.

    The builder does not select or truncate history. ContextBuilder owns those
    responsibilities. This class validates roles, trusted prompt selection,
    and the final users-turn boundary.
    """

    def __init__(
        self,
        prompt_registry: PromptRegistry,
        *,
        prompt_name: str = DEFAULT_CHAT_PROMPT_NAME,
    ) -> None:
        if not isinstance(prompt_registry, PromptRegistry):
            raise TypeError("prompt_registry must be a PromptRegistry instance.")

        if not isinstance(prompt_name, str) or not prompt_name.strip():
            raise ValueError("prompt_name must be a non-empty string.")

        self._prompt_registry = prompt_registry
        self._prompt_name = prompt_name

    @staticmethod
    def _to_llm_message(item: Any) -> LLMMessage:
        """
        Normalize an LLMMessage, mapping, or ORM-like Message object.

        SQLAlchemy Message objects expose ``role`` and ``content`` attributes.
        Mapping support keeps the builder compatible with ContextBuilder's
        dictionary output and makes isolated tests straightforward.
        """

        if isinstance(item, LLMMessage):
            return item

        if isinstance(item, Mapping):
            role = item.get("role")
            content = item.get("content")
            name = item.get("name")
        else:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
            name = getattr(item, "name", None)

        if not isinstance(role, str) or not isinstance(content, str):
            raise PromptBuildException(
                "Conversation history contains an invalid message."
            )

        try:
            return LLMMessage(
                role=role,
                content=content,
                name=name,
            )
        except (TypeError, ValueError) as exc:
            raise PromptBuildException(
                "Conversation history contains an unsupported message."
            ) from exc

    def build(
        self,
        context_messages: Iterable[Any],
        *,
        prompt_variables: Mapping[str, str] | None = None,
        prompt_version: str | PromptVersion | None = None,
    ) -> tuple[LLMMessage, ...]:
        """
        Return a trusted system message followed by chronological history.

        The latest history message must be a users message. Consecutive users
        messages are allowed because an interrupted or failed previous turn
        may leave a valid conversation ending in a users message.
        """

        try:
            template = self._prompt_registry.get(
                self._prompt_name,
                prompt_version,
                require_active=True,
            )
            system_message = template.to_llm_message(prompt_variables)

        except PromptTemplateValidationError as exc:
            raise PromptBuildException(
                "The configured chat prompt is invalid."
            ) from exc

        except ApplicationException as exc:
            # A missing, retired, or otherwise unusable configured prompt is
            # an application configuration failure at this boundary. Avoid
            # exposing registry internals as a resource lookup to API clients.
            logger.error(
                "Unable to resolve active Chat prompt: prompt_name=%s error_code=%s",
                self._prompt_name,
                exc.error_code.value,
            )
            raise PromptBuildException(
                "The configured chat prompt is unavailable."
            ) from exc

        except (TypeError, ValueError) as exc:
            raise PromptBuildException(
                "The configured chat prompt version is invalid."
            ) from exc

        try:
            normalized_history = tuple(
                self._to_llm_message(item)
                for item in context_messages
            )
        except TypeError as exc:
            raise PromptBuildException(
                "Conversation context must be an iterable of messages."
            ) from exc

        # Compatibility bridge for a context builder that still prepends the
        # configured system prompt. Only the exact trusted prompt is accepted.
        if normalized_history and normalized_history[0].role == "system":
            legacy_system_message = normalized_history[0]
            if legacy_system_message.content != system_message.content:
                raise PromptBuildException(
                    "Conversation context contains a conflicting system prompt."
                )
            normalized_history = normalized_history[1:]

        for message in normalized_history:
            if message.role not in {"users", "assistant"}:
                raise PromptBuildException(
                    "Conversation history may contain only users and assistant messages."
                )

        if not normalized_history:
            raise PromptBuildException(
                "Conversation history must contain a users message."
            )

        if normalized_history[-1].role != "users":
            raise PromptBuildException(
                "The latest conversation message must be a users message."
            )

        return (system_message, *normalized_history)


__all__ = [
    "DEFAULT_CHAT_PROMPT_NAME",
    "PromptBuildException",
    "PromptBuilder",
]
