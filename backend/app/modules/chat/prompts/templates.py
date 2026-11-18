from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from string import Formatter
from types import MappingProxyType
from typing import Mapping

from app.integrations.llm.schemas import LLMMessage

from .versioning import PromptLifecycle, PromptVersion, PromptVersionMetadata


_PROMPT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_VARIABLE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_ALLOWED_PROMPT_ROLES = frozenset({"system", "developer"})
_MAX_PROMPT_CONTENT_CHARACTERS = 200_000
_MAX_PROMPT_DESCRIPTION_CHARACTERS = 2_000
_MAX_PROMPT_CHANGELOG_CHARACTERS = 4_000
_MAX_METADATA_ENTRIES = 64
_MAX_METADATA_KEY_CHARACTERS = 128
_MAX_METADATA_VALUE_CHARACTERS = 2_000


class PromptTemplateValidationError(ValueError):
    """
    Raised when an application-authored prompt definition is invalid.

    This is a configuration/programming error, not a users request-validation
    error. It intentionally remains a ValueError so invalid prompt releases
    fail fast during application composition or deployment validation.
    """


def _extract_placeholders(template: str) -> frozenset[str]:
    """Parse and validate all format fields in a prompt template."""

    placeholders: set[str] = set()

    try:
        parsed_fields = Formatter().parse(template)
        for _literal, field_name, format_spec, conversion in parsed_fields:
            if field_name is None:
                continue

            if not _VARIABLE_PATTERN.fullmatch(field_name):
                raise PromptTemplateValidationError(
                    "Prompt placeholders must be simple snake_case identifiers."
                )

            if format_spec or conversion:
                raise PromptTemplateValidationError(
                    "Prompt placeholders cannot use format specifications or conversions."
                )

            placeholders.add(field_name)

    except ValueError as exc:
        if isinstance(exc, PromptTemplateValidationError):
            raise
        raise PromptTemplateValidationError(
            "Prompt template contains invalid brace syntax. Literal braces "
            "must be escaped as '{{' and '}}'."
        ) from exc

    return frozenset(placeholders)


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise PromptTemplateValidationError(
            f"{field_name} must be timezone-aware."
        )


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """
    One immutable prompt release.

    ``name`` is the stable logical identifier. ``version`` identifies this
    exact release. ``role`` is restricted to trusted instruction roles.
    ``required_variables`` must exactly match placeholders discovered in
    ``content`` so misspelled or unused variables fail at registration time.
    """

    name: str
    version: PromptVersion
    content: str
    role: str = "system"
    description: str = ""
    changelog: str = ""
    lifecycle: PromptLifecycle = PromptLifecycle.DRAFT
    required_variables: frozenset[str] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    activated_at: datetime | None = None
    deprecated_at: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _PROMPT_NAME_PATTERN.fullmatch(self.name):
            raise PromptTemplateValidationError(
                "Prompt name must be a lowercase identifier containing only "
                "letters, digits, dots, underscores, or hyphens."
            )

        try:
            parsed_version = PromptVersion.parse(self.version)
        except (TypeError, ValueError) as exc:
            raise PromptTemplateValidationError(
                "Prompt version must be a valid semantic version."
            ) from exc
        object.__setattr__(self, "version", parsed_version)

        if not isinstance(self.content, str) or not self.content.strip():
            raise PromptTemplateValidationError("Prompt content must be non-empty.")
        if len(self.content) > _MAX_PROMPT_CONTENT_CHARACTERS:
            raise PromptTemplateValidationError("Prompt content exceeds the configured size limit.")

        if self.role not in _ALLOWED_PROMPT_ROLES:
            raise PromptTemplateValidationError(
                "Prompt role must be either 'system' or 'developer'."
            )

        if not isinstance(self.description, str):
            raise PromptTemplateValidationError("Prompt description must be a string.")
        if len(self.description) > _MAX_PROMPT_DESCRIPTION_CHARACTERS:
            raise PromptTemplateValidationError("Prompt description is too long.")

        if not isinstance(self.changelog, str):
            raise PromptTemplateValidationError("Prompt changelog must be a string.")
        if len(self.changelog) > _MAX_PROMPT_CHANGELOG_CHARACTERS:
            raise PromptTemplateValidationError("Prompt changelog is too long.")

        if not isinstance(self.lifecycle, PromptLifecycle):
            raise PromptTemplateValidationError(
                "lifecycle must be a PromptLifecycle value."
            )

        _require_aware_datetime(self.created_at, "created_at")

        if self.activated_at is not None:
            _require_aware_datetime(self.activated_at, "activated_at")
            if self.activated_at < self.created_at:
                raise PromptTemplateValidationError(
                    "activated_at cannot be earlier than created_at."
                )

        if self.deprecated_at is not None:
            _require_aware_datetime(self.deprecated_at, "deprecated_at")
            lower_bound = self.activated_at or self.created_at
            if self.deprecated_at < lower_bound:
                raise PromptTemplateValidationError(
                    "deprecated_at cannot be earlier than activation or creation."
                )

        if self.lifecycle is PromptLifecycle.ACTIVE and self.activated_at is None:
            object.__setattr__(self, "activated_at", datetime.now(timezone.utc))

        if (
            self.lifecycle in {PromptLifecycle.DEPRECATED, PromptLifecycle.RETIRED}
            and self.deprecated_at is None
        ):
            object.__setattr__(self, "deprecated_at", datetime.now(timezone.utc))

        if self.lifecycle is PromptLifecycle.DRAFT and self.activated_at is not None:
            raise PromptTemplateValidationError(
                "A draft prompt cannot have an activated_at timestamp."
            )

        discovered = _extract_placeholders(self.content)
        declared = (
            discovered
            if self.required_variables is None
            else frozenset(self.required_variables)
        )

        for variable in declared:
            if not isinstance(variable, str) or not _VARIABLE_PATTERN.fullmatch(variable):
                raise PromptTemplateValidationError(
                    "Declared prompt variables must be simple snake_case identifiers."
                )

        if discovered != declared:
            missing_declarations = discovered - declared
            unused_declarations = declared - discovered
            details: list[str] = []
            if missing_declarations:
                details.append(
                    "undeclared placeholders: " + ", ".join(sorted(missing_declarations))
                )
            if unused_declarations:
                details.append(
                    "unused declarations: " + ", ".join(sorted(unused_declarations))
                )
            raise PromptTemplateValidationError(
                "Prompt variable declarations do not match the template ("
                + "; ".join(details)
                + ")."
            )

        if not isinstance(self.metadata, Mapping):
            raise PromptTemplateValidationError("Prompt metadata must be a mapping.")
        if len(self.metadata) > _MAX_METADATA_ENTRIES:
            raise PromptTemplateValidationError("Prompt metadata has too many entries.")

        copied_metadata: dict[str, str] = {}
        for key, value in self.metadata.items():
            if (
                not isinstance(key, str)
                or not key.strip()
                or len(key) > _MAX_METADATA_KEY_CHARACTERS
            ):
                raise PromptTemplateValidationError(
                    "Metadata keys must be non-empty strings within the length limit."
                )
            if (
                not isinstance(value, str)
                or len(value) > _MAX_METADATA_VALUE_CHARACTERS
            ):
                raise PromptTemplateValidationError(
                    "Metadata values must be strings within the length limit."
                )
            copied_metadata[key] = value

        object.__setattr__(self, "required_variables", declared)
        object.__setattr__(self, "metadata", MappingProxyType(copied_metadata))

    @property
    def version_metadata(self) -> PromptVersionMetadata:
        """Return immutable lifecycle metadata for this exact release."""

        return PromptVersionMetadata(
            version=self.version,
            lifecycle=self.lifecycle,
            changelog=self.changelog,
            created_at=self.created_at,
            activated_at=self.activated_at,
            deprecated_at=self.deprecated_at,
        )

    def render(self, variables: Mapping[str, str] | None = None) -> str:
        """
        Render with an exact set of declared string variables.

        Variable values are inserted literally and are not parsed a second
        time. This prevents values containing braces from becoming a second
        template language. Callers must not use this mechanism to merge
        untrusted conversation text into trusted system/developer messages.
        """

        if variables is None:
            supplied: dict[str, str] = {}
        elif not isinstance(variables, Mapping):
            raise PromptTemplateValidationError(
                "Prompt variables must be supplied as a mapping."
            )
        else:
            supplied = dict(variables)

        expected = self.required_variables or frozenset()
        supplied_keys = set(supplied)

        missing = expected - supplied_keys
        unexpected = supplied_keys - expected

        if missing:
            raise PromptTemplateValidationError(
                "Missing required prompt variables: "
                + ", ".join(sorted(missing))
                + "."
            )
        if unexpected:
            raise PromptTemplateValidationError(
                "Unexpected prompt variables: "
                + ", ".join(sorted(unexpected))
                + "."
            )

        for key, value in supplied.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise PromptTemplateValidationError(
                    "Prompt variable names and values must be strings."
                )

        try:
            return self.content.format_map(supplied)
        except (KeyError, ValueError, IndexError) as exc:
            # Definition-time validation should make this unreachable, but
            # retain a deterministic failure boundary if invariants regress.
            raise PromptTemplateValidationError(
                "Prompt template could not be rendered."
            ) from exc

    def to_llm_message(
        self,
        variables: Mapping[str, str] | None = None,
    ) -> LLMMessage:
        """Render this trusted prompt as a provider-neutral LLM message."""

        return LLMMessage(
            role=self.role,
            content=self.render(variables),
        )


DEFAULT_CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Provide accurate, clear, and appropriately concise answers. "
    "If you are uncertain, say so rather than inventing information."
)

DEFAULT_CHAT_SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    name="chat.system",
    version=PromptVersion(1, 0, 0),
    content=DEFAULT_CHAT_SYSTEM_PROMPT,
    role="system",
    description="Default system instructions for the Chat domain.",
    changelog="Initial version.",
    lifecycle=PromptLifecycle.ACTIVE,
)


def build_chat_system_prompt_template(
    content: str | None = None,
    *,
    version: str | PromptVersion = "1.0.0",
) -> PromptTemplate:
    """
    Build the configured Chat system prompt.

    The application composition root should pass its validated settings value.
    This module deliberately does not import global settings, avoiding
    import-time configuration side effects and keeping tests deterministic.
    """

    resolved_content = (
        DEFAULT_CHAT_SYSTEM_PROMPT if content is None else content
    )
    return PromptTemplate(
        name="chat.system",
        version=PromptVersion.parse(version),
        content=resolved_content,
        role="system",
        description="Configured system instructions for the Chat domain.",
        changelog="Configured application system prompt.",
        lifecycle=PromptLifecycle.ACTIVE,
    )


__all__ = [
    "PromptTemplateValidationError",
    "PromptTemplate",
    "DEFAULT_CHAT_SYSTEM_PROMPT",
    "DEFAULT_CHAT_SYSTEM_PROMPT_TEMPLATE",
    "build_chat_system_prompt_template",
]
