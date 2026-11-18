from __future__ import annotations

from threading import RLock
from typing import Iterable

from app.exceptions.base import ApplicationException
from app.exceptions.error_codes import ErrorCode

from .templates import (
    DEFAULT_CHAT_SYSTEM_PROMPT_TEMPLATE,
    PromptTemplate,
)
from .versioning import PromptLifecycle, PromptVersion


class PromptRegistryError(ApplicationException):
    """Base class for safe, application-level prompt registry failures."""

    def __init__(
        self,
        message: str = "The prompt configuration is invalid.",
        *,
        status_code: int = 500,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_code=ErrorCode.APPLICATION_ERROR,
        )


class PromptAlreadyRegisteredError(PromptRegistryError):
    """Raised when an exact prompt name/version pair already exists."""

    def __init__(self, name: str, version: PromptVersion) -> None:
        super().__init__(
            f"Prompt '{name}' version '{version}' is already registered.",
            status_code=409,
        )


class PromptNotFoundError(PromptRegistryError):
    """Raised when a logical prompt or requested release is not registered."""

    def __init__(
        self,
        name: str,
        version: PromptVersion | None = None,
    ) -> None:
        target = f" version '{version}'" if version is not None else ""
        super().__init__(
            f"Prompt '{name}'{target} is not registered.",
            status_code=404,
        )


class PromptNotActiveError(PromptRegistryError):
    """Raised when a release is not eligible for normal generation."""

    def __init__(self, name: str, version: PromptVersion) -> None:
        super().__init__(
            f"Prompt '{name}' version '{version}' is not active.",
            status_code=409,
        )


class PromptDefaultAlreadyConfiguredError(PromptRegistryError):
    """Raised when registration attempts to create a second default release."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"Prompt '{name}' already has a configured default release.",
            status_code=409,
        )


class PromptRegistry:
    """
    In-memory, thread-safe registry of immutable prompt releases.

    A logical prompt may have multiple releases but only one configured
    default. A default release must be ACTIVE. Explicit lookup also requires
    ACTIVE by default; non-active releases can be inspected only by passing
    ``require_active=False`` deliberately.

    The lock protects registry mutations and reads within one Python process.
    It does not coordinate multiple application workers or deployments.
    """

    def __init__(self, templates: Iterable[PromptTemplate] = ()) -> None:
        self._lock = RLock()
        self._templates: dict[tuple[str, PromptVersion], PromptTemplate] = {}
        self._defaults: dict[str, PromptVersion] = {}

        for template in templates:
            self.register(
                template,
                make_default=template.lifecycle is PromptLifecycle.ACTIVE,
            )

    def register(
        self,
        template: PromptTemplate,
        *,
        make_default: bool = False,
    ) -> None:
        """
        Register one immutable prompt release.

        Validation occurs before mutation. If any condition fails, registry
        state remains unchanged.
        """

        if not isinstance(template, PromptTemplate):
            raise TypeError("template must be a PromptTemplate instance.")
        if not isinstance(make_default, bool):
            raise TypeError("make_default must be a bool.")

        key = (template.name, template.version)

        with self._lock:
            if key in self._templates:
                raise PromptAlreadyRegisteredError(*key)

            if make_default and template.lifecycle is not PromptLifecycle.ACTIVE:
                raise PromptNotActiveError(template.name, template.version)

            if make_default and template.name in self._defaults:
                raise PromptDefaultAlreadyConfiguredError(template.name)

            self._templates[key] = template
            if make_default:
                self._defaults[template.name] = template.version

    def set_default(
        self,
        name: str,
        version: str | PromptVersion,
    ) -> None:
        """
        Atomically select an already registered ACTIVE release as default.

        This operation replaces the previous default for the logical prompt.
        It does not mutate either prompt release.
        """

        parsed_version = PromptVersion.parse(version)

        with self._lock:
            template = self._templates.get((name, parsed_version))
            if template is None:
                raise PromptNotFoundError(name, parsed_version)
            if template.lifecycle is not PromptLifecycle.ACTIVE:
                raise PromptNotActiveError(name, parsed_version)

            self._defaults[name] = parsed_version

    def get(
        self,
        name: str,
        version: str | PromptVersion | None = None,
        *,
        require_active: bool = True,
    ) -> PromptTemplate:
        """
        Resolve an explicit release or configured default.

        There is deliberately no implicit "latest" fallback. A newly
        registered release cannot silently change production behavior.
        """

        if not isinstance(require_active, bool):
            raise TypeError("require_active must be a bool.")

        with self._lock:
            if version is None:
                resolved_version = self._defaults.get(name)
                if resolved_version is None:
                    raise PromptNotFoundError(name)
            else:
                resolved_version = PromptVersion.parse(version)

            template = self._templates.get((name, resolved_version))
            if template is None:
                raise PromptNotFoundError(name, resolved_version)

            if require_active and template.lifecycle is not PromptLifecycle.ACTIVE:
                raise PromptNotActiveError(name, resolved_version)

            return template

    def get_default_version(self, name: str) -> PromptVersion:
        """Return the currently configured default release version."""

        with self._lock:
            version = self._defaults.get(name)
            if version is None:
                raise PromptNotFoundError(name)
            return version

    def list_versions(self, name: str) -> tuple[PromptVersion, ...]:
        """Return all exact registered versions in SemVer precedence order."""

        with self._lock:
            versions = [
                version
                for prompt_name, version in self._templates
                if prompt_name == name
            ]

        return tuple(sorted(versions))

    def list_templates(
        self,
        name: str | None = None,
        *,
        lifecycle: PromptLifecycle | None = None,
    ) -> tuple[PromptTemplate, ...]:
        """
        Return an immutable snapshot of registered definitions.

        This is an inspection API. It does not expose the mutable internal
        dictionaries and does not imply that the registry is persistent.
        """

        if lifecycle is not None and not isinstance(lifecycle, PromptLifecycle):
            raise TypeError("lifecycle must be a PromptLifecycle value or None.")

        with self._lock:
            templates = tuple(self._templates.values())

        if name is not None:
            templates = tuple(item for item in templates if item.name == name)
        if lifecycle is not None:
            templates = tuple(
                item for item in templates if item.lifecycle is lifecycle
            )

        return tuple(
            sorted(
                templates,
                key=lambda item: (
                    item.name,
                    item.version,
                ),
            )
        )

    def contains(
        self,
        name: str,
        version: str | PromptVersion,
    ) -> bool:
        """Return whether an exact logical-name/version key is registered."""

        parsed_version = PromptVersion.parse(version)
        with self._lock:
            return (name, parsed_version) in self._templates


def build_default_prompt_registry(
    system_prompt: str | None = None,
    *,
    version: str | PromptVersion = "1.0.0",
) -> PromptRegistry:
    """
    Construct the standard Chat prompt registry.

    The application composition root should pass the validated
    ``settings.CHAT_SYSTEM_PROMPT`` value. This function does not import
    settings or initialize global application state.

    When no custom prompt is supplied and the requested version is 1.0.0,
    the canonical default template is reused. Otherwise a fresh immutable
    release is built using the supplied content and version.
    """

    parsed_version = PromptVersion.parse(version)

    if (
        system_prompt is None
        and parsed_version == PromptVersion(1, 0, 0)
    ):
        template = DEFAULT_CHAT_SYSTEM_PROMPT_TEMPLATE
    else:
        from .templates import build_chat_system_prompt_template

        template = build_chat_system_prompt_template(
            system_prompt,
            version=parsed_version,
        )

    return PromptRegistry((template,))


__all__ = [
    "PromptRegistryError",
    "PromptAlreadyRegisteredError",
    "PromptNotFoundError",
    "PromptNotActiveError",
    "PromptDefaultAlreadyConfiguredError",
    "PromptRegistry",
    "build_default_prompt_registry",
]
