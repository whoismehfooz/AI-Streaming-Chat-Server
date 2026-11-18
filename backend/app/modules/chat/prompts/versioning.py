from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from functools import total_ordering


_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
_NUMERIC_IDENTIFIER_PATTERN = re.compile(r"^[0-9]+$")


class PromptLifecycle(StrEnum):
    """Lifecycle state of one immutable prompt release."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


def _validate_identifiers(
    identifiers: tuple[str, ...],
    *,
    field_name: str,
    reject_numeric_leading_zeroes: bool,
) -> None:
    if not isinstance(identifiers, tuple):
        raise TypeError(f"{field_name} must be a tuple of strings.")

    for identifier in identifiers:
        if (
            not isinstance(identifier, str)
            or not identifier
            or not re.fullmatch(r"[0-9A-Za-z-]+", identifier)
        ):
            raise ValueError(f"Invalid {field_name} version identifier.")

        if (
            reject_numeric_leading_zeroes
            and _NUMERIC_IDENTIFIER_PATTERN.fullmatch(identifier)
            and len(identifier) > 1
            and identifier.startswith("0")
        ):
            raise ValueError(
                "Numeric prerelease identifiers must not contain leading zeroes."
            )


@total_ordering
@dataclass(frozen=True, slots=True)
class PromptVersion:
    """
    Parsed SemVer value.

    Equality and hashing identify the complete version, including build
    metadata. Ordering follows SemVer precedence, where build metadata is
    ignored. Consequently, two distinct versions differing only in build
    metadata have equal precedence but are not the same registry key.
    """

    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...] = ()
    build: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("major", self.major),
            ("minor", self.minor),
            ("patch", self.patch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

        _validate_identifiers(
            self.prerelease,
            field_name="prerelease",
            reject_numeric_leading_zeroes=True,
        )
        _validate_identifiers(
            self.build,
            field_name="build",
            reject_numeric_leading_zeroes=False,
        )

    @classmethod
    def parse(cls, value: str | PromptVersion) -> PromptVersion:
        """Parse a strict MAJOR.MINOR.PATCH SemVer string."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("Prompt version must be a string or PromptVersion.")

        match = _SEMVER_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(
                "Prompt version must use valid MAJOR.MINOR.PATCH semantic versioning."
            )

        major, minor, patch, prerelease, build = match.groups()
        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            prerelease=tuple(prerelease.split(".")) if prerelease else (),
            build=tuple(build.split(".")) if build else (),
        )

    def _precedence_key(self) -> tuple[object, ...]:
        """
        Return a comparison key implementing SemVer prerelease precedence.

        A normal release has higher precedence than its prerelease. Numeric
        prerelease identifiers sort below non-numeric identifiers, and
        shorter equal-prefix prerelease sequences sort first.
        """

        core = (self.major, self.minor, self.patch)

        if not self.prerelease:
            prerelease_key: tuple[object, ...] = ((2, ""),)
        else:
            identifiers: list[tuple[int, int | str]] = []
            for identifier in self.prerelease:
                if _NUMERIC_IDENTIFIER_PATTERN.fullmatch(identifier):
                    identifiers.append((0, int(identifier)))
                else:
                    identifiers.append((1, identifier))

            # The leading marker makes every prerelease lower than a normal
            # release with the same MAJOR.MINOR.PATCH values.
            prerelease_key = ((1, ""), *identifiers)

        return (*core, prerelease_key)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, PromptVersion):
            return NotImplemented
        return self._precedence_key() < other._precedence_key()

    def __str__(self) -> str:
        value = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            value += "-" + ".".join(self.prerelease)
        if self.build:
            value += "+" + ".".join(self.build)
        return value


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")


@dataclass(frozen=True, slots=True)
class PromptVersionMetadata:
    """
    Immutable audit metadata for a prompt release.

    Lifecycle changes are represented by a new prompt release rather than
    mutating an existing release in place. Timestamps are stored in UTC by
    convention and must always be timezone-aware.
    """

    version: PromptVersion
    lifecycle: PromptLifecycle = PromptLifecycle.DRAFT
    changelog: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    activated_at: datetime | None = None
    deprecated_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", PromptVersion.parse(self.version))

        if not isinstance(self.lifecycle, PromptLifecycle):
            raise TypeError("lifecycle must be a PromptLifecycle value.")
        if not isinstance(self.changelog, str):
            raise TypeError("changelog must be a string.")

        _require_aware_datetime(self.created_at, "created_at")

        if self.activated_at is not None:
            _require_aware_datetime(self.activated_at, "activated_at")
            if self.activated_at < self.created_at:
                raise ValueError("activated_at cannot be earlier than created_at.")

        if self.deprecated_at is not None:
            _require_aware_datetime(self.deprecated_at, "deprecated_at")
            lower_bound = self.activated_at or self.created_at
            if self.deprecated_at < lower_bound:
                raise ValueError(
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
            raise ValueError("A draft prompt cannot have an activated_at timestamp.")


__all__ = [
    "PromptLifecycle",
    "PromptVersion",
    "PromptVersionMetadata",
]
