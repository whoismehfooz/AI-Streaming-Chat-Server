"""
Application exception hierarchy.

This module is the single shared foundation for application-level errors.

Responsibilities:
    - Provide a stable exception contract for every application domain.
    - Preserve the original exception API: message and status_code.
    - Add machine-readable error codes and optional HTTP response headers.
    - Keep legacy AI, chat, users, and account exceptions available.
    - Provide one base class for authentication and authorization errors.

This module deliberately contains no FastAPI dependencies. Domain and
service code can raise these exceptions without depending on the HTTP
transport layer. HTTP response translation belongs in handlers.py.

Security notes:
    - Exception messages are public API text. Never put secrets, tokens,
      passwords, SQL statements, provider payloads, or stack traces in them.
    - Do not use exception classes as a substitute for authorization checks,
      transaction management, token revocation, or stream cleanup.
"""

import re
from collections.abc import Mapping
from typing import ClassVar

from app.exceptions.error_codes import ErrorCode


_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


class ApplicationException(Exception):
    """
    Base class for expected application-level failures.

    Args:
        message: Safe, users-facing error message.
        status_code: HTTP status code to use at the API boundary.
        error_code: Stable machine-readable application error identifier.
        headers: Optional HTTP response headers, for example
            ``WWW-Authenticate`` or ``Retry-After``.

    The positional ``message, status_code`` form is intentionally retained
    for compatibility with the original base.py classes.
    New domain exceptions should normally use keyword arguments.

    This class represents client/application errors (400-599), not successful
    HTTP control-flow responses such as redirects.
    """

    default_message: ClassVar[str] = "An application error occurred."
    default_status_code: ClassVar[int] = 500
    default_error_code: ClassVar[ErrorCode] = ErrorCode.APPLICATION_ERROR

    def __init__(
        self,
        message: str | None = None,
        status_code: int | None = None,
        *,
        error_code: ErrorCode | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        resolved_message = (
            self.default_message if message is None else message
        )
        resolved_status_code = (
            self.default_status_code
            if status_code is None
            else status_code
        )
        resolved_error_code = (
            self.default_error_code
            if error_code is None
            else error_code
        )

        if not isinstance(resolved_message, str) or not resolved_message.strip():
            raise ValueError("Application exception message must be a non-empty string.")

        if (
            isinstance(resolved_status_code, bool)
            or not isinstance(resolved_status_code, int)
            or not 400 <= resolved_status_code <= 599
        ):
            raise ValueError(
                "Application exception status_code must be an integer between 400 and 599."
            )

        if not isinstance(resolved_error_code, ErrorCode):
            raise TypeError("error_code must be an ErrorCode member.")

        normalized_headers = self._validate_headers(headers)

        self.message = resolved_message
        self.status_code = resolved_status_code
        self.error_code = resolved_error_code
        # Always expose a fresh dictionary. Callers cannot mutate the
        # exception's headers by changing the mapping passed to __init__.
        self.headers: dict[str, str] = normalized_headers

        super().__init__(self.message)

    @staticmethod
    def _validate_headers(
        headers: Mapping[str, str] | None,
    ) -> dict[str, str]:
        """
        Validate and copy response headers.

        Header names are checked against the HTTP token character set.
        Header values reject CR/LF to prevent response-splitting attacks.
        """
        if headers is None:
            return {}

        if not isinstance(headers, Mapping):
            raise TypeError("headers must be a mapping of string keys to string values.")

        validated: dict[str, str] = {}

        for name, value in headers.items():
            if not isinstance(name, str) or not _HEADER_NAME_PATTERN.fullmatch(name):
                raise ValueError("An application exception contains an invalid HTTP header name.")

            if not isinstance(value, str):
                raise TypeError("HTTP response header values must be strings.")

            if "\r" in value or "\n" in value:
                raise ValueError("HTTP response header values must not contain CR or LF.")

            validated[name] = value

        return validated

    def __str__(self) -> str:
        """Return the safe, public-facing exception message."""
        return self.message


class AIProviderException(ApplicationException):
    """
    Raised when an AI provider cannot successfully process a request.

    Provider diagnostics belong in protected server-side logs, not in the
    public exception message.
    """

    def __init__(
        self,
        message: str = "The AI provider could not process the request.",
        status_code: int = 502,
    ) -> None:
        super().__init__(
            message,
            status_code,
            error_code=ErrorCode.AI_PROVIDER_ERROR,
        )


class ContextBudgetExceededException(ApplicationException):
    """Raised when the conversation context exceeds its permitted token budget."""

    def __init__(
        self,
        message: str = "The conversation context exceeds the allowed token budget.",
        status_code: int = 400,
    ) -> None:
        super().__init__(
            message,
            status_code,
            error_code=ErrorCode.CONTEXT_BUDGET_EXCEEDED,
        )

class EmailAlreadyRegisteredException(ApplicationException):
    """Raised when an email address is already associated with an account."""

    def __init__(
        self,
        message: str = "An account with this email already exists.",
        status_code: int = 409,
    ) -> None:
        super().__init__(
            message,
            status_code,
            error_code=ErrorCode.EMAIL_ALREADY_REGISTERED,
        )


__all__ = [
    "ApplicationException",
    "AIProviderException",
    "ContextBudgetExceededException",
    "EmailAlreadyRegisteredException"
]
