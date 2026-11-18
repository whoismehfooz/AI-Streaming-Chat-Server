"""
Provider-neutral exceptions for the LLM integration boundary.

Concrete provider adapters translate vendor SDK failures into these
exceptions. Application services can then handle provider failures without
importing or catching vendor-specific exception classes.

These exceptions inherit from the existing ``AIProviderException`` so
existing application code that catches that legacy exception continues to
catch the new provider-specific failures.

Important separation:
    - Provider authentication failures concern the server's credentials
      with the upstream LLM vendor. They are not users AuthenticationException
      failures.
    - Model access, users quotas, token budgets, and feature permissions are
      application authorization concerns. Raise the existing exceptions from
      ``app.exceptions.authorization`` in the relevant application service,
      not from this external-provider adapter.
    - Resource-not-found exceptions are not provider transport failures.
    - asyncio cancellation must propagate as cancellation and must not be
      translated into one of these provider exceptions.
"""

from __future__ import annotations

from typing import ClassVar

from app.exceptions.base import AIProviderException
from app.exceptions.error_codes import ErrorCode


class LLMProviderException(AIProviderException):
    """
    Base exception for failures originating at the LLM integration boundary.

    The public message must be safe for API consumers. Raw provider response
    bodies, credentials, request headers, SDK exception text, and stack
    traces must remain in protected server-side logs.

    Attributes:
        error_code: Stable application error code.
        provider_name: Safe provider identifier, never a credential.
        provider_request_id: Upstream request identifier when available.
        retryable: Whether a retry may be appropriate under the configured
            retry policy. Streaming adapters must additionally ensure that
            no output has already been emitted.
        retry_after_seconds: Optional upstream retry delay.
    """

    default_message: ClassVar[str] = "The LLM provider could not process the request."
    default_status_code: ClassVar[int] = 502
    default_error_code: ClassVar[ErrorCode] = ErrorCode.AI_PROVIDER_ERROR
    default_retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str | None = None,
        *,
        status_code: int | None = None,
        error_code: ErrorCode | None = None,
        provider_name: str | None = None,
        provider_request_id: str | None = None,
        retryable: bool | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        resolved_message = (
            self.default_message if message is None else message
        )
        resolved_status_code = (
            self.default_status_code if status_code is None else status_code
        )
        resolved_error_code = (
            self.default_error_code if error_code is None else error_code
        )
        resolved_retryable = (
            self.default_retryable if retryable is None else retryable
        )

        if not isinstance(resolved_message, str) or not resolved_message.strip():
            raise ValueError("LLM provider exception message must be a non-empty string.")

        if (
            isinstance(resolved_status_code, bool)
            or not isinstance(resolved_status_code, int)
            or not 400 <= resolved_status_code <= 599
        ):
            raise ValueError(
                "LLM provider exception status_code must be an integer between 400 and 599."
            )

        if not isinstance(resolved_error_code, ErrorCode):
            raise TypeError("error_code must be an ErrorCode member.")

        if not isinstance(resolved_retryable, bool):
            raise TypeError("retryable must be a boolean.")

        if provider_name is not None:
            if (
                not isinstance(provider_name, str)
                or not provider_name.strip()
                or len(provider_name) > 64
            ):
                raise ValueError("provider_name must be a non-empty string of at most 64 characters.")

        if provider_request_id is not None:
            if (
                not isinstance(provider_request_id, str)
                or not provider_request_id.strip()
                or len(provider_request_id) > 256
            ):
                raise ValueError(
                    "provider_request_id must be a non-empty string of at most 256 characters."
                )

        if retry_after_seconds is not None:
            if (
                isinstance(retry_after_seconds, bool)
                or not isinstance(retry_after_seconds, int)
                or retry_after_seconds < 0
            ):
                raise ValueError(
                    "retry_after_seconds must be a non-negative integer."
                )

        # Preserve the existing AIProviderException inheritance and its
        # constructor contract for callers that already catch that type.
        super().__init__(
            message=resolved_message,
            status_code=resolved_status_code,
        )

        # AIProviderException uses the generic AI_PROVIDER_ERROR code.
        # Specializations set their more precise stable code here.
        self.error_code = resolved_error_code
        self.provider_name = provider_name
        self.provider_request_id = provider_request_id
        self.retryable = resolved_retryable
        self.retry_after_seconds = retry_after_seconds

        if retry_after_seconds is not None:
            # Numeric Retry-After is safe and conforms to the HTTP delta-seconds
            # form. The shared base exception already owns a validated dict.
            self.headers["Retry-After"] = str(retry_after_seconds)


class LLMProviderTimeoutException(LLMProviderException):
    """Raised when an upstream LLM operation exceeds its configured deadline."""

    default_message = "The LLM provider request timed out."
    default_status_code = 504
    default_error_code = ErrorCode.AI_PROVIDER_TIMEOUT
    default_retryable = True


class LLMProviderRateLimitException(LLMProviderException):
    """
    Raised when the upstream provider throttles this application's request.

    This is distinct from an application's own users quota, which should use
    AIUsageQuotaExceededException and the authorization layer. A provider-side
    throttle is represented as 503 because the dependency is temporarily
    unable to serve the application.
    """

    default_message = "The LLM provider is temporarily rate limiting requests."
    default_status_code = 503
    default_error_code = ErrorCode.AI_PROVIDER_RATE_LIMITED
    default_retryable = True


class LLMProviderUnavailableException(LLMProviderException):
    """Raised when the upstream LLM service is unavailable."""

    default_message = "The LLM provider is temporarily unavailable."
    default_status_code = 503
    default_error_code = ErrorCode.AI_PROVIDER_UNAVAILABLE
    default_retryable = True


class LLMProviderConnectionException(LLMProviderUnavailableException):
    """Raised when a connection to the upstream LLM provider cannot be made."""

    default_message = "The application could not connect to the LLM provider."


class LLMProviderAuthenticationException(LLMProviderException):
    """
    Raised when the upstream provider rejects this application's credentials.

    This indicates provider-side configuration/credential failure, not an
    unauthenticated end users. It must never be converted to a users 401.
    """

    default_message = "The LLM provider rejected the configured credentials."
    default_status_code = 502
    default_error_code = ErrorCode.AI_PROVIDER_ERROR
    default_retryable = False


class LLMProviderConfigurationException(LLMProviderException):
    """Raised when local provider configuration is missing or invalid."""

    default_message = "The LLM provider is not configured correctly."
    default_status_code = 500
    default_error_code = ErrorCode.AI_PROVIDER_ERROR
    default_retryable = False


class LLMProviderResponseException(LLMProviderException):
    """
    Raised when the provider returns a malformed or unusable response.

    Do not include the raw response body in the public message.
    """

    default_message = "The LLM provider returned an invalid response."
    default_status_code = 502
    default_error_code = ErrorCode.AI_PROVIDER_ERROR
    default_retryable = False


class LLMProviderRequestRejectedException(LLMProviderException):
    """
    Raised when the upstream provider rejects a request the adapter sent.

    Since the application's request has already passed local validation,
    this commonly indicates an unsupported model/parameter combination or
    an upstream contract mismatch. The adapter should log safe diagnostics.
    """

    default_message = "The LLM provider rejected the generation request."
    default_status_code = 502
    default_error_code = ErrorCode.AI_PROVIDER_ERROR
    default_retryable = False


class LLMModelNotSupportedException(LLMProviderException):
    """Raised when the selected model is not supported by this adapter."""

    default_message = "The selected model is not supported by this LLM provider."
    default_status_code = 400
    default_error_code = ErrorCode.AI_PROVIDER_ERROR
    default_retryable = False


class LLMStreamingNotSupportedException(LLMProviderException):
    """Raised when streaming is requested from a provider/model that lacks it."""

    default_message = "Streaming is not supported by the selected LLM provider or model."
    default_status_code = 400
    default_error_code = ErrorCode.AI_PROVIDER_ERROR
    default_retryable = False


class LLMStreamInterruptedException(LLMProviderException):
    """
    Raised when an upstream stream fails after it has been established.

    The exception records whether output has already been delivered and how
    many chunks were emitted. It intentionally does not retain partial text:
    generated content may be sensitive and should be managed by the owning
    chat/stream service.

    A stream that emitted any output is never marked retryable because
    replaying it could duplicate already-delivered content.
    """

    default_message = "The LLM response stream was interrupted."
    default_status_code = 502
    default_error_code = ErrorCode.AI_PROVIDER_ERROR
    default_retryable = False

    def __init__(
        self,
        message: str | None = None,
        *,
        provider_name: str | None = None,
        provider_request_id: str | None = None,
        chunks_emitted: int = 0,
        partial_output_emitted: bool | None = None,
        retryable: bool | None = None,
    ) -> None:
        if (
            isinstance(chunks_emitted, bool)
            or not isinstance(chunks_emitted, int)
            or chunks_emitted < 0
        ):
            raise ValueError("chunks_emitted must be a non-negative integer.")

        emitted = (
            chunks_emitted > 0
            if partial_output_emitted is None
            else partial_output_emitted
        )

        if not isinstance(emitted, bool):
            raise TypeError("partial_output_emitted must be a boolean.")

        if emitted and chunks_emitted == 0:
            raise ValueError(
                "chunks_emitted must be greater than zero when partial output was emitted."
            )

        # A stream cannot be safely replayed after any output has escaped.
        resolved_retryable = False if emitted else retryable

        super().__init__(
            message=message,
            provider_name=provider_name,
            provider_request_id=provider_request_id,
            retryable=resolved_retryable,
        )

        self.chunks_emitted = chunks_emitted
        self.partial_output_emitted = emitted


__all__ = [
    "LLMProviderException",
    "LLMProviderTimeoutException",
    "LLMProviderRateLimitException",
    "LLMProviderUnavailableException",
    "LLMProviderConnectionException",
    "LLMProviderAuthenticationException",
    "LLMProviderConfigurationException",
    "LLMProviderResponseException",
    "LLMProviderRequestRejectedException",
    "LLMModelNotSupportedException",
    "LLMStreamingNotSupportedException",
    "LLMStreamInterruptedException",
]
