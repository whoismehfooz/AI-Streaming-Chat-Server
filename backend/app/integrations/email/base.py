"""Provider-neutral contracts and errors for transactional email delivery.

Auth services depend on these contracts rather than a concrete provider SDK.
The concrete Resend adapter implements both methods structurally, so it can be
injected into the existing ``AuthService`` and ``PasswordService`` without
changing their public interfaces.

Security boundary:
- Raw verification/reset tokens may appear in message content because the
  recipient must receive them, but they must never appear in logs or errors.
- Provider exceptions are translated into ``EmailProviderError`` with only
  safe, non-sensitive metadata. The original exception is suppressed when
  re-raising so provider payloads cannot leak through chained tracebacks.
- Delivery is an awaited operation in the request lifecycle. This module does
  not create background tasks, queues, retries, or database transactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class EmailContent:
    """Fully rendered email content, independent of any delivery provider."""

    subject: str
    html: str
    text: str

    def __post_init__(self) -> None:
        for field_name in ("subject", "html", "text"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Email {field_name} must be a non-empty string.")


class EmailProviderError(RuntimeError):
    """Safe, provider-neutral failure raised when email delivery fails.

    ``operation`` is an internal operation label, not user input.
    ``provider_code`` is optional and must contain a provider error identifier,
    never a raw response body or message. ``retryable`` is advisory only; this
    integration deliberately does not retry automatically.
    """

    def __init__(
        self,
        *,
        operation: str,
        provider_code: str | None = None,
        retryable: bool = False,
    ) -> None:
        if not operation or not operation.isidentifier():
            raise ValueError("operation must be a non-empty identifier.")
        if provider_code is not None and (
            not isinstance(provider_code, str)
            or not provider_code
            or len(provider_code) > 64
            or not provider_code.replace("_", "").replace("-", "").replace(".", "").isalnum()
        ):
            raise ValueError("provider_code must be a short safe identifier.")

        self.operation = operation
        self.provider_code = provider_code
        self.retryable = bool(retryable)
        super().__init__(f"Email delivery failed during {operation}.")


@runtime_checkable
class VerificationEmailSender(Protocol):
    """Contract consumed by ``EmailVerificationService``."""

    async def send_verification_email(
        self,
        *,
        recipient: str,
        token: str,
    ) -> None:
        """Deliver an account-verification token to ``recipient``."""
        ...


@runtime_checkable
class PasswordResetEmailSender(Protocol):
    """Contract consumed by ``PasswordService``."""

    async def send_password_reset_email(
        self,
        *,
        recipient: str,
        token: str,
    ) -> None:
        """Deliver a password-reset token to ``recipient``."""
        ...


@runtime_checkable
class AuthEmailSender(VerificationEmailSender, PasswordResetEmailSender, Protocol):
    """Combined email contract implemented by the application adapter."""


__all__ = [
    "AuthEmailSender",
    "EmailContent",
    "EmailProviderError",
    "PasswordResetEmailSender",
    "VerificationEmailSender",
]
