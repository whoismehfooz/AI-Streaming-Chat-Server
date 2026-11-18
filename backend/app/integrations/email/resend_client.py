"""Asynchronous Resend adapter for authentication email delivery.

The application lifespan owns global Resend SDK configuration (API key and
async HTTPX client). This adapter intentionally does not mutate SDK globals,
which avoids per-request credential races and remains compatible with the
existing ``core/lifespan.py`` implementation.

The adapter implements the exact keyword-only methods already injected into
``EmailVerificationService`` and ``PasswordService``. Database transactions,
token generation, token hashing, and account-state decisions remain in Auth.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import resend
from pydantic import EmailStr, TypeAdapter, ValidationError

from app.core.config import Settings, settings
from app.integrations.email.base import EmailProviderError
from app.integrations.email.templates import (
    render_password_reset_email,
    render_verification_email,
    render_email_change_verification_email,
)


_EMAIL_ADAPTER = TypeAdapter(EmailStr)
_SAFE_PROVIDER_CODE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class ResendEmailClient:
    """Deliver account-verification and password-reset messages through Resend."""

    def __init__(self, config: Settings = settings) -> None:
        if config is None:
            raise ValueError("config is required.")
        if config.RESEND_API_KEY is None:
            raise RuntimeError(
                "RESEND_API_KEY must be configured before creating ResendEmailClient."
            )

        self._config = config
        self._sender = self._format_sender(
            str(config.RESEND_FROM_NAME),
            str(config.RESEND_FROM_EMAIL),
        )
        self._base_url = str(config.BACKEND_URL).rstrip("/")

    @staticmethod
    def _format_sender(name: str, email: str) -> str:
        normalized_name = name.strip()
        normalized_email = email.strip()
        if not normalized_name or not normalized_email:
            raise ValueError("A valid sender name and email are required.")
        # EmailStr validation has already occurred in Settings. Reject header
        # control characters to prevent malformed/injected display-name headers.
        if any(character in normalized_name for character in "\r\n"):
            raise ValueError("Sender name must not contain line breaks.")
        return f"{normalized_name} <{normalized_email}>"

    @staticmethod
    def _validate_recipient(recipient: str) -> str:
        if not isinstance(recipient, str) or not recipient.strip():
            raise ValueError("recipient must be a non-empty email address.")
        try:
            return str(_EMAIL_ADAPTER.validate_python(recipient.strip()))
        except ValidationError as exc:
            # Do not include the supplied address in this exception text.
            raise ValueError("recipient must be a valid email address.") from None

    @staticmethod
    def _validate_token(token: str) -> str:
        if not isinstance(token, str) or not token or len(token) > 8192:
            raise ValueError("token must be a non-empty opaque token.")
        if any(character.isspace() for character in token):
            raise ValueError("token must not contain whitespace.")
        return token

    @staticmethod
    def _provider_error_code(error: Exception) -> str | None:
        candidate: Any = getattr(error, "code", None)
        if candidate is None:
            candidate = getattr(error, "status_code", None)
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            candidate = str(candidate)
        if isinstance(candidate, str) and _SAFE_PROVIDER_CODE.fullmatch(candidate):
            return candidate
        return None

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return True

        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code == 429 or 500 <= status_code <= 599

        code = getattr(error, "code", None)
        return isinstance(code, str) and code.lower() in {
            "rate_limit_exceeded",
            "temporarily_unavailable",
            "timeout",
        }

    async def _send(
        self,
        *,
        operation: str,
        recipient: str,
        subject: str,
        html: str,
        text: str,
        tag: str,
    ) -> None:
        """Send one transactional message and normalize provider failures."""

        if not self._config.RESEND_API_KEY:
            # Defensive guard for nonstandard Settings test doubles or runtime
            # mutation. Settings itself is frozen in the supplied application.
            raise RuntimeError("Resend API key is not configured.")

        params: resend.Emails.SendParams = {
            "from": self._sender,
            "to": [recipient],
            "subject": subject,
            "html": html,
            "text": text,
            "tags": [{"name": "category", "value": tag}],
        }

        try:
            response = await resend.Emails.send_async(params)
        except asyncio.CancelledError:
            # Preserve request/task cancellation. Never convert it into a
            # provider failure because cancellation must propagate promptly.
            raise
        except Exception as exc:
            raise EmailProviderError(
                operation=operation,
                provider_code=self._provider_error_code(exc),
                retryable=self._is_retryable(exc),
            ) from None

        # The SDK normally raises on provider/API errors. Still reject an
        # impossible empty response rather than reporting delivery success.
        if response is None:
            raise EmailProviderError(
                operation=operation,
                provider_code="empty_response",
                retryable=True,
            )

    async def send_verification_email(
        self,
        *,
        recipient: str,
        token: str,
    ) -> None:
        """Send the account-verification email expected by Auth services."""

        normalized_recipient = self._validate_recipient(recipient)
        raw_token = self._validate_token(token)
        content = render_verification_email(
            token=raw_token,
            base_url=self._base_url,
            app_name=self._config.APP_NAME,
            expires_in_minutes=self._config.EMAIL_VERIFICATION_TOKEN_EXPIRE_MINUTES,
        )
        await self._send(
            operation="send_verification_email",
            recipient=normalized_recipient,
            subject=content.subject,
            html=content.html,
            text=content.text,
            tag="email_verification",
        )

    async def send_password_reset_email(
        self,
        *,
        recipient: str,
        token: str,
    ) -> None:
        """Send the password-reset email expected by PasswordService."""

        normalized_recipient = self._validate_recipient(recipient)
        raw_token = self._validate_token(token)
        content = render_password_reset_email(
            token=raw_token,
            base_url=self._base_url,
            app_name=self._config.APP_NAME,
            expires_in_minutes=self._config.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES,
        )
        await self._send(
            operation="send_password_reset_email",
            recipient=normalized_recipient,
            subject=content.subject,
            html=content.html,
            text=content.text,
            tag="password_reset",
        )

    async def send_email_change_verification_email(
        self, *, recipient: str, token: str,
    ) -> None:
        """Send proof-of-control message to the requested new address."""
        normalized_recipient = self._validate_recipient(recipient)
        raw_token = self._validate_token(token)
        content = render_email_change_verification_email(
            token=raw_token, recipient=normalized_recipient,
            base_url=self._base_url, app_name=self._config.APP_NAME,
            expires_in_minutes=self._config.EMAIL_VERIFICATION_TOKEN_EXPIRE_MINUTES,
        )
        await self._send(
            operation="send_email_change_verification_email",
            recipient=normalized_recipient, subject=content.subject,
            html=content.html, text=content.text, tag="email_change_verification",
        )


__all__ = ["ResendEmailClient"]
