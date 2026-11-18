from __future__ import annotations

from urllib.parse import urlencode, urlsplit, urlunsplit

from jinja2 import Environment, select_autoescape

from app.integrations.email.base import EmailContent


_TEMPLATE_ENVIRONMENT = Environment(
    autoescape=select_autoescape(
        enabled_extensions=("html", "htm", "xml"),
        default_for_string=True,
    ),
    trim_blocks=True,
    lstrip_blocks=True,
)

_VERIFICATION_SUBJECT = "Verify your email address"
_PASSWORD_RESET_SUBJECT = "Reset your password"
_EMAIL_CHANGE_SUBJECT = "Confirm your new email address"

_VERIFICATION_HTML = _TEMPLATE_ENVIRONMENT.from_string(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ subject }}</title>
</head>
<body style="margin:0;padding:24px;background:#f5f7fa;color:#172033;font-family:Arial,Helvetica,sans-serif;line-height:1.6;">
  <main style="max-width:560px;margin:0 auto;padding:32px;background:#ffffff;border:1px solid #e5e9f0;border-radius:12px;">
    <p style="margin:0 0 12px;color:#526078;font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;">{{ app_name }}</p>
    <h1 style="margin:0 0 16px;font-size:24px;line-height:1.25;">Verify your email address</h1>
    <p style="margin:0 0 24px;">Confirm this email address to finish creating your account.</p>
    <p style="margin:0 0 24px;">
      <a href="{{ action_url }}" style="display:inline-block;padding:12px 20px;border-radius:8px;background:#2457d6;color:#ffffff;text-decoration:none;font-weight:700;">Verify email address</a>
    </p>
    <p style="margin:0 0 8px;">This link expires in {{ expires_in_minutes }} minutes.</p>
    <p style="margin:0 0 8px;color:#526078;font-size:13px;">If the button does not work, copy and paste this URL into your browser:</p>
    <p style="margin:0 0 24px;overflow-wrap:anywhere;font-size:13px;"><a href="{{ action_url }}">{{ action_url }}</a></p>
    <p style="margin:0;color:#526078;font-size:13px;">If you did not create this account, you can safely ignore this email.</p>
  </main>
</body>
</html>"""
)

_VERIFICATION_TEXT = _TEMPLATE_ENVIRONMENT.from_string(
    """{{ app_name }}

Verify your email address

Confirm this email address to finish creating your account:
{{ action_url }}

This link expires in {{ expires_in_minutes }} minutes.
If you did not create this account, you can safely ignore this email.
"""
)

_EMAIL_CHANGE_HTML = _TEMPLATE_ENVIRONMENT.from_string(
    """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{{ subject }}</title></head>
<body style="margin:0;padding:24px;background:#f5f7fa;color:#172033;font-family:Arial,Helvetica,sans-serif;line-height:1.6;"><main style="max-width:560px;margin:0 auto;padding:32px;background:#fff;border:1px solid #e5e9f0;border-radius:12px;">
<p>{{ app_name }}</p><h1>Confirm your new email address</h1><p>We received a request to change the email address on your account to {{ recipient }}.</p>
<p><a href="{{ action_url }}">Confirm new email address</a></p><p>This link expires in {{ expires_in_minutes }} minutes and can be used only once.</p>
<p>If you did not request this change, ignore this message. Your current email address will remain unchanged.</p></main></body></html>"""
)
_EMAIL_CHANGE_TEXT = _TEMPLATE_ENVIRONMENT.from_string(
    "{{ app_name }}\n\nConfirm your new email address\n\nWe received a request to change your account email to {{ recipient }}.\n{{ action_url }}\n\nThis link expires in {{ expires_in_minutes }} minutes and can be used only once.\nIf you did not request this change, ignore this message. Your current email address will remain unchanged.\n"
)

_PASSWORD_RESET_HTML = _TEMPLATE_ENVIRONMENT.from_string(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ subject }}</title>
</head>
<body style="margin:0;padding:24px;background:#f5f7fa;color:#172033;font-family:Arial,Helvetica,sans-serif;line-height:1.6;">
  <main style="max-width:560px;margin:0 auto;padding:32px;background:#ffffff;border:1px solid #e5e9f0;border-radius:12px;">
    <p style="margin:0 0 12px;color:#526078;font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;">{{ app_name }}</p>
    <h1 style="margin:0 0 16px;font-size:24px;line-height:1.25;">Reset your password</h1>
    <p style="margin:0 0 24px;">We received a request to reset the password for your account. Use the link below to continue.</p>
    <p style="margin:0 0 24px;">
      <a href="{{ action_url }}" style="display:inline-block;padding:12px 20px;border-radius:8px;background:#2457d6;color:#ffffff;text-decoration:none;font-weight:700;">Reset password</a>
    </p>
    <p style="margin:0 0 8px;">This link expires in {{ expires_in_minutes }} minutes and can be used only once.</p>
    <p style="margin:0 0 8px;color:#526078;font-size:13px;">If the button does not work, copy and paste this URL into your browser:</p>
    <p style="margin:0 0 24px;overflow-wrap:anywhere;font-size:13px;"><a href="{{ action_url }}">{{ action_url }}</a></p>
    <p style="margin:0;color:#526078;font-size:13px;">If you did not request a password reset, ignore this email. Your password will not change unless the reset flow is completed.</p>
  </main>
</body>
</html>"""
)

_PASSWORD_RESET_TEXT = _TEMPLATE_ENVIRONMENT.from_string(
    """{{ app_name }}

Reset your password

We received a request to reset the password for your account. Use this link to continue:
{{ action_url }}

This link expires in {{ expires_in_minutes }} minutes and can be used only once.
If you did not request a password reset, ignore this email. Your password will not change unless the reset flow is completed.
"""
)


def _build_action_url(base_url: str, *, path: str, token: str) -> str:
    """Build a same-origin action URL without accepting arbitrary URL input."""

    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty URL string.")
    if not isinstance(token, str) or not token or len(token) > 8192:
        raise ValueError("token must be a non-empty opaque token.")

    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url must not contain URL credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query or fragment.")

    base_path = parsed.path.rstrip("/")
    normalized_path = f"{base_path}/{path.lstrip('/')}"
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            normalized_path,
            urlencode({"token": token}),
            "",
        )
    )


def build_verification_url(base_url: str, token: str) -> str:
    """Build the verification URL consumed by the future Auth API route."""

    return _build_action_url(
        base_url,
        path="/api/v1/auth/verify-email",
        token=token,
    )


def build_password_reset_url(base_url: str, token: str) -> str:
    """Build the password-reset URL consumed by the future Auth UI/API flow."""

    return _build_action_url(
        base_url,
        path="/api/v1/auth/reset-password",
        token=token,
    )


def _validate_template_context(
    *,
    app_name: str,
    expires_in_minutes: int,
) -> None:
    if not isinstance(app_name, str) or not app_name.strip():
        raise ValueError("app_name must be a non-empty string.")
    if (
        isinstance(expires_in_minutes, bool)
        or not isinstance(expires_in_minutes, int)
        or expires_in_minutes <= 0
    ):
        raise ValueError("expires_in_minutes must be a positive integer.")


def render_verification_email(
    *,
    token: str,
    base_url: str,
    app_name: str,
    expires_in_minutes: int,
) -> EmailContent:
    """Render HTML and plain-text account-verification content."""

    _validate_template_context(
        app_name=app_name,
        expires_in_minutes=expires_in_minutes,
    )
    action_url = build_verification_url(base_url, token)
    context = {
        "subject": _VERIFICATION_SUBJECT,
        "app_name": app_name.strip(),
        "action_url": action_url,
        "expires_in_minutes": expires_in_minutes,
    }
    return EmailContent(
        subject=_VERIFICATION_SUBJECT,
        html=_VERIFICATION_HTML.render(**context),
        text=_VERIFICATION_TEXT.render(**context),
    )


def build_email_change_verification_url(base_url: str, token: str) -> str:
    return _build_action_url(base_url, path="/api/v1/auth/verify-email-change", token=token)


def render_email_change_verification_email(
    *, token: str, recipient: str, base_url: str, app_name: str, expires_in_minutes: int,
) -> EmailContent:
    _validate_template_context(app_name=app_name, expires_in_minutes=expires_in_minutes)
    action_url = build_email_change_verification_url(base_url, token)
    context = {"subject": _EMAIL_CHANGE_SUBJECT, "app_name": app_name.strip(),
               "recipient": recipient, "action_url": action_url,
               "expires_in_minutes": expires_in_minutes}
    return EmailContent(subject=_EMAIL_CHANGE_SUBJECT,
                        html=_EMAIL_CHANGE_HTML.render(**context),
                        text=_EMAIL_CHANGE_TEXT.render(**context))


def render_password_reset_email(
    *,
    token: str,
    base_url: str,
    app_name: str,
    expires_in_minutes: int,
) -> EmailContent:
    """Render HTML and plain-text password-reset content."""

    _validate_template_context(
        app_name=app_name,
        expires_in_minutes=expires_in_minutes,
    )
    action_url = build_password_reset_url(base_url, token)
    context = {
        "subject": _PASSWORD_RESET_SUBJECT,
        "app_name": app_name.strip(),
        "action_url": action_url,
        "expires_in_minutes": expires_in_minutes,
    }
    return EmailContent(
        subject=_PASSWORD_RESET_SUBJECT,
        html=_PASSWORD_RESET_HTML.render(**context),
        text=_PASSWORD_RESET_TEXT.render(**context),
    )


__all__ = [
    "build_password_reset_url",
    "build_email_change_verification_url",
    "render_email_change_verification_email",
    "build_verification_url",
    "render_password_reset_email",
    "render_verification_email",
]
