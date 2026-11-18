"""
Authentication exceptions.

This module contains exceptions related to identity verification,
credentials, account authentication state, access tokens, refresh tokens,
sessions, email verification, password recovery and authentication
security controls.

Authentication answers the question:

    Who is making this request?

Authentication failures must remain separate from authorization failures.

HTTP conventions:
    - 401: Missing or invalid authentication credentials.
    - 403: An authenticated account cannot currently perform the
      requested authentication-dependent operation.
    - 423: An account is temporarily locked.
    - 429: Authentication rate limits have been exceeded.

Sensitive information must never be included in exception messages.
"""

from typing import ClassVar

from app.exceptions.base import ApplicationException
from app.exceptions.error_codes import ErrorCode


# ================================================================
# AUTHENTICATION CONSTANTS
# ================================================================

DEFAULT_AUTHENTICATION_MESSAGE = "Authentication failed."

DEFAULT_AUTHENTICATION_REQUIRED_MESSAGE = (
    "Authentication credentials are required."
)

DEFAULT_INVALID_CREDENTIALS_MESSAGE = (
    "The provided credentials are invalid."
)

DEFAULT_INVALID_ACCESS_TOKEN_MESSAGE = (
    "The access token is invalid."
)

DEFAULT_EXPIRED_ACCESS_TOKEN_MESSAGE = (
    "The access token has expired."
)

DEFAULT_INVALID_REFRESH_TOKEN_MESSAGE = (
    "The refresh token is invalid."
)

DEFAULT_EXPIRED_REFRESH_TOKEN_MESSAGE = (
    "The refresh token has expired."
)


def _bearer_challenge(error: str | None = None) -> dict[str, str]:
    """
    Build the HTTP Bearer authentication challenge.

    The challenge is intended for protected endpoints that authenticate
    requests using Bearer access tokens.

    Login and refresh-token endpoints should not automatically expose
    this challenge because their credential mechanisms may differ.
    """

    if error is None:
        return {
            "WWW-Authenticate": "Bearer",
        }

    return {
        "WWW-Authenticate": f'Bearer error="{error}"',
    }


# ================================================================
# BASE AUTHENTICATION EXCEPTION
# ================================================================

class AuthenticationException(ApplicationException):
    """
    Base exception for authentication-related failures.

    All authentication exceptions inherit from this class.

    The default status code is 401, but specialized subclasses may use
    another HTTP status when required by the authentication workflow.
    """

    default_message: ClassVar[str] = DEFAULT_AUTHENTICATION_MESSAGE

    default_status_code: ClassVar[int] = 401

    default_error_code: ClassVar[ErrorCode] = (
        ErrorCode.AUTHENTICATION_FAILED
    )

    def __init__(
        self,
        message: str | None = None,
        *,
        status_code: int | None = None,
        error_code: ErrorCode | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        resolved_message = (
            message
            if message is not None
            else self.default_message
        )

        resolved_status_code = (
            status_code
            if status_code is not None
            else self.default_status_code
        )

        resolved_error_code = (
            error_code
            if error_code is not None
            else self.default_error_code
        )

        super().__init__(
            message=resolved_message,
            status_code=resolved_status_code,
            error_code=resolved_error_code,
            headers=headers,
        )


# ================================================================
# GENERAL AUTHENTICATION ERRORS
# ================================================================

class AuthenticationRequiredException(AuthenticationException):
    """
    Raised when a protected operation requires authentication but
    the request contains no usable authentication credentials.
    """

    default_message = DEFAULT_AUTHENTICATION_REQUIRED_MESSAGE

    default_error_code = ErrorCode.AUTHENTICATION_REQUIRED

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            headers=_bearer_challenge(),
        )


class InvalidCredentialsException(AuthenticationException):
    """
    Raised when supplied login credentials cannot authenticate a users.

    The message intentionally does not reveal whether the email address
    or password was incorrect.
    """

    default_message = DEFAULT_INVALID_CREDENTIALS_MESSAGE

    default_error_code = ErrorCode.INVALID_CREDENTIALS


class InvalidPasswordException(AuthenticationException):
    """
    Raised when a supplied password does not satisfy the expected
    authentication or password verification requirements.
    """

    default_message = "The supplied password is invalid."

    default_error_code = ErrorCode.INVALID_PASSWORD


class PasswordMismatchException(AuthenticationException):
    """
    Raised when password confirmation does not match the new password.
    """

    default_message = "The password confirmation does not match."

    default_status_code = 400

    default_error_code = ErrorCode.PASSWORD_MISMATCH


class PasswordPolicyViolationException(AuthenticationException):
    """
    Raised when a new password violates the application's password
    security policy.
    """

    default_message = (
        "The password does not satisfy the password security policy."
    )

    default_status_code = 400

    default_error_code = ErrorCode.PASSWORD_POLICY_VIOLATION


class ReauthenticationRequiredException(AuthenticationException):
    """
    Raised when a sensitive operation requires the users to verify
    their identity again.
    """

    default_message = (
        "Reauthentication is required to perform this operation."
    )

    default_error_code = ErrorCode.REAUTHENTICATION_REQUIRED


# ================================================================
# ACCOUNT STATE ERRORS
# ================================================================

class AccountInactiveException(AuthenticationException):
    """
    Raised when an inactive account attempts to authenticate.
    """

    default_message = "This account is inactive."

    default_status_code = 403

    default_error_code = ErrorCode.ACCOUNT_INACTIVE


class AccountSuspendedException(AuthenticationException):
    """
    Raised when a suspended account attempts to authenticate.
    """

    default_message = "This account has been suspended."

    default_status_code = 403

    default_error_code = ErrorCode.ACCOUNT_SUSPENDED


class AccountDeactivatedException(AuthenticationException):
    """
    Raised when a deactivated account attempts to authenticate.
    """

    default_message = "This account has been deactivated."

    default_status_code = 403

    default_error_code = ErrorCode.ACCOUNT_DEACTIVATED


class AccountDeletedException(AuthenticationException):
    """
    Raised when authentication is attempted for a deleted account.
    """

    default_message = "This account is no longer available."

    default_status_code = 403

    default_error_code = ErrorCode.ACCOUNT_DELETED


class AccountLockedException(AuthenticationException):
    """
    Raised when an account is temporarily locked because of security
    controls such as repeated failed authentication attempts.
    """

    default_message = (
        "This account is temporarily locked. Try again later."
    )

    default_status_code = 423

    default_error_code = ErrorCode.ACCOUNT_LOCKED


class AccountNotVerifiedException(AuthenticationException):
    """
    Raised when an account has not completed the required verification
    process.
    """

    default_message = (
        "Account verification is required before continuing."
    )

    default_status_code = 403

    default_error_code = ErrorCode.ACCOUNT_NOT_VERIFIED


class EmailNotVerifiedException(AuthenticationException):
    """
    Raised when an operation requires a verified email address.
    """

    default_message = (
        "Email verification is required before continuing."
    )

    default_status_code = 403

    default_error_code = ErrorCode.EMAIL_NOT_VERIFIED


# ================================================================
# ACCESS TOKEN ERRORS
# ================================================================

class InvalidAccessTokenException(AuthenticationException):
    """
    Raised when an access token cannot be accepted.

    Examples:
        - Invalid token signature.
        - Invalid token structure.
        - Invalid claims.
        - Invalid issuer.
        - Invalid audience.
        - Invalid token type.
    """

    default_message = DEFAULT_INVALID_ACCESS_TOKEN_MESSAGE

    default_error_code = ErrorCode.INVALID_ACCESS_TOKEN

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            headers=_bearer_challenge("invalid_token"),
        )


class ExpiredAccessTokenException(AuthenticationException):
    """
    Raised when an access token has expired.

    Clients should obtain a new access token through the refresh-token
    workflow rather than continuing to use the expired token.
    """

    default_message = DEFAULT_EXPIRED_ACCESS_TOKEN_MESSAGE

    default_error_code = ErrorCode.EXPIRED_ACCESS_TOKEN

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            headers=_bearer_challenge("invalid_token"),
        )


class RevokedAccessTokenException(AuthenticationException):
    """
    Raised when an access token is known to have been revoked.

    Access-token revocation checks should only be implemented if the
    application's token strategy supports them.
    """

    default_message = "The access token has been revoked."

    default_error_code = ErrorCode.REVOKED_ACCESS_TOKEN

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            headers=_bearer_challenge("invalid_token"),
        )


class MalformedAccessTokenException(AuthenticationException):
    """
    Raised when an access token has an invalid structure.
    """

    default_message = "The access token is malformed."

    default_error_code = ErrorCode.MALFORMED_ACCESS_TOKEN

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            headers=_bearer_challenge("invalid_token"),
        )


class InvalidTokenSignatureException(AuthenticationException):
    """
    Raised when JWT signature verification fails.
    """

    default_message = "The authentication token is invalid."

    default_error_code = ErrorCode.INVALID_TOKEN_SIGNATURE

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            headers=_bearer_challenge("invalid_token"),
        )


class InvalidTokenIssuerException(AuthenticationException):
    """
    Raised when the JWT issuer does not match the configured issuer.
    """

    default_message = "The authentication token is invalid."

    default_error_code = ErrorCode.INVALID_TOKEN_ISSUER

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            headers=_bearer_challenge("invalid_token"),
        )


class InvalidTokenAudienceException(AuthenticationException):
    """
    Raised when the JWT audience does not match the intended recipient.
    """

    default_message = "The authentication token is invalid."

    default_error_code = ErrorCode.INVALID_TOKEN_AUDIENCE

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            headers=_bearer_challenge("invalid_token"),
        )


class InvalidTokenClaimsException(AuthenticationException):
    """
    Raised when token claims fail application-level validation.
    """

    default_message = "The authentication token contains invalid claims."

    default_error_code = ErrorCode.INVALID_TOKEN_CLAIMS

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            headers=_bearer_challenge("invalid_token"),
        )


class MissingTokenClaimsException(AuthenticationException):
    """
    Raised when required JWT claims are absent.
    """

    default_message = "The authentication token is invalid."

    default_error_code = ErrorCode.MISSING_TOKEN_CLAIMS

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            headers=_bearer_challenge("invalid_token"),
        )


class InvalidTokenAlgorithmException(AuthenticationException):
    """
    Raised when a JWT uses an unsupported or disallowed signing
    algorithm.
    """

    default_message = "The authentication token is invalid."

    default_error_code = ErrorCode.INVALID_TOKEN_ALGORITHM

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            headers=_bearer_challenge("invalid_token"),
        )


class UnsupportedTokenTypeException(AuthenticationException):
    """
    Raised when a token of an unexpected type is supplied to an
    authentication operation.
    """

    default_message = "The supplied token type is not supported."

    default_error_code = ErrorCode.UNSUPPORTED_TOKEN_TYPE

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            headers=_bearer_challenge("invalid_token"),
        )


# ================================================================
# REFRESH TOKEN ERRORS
# ================================================================

class InvalidRefreshTokenException(AuthenticationException):
    """
    Raised when a refresh token cannot be validated.
    """

    default_message = DEFAULT_INVALID_REFRESH_TOKEN_MESSAGE

    default_error_code = ErrorCode.INVALID_REFRESH_TOKEN


class ExpiredRefreshTokenException(AuthenticationException):
    """
    Raised when a refresh token has expired.
    """

    default_message = DEFAULT_EXPIRED_REFRESH_TOKEN_MESSAGE

    default_error_code = ErrorCode.EXPIRED_REFRESH_TOKEN


class RevokedRefreshTokenException(AuthenticationException):
    """
    Raised when a refresh token has already been revoked.
    """

    default_message = "The refresh token has been revoked."

    default_error_code = ErrorCode.REVOKED_REFRESH_TOKEN


class RefreshTokenReuseDetectedException(AuthenticationException):
    """
    Raised when a previously rotated refresh token is reused.

    The authentication service must perform the appropriate security
    response, such as revoking the affected token family, before
    propagating this exception.

    Raising this exception alone does not revoke tokens.
    """

    default_message = (
        "Refresh token reuse was detected. Please authenticate again."
    )

    default_error_code = ErrorCode.REFRESH_TOKEN_REUSE_DETECTED


class RefreshTokenRotationFailedException(AuthenticationException):
    """
    Raised when the refresh-token rotation workflow cannot complete.
    """

    default_message = "The refresh-token operation could not be completed."

    default_error_code = ErrorCode.REFRESH_TOKEN_ROTATION_FAILED


# ================================================================
# SESSION ERRORS
# ================================================================


class SessionExpiredException(AuthenticationException):
    """
    Raised when an authentication session has expired.
    """

    default_message = "The authentication session has expired."

    default_error_code = ErrorCode.SESSION_EXPIRED


class SessionRevokedException(AuthenticationException):
    """
    Raised when an authentication session has been revoked.
    """

    default_message = "The authentication session has been revoked."

    default_error_code = ErrorCode.SESSION_REVOKED


class SessionInactiveException(AuthenticationException):
    """
    Raised when an inactive session cannot be used for authentication.
    """

    default_message = "The authentication session is inactive."

    default_error_code = ErrorCode.SESSION_INACTIVE


class SessionLimitExceededException(AuthenticationException):
    """
    Raised when the configured concurrent-session limit is exceeded.
    """

    default_message = (
        "The maximum number of active sessions has been reached."
    )

    default_status_code = 429

    default_error_code = ErrorCode.SESSION_LIMIT_EXCEEDED


class InvalidSessionException(AuthenticationException):
    """
    Raised when session information cannot be validated.
    """

    default_message = "The authentication session is invalid."

    default_error_code = ErrorCode.INVALID_SESSION


# ================================================================
# EMAIL VERIFICATION ERRORS
# ================================================================

class EmailAlreadyVerifiedException(AuthenticationException):
    """
    Raised when an operation requires an unverified email address,
    but the email address has already been verified.
    """

    default_message = "The email address has already been verified."

    default_status_code = 409

    default_error_code = ErrorCode.EMAIL_ALREADY_VERIFIED


class InvalidEmailVerificationTokenException(AuthenticationException):
    """
    Raised when an email-verification token is invalid.
    """

    default_message = "The email-verification token is invalid."

    default_status_code = 400

    default_error_code = ErrorCode.INVALID_EMAIL_VERIFICATION_TOKEN


class ExpiredEmailVerificationTokenException(AuthenticationException):
    """
    Raised when an email-verification token has expired.
    """

    default_message = "The email-verification token has expired."

    default_status_code = 400

    default_error_code = ErrorCode.EXPIRED_EMAIL_VERIFICATION_TOKEN


class UsedEmailVerificationTokenException(AuthenticationException):
    """
    Raised when an email-verification token has already been consumed.
    """

    default_message = "The email-verification token has already been used."

    default_status_code = 400

    default_error_code = ErrorCode.USED_EMAIL_VERIFICATION_TOKEN


class EmailVerificationRequiredException(AuthenticationException):
    """
    Raised when an operation requires successful email verification.
    """

    default_message = "Email verification is required."

    default_status_code = 403

    default_error_code = ErrorCode.EMAIL_VERIFICATION_REQUIRED


class EmailVerificationFailedException(AuthenticationException):
    """
    Raised when the email-verification workflow cannot be completed.
    """

    default_message = "Email verification could not be completed."

    default_status_code = 400

    default_error_code = ErrorCode.EMAIL_VERIFICATION_FAILED


# ================================================================
# PASSWORD RESET AND RECOVERY ERRORS
# ================================================================

class InvalidPasswordResetTokenException(AuthenticationException):
    """
    Raised when a password-reset token is invalid.
    """

    default_message = "The password-reset token is invalid."

    default_status_code = 400

    default_error_code = ErrorCode.INVALID_PASSWORD_RESET_TOKEN


class ExpiredPasswordResetTokenException(AuthenticationException):
    """
    Raised when a password-reset token has expired.
    """

    default_message = "The password-reset token has expired."

    default_status_code = 400

    default_error_code = ErrorCode.EXPIRED_PASSWORD_RESET_TOKEN


class UsedPasswordResetTokenException(AuthenticationException):
    """
    Raised when a password-reset token has already been consumed.
    """

    default_message = "The password-reset token has already been used."

    default_status_code = 400

    default_error_code = ErrorCode.USED_PASSWORD_RESET_TOKEN


class PasswordResetRequiredException(AuthenticationException):
    """
    Raised when the users must complete password recovery before
    continuing.
    """

    default_message = "A password reset is required."

    default_status_code = 403

    default_error_code = ErrorCode.PASSWORD_RESET_REQUIRED


class PasswordResetTokenRequiredException(AuthenticationException):
    """
    Raised when a password-reset operation does not include the
    required reset token.
    """

    default_message = "A password-reset token is required."

    default_status_code = 400

    default_error_code = ErrorCode.PASSWORD_RESET_TOKEN_REQUIRED


class PasswordResetFailedException(AuthenticationException):
    """
    Raised when the password-reset workflow cannot be completed.
    """

    default_message = "The password-reset operation could not be completed."

    default_status_code = 400

    default_error_code = ErrorCode.PASSWORD_RESET_FAILED


class PasswordChangeFailedException(AuthenticationException):
    """
    Raised when an authenticated password-change operation fails.
    """

    default_message = "The password could not be changed."

    default_status_code = 400

    default_error_code = ErrorCode.PASSWORD_CHANGE_FAILED


# ================================================================
# AUTHENTICATION RATE LIMITING
# ================================================================

class AuthenticationRateLimitExceededException(AuthenticationException):
    """
    Raised when an authentication-related rate limit is exceeded.

    retry_after_seconds can be supplied by the rate-limiting service.
    When supplied, it is exposed through the standard Retry-After
    response header.
    """

    default_message = (
        "Too many authentication attempts. Please try again later."
    )

    default_status_code = 429

    default_error_code = ErrorCode.AUTHENTICATION_RATE_LIMIT_EXCEEDED

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        headers: dict[str, str] = {}

        if retry_after_seconds is not None:
            if (
                isinstance(retry_after_seconds, bool)
                or not isinstance(retry_after_seconds, int)
                or retry_after_seconds < 0
            ):
                raise ValueError(
                    "retry_after_seconds must be a non-negative integer."
                )

            headers["Retry-After"] = str(retry_after_seconds)

        self.retry_after_seconds = retry_after_seconds

        super().__init__(
            message=message,
            headers=headers or None,
        )


__all__ = [
    "AuthenticationException",
    "AuthenticationRequiredException",
    "InvalidCredentialsException",
    "InvalidPasswordException",
    "PasswordMismatchException",
    "PasswordPolicyViolationException",
    "ReauthenticationRequiredException",
    "AccountInactiveException",
    "AccountSuspendedException",
    "AccountDeactivatedException",
    "AccountDeletedException",
    "AccountLockedException",
    "AccountNotVerifiedException",
    "EmailNotVerifiedException",
    "InvalidAccessTokenException",
    "ExpiredAccessTokenException",
    "RevokedAccessTokenException",
    "MalformedAccessTokenException",
    "InvalidTokenSignatureException",
    "InvalidTokenIssuerException",
    "InvalidTokenAudienceException",
    "InvalidTokenClaimsException",
    "MissingTokenClaimsException",
    "InvalidTokenAlgorithmException",
    "UnsupportedTokenTypeException",
    "InvalidRefreshTokenException",
    "ExpiredRefreshTokenException",
    "RevokedRefreshTokenException",
    "RefreshTokenReuseDetectedException",
    "RefreshTokenRotationFailedException",
    "SessionExpiredException",
    "SessionRevokedException",
    "SessionInactiveException",
    "SessionLimitExceededException",
    "InvalidSessionException",
    "EmailAlreadyVerifiedException",
    "InvalidEmailVerificationTokenException",
    "ExpiredEmailVerificationTokenException",
    "UsedEmailVerificationTokenException",
    "EmailVerificationRequiredException",
    "EmailVerificationFailedException",
    "InvalidPasswordResetTokenException",
    "ExpiredPasswordResetTokenException",
    "UsedPasswordResetTokenException",
    "PasswordResetRequiredException",
    "PasswordResetTokenRequiredException",
    "PasswordResetFailedException",
    "PasswordChangeFailedException",
    "AuthenticationRateLimitExceededException",
]
