from enum import Enum


class AuthTokenType(str, Enum):
    """Purpose/type claim for credentials managed by the authentication domain."""

    ACCESS = "access"
    REFRESH = "refresh"
    EMAIL_VERIFICATION = "email_verification"
    PASSWORD_RESET = "password_reset"


class SessionRevocationReason(str, Enum):
    """Reason an authenticated session was invalidated."""

    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    REFRESH_TOKEN_REUSE = "refresh_token_reuse"
    ACCOUNT_DEACTIVATED = "account_deactivated"
    SECURITY_EVENT = "security_event"
