from app.modules.users.models.user import User
from app.modules.conversations.models.conversation import Conversation
from app.modules.conversations.models.message import Message
from app.modules.auth.models.session import AuthSession
from app.modules.auth.models.refresh_token import RefreshToken
from app.modules.auth.models.email_verification_token import EmailVerificationToken
from app.modules.auth.models.password_reset_token import PasswordResetToken
from app.modules.auth.models.rate_limit import AuthenticationRateLimitEntry


MODEL_REGISTRY = (
    User,
    Conversation,
    Message,
    AuthSession,
    RefreshToken,
    EmailVerificationToken,
    PasswordResetToken,
    AuthenticationRateLimitEntry,
)

__all__ = [
    "User",
    "Conversation",
    "Message",
    "AuthSession",
    "RefreshToken",
    "EmailVerificationToken",
    "PasswordResetToken",
    "AuthenticationRateLimitEntry",
    "MODEL_REGISTRY",
]
