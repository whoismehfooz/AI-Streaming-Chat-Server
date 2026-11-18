"""
Authorization exceptions.

This module contains exceptions related to permissions, roles,
resource ownership, object-level access control, conversation and
message isolation, streaming access, and AI application permissions.

Authorization answers the question:

    Is the authenticated users allowed to perform this operation?

Authentication and authorization must remain separate.

HTTP conventions:
    - 403: The authenticated users does not have the required permission.
    - 404: A resource is deliberately treated as unavailable to avoid
      disclosing its existence to an unauthorized users.
    - 429: A usage quota or concurrency limit has been exceeded.

Authorization must follow a default-deny approach.

Services must enforce resource ownership independently of route-level
authorization dependencies.
"""

from typing import ClassVar

from app.exceptions.base import ApplicationException
from app.exceptions.error_codes import ErrorCode


# ================================================================
# AUTHORIZATION CONSTANTS
# ================================================================

DEFAULT_AUTHORIZATION_MESSAGE = (
    "You do not have permission to perform this operation."
)

DEFAULT_ACCESS_DENIED_MESSAGE = (
    "Access to this resource is denied."
)

DEFAULT_RESOURCE_ACCESS_DENIED_MESSAGE = (
    "The requested resource is not available."
)


# ================================================================
# BASE AUTHORIZATION EXCEPTION
# ================================================================

class AuthorizationException(ApplicationException):
    """
    Base exception for authorization-related failures.

    All authorization exceptions inherit from this class.

    The default HTTP status code is 403.

    Specialized resource-level exceptions may use 404 when concealing
    resource existence is necessary to prevent information disclosure.
    """

    default_message: ClassVar[str] = DEFAULT_AUTHORIZATION_MESSAGE

    default_status_code: ClassVar[int] = 403

    default_error_code: ClassVar[ErrorCode] = (
        ErrorCode.AUTHORIZATION_FAILED
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
# GENERAL ACCESS CONTROL ERRORS
# ================================================================

class AccessDeniedException(AuthorizationException):
    """
    Raised when access to a protected operation or resource is denied.
    """

    default_message = DEFAULT_ACCESS_DENIED_MESSAGE

    default_error_code = ErrorCode.ACCESS_DENIED


class PermissionDeniedException(AuthorizationException):
    """
    Raised when the authenticated users lacks permission to perform
    a requested operation.
    """

    default_message = (
        "You do not have permission to perform this operation."
    )

    default_error_code = ErrorCode.PERMISSION_DENIED


class InsufficientPermissionsException(AuthorizationException):
    """
    Raised when the authenticated users's effective permissions are
    insufficient for an operation.
    """

    default_message = (
        "You do not have sufficient permissions to perform this operation."
    )

    default_error_code = ErrorCode.INSUFFICIENT_PERMISSIONS


class RequiredRoleMissingException(AuthorizationException):
    """
    Raised when an operation requires a role that the authenticated
    users does not possess.
    """

    default_message = (
        "You do not have the required role to perform this operation."
    )

    default_error_code = ErrorCode.REQUIRED_ROLE_MISSING


class PrivilegeEscalationDeniedException(AuthorizationException):
    """
    Raised when a users attempts to grant or obtain privileges beyond
    their own authorization scope.
    """

    default_message = (
        "The requested privilege escalation is not permitted."
    )

    default_error_code = ErrorCode.PRIVILEGE_ESCALATION_DENIED


class InvalidRoleAssignmentException(AuthorizationException):
    """
    Raised when a role assignment violates authorization rules.
    """

    default_message = "The requested role assignment is invalid."

    default_status_code = 400

    default_error_code = ErrorCode.INVALID_ROLE_ASSIGNMENT


class InvalidPermissionAssignmentException(AuthorizationException):
    """
    Raised when a permission assignment violates authorization rules.
    """

    default_message = "The requested permission assignment is invalid."

    default_status_code = 400

    default_error_code = ErrorCode.INVALID_PERMISSION_ASSIGNMENT


# ================================================================
# RESOURCE-LEVEL AUTHORIZATION
# ================================================================

class ResourceAccessDeniedException(AuthorizationException):
    """
    Raised when an authenticated users cannot access a requested resource.

    The default response message does not disclose resource ownership
    or the existence of a protected resource.
    """

    default_message = DEFAULT_RESOURCE_ACCESS_DENIED_MESSAGE

    default_error_code = ErrorCode.RESOURCE_ACCESS_DENIED


class ResourceOwnershipViolationException(AuthorizationException):
    """
    Raised when a resource ownership check fails.

    This exception defaults to HTTP 404 to avoid confirming that a
    resource belonging to another users exists.

    Services should use this behavior consistently for private resources
    where resource enumeration is a security concern.
    """

    default_message = DEFAULT_RESOURCE_ACCESS_DENIED_MESSAGE

    default_status_code = 404

    default_error_code = ErrorCode.RESOURCE_OWNERSHIP_VIOLATION


class ResourceNotOwnedException(ResourceOwnershipViolationException):
    """
    Raised when a users attempts to access or modify a resource that
    they do not own.
    """

    default_error_code = ErrorCode.RESOURCE_NOT_OWNED


class CrossUserAccessDeniedException(ResourceOwnershipViolationException):
    """
    Raised when an operation attempts to cross users ownership boundaries.

    This is particularly important for preventing BOLA and IDOR
    vulnerabilities.
    """

    default_error_code = ErrorCode.CROSS_USER_ACCESS_DENIED


class BulkResourceAccessDeniedException(AuthorizationException):
    """
    Raised when a bulk operation contains one or more resources that
    the authenticated users is not authorized to access.

    The operation should fail safely rather than partially processing
    unauthorized resources unless partial processing is explicitly
    part of the business contract.
    """

    default_message = (
        "Access to one or more requested resources is denied."
    )

    default_error_code = ErrorCode.BULK_RESOURCE_ACCESS_DENIED


class FieldAccessDeniedException(AuthorizationException):
    """
    Raised when the authenticated users attempts to read or modify
    a field that is not available within their authorization scope.
    """

    default_message = (
        "You do not have permission to access this field."
    )

    default_error_code = ErrorCode.FIELD_ACCESS_DENIED


class ActionNotPermittedException(AuthorizationException):
    """
    Raised when a users is not permitted to perform a particular action
    on a resource.
    """

    default_message = (
        "You are not permitted to perform this action."
    )

    default_error_code = ErrorCode.ACTION_NOT_PERMITTED


# ================================================================
# CONVERSATION AUTHORIZATION
# ================================================================

class ConversationAccessDeniedException(ResourceOwnershipViolationException):
    """
    Raised when a users attempts to access a conversation they do not own.

    The default HTTP 404 response helps prevent conversation ID
    enumeration and cross-users information disclosure.
    """

    default_message = "The requested conversation is not available."

    default_error_code = ErrorCode.CONVERSATION_ACCESS_DENIED


class ConversationOwnershipViolationException(
    ConversationAccessDeniedException
):
    """
    Raised when a conversation ownership validation fails.
    """

    default_error_code = ErrorCode.CONVERSATION_OWNERSHIP_VIOLATION


class ConversationOperationDeniedException(AuthorizationException):
    """
    Raised when a users owns or can access a conversation but is not
    permitted to perform the requested operation on it.

    Use this for operation-specific restrictions that do not require
    concealing the existence of the conversation.
    """

    default_message = (
        "You are not permitted to perform this operation on the conversation."
    )

    default_error_code = ErrorCode.CONVERSATION_OPERATION_DENIED


# ================================================================
# MESSAGE AUTHORIZATION
# ================================================================

class MessageAccessDeniedException(ResourceOwnershipViolationException):
    """
    Raised when a users attempts to access a message outside their
    authorized conversation scope.
    """

    default_message = "The requested message is not available."

    default_error_code = ErrorCode.MESSAGE_ACCESS_DENIED


class MessageOwnershipViolationException(MessageAccessDeniedException):
    """
    Raised when message ownership or its parent conversation ownership
    validation fails.
    """

    default_error_code = ErrorCode.MESSAGE_OWNERSHIP_VIOLATION


class MessageOperationDeniedException(AuthorizationException):
    """
    Raised when a message operation is not permitted for the users.
    """

    default_message = (
        "You are not permitted to perform this operation on the message."
    )

    default_error_code = ErrorCode.MESSAGE_OPERATION_DENIED


class ChatHistoryAccessDeniedException(ResourceOwnershipViolationException):
    """
    Raised when a users attempts to access chat history outside their
    own authorized conversation scope.
    """

    default_message = "The requested chat history is not available."

    default_error_code = ErrorCode.CHAT_HISTORY_ACCESS_DENIED


class ConversationContextAccessDeniedException(
    ResourceOwnershipViolationException
):
    """
    Raised when conversation context cannot be accessed within the
    authenticated users's ownership boundary.

    This is particularly important when constructing AI provider
    requests to prevent one users's conversation context from entering
    another users's generation request.
    """

    default_message = (
        "The requested conversation context is not available."
    )

    default_error_code = ErrorCode.CONVERSATION_CONTEXT_ACCESS_DENIED


# ================================================================
# STREAMING AUTHORIZATION
# ================================================================

class StreamAccessDeniedException(ResourceOwnershipViolationException):
    """
    Raised when a users attempts to access a generation stream that
    is outside their authorization scope.
    """

    default_message = "The requested stream is not available."

    default_error_code = ErrorCode.STREAM_ACCESS_DENIED


class StreamOwnershipViolationException(StreamAccessDeniedException):
    """
    Raised when an active stream does not belong to the authenticated
    users who is attempting to access it.
    """

    default_error_code = ErrorCode.STREAM_OWNERSHIP_VIOLATION


class StreamInitializationDeniedException(AuthorizationException):
    """
    Raised when a users is not permitted to initialize a generation stream.

    Examples include restricted features, missing permissions or
    policy-based restrictions.
    """

    default_message = (
        "You are not permitted to initialize this generation stream."
    )

    default_error_code = ErrorCode.STREAM_INITIALIZATION_DENIED


class StreamCancellationDeniedException(StreamAccessDeniedException):
    """
    Raised when a users attempts to cancel a generation stream initiated
    by another users.

    The default HTTP 404 response avoids disclosing the existence of
    another users's active stream.
    """

    default_message = "The requested stream is not available."

    default_error_code = ErrorCode.STREAM_CANCELLATION_DENIED


class StreamStatusAccessDeniedException(StreamAccessDeniedException):
    """
    Raised when a users attempts to inspect the status of a stream
    outside their ownership boundary.
    """

    default_error_code = ErrorCode.STREAM_STATUS_ACCESS_DENIED


class StreamConnectionDeniedException(AuthorizationException):
    """
    Raised when a streaming connection cannot be established because
    the users is not authorized to connect.
    """

    default_message = (
        "You are not permitted to establish this streaming connection."
    )

    default_error_code = ErrorCode.STREAM_CONNECTION_DENIED


class StreamLimitExceededException(AuthorizationException):
    """
    Raised when the configured number of concurrent streams per users
    has been exceeded.
    """

    default_message = (
        "The maximum number of concurrent streams has been reached."
    )

    default_status_code = 429

    default_error_code = ErrorCode.STREAM_LIMIT_EXCEEDED


class StreamTimeoutException(AuthorizationException):
    """
    Raised when a stream exceeds the permitted execution duration.

    This exception represents an application-level stream timeout.
    It does not itself cancel or clean up the underlying task.
    """

    default_message = "The generation stream exceeded its time limit."

    default_status_code = 408

    default_error_code = ErrorCode.STREAM_TIMEOUT


# ================================================================
# AI APPLICATION AUTHORIZATION AND USAGE LIMITS
# ================================================================

class AIModelAccessDeniedException(AuthorizationException):
    """
    Raised when a users is not permitted to use a requested AI model.
    """

    default_message = (
        "You are not permitted to use the requested AI model."
    )

    default_error_code = ErrorCode.AI_MODEL_ACCESS_DENIED


class AIFeatureAccessDeniedException(AuthorizationException):
    """
    Raised when a users attempts to access an AI feature outside
    their authorization scope.
    """

    default_message = (
        "You are not permitted to use this AI feature."
    )

    default_error_code = ErrorCode.AI_FEATURE_ACCESS_DENIED


class AIUsageQuotaExceededException(AuthorizationException):
    """
    Raised when a users exceeds their configured AI usage quota.

    retry_after_seconds may be supplied when the quota has a known
    reset or retry time.
    """

    default_message = (
        "Your AI usage quota has been exceeded."
    )

    default_status_code = 429

    default_error_code = ErrorCode.AI_USAGE_QUOTA_EXCEEDED

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


class AITokenLimitExceededException(AuthorizationException):
    """
    Raised when an AI request exceeds the token allowance assigned
    to the users or operation.
    """

    default_message = (
        "The AI request exceeds the permitted token limit."
    )

    default_status_code = 429

    default_error_code = ErrorCode.AI_TOKEN_LIMIT_EXCEEDED


class AIRequestLimitExceededException(AuthorizationException):
    """
    Raised when a users exceeds the permitted AI request quota.
    """

    default_message = (
        "The maximum number of AI requests has been reached."
    )

    default_status_code = 429

    default_error_code = ErrorCode.AI_REQUEST_LIMIT_EXCEEDED


class AIConcurrentGenerationLimitExceededException(
    AuthorizationException
):
    """
    Raised when a users exceeds the permitted number of concurrent
    AI generation operations.
    """

    default_message = (
        "The maximum number of concurrent AI generations has been reached."
    )

    default_status_code = 429

    default_error_code = (
        ErrorCode.AI_CONCURRENT_GENERATION_LIMIT_EXCEEDED
    )


class AICostLimitExceededException(AuthorizationException):
    """
    Raised when a users exceeds the configured AI usage cost allowance.
    """

    default_message = (
        "The permitted AI usage cost limit has been exceeded."
    )

    default_status_code = 429

    default_error_code = ErrorCode.AI_COST_LIMIT_EXCEEDED


__all__ = [
    "AuthorizationException",
    "AccessDeniedException",
    "PermissionDeniedException",
    "InsufficientPermissionsException",
    "RequiredRoleMissingException",
    "PrivilegeEscalationDeniedException",
    "InvalidRoleAssignmentException",
    "InvalidPermissionAssignmentException",
    "ResourceAccessDeniedException",
    "ResourceOwnershipViolationException",
    "ResourceNotOwnedException",
    "CrossUserAccessDeniedException",
    "BulkResourceAccessDeniedException",
    "FieldAccessDeniedException",
    "ActionNotPermittedException",
    "ConversationAccessDeniedException",
    "ConversationOwnershipViolationException",
    "ConversationOperationDeniedException",
    "MessageAccessDeniedException",
    "MessageOwnershipViolationException",
    "MessageOperationDeniedException",
    "ChatHistoryAccessDeniedException",
    "ConversationContextAccessDeniedException",
    "StreamAccessDeniedException",
    "StreamOwnershipViolationException",
    "StreamInitializationDeniedException",
    "StreamCancellationDeniedException",
    "StreamStatusAccessDeniedException",
    "StreamConnectionDeniedException",
    "StreamLimitExceededException",
    "StreamTimeoutException",
    "AIModelAccessDeniedException",
    "AIFeatureAccessDeniedException",
    "AIUsageQuotaExceededException",
    "AITokenLimitExceededException",
    "AIRequestLimitExceededException",
    "AIConcurrentGenerationLimitExceededException",
    "AICostLimitExceededException",
]
