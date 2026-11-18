"""
Resource not-found exceptions.

This module defines application-level exceptions for resources that are
missing or intentionally treated as unavailable. It includes the resource
domains shown in the backend architecture and the not-found exceptions
currently defined in base.py, authentication.py, and authorization.py
that belong to resource lookup.

Resource lookup failures are distinct from authorization failures:
    - Use a not-found exception when the resource does not exist.
    - Use an authorization exception when access is denied.
    - For private resources, services may deliberately raise a not-found
      exception for both missing and unauthorized resources to avoid
      disclosing whether the resource exists.

Resource identifiers and internal lookup details should not be included
in public exception messages.
"""

from typing import ClassVar

from app.exceptions.base import ApplicationException
from app.exceptions.error_codes import ErrorCode


# ================================================================
# RESOURCE EXCEPTION CONSTANTS
# ================================================================

DEFAULT_RESOURCE_NOT_FOUND_MESSAGE = "The requested resource was not found."

DEFAULT_USER_NOT_FOUND_MESSAGE = "User not found."

DEFAULT_WORKSPACE_NOT_FOUND_MESSAGE = "The requested workspace was not found."

DEFAULT_KNOWLEDGE_BASE_NOT_FOUND_MESSAGE = (
    "The requested knowledge base was not found."
)

DEFAULT_DOCUMENT_NOT_FOUND_MESSAGE = "The requested document was not found."

DEFAULT_CONVERSATION_NOT_FOUND_MESSAGE = "Conversation not found."

DEFAULT_MESSAGE_NOT_FOUND_MESSAGE = "Message not found."

DEFAULT_INGESTION_JOB_NOT_FOUND_MESSAGE = (
    "The requested ingestion job was not found."
)

DEFAULT_API_KEY_NOT_FOUND_MESSAGE = "The requested API key was not found."


# ================================================================
# BASE RESOURCE NOT-FOUND EXCEPTION
# ================================================================

class ResourceNotFoundException(ApplicationException):
    """
    Base exception for missing or unavailable application resources.

    Specialized resource exceptions inherit this class. The default
    status code is 404 and the default error code is the shared
    RESOURCE_NOT_FOUND code.

    Avoid putting resource identifiers, tenant details, or internal
    lookup information in the public message.
    """

    default_message: ClassVar[str] = DEFAULT_RESOURCE_NOT_FOUND_MESSAGE

    default_status_code: ClassVar[int] = 404

    default_error_code: ClassVar[ErrorCode] = ErrorCode.RESOURCE_NOT_FOUND

    def __init__(
        self,
        message: str | None = None,
        status_code: int | None = None,
        *,
        error_code: ErrorCode | None = None,
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
        )


# ================================================================
# USER AND ACCOUNT RESOURCES
# ================================================================

class UserNotFoundException(ResourceNotFoundException):
    """Raised when the requested users does not exist or is unavailable."""

    default_message = DEFAULT_USER_NOT_FOUND_MESSAGE

    default_error_code = ErrorCode.USER_NOT_FOUND


# ================================================================
# WORKSPACE RESOURCES
# ================================================================

class WorkspaceNotFoundException(ResourceNotFoundException):
    """Raised when a workspace does not exist or is unavailable."""

    default_message = DEFAULT_WORKSPACE_NOT_FOUND_MESSAGE


# ================================================================
# KNOWLEDGE-BASE RESOURCES
# ================================================================

class KnowledgeBaseNotFoundException(ResourceNotFoundException):
    """Raised when a knowledge base does not exist or is unavailable."""

    default_message = DEFAULT_KNOWLEDGE_BASE_NOT_FOUND_MESSAGE


# ================================================================
# DOCUMENT RESOURCES
# ================================================================

class DocumentNotFoundException(ResourceNotFoundException):
    """Raised when a document does not exist or is unavailable."""

    default_message = DEFAULT_DOCUMENT_NOT_FOUND_MESSAGE


class DocumentChunkNotFoundException(ResourceNotFoundException):
    """Raised when a document chunk does not exist or is unavailable."""

    default_message = "The requested document chunk was not found."


# ================================================================
# CONVERSATION AND MESSAGE RESOURCES
# ================================================================

class ConversationNotFoundException(ResourceNotFoundException):
    """Raised when the requested conversation does not exist or is unavailable."""

    default_message = DEFAULT_CONVERSATION_NOT_FOUND_MESSAGE

    default_error_code = ErrorCode.CONVERSATION_NOT_FOUND


class MessageNotFoundException(ResourceNotFoundException):
    """Raised when the requested message does not exist or is unavailable."""

    default_message = DEFAULT_MESSAGE_NOT_FOUND_MESSAGE

    default_error_code = ErrorCode.MESSAGE_NOT_FOUND


class ChatSessionNotFoundException(ResourceNotFoundException):
    """
    Raised when a chat session resource does not exist or is unavailable.

    Authentication sessions use SessionNotFoundException.
    """

    default_message = "The requested chat session was not found."


class SessionNotFoundException(ResourceNotFoundException):
    """Raised when the requested authentication session is unavailable."""

    default_message = "The authentication session was not found."

    default_error_code = ErrorCode.SESSION_NOT_FOUND


# ================================================================
# INGESTION RESOURCES
# ================================================================

class IngestionJobNotFoundException(ResourceNotFoundException):
    """Raised when an ingestion job does not exist or is unavailable."""

    default_message = DEFAULT_INGESTION_JOB_NOT_FOUND_MESSAGE


class IngestionRunNotFoundException(ResourceNotFoundException):
    """Raised when an ingestion run does not exist or is unavailable."""

    default_message = "The requested ingestion run was not found."


# ================================================================
# RETRIEVAL RESOURCES
# ================================================================

class RetrievalResultNotFoundException(ResourceNotFoundException):
    """Raised when a requested retrieval result is unavailable."""

    default_message = "The requested retrieval result was not found."


# ================================================================
# AUTHORIZATION DEFINITIONS AS RESOURCES
# ================================================================

class RoleNotFoundException(ResourceNotFoundException):
    """Raised when a requested role definition does not exist."""

    default_message = "The requested role was not found."

    default_error_code = ErrorCode.ROLE_NOT_FOUND


class PermissionNotFoundException(ResourceNotFoundException):
    """Raised when a requested permission definition does not exist."""

    default_message = "The requested permission was not found."

    default_error_code = ErrorCode.PERMISSION_NOT_FOUND


# ================================================================
# API KEY AND EVALUATION RESOURCES
# ================================================================

class APIKeyNotFoundException(ResourceNotFoundException):
    """Raised when an API key resource does not exist or is unavailable."""

    default_message = DEFAULT_API_KEY_NOT_FOUND_MESSAGE


class EvaluationNotFoundException(ResourceNotFoundException):
    """Raised when an evaluation does not exist or is unavailable."""

    default_message = "The requested evaluation was not found."


__all__ = [
    "ResourceNotFoundException",
    "UserNotFoundException",
    "WorkspaceNotFoundException",
    "KnowledgeBaseNotFoundException",
    "DocumentNotFoundException",
    "DocumentChunkNotFoundException",
    "ConversationNotFoundException",
    "MessageNotFoundException",
    "ChatSessionNotFoundException",
    "SessionNotFoundException",
    "IngestionJobNotFoundException",
    "IngestionRunNotFoundException",
    "RetrievalResultNotFoundException",
    "RoleNotFoundException",
    "PermissionNotFoundException",
    "APIKeyNotFoundException",
    "EvaluationNotFoundException",
]