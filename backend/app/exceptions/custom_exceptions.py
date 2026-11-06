
class ApplicationException(Exception):

    def __init__(self, message: str, status_code: int) -> None:
        if not 400 <= status_code <= 599:
            raise ValueError(
                "Application exception status_code must be between 400 and 599."
            )

        self.status_code = status_code
        super().__init__(message)


class ConversationNotFoundException(ApplicationException):
    def __init__(
        self,
        message: str = "Conversation not found.",
        status_code: int = 404,
    ) -> None:
        super().__init__(message, status_code)


class MessageNotFoundException(ApplicationException):
    def __init__(
        self,
        message: str = "Message not found.",
        status_code: int = 404,
    ) -> None:
        super().__init__(message, status_code)


class AIProviderException(ApplicationException):
    def __init__(
        self,
        message: str = "The AI provider could not process the request.",
        status_code: int = 502,
    ) -> None:
        super().__init__(message, status_code)


class ContextBudgetExceededException(ApplicationException):
    def __init__(
        self,
        message: str = "The conversation context exceeds the allowed token budget.",
        status_code: int = 400,
    ) -> None:
        super().__init__(message, status_code)


class UserNotFoundException(ApplicationException):
    def __init__(
        self,
        message: str = "User not found.",
        status_code: int = 404,
    ) -> None:
        super().__init__(message, status_code)


class EmailAlreadyRegisteredException(ApplicationException):
    def __init__(
        self,
        message: str = "An account with this email already exists.",
        status_code: int = 409,
    ) -> None:
        super().__init__(message, status_code)
