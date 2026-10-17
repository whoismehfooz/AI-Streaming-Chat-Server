class ConversationNotFoundException(Exception):
    def __init__(self, message: str, status_code: int=404):
        self.status_code=status_code
        super().__init__(message)


class AIProviderException(Exception):
    def __init__(self,message: str, status_code: int = 502):
        self.status_code=status_code
        super().__init__(message)


class ContextBudgetExceededException(Exception):
    def __init__(self,message: str, status_code=400):
        self.status_code=status_code
        super().__init__(message)
