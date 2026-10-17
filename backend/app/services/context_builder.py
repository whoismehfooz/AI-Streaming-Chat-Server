from app.exceptions.custom_exceptions import ContextBudgetExceededException
from app.models import Message
from app.services.token_counter import TokenCounter


class ContextBuilder:
    def __init__(
            self,
            token_counter: TokenCounter,
            context_token_budget: int,
            system_prompt: str | None = None
    ):
        if context_token_budget <= 0:
            raise ValueError("Context token budget must be greater than zero.")

        self.token_counter = token_counter
        self.context_token_budget = context_token_budget
        self.system_prompt = system_prompt

    def build(
            self,
            messages: list[Message],
    ) -> list[dict[str, str]]:
        system_message : dict[str, str] | None = None
        system_tokens = 0

        if self.system_prompt:
            system_message = {
                "role":"system",
                "content":self.system_prompt
            }

            system_tokens = self.token_counter.estimate(
                self.system_prompt
            )

        if system_tokens > self.context_token_budget:
            raise ContextBudgetExceededException(
                "System prompt exceeds the configured context token budget."
            )

        remaining_budget = (
            self.context_token_budget -  system_tokens
        )

        selected_messages: list[Message] = []
        consumed_tokens = 0

        for message in reversed(messages):
            message_tokens = self.token_counter.estimate(
                message.content
            )

            if message_tokens > remaining_budget - consumed_tokens:
                if not selected_messages:
                    raise ContextBudgetExceededException(
                        "Latest message exceeds the configured context token budget."
                    )
                break


            selected_messages.append(message)
            consumed_tokens += message_tokens

        selected_messages.reverse()

        context: list[dict[str, str]] = []

        if system_message:
            context.append(system_message)

        context.extend(
            {
                "role":message.role,
                "content": message.content
            }
            for message in selected_messages
        )

        return context
