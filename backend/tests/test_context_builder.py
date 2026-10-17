import pytest

from app.exceptions.custom_exceptions import ContextBudgetExceededException
from app.models import Message
from app.services import ContextBuilder


class FakeTokenCounter:
    def __init__(self, token_counts: dict[str, str]):
        self.token_counts = token_counts

    def estimate(self, text: str) -> int:
        return self.token_counts[text]


@pytest.mark.asyncio
async def test_context_builder_normal_context():
    token_counter = FakeTokenCounter(
        {
            "You are a helpful assistant.": 5,
            "Hello": 2,
            "Hi there!": 2,
            "How are you?": 3,
        }
    )

    builder = ContextBuilder(
        token_counter=token_counter,
        context_token_budget=20,
        system_prompt="You are a helpful assistant.",
    )

    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
        Message(role="user", content="How are you?")
    ]

    context = builder.build(messages)

    assert context == [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hi there!"
        },
        {
            "role": "user",
            "content": "How are you?"
        }
    ]


@pytest.mark.asyncio
async def test_context_builder_system_message_is_always_first():
    token_counter = FakeTokenCounter(
        {
            "System instructions": 5,
            "message 1": 3,
            "message 2": 3,
            "message 3": 3
        }
    )

    builder = ContextBuilder(
        token_counter=token_counter,
        context_token_budget=14,
        system_prompt="System instructions"
    )

    messages = [
        Message(role="user", content="message 1"),
        Message(role="assistant", content="message 2"),
        Message(role="user", content="message 3")
    ]

    context = builder.build(messages)

    assert context[0] == {
        "role": "system",
        "content": "System instructions"
    }

    assert context[1:] == [
        {
            "role": "assistant" if False else "user",
            "content": "message 1"
        },
        {
            "role": "assistant",
            "content": "message 2",
        },
        {
            "role": "user",
            "content": "message 3"
        }
    ]

    assert context[0]["role"] == "system"


@pytest.mark.asyncio
async def test_context_builder_truncates_old_messages():
    token_counter = FakeTokenCounter(
        {
            "message 1": 10,
            "message 2": 10,
            "message 3": 10,
            "message 4": 10,
        }
    )

    builder = ContextBuilder(
        token_counter=token_counter,
        context_token_budget=20
    )

    messages = [
        Message(role="user", content="message 1"),
        Message(role="assistant", content="message 2"),
        Message(role="user", content="message 3"),
        Message(role="assistant", content="message 4")
    ]

    context = builder.build(messages)

    assert context == [
        {
            "role": "user",
            "content": "message 3"
        },
        {
            "role": "assistant",
            "content": "message 4"
        }
    ]


@pytest.mark.asyncio
async def test_context_builder_preserves_choronological_order():
    token_counter = FakeTokenCounter(
        {
            "message 1": 10,
            "message 2": 10,
            "message 3": 10,
            "message 4": 10
        }
    )

    builder = ContextBuilder(
        token_counter=token_counter,
        context_token_budget=20,
    )

    messages = [
        Message(role="user", content="message 1"),
        Message(role="assistant", content="message 2"),
        Message(role="user", content="message 3"),
        Message(role="assistant", content="message 4")
    ]

    context = builder.build(messages)

    assert [message["content"] for message in context] == [
        "message 3",
        "message 4"
    ]

    assert [message["role"] for message in context ] == [
        "user",
        "assistant"
    ]


@pytest.mark.asyncio
async def test_context_builder_latest_message_too_large():
    token_counter = FakeTokenCounter(
        {
            "small message": 5,
            "huge message" : 25,
        }
    )

    builder = ContextBuilder(
        token_counter=token_counter,
        context_token_budget=20
    )

    messages = [
        Message(role="uesr", content="small message"),
        Message(role="assistant", content="huge message")
    ]

    with pytest.raises(ContextBudgetExceededException):
        builder.build(messages)


@pytest.mark.asyncio
async def test_context_builder_rejects_invalid_or_excessive_budget():
    token_counter = FakeTokenCounter(
        {
            "Very large system prompt": 20,
        }
    )

    with pytest.raises(
        ValueError,
        match="Context token budget must be greater than zero."
    ):
        ContextBuilder(
            token_counter=token_counter,
            context_token_budget=0
        )

    builder = ContextBuilder(
        token_counter=token_counter,
        context_token_budget=10,
        system_prompt="Very large system prompt"
    )

    with pytest.raises(
        ContextBudgetExceededException,
        match="System prompt exceeds the configured context token budget.",
    ):
        builder.build([])
