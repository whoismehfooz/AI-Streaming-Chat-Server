import pytest
from pydantic import ValidationError

from app.schemas.message import MessageCreate


@pytest.mark.parametrize(
    "role",
    [
        "user",
        "assistant",
        "system"
    ],
)
def test_message_create_accepts_valid_roles(role):
    message = MessageCreate(
        role=role,
        content="Hello"
    )

    assert message.role == role
    assert message.content == "Hello"


@pytest.mark.parametrize(
    "role",
    [
        "banana",
        "admin",
        "USER",
        "",
        "tool",
        "developer"
    ]
)
def test_message_create_rejects_invalid_roles(role):
    with pytest.raises(ValidationError):
        MessageCreate(
            role=role,
            content="Hello"
        )


def test_message_create_rejects_empyt_content():
    with pytest.raises(ValidationError):
        MessageCreate(
            role="user",
            content=""
        )
