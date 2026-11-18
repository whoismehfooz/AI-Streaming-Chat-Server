"""
Public request contracts for the Chat domain.

These models are API boundary DTOs. They deliberately contain no provider
selection, model identifier, provider credentials, internal retry controls,
or transport-specific fields. The application resolves trusted defaults
and model capabilities before constructing an LLMRequest.

Conversation ownership and authenticated identity are supplied by the API
and application-service boundary, not trusted from this request body.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


# This is a request-size guardrail, not a substitute for token budgeting.
# ContextBuilder and the selected provider/model limits remain authoritative.
MAX_CHAT_MESSAGE_CHARACTERS = 32_000
MAX_STOP_SEQUENCES = 16
MAX_STOP_SEQUENCE_CHARACTERS = 256


class ChatGenerationOptions(BaseModel):
    """
    Provider-neutral generation overrides supplied by the client.

    Every field is optional so application configuration remains the source
    of defaults. The application must resolve these values against the
    selected model's capabilities and configured limits before dispatch.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    max_output_tokens: Annotated[
        int | None,
        Field(default=None, strict=True, gt=0),
    ]
    temperature: Annotated[
        float | None,
        Field(default=None, ge=0.0, le=2.0),
    ]
    top_p: Annotated[
        float | None,
        Field(default=None, gt=0.0, le=1.0),
    ]
    frequency_penalty: Annotated[
        float | None,
        Field(default=None, ge=-2.0, le=2.0),
    ]
    presence_penalty: Annotated[
        float | None,
        Field(default=None, ge=-2.0, le=2.0),
    ]
    stop_sequences: Annotated[
        tuple[
            Annotated[
                str,
                Field(
                    min_length=1,
                    max_length=MAX_STOP_SEQUENCE_CHARACTERS,
                    strict=True,
                ),
            ],
            ...
        ]
        | None,
        Field(default=None, max_length=MAX_STOP_SEQUENCES),
    ]

    @field_validator("stop_sequences")
    @classmethod
    def stop_sequences_must_not_be_blank(
        cls,
        value: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        if value is None:
            return None

        if any(not sequence.strip() for sequence in value):
            raise ValueError("Stop sequences cannot be empty or whitespace-only.")

        return value


class ChatRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    content: Annotated[
        str,
        Field(
            strict=True,
            min_length=1,
            max_length=MAX_CHAT_MESSAGE_CHARACTERS,
        ),
    ]
    generation: ChatGenerationOptions | None = None

    @field_validator("content")
    @classmethod
    def content_must_contain_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "Message content cannot be empty or whitespace-only."
            )
        return value


__all__ = [
    "MAX_CHAT_MESSAGE_CHARACTERS",
    "MAX_STOP_SEQUENCES",
    "MAX_STOP_SEQUENCE_CHARACTERS",
    "ChatGenerationOptions",
    "ChatRequest",
]
