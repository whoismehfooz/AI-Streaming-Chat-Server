from pathlib import Path
from typing import Literal

from pydantic import (
    AnyHttpUrl,
    EmailStr,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):

    APP_NAME: str = Field(
        default="AI Streaming Chat Server",
        min_length=1,
        max_length=120,
    )
    APP_ENV: Literal[
        "development",
        "testing",
        "staging",
        "production",
    ] = "development"
    DEBUG: bool = False

    BACKEND_URL: AnyHttpUrl = AnyHttpUrl("http://localhost:8000")

    CHAT_MAX_OUTPUT_TOKENS: int = Field(default=512, gt=0)
    CHAT_CONTEXT_TOKEN_BUDGET: int = Field(default=4000, gt=0)
    CHAT_SYSTEM_PROMPT: str = Field(
        default="You are a helpful assistant.",
        min_length=1,
    )
    STREAM_CHUNK_DELAY: float = Field(default=0.0, ge=0.0)

    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = Field(default="openai/gpt-oss-20b", min_length=1)

    DATABASE_URL: str
    TEST_DATABASE_URL: str | None = None

    JWT_SECRET_KEY: SecretStr = Field(min_length=32)
    JWT_ALGORITHM: Literal["HS256"] = "HS256"
    JWT_ISSUER: str = Field(
        default="ai-streaming-chat-server",
        min_length=1,
    )
    JWT_AUDIENCE: str = Field(
        default="ai-streaming-chat-server-api",
        min_length=1,
    )

    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15, gt=0)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, gt=0)

    EMAIL_VERIFICATION_TOKEN_EXPIRE_MINUTES: int = Field(
        default=1440,
        gt=0,
    )
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        gt=0,
    )

    RESEND_API_KEY: SecretStr | None = None
    RESEND_FROM_EMAIL: EmailStr = "onboarding@resend.dev"
    RESEND_FROM_NAME: str = Field(
        default="AI Streaming Chat",
        min_length=1,
        max_length=120,
    )
    RESEND_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        gt=0.0,
        le=120.0,
    )

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
        validate_default=True,
        hide_input_in_errors=True,
        frozen=True,
    )

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        """Require the async PostgreSQL driver used by this application."""

        normalized = value.strip()
        if not normalized.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must use the postgresql+asyncpg driver."
            )
        return normalized

    @field_validator("TEST_DATABASE_URL")
    @classmethod
    def validate_test_database_url(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        if not normalized.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "TEST_DATABASE_URL must use the postgresql+asyncpg driver."
            )
        return normalized

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret_key(cls, value: SecretStr) -> SecretStr:
        secret_bytes = value.get_secret_value().encode("utf-8")
        if len(secret_bytes) < 32:
            raise ValueError(
                "JWT_SECRET_KEY must contain at least 32 UTF-8 bytes."
            )
        return value

    @field_validator("JWT_ISSUER", "JWT_AUDIENCE")
    @classmethod
    def normalize_jwt_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("JWT issuer and audience cannot be empty.")
        return normalized

    @field_validator("RESEND_FROM_NAME")
    @classmethod
    def normalize_resend_from_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("RESEND_FROM_NAME cannot be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_environment_configuration(self) -> "Settings":

        if (
            self.TEST_DATABASE_URL is not None
            and self.TEST_DATABASE_URL == self.DATABASE_URL
        ):
            raise ValueError(
                "TEST_DATABASE_URL must not equal DATABASE_URL."
            )

        if self.APP_ENV == "production":
            if self.DEBUG:
                raise ValueError("DEBUG must be false in production.")

            if self.BACKEND_URL.scheme != "https":
                raise ValueError(
                    "BACKEND_URL must use HTTPS in production."
                )

            if self.RESEND_API_KEY is None:
                raise ValueError(
                    "RESEND_API_KEY is required in production."
                )

        return self

# noinspection PyArgumentList
settings = Settings()
