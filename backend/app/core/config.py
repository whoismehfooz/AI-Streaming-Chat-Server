from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR=Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    APP_NAME: str="AI Streaming Chat Server"
    APP_ENV: str="developement"
    DEBUG: bool=False

    CHAT_MAX_OUTPUT_TOKENS: int=512
    CHAT_CONTEXT_TOKEN_BUDGET: int=4000
    CHAT_SYSTEM_PROMPT: str="You are a helpful assistant."
    STREAM_CHUNK_DELAY: float=0.0

    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str="openai/gpt-oss-20b"

    DATABASE_URL: str
    TEST_DATABASE_URL: str

    model_config=SettingsConfigDict(
        env_file=BACKEND_DIR/".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

settings = Settings()
