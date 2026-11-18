"""
Factory for constructing the configured LLM provider adapter.

The factory is the composition point between application configuration
and the provider-neutral LLM interface. It creates a fresh provider
instance; it does not cache clients globally.

The current application configuration defines Groq credentials/model
settings. Groq exposes an OpenAI-compatible API, so it is accessed through
the OpenAI asynchronous SDK and the shared OpenAI-compatible adapter.

Direct OpenAI configuration is also supported when the supplied settings
object exposes OPENAI_API_KEY and OPENAI_MODEL. The existing Settings class
must declare those fields before values in its .env file can be loaded by
Pydantic Settings.

Lifecycle:
    - Call ``create_llm_provider`` once during application startup.
    - Store the returned provider in application lifespan/state.
    - Inject that instance into services that need it.
    - Call ``await provider.close()`` during application shutdown.

Do not call this factory for every incoming request. Each created provider
owns a network client and therefore consumes connection-pool resources.
"""

from __future__ import annotations

import re
from typing import Any

from app.integrations.llm.base import BaseLLMProvider
from app.integrations.llm.exceptions import LLMProviderConfigurationException
from app.integrations.llm.openai_client import OpenAICompatibleLLMProvider
from app.integrations.llm.schemas import LLMProviderCapabilities


GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _read_required_setting(
    config: object,
    setting_name: str,
    *,
    provider_name: str,
) -> str:
    """Read and validate a required string setting without exposing its value."""
    value = getattr(config, setting_name, None)

    if not isinstance(value, str) or not value.strip():
        raise LLMProviderConfigurationException(
            message=(
                f"The {provider_name} LLM provider is missing the "
                f"required {setting_name} configuration."
            ),
            provider_name=provider_name,
        )

    return value.strip()


def create_llm_provider(
    config: object | None = None,
    *,
    provider: str | None = None,
) -> BaseLLMProvider:
    """
    Create the configured asynchronous LLM provider.

    Args:
        config:
            Settings-like object. When omitted, the application's
            ``app.core.config.settings`` instance is loaded lazily.
            Lazy loading keeps importing this module side-effect-light and
            allows tests to pass a small settings stub.
        provider:
            Optional explicit provider override. Supported values are
            ``"groq"`` and ``"openai"``. When omitted, ``LLM_PROVIDER`` is
            read from the settings object; the current application defaults
            to Groq because that is the provider configured by the existing
            AI service.

    Returns:
        A newly constructed ``BaseLLMProvider`` implementation.

    Raises:
        LLMProviderConfigurationException:
            If the provider is unsupported or required configuration is
            missing/invalid.

    The function intentionally does not catch SDK/client-construction
    failures indiscriminately. Invalid application configuration is
    normalized here; unexpected programming/runtime errors should retain
    their traceback and be handled by application startup error handling.
    """
    if config is None:
        from app.core.config import settings

        config = settings

    configured_provider: Any = (
        provider
        if provider is not None
        else getattr(config, "LLM_PROVIDER", "groq")
    )

    if not isinstance(configured_provider, str) or not configured_provider.strip():
        raise LLMProviderConfigurationException(
            message="The configured LLM provider name is missing or invalid."
        )

    provider_name = configured_provider.strip().lower()

    if (
        len(provider_name) > 64
        or re.fullmatch(r"[a-z0-9_-]+", provider_name) is None
    ):
        raise LLMProviderConfigurationException(
            message="The configured LLM provider name is invalid."
        )

    if provider_name == "groq":
        api_key = _read_required_setting(
            config,
            "GROQ_API_KEY",
            provider_name=provider_name,
        )
        model = _read_required_setting(
            config,
            "GROQ_MODEL",
            provider_name=provider_name,
        )
        base_url = GROQ_BASE_URL

    elif provider_name == "openai":
        api_key = _read_required_setting(
            config,
            "OPENAI_API_KEY",
            provider_name=provider_name,
        )
        model = _read_required_setting(
            config,
            "OPENAI_MODEL",
            provider_name=provider_name,
        )
        configured_base_url = getattr(config, "OPENAI_BASE_URL", None)

        if configured_base_url is not None and (
            not isinstance(configured_base_url, str)
            or not configured_base_url.strip()
        ):
            raise LLMProviderConfigurationException(
                message="The configured OpenAI base URL is invalid.",
                provider_name=provider_name,
            )

        base_url = (
            configured_base_url.strip()
            if isinstance(configured_base_url, str)
            else None
        )

    else:
        raise LLMProviderConfigurationException(
            message=(
                f"Unsupported LLM provider '{provider_name}'. "
                "Supported providers are: groq, openai."
            ),
            provider_name=provider_name,
        )

    capabilities = LLMProviderCapabilities(
        provider_name=provider_name,
        # Do not invent a model catalogue or context limits. The provider
        # adapter validates the configured model against upstream responses;
        # model-specific capability metadata can be added when verified.
        models=(),
        supports_streaming=True,
        supports_streaming_usage=(provider_name == "openai"),
        supports_idempotency_keys=False,
        reports_usage=True,
    )

    return OpenAICompatibleLLMProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider_name=provider_name,
        capabilities=capabilities,
    )


__all__ = [
    "GROQ_BASE_URL",
    "create_llm_provider",
]
