"""
Provider-agnostic model resolver for ADK agents.

Reads `settings.adk_model` and returns either:
  - A bare string for native Google Gemini models (ADK consumes it directly).
  - A `LiteLlm(...)` instance for any model whose name carries a provider
    prefix understood by LiteLLM (e.g. "anthropic/...", "openai/...").

Keeping this in one place means swapping providers is a single .env edit:
each agent calls `get_model()` instead of hard-coding a model string.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.settings import settings

# LiteLLM provider prefixes we route through google.adk.models.lite_llm.
_LITELLM_PREFIXES = ("anthropic/", "openai/", "azure/", "bedrock/", "ollama/", "groq/")


@lru_cache(maxsize=1)
def get_model() -> Any:
    name = settings.adk_model
    if any(name.startswith(p) for p in _LITELLM_PREFIXES):
        # Imported lazily so plain Gemini setups don't need litellm installed.
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(model=name)
    return name
