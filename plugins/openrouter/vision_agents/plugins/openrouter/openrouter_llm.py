"""OpenRouter LLM implementation using OpenAI-compatible API."""
import os
from typing import Any

from vision_agents.plugins.openai import LLM as OpenAILLM


class OpenRouterLLM(OpenAILLM):
    """OpenRouter LLM that extends OpenAI LLM with OpenRouter-specific configuration.
    
    OpenRouter provides an OpenAI-compatible API, so we can reuse the OpenAI plugin
    implementation with OpenRouter-specific defaults.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openrouter/andromeda-alpha",
        **kwargs: Any,
    ) -> None:
        """Initialize OpenRouter LLM.
        
        Args:
            api_key: OpenRouter API key. If not provided, uses OPENROUTER_API_KEY env var.
            base_url: OpenRouter API base URL.
            model: Model to use. Defaults to openai/gpt-4o.
            **kwargs: Additional arguments passed to OpenAI LLM.
        """
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs,
        )

