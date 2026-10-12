"""LLM provider abstraction for answer generation.

Supports OpenAI with extensibility for other providers.
"""

import json
import urllib.error
import urllib.request
from typing import Optional
from config import Config


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMConfigurationError(LLMProviderError):
    """Raised when LLM configuration is missing or invalid."""
    pass


class LLMProvider:
    """Abstract interface for LLM providers."""

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt to send to the LLM.
            system_prompt: Optional system prompt to guide behavior.

        Returns:
            The generated text response.

        Raises:
            LLMProviderError: If generation fails.
        """
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model name. If None, uses LLM_MODEL env var or default.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.LLM_MODEL
        self.timeout = timeout

        if not self.api_key:
            raise LLMConfigurationError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
            )

        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise LLMConfigurationError(
                "OpenAI package not installed. Install with: pip install openai"
            )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using OpenAI API.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            Generated text response.

        Raises:
            LLMProviderError: If API call fails.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=self.timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMProviderError(f"OpenAI API call failed: {str(e)}")


class OllamaProvider(LLMProvider):
    """Generate responses through a local Ollama chat endpoint."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.model = model or Config.OLLAMA_MODEL
        self.base_url = (base_url or Config.OLLAMA_BASE_URL).rstrip('/')
        self.timeout = timeout if timeout is not None else Config.OLLAMA_TIMEOUT

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using Ollama's non-streaming chat API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMProviderError(
                f"Ollama returned HTTP {e.code} while generating a response."
            ) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMProviderError(f"Could not reach the local Ollama service: {e}") from e
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise LLMProviderError("Ollama returned an invalid response.") from e

        if not isinstance(result, dict):
            raise LLMProviderError("Ollama returned an invalid response.")
        if result.get("error"):
            raise LLMProviderError(f"Ollama generation failed: {result['error']}")

        try:
            answer = result["message"]["content"]
        except (KeyError, TypeError) as e:
            raise LLMProviderError("Ollama response did not contain generated text.") from e
        if not isinstance(answer, str):
            raise LLMProviderError("Ollama response did not contain generated text.")
        return answer


def get_llm_provider() -> LLMProvider:
    """Factory function to get configured LLM provider.

    Returns:
        Configured LLM provider instance.

    Raises:
        LLMConfigurationError: If provider is not configured.
    """
    provider = Config.LLM_PROVIDER.lower()

    if provider == "ollama":
        return OllamaProvider()
    if provider == "openai":
        return OpenAIProvider(
            timeout=Config.LLM_TIMEOUT
        )
    else:
        raise LLMConfigurationError(
            f"Unsupported LLM provider: {provider}. Supported: ollama, openai"
        )
