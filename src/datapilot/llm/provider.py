import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

import httpx
import openai

from datapilot.config import GROQ_BASE_URL, Settings, get_settings


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatModel(Protocol):
    @property
    def model(self) -> str: ...

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> LLMResponse: ...


class LLMError(RuntimeError):
    """The provider could not produce a reply, even after the SDK's own retries."""


@dataclass
class OpenAICompatibleModel:
    """Chat model for Groq or any server that speaks the OpenAI chat completions API."""

    client: openai.OpenAI
    model: str
    temperature: float = 0.0

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(**request)
        except openai.OpenAIError as error:
            raise LLMError(f"{self.model} request failed: {error}") from error
        latency_ms = (time.perf_counter() - started) * 1000

        usage = completion.usage
        return LLMResponse(
            text=completion.choices[0].message.content or "",
            model=completion.model or self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
        )


def build_chat_model(
    settings: Settings | None = None,
    http_client: httpx.Client | None = None,
) -> ChatModel:
    settings = settings or get_settings()
    if settings.llm_provider == "groq":
        api_key, base_url, key_name = settings.groq_api_key, GROQ_BASE_URL, "GROQ_API_KEY"
    else:
        api_key, base_url, key_name = (
            settings.openai_api_key,
            settings.openai_base_url,
            "OPENAI_API_KEY",
        )
    if api_key is None or not api_key.get_secret_value():
        raise LLMError(f"{key_name} is not set. Copy .env.example to .env and add your key.")

    # The SDK retries 429s and 5xx responses with exponential backoff, which matters
    # on Groq's free tier where rate limits are hit quickly during evaluation runs.
    client = openai.OpenAI(
        api_key=api_key.get_secret_value(),
        base_url=base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        http_client=http_client,
    )
    model: ChatModel = OpenAICompatibleModel(
        client=client, model=settings.llm_model, temperature=settings.llm_temperature
    )
    if settings.llm_cache_enabled:
        from datapilot.llm.cache import CachedChatModel

        model = CachedChatModel(inner=model, path=settings.llm_cache_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send one tiny prompt to check that the LLM settings in .env work."
    )
    parser.parse_args()

    settings = get_settings()
    chat_model = build_chat_model(settings)
    response = chat_model.complete([{"role": "user", "content": "Reply with the word: pong"}])
    print(f"provider:  {settings.llm_provider}")
    print(f"model:     {response.model}")
    print(f"reply:     {response.text.strip()}")
    print(f"tokens:    {response.prompt_tokens} in / {response.completion_tokens} out")
    print(f"latency:   {response.latency_ms:.0f} ms")
