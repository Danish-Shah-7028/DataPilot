from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from datapilot.llm.provider import ChatMessage, LLMError, LLMResponse

Responder = Callable[[Sequence[ChatMessage]], str]


@dataclass
class ScriptedChatModel:
    """A stand-in chat model that replays canned replies.

    Tests use it to drive the agents without network access. Pass either a list of
    replies (consumed in order) or a function that picks a reply from the messages.
    """

    replies: list[str] | Responder
    model: str = "scripted"
    calls: list[list[ChatMessage]] = field(default_factory=list)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if callable(self.replies):
            text = self.replies(messages)
        elif self.replies:
            text = self.replies.pop(0)
        else:
            raise LLMError("ScriptedChatModel ran out of replies")

        prompt_chars = sum(len(message["content"]) for message in messages)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=prompt_chars // 4,
            completion_tokens=len(text) // 4,
            latency_ms=1.0,
        )
